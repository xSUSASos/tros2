"""Контур управления.

Позиционный режим у привода T3D доступен только с импульсного входа, поэтому
контур положения замыкает хост: каждый цикл читаются позиции и моменты всех
осей, считается положение платформы, и на приводы уходят уставки скорости.

Слежение идёт ПО ДЛИНАМ ТРОСОВ, а не по декартовым координатам. Так надёжнее:
длина каждого троса измеряется напрямую своим энкодером, а положение платформы
— величина вычисленная, и её ошибка вошла бы в контур как шум. Декартовы
координаты нужны только для того, чтобы задать цель.

Отдельного регулятора натяжения здесь нет, и это не упущение. При четырёх
тросах и трёх координатах общий уровень натяжения — не свободный параметр:
его однозначно задаёт равновесие, а значит геометрия. Выбирается он высотой
рабочей плоскости, а не алгоритмом. Роль, которую в жёстких системах играл бы
преднатяг, здесь выполняет поправка на вытяжку: трос под нагрузкой длиннее
свободного, и стравить надо ровно на эту разницу меньше.

Различаются две длины, и путать их нельзя:

    свободная    сколько троса стравлено с барабана — это меряет энкодер
    геометрическая  расстояние от схода до платформы — это видит геометрия

Связь между ними — вытяжка под текущим натяжением. При капроне ⌀0.3 мм она
доходит до сотни миллиметров, то есть это не поправка второго порядка, а
главный член.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from cdpr import tension as T
from cdpr.config import DriveProfile, MachineConfig
from cdpr.kinematics import CDPRKinematics
from cdpr.line import LineModel, build_line_models
from cdpr.modes.base import IdleMode, Mode, ModeOutput
from cdpr.safety import SafetyMonitor
from cdpr.state import Health, MachineState, ModeName
from cdpr.workspace import box_limits
from drives.base import DriveGroup

log = logging.getLogger(__name__)


class Controller:
    """Цикл управления и переключение режимов."""

    def __init__(
        self,
        machine: MachineConfig,
        profile: DriveProfile,
        drives: DriveGroup,
    ) -> None:
        self.machine = machine
        self.profile = profile
        self.drives = drives
        self.kinematics = CDPRKinematics.from_config(machine)
        self.lines: list[LineModel] = build_line_models(machine.ordered_winches())
        self.winches = machine.ordered_winches()
        self.safety = SafetyMonitor(machine)

        self.state = MachineState()
        self.state.online = [False] * drives.n_axes
        self.state.alarms = [0] * drives.n_axes

        self._mode: Mode = IdleMode()
        self._pending_mode: Mode | None = None
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._last_pose: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._target_tension_n = machine.tension.target_n
        self._last_desired: np.ndarray | None = None
        self._tension_filtered: np.ndarray | None = None
        self._setpoint: np.ndarray | None = None
        self._setpoint_speed = 0.0
        self._hold_target: np.ndarray | None = None
        self._listeners: list[Any] = []
        self._last_contact = time.perf_counter()
        self.box_low, self.box_high = box_limits(machine, self.kinematics)

        # Границы для прямой задачи в пространственном случае: они должны быть
        # шире рабочей зоны, чтобы задача умела показать выезд за пределы, но
        # обязаны отсекать зеркальное решение выше плоскости якорей.
        # На плоской машине они не нужны — там решение замкнутое и единственное.
        anchors = self.kinematics.anchors
        reach = float(np.max(anchors.max(axis=0) - anchors.min(axis=0)))
        self.fk_low = anchors.min(axis=0) - 0.5 * reach
        self.fk_high = anchors.max(axis=0) + 0.5 * reach
        self.fk_high[2] = float(anchors[:, 2].min()) - 1.0
        self.fk_low[2] = min(machine.workspace.z_min_mm, self.fk_high[2] - reach)

    # ------------------------------------------------------------------ #
    #  Свойства и управление извне
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def target_tension_n(self) -> float:
        return self._target_tension_n

    def set_target_tension(self, value: float) -> None:
        limits = self.machine.tension
        self._target_tension_n = float(np.clip(value, limits.min_n, limits.max_n))

    def set_mode(self, mode: Mode) -> None:
        """Переключение происходит в начале цикла, а не посреди расчёта."""
        with self._lock:
            self._pending_mode = mode

    def estop(self, reason: str = "кнопка в панели") -> None:
        """Программный стоп: уставки в ноль немедленно, не дожидаясь цикла.

        Настоящий аварийный стоп на этой машине — физическая кнопка, снимающая
        питание с приводов. Софт им не управляет и не притворяется, что может.
        Абсолютный энкодер переживает выключение, поэтому после обратного
        включения отсчёты остаются верными и привязка не теряется.
        """
        self.safety.trigger_estop(reason)
        try:
            self.drives.stop()
        except Exception as exc:  # noqa: BLE001 — стоп обязан пройти до конца
            log.error("при аварийном стопе: %s", exc)
        log.warning("СТОП: %s", reason)

    def clear_estop(self) -> None:
        self.safety.clear_estop()
        self.set_mode(IdleMode())

    def allow_motion(self, on: bool) -> None:
        """Программное разрешение движения.

        Это не SON и не силовое разрешение — приводы включены всегда. Это
        предохранитель от случайной команды: пока он снят, на шину уходят
        нули, что бы ни насчитал режим.
        """
        if on and self.safety.estop:
            raise RuntimeError("сначала снимите аварийный стоп")
        if not on:
            self.drives.stop()
        self.state.enabled = on

    def add_listener(self, callback) -> None:
        """Колбэк на каждый цикл — так панель получает телеметрию."""
        self._listeners.append(callback)

    # ------------------------------------------------------------------ #
    #  Жизненный цикл потока
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="cdpr-control", daemon=True)
        self._thread.start()
        log.info("контур управления запущен на %.0f Гц", self.machine.control.loop_hz)

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.drives.stop()
        except Exception as exc:  # noqa: BLE001
            log.error("при останове: %s", exc)

    def _loop(self) -> None:
        period = self.machine.control.dt
        next_tick = time.perf_counter()
        previous = next_tick
        while self._running.is_set():
            now = time.perf_counter()
            dt = now - previous
            previous = now
            try:
                self.cycle(dt)
            except Exception as exc:  # noqa: BLE001 — цикл не имеет права падать
                log.exception("сбой в цикле управления: %s", exc)
                self.estop(f"внутренняя ошибка: {exc}")
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # не успели за период — не копим долг, иначе цикл пойдёт вразнос
                next_tick = time.perf_counter()

    # ------------------------------------------------------------------ #
    #  Один цикл
    # ------------------------------------------------------------------ #
    def cycle(self, dt: float) -> MachineState:
        with self._lock:
            if self._pending_mode is not None:
                self._switch_mode(self._pending_mode)
                self._pending_mode = None
            mode = self._mode

        states = self.drives.read_states()
        st = self.state
        st.cycle += 1
        st.stamp = time.time()
        st.loop_hz_actual = 1.0 / dt if dt > 0 else 0.0
        st.mode = mode.name
        st.online = [s.online for s in states]
        st.alarms = [s.alarm for s in states]
        st.speeds_rpm = np.array([s.speed_rpm for s in states])
        st.estop = self.safety.estop

        counts = np.array([s.position_counts for s in states])
        tensions = self._tensions_from_states(states, counts)
        st.tensions_n = tensions
        smooth = self._smooth_tensions(tensions, dt)
        st.tension_min_n = float(tensions.min()) if tensions.size else 0.0
        st.tension_max_n = float(tensions.max()) if tensions.size else 0.0

        free = self._free_lengths(counts)
        geometric = self._geometric_lengths(free, smooth)
        st.free_lengths_mm = free
        st.lengths_mm = geometric

        pose, residual = self._solve_pose(geometric)
        st.pose_mm = pose
        st.fk_residual_mm = residual
        st.homed = self.machine.is_calibrated

        # Сторожевой таймер шины: если ни одна ось не отозвалась дольше
        # положенного, дальше считать не по чему, и продолжать выдавать
        # прежние уставки — худшее из возможного.
        if any(s.online for s in states):
            self._last_contact = time.perf_counter()
        silence_ms = (time.perf_counter() - self._last_contact) * 1000.0
        bus_dead = silence_ms > self.machine.control.watchdog_ms

        verdict = self.safety.check(
            states=states, pose=pose, tensions=tensions,
            fk_residual_mm=residual, moving=not isinstance(mode, IdleMode),
            guard_model=not mode.tolerates_slack,
            tension_ceiling_n=mode.tension_ceiling_n,
        )
        st.health = verdict.health
        st.messages = list(verdict.reasons)

        if bus_dead:
            st.health = Health.FAULT
            st.messages.append(
                f"шина молчит {silence_ms:.0f} мс при пределе "
                f"{self.machine.control.watchdog_ms:.0f} мс — уставки обнулены"
            )

        if verdict.stop or bus_dead or not st.enabled:
            self.drives.stop()
            st.commands_rpm = np.zeros(self.drives.n_axes)
            if not st.enabled and not verdict.stop and not bus_dead:
                st.messages.append("движение не разрешено")
            # Остановка обязана ЗАВЕРШИТЬ операцию, а не подвесить её. Пока
            # машина стоит, mode.update не вызывается — значит режим не
            # отсчитывает своё время, не видит таймаута и не может закончиться
            # сам. Оператор в такой ситуации смотрит на замерший экран и не
            # понимает, идёт что-то или нет.
            if (verdict.stop or bus_dead) and not isinstance(mode, IdleMode):
                log.warning("режим %s прерван: %s", mode.name.value,
                            "; ".join(st.messages[:2]) or "остановка")
                self.set_mode(IdleMode())
            self._notify(st)
            return st

        output = mode.update(self, dt)
        if output.message:
            st.messages.append(output.message)
        rpm = self._compute_commands(output, pose, free, geometric, smooth, dt)
        st.commands_rpm = rpm
        self.drives.set_speeds(rpm)

        if output.done:
            self.set_mode(IdleMode())
        self._notify(st)
        return st

    def _switch_mode(self, mode: Mode) -> None:
        if mode.requires_homing and not self.machine.is_calibrated:
            log.warning(
                "режим %s требует привязки: без неё отсчёт энкодера "
                "не перевести в длину троса", mode.name.value,
            )
        try:
            self._mode.exit(self)
        except Exception as exc:  # noqa: BLE001
            log.error("при выходе из режима %s: %s", self._mode.name.value, exc)
        self._mode = mode
        self._setpoint = None
        self._setpoint_speed = 0.0
        self._hold_target = None
        mode.enter(self)
        log.info("режим: %s", mode.describe())

    def _notify(self, state: MachineState) -> None:
        for callback in self._listeners:
            try:
                callback(state)
            except Exception as exc:  # noqa: BLE001 — слушатель не роняет цикл
                log.debug("слушатель телеметрии упал: %s", exc)

    # ------------------------------------------------------------------ #
    #  Измерения
    # ------------------------------------------------------------------ #
    def _tensions_from_states(self, states, counts: np.ndarray) -> np.ndarray:
        """Натяжения из моментов моторов с учётом текущего радиуса намотки."""
        out = np.zeros(len(states))
        for i, (state, winch, line) in enumerate(zip(states, self.winches, self.lines, strict=True)):
            try:
                radius = line.radius_at(line.turns_at(int(counts[i])))
            except Exception:  # noqa: BLE001 — до привязки радиус берём первого слоя
                radius = winch.first_layer_radius_mm
            out[i] = abs(winch.torque_percent_to_force(state.torque_percent, radius))
        return out

    def _smooth_tensions(self, tensions: np.ndarray, dt: float) -> np.ndarray:
        """Сглаженное натяжение — только для расчёта вытяжки.

        Защиты смотрят на сырое значение: там важно не проспать всплеск. А вот
        в длину этот шум переходить не должен: момент квантован целыми
        процентами, и при мягком тросе один процент — это больше сантиметра
        кажущейся длины.
        """
        tau = self.machine.control.tension_filter_s
        if self._tension_filtered is None or self._tension_filtered.shape != tensions.shape:
            self._tension_filtered = tensions.copy()
        elif tau > 0 and dt > 0:
            self._tension_filtered += min(1.0, dt / tau) * (tensions - self._tension_filtered)
        else:
            self._tension_filtered = tensions.copy()
        return self._tension_filtered

    def _ea_or_inf(self) -> np.ndarray:
        """Жёсткость тросов; для незаданной — бесконечность, то есть без поправки."""
        return np.array([w.ea_n if w.ea_n else np.inf for w in self.winches], dtype=float)

    def _free_lengths(self, counts: np.ndarray) -> np.ndarray | None:
        """Отсчёты энкодеров -> сколько троса стравлено с барабана.

        Это ровно та величина, которой управляет мотор, и именно по ней идёт
        слежение. Расстояние до платформы отсюда получается прибавлением
        вытяжки — см. `_geometric_lengths`.
        """
        if not self.machine.is_calibrated:
            return None
        return np.array([
            line.length_from_counts(int(counts[i])) for i, line in enumerate(self.lines)
        ])

    def _geometric_lengths(
        self, free: np.ndarray | None, tensions: np.ndarray
    ) -> np.ndarray | None:
        """Свободная длина + вытяжка под нагрузкой = расстояние до платформы.

        Геометрия «видит» именно растянутый трос, поэтому прямую задачу надо
        решать по этим длинам, а не по показаниям энкодера напрямую.
        """
        if free is None:
            return None
        return np.array([
            line.stretched(float(free[i]), float(tensions[i]))
            for i, line in enumerate(self.lines)
        ])

    def _solve_pose(self, lengths: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        if lengths is None:
            return None, 0.0
        if self.machine.geometry.is_planar:
            # Высота известна заранее, поэтому решение замкнутое: ни итераций,
            # ни начального приближения, ни зеркального решения.
            pose, residual = self.kinematics.forward_planar(
                lengths, self.machine.geometry.plane_z_mm
            )
        else:
            guess = self._last_pose
            if guess is None:
                guess = 0.5 * (self.box_low + self.box_high)
            pose, residual = self.kinematics.forward(
                lengths, guess=guess, bounds=(self.fk_low, self.fk_high)
            )
        self._last_pose = pose
        return pose, residual

    # ------------------------------------------------------------------ #
    #  Формирование команд
    # ------------------------------------------------------------------ #
    def _compute_commands(
        self,
        output: ModeOutput,
        pose: np.ndarray | None,
        free: np.ndarray | None,
        geometric: np.ndarray | None,
        tensions: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        n = self.drives.n_axes

        if output.cable_velocity_mms is not None:
            # Режим правит тросы напрямую (выборка слабины, хоминг) — там
            # положение платформы либо неизвестно, либо неважно.
            return self._to_rpm(np.asarray(output.cable_velocity_mms, dtype=float), free)

        if pose is None or free is None or geometric is None:
            return np.zeros(n)

        # Ожидание — это не «ничего не делать»: держать надо и положение, и
        # натяжение. Цель при этом просто равна текущему положению, и общий
        # закон управления работает без исключений.
        hold = output.hold or output.target_pose is None
        winding = self._track_pose(output, pose, geometric, tensions, dt, hold=hold)
        return self._to_rpm(winding, free)

    def _track_pose(
        self, output: ModeOutput, pose: np.ndarray, geometric: np.ndarray,
        tensions: np.ndarray, dt: float, *, hold: bool,
    ) -> np.ndarray:
        """Слежение за целевым положением по длинам тросов.

        Ошибка считается в ГЕОМЕТРИЧЕСКИХ длинах — расстояниях до якоря, — а не
        в свободных, которые меряет энкодер. Разница принципиальная. Свободная
        длина связана с геометрической через натяжение; если сравнивать
        измеренную свободную с желаемой свободной, натяжение входит в ошибку
        дважды и с разных сторон. Получается алгебраическая петля: контур
        начинает гоняться за собственным хвостом и уводит платформу вместо
        того, чтобы её удерживать. В геометрических длинах эти члены
        сокращаются, и остаётся честная ошибка положения.
        """
        if hold:
            # Точка удержания запоминается один раз. Брать текущее положение
            # каждый цикл нельзя: тогда любой снос немедленно становится новой
            # целью, восстанавливающей силы не остаётся вовсе, и платформа
            # медленно уползает — тем быстрее, чем шумнее измерения.
            if self._hold_target is None:
                self._hold_target = pose.copy()
            target = self._hold_target
        else:
            self._hold_target = None
            target = np.clip(np.asarray(output.target_pose, dtype=float), self.box_low, self.box_high)
        self._target_pose = target
        self.state.target_mm = target

        # ── ползущая уставка ─────────────────────────────────────────────
        # Между текущим положением и целью движется отдельная точка — она и
        # есть то, за чем следит контур. Без неё поправка по длине оказалась бы
        # пропорциональна всему расстоянию до цели и перебивала бы подачу:
        # команда «ехать 20 мм/с» на дальнюю точку выливалась бы в рывок на
        # предельных оборотах. С уставкой подача означает ровно то, что
        # написано, а поправка остаётся маленькой — она правит только ошибку
        # слежения, а не всю дистанцию.
        setpoint, velocity = self._advance_setpoint(pose, target, output, dt, hold=hold)
        target_geometric = self.kinematics.inverse(setpoint)
        self.state.target_lengths_mm = target_geometric
        self.state.target_tensions_n = self._equilibrium_tensions(setpoint)

        distance = float(np.linalg.norm(target - pose))
        self.state.arrived = bool(hold or distance <= self.machine.control.arrival_tolerance_mm)

        # ── прямая связь: скорость уставки -> скорости тросов ────────────
        # Все четыре команды выведены из одного вектора скорости, поэтому в
        # движении они согласованы по построению и тросы не тянут друг против
        # друга, что бы ни было с моделью.
        feedforward = self.kinematics.winding_rates(pose, velocity)

        # ── поправка по длинам ───────────────────────────────────────────
        # Длина измеряется энкодером напрямую, поэтому именно она убирает
        # накопленную ошибку. Она же подбирает провисший трос: у него
        # расстояние до якоря выходит больше, чем требует геометрия.
        error = geometric - target_geometric
        deadband = self.machine.control.length_deadband_mm
        if deadband > 0:
            error = np.where(np.abs(error) < deadband, 0.0, error)
        # Чтобы изменить расстояние до якоря на dx, стравить надо чуть меньше:
        # под нагрузкой трос длиннее свободного. Множитель близок к единице и
        # на устойчивость не влияет, но пусть будет верным.
        scale = 1.0 / (1.0 + np.asarray(tensions, dtype=float) / self._ea_or_inf())
        correction = self.machine.control.position_kp * error * scale
        return feedforward + correction

    def _advance_setpoint(
        self, pose: np.ndarray, target: np.ndarray, output: ModeOutput,
        dt: float, *, hold: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Двигает уставку к цели с заданной подачей и ограничением разгона.

        Возвращает новое положение уставки и её скорость. Разгон ограничен не
        ради красоты профиля: на мягком тросе рывок раскачивает коробку, и
        успокаивается она долго.
        """
        n_dim = len(target)
        if self._setpoint is None:
            self._setpoint = pose.copy()
            self._setpoint_speed = 0.0
        if hold:
            self._setpoint = target.copy()
            self._setpoint_speed = 0.0
            return self._setpoint, np.zeros(n_dim)

        limit = self.machine.motion.max_velocity_mms
        feed = min(output.feed_mms if output.feed_mms is not None else limit, limit)

        delta = target - self._setpoint
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            self._setpoint_speed = 0.0
            return self._setpoint, np.zeros(n_dim)

        accel = self.machine.motion.max_acceleration_mms2
        # Тормозить надо заранее: с этой скорости до нуля нужно v^2/(2a).
        stopping = min(feed, float(np.sqrt(2.0 * accel * distance)))
        step = accel * dt
        self._setpoint_speed = float(np.clip(stopping, self._setpoint_speed - step,
                                             self._setpoint_speed + step))

        direction = delta / distance
        travel = self._setpoint_speed * dt
        if travel >= distance:
            self._setpoint = target.copy()
            return self._setpoint, direction * (distance / dt if dt > 0 else 0.0)
        self._setpoint = self._setpoint + direction * travel
        return self._setpoint, direction * self._setpoint_speed

    def _equilibrium_tensions(self, pose: np.ndarray) -> np.ndarray:
        """Натяжения, при которых платформа в этой позе стоит в равновесии.

        Используется как модель, а не как регулятор: по ним считается вытяжка
        троса. Регулировать натяжение отдельно в такой машине нельзя — при
        четырёх тросах и трёх координатах равновесие задаёт его однозначно.
        """
        n = self.drives.n_axes
        fallback = np.full(n, self._target_tension_n)
        try:
            W = self.kinematics.structure_matrix(pose)
        except Exception:  # noqa: BLE001 — вырожденная поза, дальше не считаем
            return fallback

        wrench = T.gravity_wrench(self.machine.platform.mass_kg)
        solution = T.distribute(
            W, wrench,
            f_min=self.machine.tension.min_n,
            f_max=self.machine.tension.max_n,
            f_target=self._target_tension_n,
            # Предпочитаем прошлое решение, а не текущие измерения: при
            # симметричном состоянии оба варианта равноудалены от измерений,
            # и без памяти выбор скакал бы каждый цикл.
            f_prefer=self._last_desired,
        )
        self.state.margin_n = solution.margin_n
        if not solution.feasible:
            return fallback
        self._last_desired = solution.forces.copy()
        return solution.forces

    def _to_rpm(self, winding_mms: np.ndarray, free: np.ndarray | None) -> np.ndarray:
        """Скорости выборки троса -> уставки моторов, с ограничением по оборотам."""
        rpm = np.zeros(len(winding_mms))
        for i, (line, winch) in enumerate(zip(self.lines, self.winches, strict=True)):
            speed = float(winding_mms[i])
            try:
                count = line.counts_from_length(float(free[i])) if free is not None else 0
                value = line.rpm_for_line_speed(speed, count)
            except Exception:  # noqa: BLE001 — до привязки считаем по первому слою
                radius = winch.first_layer_radius_mm
                value = winch.direction * speed * 60.0 / (2 * np.pi * radius) * winch.gear_ratio
            rpm[i] = float(np.clip(value, -winch.max_rpm, winch.max_rpm))
        return rpm

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        shape = "плоская" if self.machine.geometry.is_planar else "пространственная"
        return (
            f"машина {self.machine.name!r}: {self.drives.n_axes} тросов, {shape}, "
            f"цикл {self.machine.control.loop_hz:.0f} Гц, режим {self._mode.name.value}, "
            f"{'привязана' if self.machine.is_calibrated else 'НЕ привязана'}"
        )
