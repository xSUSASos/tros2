"""Контур управления.

Позиционный режим у привода T3D доступен только с импульсного входа, поэтому
контур положения замыкает хост: каждый цикл читаются позиции и моменты всех
осей, считается положение платформы, и на приводы уходят уставки скорости.

Слежение идёт ПО ДЛИНАМ ТРОСОВ, а не по декартовым координатам. Так надёжнее:
длина каждого троса измеряется напрямую своим энкодером, а положение платформы
— величина вычисленная, и её ошибка вошла бы в контур как шум. Декартовы
координаты нужны только для того, чтобы задать цель.

Натяжение правится отдельным слагаемым и только вдоль нуль-пространства
структурной матрицы: это единственное изменение длин, которое меняет натяжение,
не сдвигая платформу. Для четырёх тросов такое направление одно — перетяжка
диагональных пар.
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
        self._listeners: list[Any] = []
        self.box_low, self.box_high = box_limits(machine, self.kinematics)
        # Для прямой задачи границы шире рабочей зоны: она должна уметь
        # показать, что платформа выехала за пределы, а не упереться в них.
        # Сверху жёстко ограничиваем плоскостью якорей — выше платформа
        # оказаться не может, а зеркальное решение лежит именно там.
        anchors = self.kinematics.anchors
        reach = float(np.max(anchors.max(axis=0) - anchors.min(axis=0)))
        self.fk_low = anchors.min(axis=0) - 0.5 * reach
        self.fk_high = anchors.max(axis=0) + 0.5 * reach
        # Якоря обычно лежат в одной плоскости, поэтому вертикальный размах
        # сам по себе нулевой — границы по высоте задаются отдельно.
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
        """Аварийный стоп. Уставки обнуляются немедленно, не дожидаясь цикла."""
        self.safety.trigger_estop(reason)
        try:
            self.drives.stop()
            self.drives.enable(False)
        except Exception as exc:  # noqa: BLE001 — стоп обязан пройти до конца
            log.error("при аварийном стопе: %s", exc)
        log.warning("АВАРИЙНЫЙ СТОП: %s", reason)

    def clear_estop(self) -> None:
        self.safety.clear_estop()
        self.set_mode(IdleMode())

    def enable(self, on: bool) -> None:
        if on and self.safety.estop:
            raise RuntimeError("сначала снимите аварийный стоп")
        self.drives.enable(on)
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
        st.enabled = self.drives.axes[0].state.enabled if hasattr(self.drives, "axes") else st.enabled

        counts = np.array([s.position_counts for s in states])
        tensions = self._tensions_from_states(states, counts)
        st.tensions_n = tensions
        st.tension_min_n = float(tensions.min()) if tensions.size else 0.0
        st.tension_max_n = float(tensions.max()) if tensions.size else 0.0

        lengths = self._lengths_from_counts(counts, tensions)
        st.lengths_mm = lengths

        pose, residual = self._solve_pose(lengths)
        st.pose_mm = pose
        st.fk_residual_mm = residual
        st.homed = self.machine.is_calibrated

        verdict = self.safety.check(
            states=states, pose=pose, tensions=tensions,
            fk_residual_mm=residual, moving=not isinstance(mode, IdleMode),
        )
        st.health = verdict.health
        st.messages = verdict.reasons

        if verdict.stop:
            self.drives.stop()
            if verdict.disable and self.state.enabled:
                self.drives.enable(False)
                self.state.enabled = False
            st.commands_rpm = np.zeros(self.drives.n_axes)
            self._notify(st)
            return st

        output = mode.update(self, dt)
        if output.message:
            st.messages.append(output.message)
        rpm = self._compute_commands(output, pose, lengths, tensions, dt)
        st.commands_rpm = rpm
        self.drives.set_speeds(rpm)

        if output.done:
            self.set_mode(IdleMode())
        self._notify(st)
        return st

    def _switch_mode(self, mode: Mode) -> None:
        if mode.requires_homing and not self.machine.is_calibrated:
            log.warning(
                "режим %s требует калибровки: без неё отсчёт энкодера "
                "не перевести в длину троса", mode.name.value,
            )
        try:
            self._mode.exit(self)
        except Exception as exc:  # noqa: BLE001
            log.error("при выходе из режима %s: %s", self._mode.name.value, exc)
        self._mode = mode
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
            except Exception:  # noqa: BLE001 — до калибровки радиус берём первого слоя
                radius = winch.first_layer_radius_mm
            out[i] = abs(winch.torque_percent_to_force(state.torque_percent, radius))
        return out

    def _lengths_from_counts(self, counts: np.ndarray, tensions: np.ndarray) -> np.ndarray | None:
        """Отсчёты энкодеров -> расстояния от схода до платформы.

        Свободная длина берётся по послойной модели намотки, затем к ней
        добавляется вытяжка под текущим натяжением — геометрия платформы
        «видит» именно растянутый трос.
        """
        if not self.machine.is_calibrated:
            return None
        out = np.zeros(len(counts))
        for i, line in enumerate(self.lines):
            free = line.length_from_counts(int(counts[i]))
            out[i] = line.stretched(free, float(tensions[i]))
        return out

    def _solve_pose(self, lengths: np.ndarray | None) -> tuple[np.ndarray | None, float]:
        if lengths is None:
            return None, 0.0
        if self._last_pose is not None:
            guess = self._last_pose
        else:
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
        lengths: np.ndarray | None,
        tensions: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        n = self.drives.n_axes

        if output.cable_velocity_mms is not None:
            # Режим правит тросы напрямую (выборка слабины, калибровка) —
            # вмешиваться в это регулировкой натяжения нельзя.
            return self._to_rpm(np.asarray(output.cable_velocity_mms, dtype=float), lengths)

        if output.hold or output.target_pose is None or pose is None or lengths is None:
            winding = np.zeros(n)
        else:
            winding = self._track_pose(output, pose, lengths)

        # Натяжение держится всегда, в том числе в ожидании: стоять на месте
        # и при этом дать тросам провиснуть — значит потерять управляемость
        # ровно в тот момент, когда её труднее всего вернуть.
        if pose is not None and lengths is not None:
            winding = winding + self._tension_correction(pose, tensions)

        return self._to_rpm(winding, lengths)

    def _track_pose(self, output: ModeOutput, pose: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Слежение по длинам тросов за целевым положением."""
        target = np.asarray(output.target_pose, dtype=float)
        target = np.clip(target, self.box_low, self.box_high)
        self._target_pose = target
        self.state.target_mm = target

        target_lengths = self.kinematics.inverse(target)
        self.state.target_lengths_mm = target_lengths

        feed = output.feed_mms if output.feed_mms is not None else self.machine.motion.max_velocity_mms
        feed = min(feed, self.machine.motion.max_velocity_mms)

        # желаемая скорость платформы: пропорционально ошибке, с ограничением
        error = target - pose
        distance = float(np.linalg.norm(error))
        if distance < 1e-6:
            velocity = np.zeros(3)
        else:
            speed = min(feed, self.machine.control.position_kp * distance)
            velocity = error / distance * speed

        feedforward = self.kinematics.winding_rates(pose, velocity)

        # Поправка по длинам держит точность, потому что длина измеряется
        # напрямую, а положение платформы — величина вычисленная.
        #
        # Но задать все четыре длины из обратной задачи нельзя: степеней
        # свободы три, и четвёртая длина не свободна — именно она отвечает за
        # внутреннее натяжение. Если править её наравне с остальными, контур
        # положения переопределит систему и погасит регулировку натяжения.
        # Поэтому ошибка длин проецируется на то подпространство, которое
        # соответствует движению платформы; оставшееся направление отдано
        # контуру натяжения, и они не мешают друг другу.
        error = lengths - target_lengths
        projected = self._project_to_motion(pose, error)
        return feedforward + self.machine.control.position_kp * projected

    def _project_to_motion(self, pose: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Оставляет только ту часть вектора длин, которая двигает платформу."""
        try:
            basis, _ = np.linalg.qr(self.kinematics.jacobian(pose))
        except np.linalg.LinAlgError:
            return vector
        return basis @ (basis.T @ vector)

    def _tension_correction(self, pose: np.ndarray, tensions: np.ndarray) -> np.ndarray:
        """Поправка натяжения вдоль нуль-пространства структурной матрицы.

        Только это направление меняет натяжение, не сдвигая платформу.
        Всё остальное двигало бы её вместо того, чтобы подтянуть тросы.
        """
        if tensions.size != self.drives.n_axes:
            return np.zeros(self.drives.n_axes)
        try:
            W = self.kinematics.structure_matrix(pose)
        except Exception:  # noqa: BLE001
            return np.zeros(self.drives.n_axes)

        wrench = T.gravity_wrench(self.machine.platform.mass_kg)
        desired = T.distribute(
            W, wrench,
            f_min=self.machine.tension.min_n,
            f_max=self.machine.tension.max_n,
            f_target=self._target_tension_n,
            # Предпочитаем прошлое решение, а не текущие измерения: при
            # симметричном состоянии оба варианта равноудалены от измерений,
            # и без памяти выбор скакал бы каждый цикл.
            f_prefer=self._last_desired if self._last_desired is not None else tensions,
        )
        if desired.feasible:
            self._last_desired = desired.forces.copy()
        self.state.target_tensions_n = desired.forces
        self.state.margin_n = desired.margin_n
        if not desired.feasible:
            return np.zeros(self.drives.n_axes)

        basis = T.null_space(W)
        if basis.shape[1] == 0:
            return np.zeros(self.drives.n_axes)
        direction = basis[:, 0] / np.linalg.norm(basis[:, 0])
        error = float(direction @ (desired.forces - tensions))
        if abs(error) < self.machine.control.tension_deadband_n:
            return np.zeros(self.drives.n_axes)
        return self.machine.control.tension_kp * error * direction

    def _to_rpm(self, winding_mms: np.ndarray, lengths: np.ndarray | None) -> np.ndarray:
        """Скорости выборки троса -> уставки моторов, с ограничением по оборотам."""
        rpm = np.zeros(len(winding_mms))
        for i, (line, winch) in enumerate(zip(self.lines, self.winches, strict=True)):
            speed = float(winding_mms[i])
            try:
                count = line.counts_from_length(float(lengths[i])) if lengths is not None else 0
                value = line.rpm_for_line_speed(speed, count)
            except Exception:  # noqa: BLE001 — до калибровки считаем по первому слою
                radius = winch.first_layer_radius_mm
                value = winch.direction * speed * 60.0 / (2 * np.pi * radius) * winch.gear_ratio
            rpm[i] = float(np.clip(value, -winch.max_rpm, winch.max_rpm))
        return rpm

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        return (
            f"машина {self.machine.name!r}: {self.drives.n_axes} тросов, "
            f"цикл {self.machine.control.loop_hz:.0f} Гц, режим {self._mode.name.value}, "
            f"{'откалибрована' if self.machine.is_calibrated else 'НЕ откалибрована'}"
        )
