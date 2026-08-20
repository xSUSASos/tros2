"""Симулятор приводов T3D.

Даёт разрабатывать и прогонять весь верхний уровень — кинематику, контур
управления, режимы, G-code, панель — не выходя к железу. Модель намеренно
воспроизводит не только «мотор крутится», но и то, что мешает в реальности:
задержку транзакции, квантование уставки целыми об/мин, ограничение по
разгону, шум измерения момента и, по желанию, потери кадров на линии.

Карта регистров у симулятора известна полностью (в отличие от настоящего
привода, где её ещё предстоит снять), поэтому `make_sim_profile` возвращает
копию профиля с проставленными адресами.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Sequence

from cdpr.config import ConfigError, DriveProfile
from drives.base import BusTimeout, CrcError, ModbusException, Transport, TransportError

log = logging.getLogger(__name__)

class VirtualClock:
    """Часы, которыми управляет вызывающий.

    Модель по умолчанию интегрируется по реальному времени — так она ведёт
    себя как железо. Но в тестах это делает результат зависящим от того,
    насколько быстро сегодня работает машина, поэтому там время двигают
    вручную: один шаг цикла — один шаг модели.
    """

    __slots__ = ("now",)

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> float:
        self.now += float(dt)
        return self.now


#: куда симулятор кладёт параметры и мониторы
SIM_PARAM_BASE = 0x0000
SIM_MONITOR_BASE = 0x1000


def make_sim_profile(profile: DriveProfile) -> DriveProfile:
    """Копия профиля с полностью проставленными адресами — для симулятора."""
    sim = profile.model_copy(deep=True)
    sim.name = f"{profile.name} (симулятор)"
    sim.addressing.param_base = SIM_PARAM_BASE
    sim.addressing.param_ram_base = SIM_PARAM_BASE
    sim.addressing.monitor_base = SIM_MONITOR_BASE
    sim.addressing.monitor_function = sim.function_codes.read_holding
    sim.addressing.word_order = "lo_hi"
    sim.eeprom_safe = True

    addr = SIM_MONITOR_BASE
    for spec in sorted(sim.monitors.values(), key=lambda s: (s.order is None, s.order)):
        spec.address = addr
        addr += spec.words
    for spec in sim.params.values():
        # Параметру с неизвестным номером P-xxx адрес не выдаётся: такого
        # регистра у привода может не быть вовсе, и модель не имеет права
        # придумывать возможность, которой на железе не окажется.
        spec.address = None if spec.p is None else SIM_PARAM_BASE + spec.p
    return sim


# --------------------------------------------------------------------------- #
#  Модель одной оси
# --------------------------------------------------------------------------- #
class SimAxis:
    """Мотор с ограничением разгона и абсолютным энкодером.

    Внешняя нагрузка задаётся колбэком, чтобы сюда можно было подключить
    физику платформы (натяжение тросов), не переписывая транспорт.
    """

    def __init__(self, slave: int, counts_per_rev: int, max_rpm: float,
                 clock=time.perf_counter) -> None:
        self.slave = slave
        self.counts_per_rev = counts_per_rev
        self.max_rpm = max_rpm

        self.position_counts: float = 0.0
        self.command_counts: int = 0   # приходит импульсным входом, у нас не используется
        self.speed_rpm: float = 0.0
        self.target_rpm: float = 0.0
        self.torque_percent: float = 0.0
        self.alarm: int = 0
        self.enabled: bool = False

        # разгон как у привода: P-060 — мс на 1000 об/мин
        self.accel_time_ms: float = 30.0
        self.params: dict[int, int] = {}
        self._t = clock()

    def step(self, now: float) -> None:
        """Двигает вал. Нагрузка считается отдельно, когда все оси уже
        сдвинулись: иначе модель платформы видела бы полуобновлённое
        состояние, и натяжения «перескакивали» бы между тросами."""
        dt = now - self._t
        if dt <= 0:
            return
        self._t = now

        target = self.target_rpm if self.enabled and not self.alarm else 0.0
        target = max(-self.max_rpm, min(self.max_rpm, target))

        rate = 1000.0 / max(self.accel_time_ms, 1.0) * 1000.0  # об/мин в секунду
        delta = target - self.speed_rpm
        step = rate * dt
        self.speed_rpm += max(-step, min(step, delta))

        self.position_counts += self.speed_rpm / 60.0 * dt * self.counts_per_rev


#: колбэк нагрузки: (ось, позиция в импульсах, скорость об/мин) -> момент в %
LoadModel = Callable[[int, float, float], float]


def viscous_load(axis: int, position_counts: float, speed_rpm: float) -> float:
    """Нагрузка по умолчанию: сухое трение плюс вязкое.

    Это не физика тросовой системы, а разумная заглушка, чтобы момент не был
    нулевым до того, как подключится модель платформы (`cdpr/simulation.py`).
    """
    return 4.0 * (1.0 if speed_rpm >= 0 else -1.0) + 0.01 * speed_rpm


class SimTransport(Transport):
    """Шина с виртуальными приводами.

    Полностью заменяет ModbusRtuTransport: тот же интерфейс, поэтому верхние
    слои не знают, работают они с железом или с моделью.
    """

    def __init__(
        self,
        name: str,
        profile: DriveProfile,
        slaves: Sequence[int],
        *,
        counts_per_rev: int = 8_388_608,
        max_rpm: float = 3000.0,
        latency_ms: float = 2.0,
        write_latency_ms: float | None = None,
        busy_on_write: bool = False,
        loss_rate: float = 0.0,
        seed: int | None = 0,
        clock=None,
        baudrate: int = 115200,
        parity: str = "E",
        stopbits: int = 1,
        encoder_bits: int = 23,
    ) -> None:
        super().__init__(name)
        self.profile = make_sim_profile(profile) if profile.addressing.param_base is None else profile
        self.latency_ms = latency_ms
        # Отдельная задержка записи моделирует привод, у которого параметр
        # ложится в EEPROM: именно по этой разнице разведка ставит диагноз.
        self.write_latency_ms = latency_ms if write_latency_ms is None else write_latency_ms
        self.busy_on_write = busy_on_write
        self.loss_rate = loss_rate
        self._rng = random.Random(seed)
        self.clock = clock if clock is not None else time.perf_counter
        self._lock = threading.RLock()
        self._open = False
        self.load_model: LoadModel = viscous_load

        self.axes: dict[int, SimAxis] = {
            s: SimAxis(s, counts_per_rev, max_rpm, self.clock) for s in slaves
        }
        # у настоящего привода энкодер абсолютный многооборотный: позиция
        # переживает выключение, поэтому старт не с нуля
        for i, ax in enumerate(self.axes.values()):
            ax.position_counts = float(self._rng.randint(0, counts_per_rev) + i * 1000)

        # Адрес мог прийти либо явным полем address, либо базой + номером
        # параметра / order — считаем его так же, как считает сам профиль,
        # иначе симулятор и настоящий драйвер разойдутся в том, что лежит
        # по какому адресу (P-xxx с известным param_base, но без явного
        # address, иначе был бы виден как "несуществующий регистр").
        self._addr_param: dict[int, str] = {}
        self._addr_monitor: dict[int, tuple[str, int]] = {}
        for pname, spec in self.profile.params.items():
            try:
                address = self.profile.param_address(pname)
            except ConfigError:
                continue
            self._addr_param[address] = pname
        for mname, spec in self.profile.monitors.items():
            try:
                address = self.profile.monitor_address(mname)
            except ConfigError:
                continue
            for w in range(spec.words):
                self._addr_monitor[address + w] = (mname, w)

        # Параметры связи должны отражать то, как мы на самом деле подключены —
        # именно на этом совпадении держится восстановление карты регистров
        # (drives/scanner.py), поэтому симулятор обязан вести себя честно.
        prof = self.profile
        for slave, ax in self.axes.items():
            preset = {
                "slave_id": slave,
                "baudrate": prof.encode_value("baudrate", baudrate),
                "serial_format": prof.encode_value("serial_format", f"8{parity}{stopbits}"),
                "encoder_bits": encoder_bits,
                # паспортные значения читаются приводом из энкодера сам;
                # в модели держим то, что написано на шильдике 80AST-A1C04025Z1
                "motor_rated_speed": 3000,
                "motor_rated_current": 46,
                "motor_pole_pairs": 4,
                "control_mode": prof.encode_value("control_mode", "position"),
                "speed_source": prof.encode_value("speed_source", "analog"),
            }
            for pname, value in preset.items():
                if pname not in prof.params:
                    continue
                try:
                    address = prof.param_address(pname)
                except ConfigError:
                    continue
                ax.params[address] = int(value) & 0xFFFF

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        self._open = True
        log.info("шина %s: симулятор поднят, приводов %d", self.name, len(self.axes))

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def describe(self) -> str:
        return f"Симулятор({self.name}, приводы {sorted(self.axes)})"

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        now = self.clock()
        for ax in self.axes.values():
            ax.step(now)
        # состояние согласовано — только теперь считаем нагрузку
        for ax in self.axes.values():
            ax.torque_percent = self.load_model(ax.slave, ax.position_counts, ax.speed_rpm)

    def _simulate_link(self, slave: int, *, writing: bool = False) -> SimAxis:
        """Имитирует задержку и потери, возвращает ось или бросает ошибку."""
        if not self._open:
            raise TransportError(f"шина {self.name} не открыта")
        self.stats.requests += 1
        delay = self.write_latency_ms if writing else self.latency_ms
        if delay:
            time.sleep(delay / 1000.0)
        if self.loss_rate and self._rng.random() < self.loss_rate:
            if self._rng.random() < 0.5:
                self.stats.timeouts += 1
                raise BusTimeout(f"шина {self.name}: привод {slave} не ответил (модель помех)")
            self.stats.crc_errors += 1
            raise CrcError(f"шина {self.name}: битая CRC от привода {slave} (модель помех)")
        ax = self.axes.get(slave)
        if ax is None:
            self.stats.timeouts += 1
            raise BusTimeout(f"шина {self.name}: на адресе {slave} никого нет")
        return ax

    # ------------------------------------------------------------------ #
    #  Значения мониторов
    # ------------------------------------------------------------------ #
    def _monitor_raw(self, ax: SimAxis, name: str) -> int:
        cpr = ax.counts_per_rev
        pos = int(ax.position_counts)
        table: dict[str, int] = {
            "actual_speed": int(round(ax.speed_rpm)),
            "encoder_single_turn": pos % cpr,
            "position_command": ax.command_counts,
            "actual_position": pos,
            "position_error": 0,
            "actual_torque": int(round(ax.torque_percent)),
            "peak_torque": int(round(abs(ax.torque_percent) * 1.2)),
            "actual_current": int(round(abs(ax.torque_percent) * 0.046)),
            "peak_current": int(round(abs(ax.torque_percent) * 0.055)),
            "pulse_freq": 0,
            "speed_command": int(round(ax.target_rpm)),
            "torque_command": int(round(ax.torque_percent)),
            "di_state": 0b1 if ax.enabled else 0b0,
            "do_state": 0b11 if ax.enabled and not ax.alarm else 0b01,
            "encoder_bits_now": 23,
            "encoder_abs_pos": (pos % cpr) >> 7,
            "avg_load": int(round(abs(ax.torque_percent))),
            "regen_load": 0,
            "control_mode_now": 1,
            "alarm_code": ax.alarm,
            "alarm_history": 0,
            "dc_bus_voltage": 310,
            "encoder_multiturn": pos // cpr,
            "command_position": ax.command_counts,
        }
        return table.get(name, 0)

    def _split(self, value: int, words: int) -> list[int]:
        raw = int(value) & (0xFFFFFFFF if words == 2 else 0xFFFF)
        if words == 1:
            return [raw]
        return [raw & 0xFFFF, (raw >> 16) & 0xFFFF]  # порядок lo_hi

    # ------------------------------------------------------------------ #
    #  Реализация интерфейса Transport
    # ------------------------------------------------------------------ #
    def read_registers(self, slave: int, address: int, count: int, *, function: int = 3) -> list[int]:
        with self._lock:
            ax = self._simulate_link(slave)
            self._tick()
            # Мониторы и параметры обычно живут в разных диапазонах регистров
            # Modbus (input FC04 и holding FC03) и не делят адреса — но
            # у профиля без подтверждённой карты (репетиция разведки) обе
            # таблицы намеренно совпадают по адресам и function, поэтому
            # function решает только когда адрес занят в ОБЕИХ таблицах
            # сразу; иначе смотрим туда, где адрес реально есть.
            is_monitor_function = function == self.profile.monitor_function_code()
            out: list[int] = []
            for a in range(address, address + count):
                in_monitor = a in self._addr_monitor
                in_param = a in self._addr_param
                if in_monitor and in_param:
                    use_monitor = is_monitor_function
                else:
                    use_monitor = in_monitor
                if use_monitor:
                    name, word = self._addr_monitor[a]
                    spec = self.profile.monitors[name]
                    out.append(self._split(self._monitor_raw(ax, name), spec.words)[word])
                elif in_param:
                    out.append(ax.params.get(a, self._param_default(a)) & 0xFFFF)
                else:
                    self.stats.exceptions += 1
                    raise ModbusException(slave, function, 2)
            self.stats.record_ok(self.latency_ms)
            return out

    def _param_default(self, address: int) -> int:
        name = self._addr_param.get(address)
        if name is None:
            return 0
        spec = self.profile.params[name]
        if spec.default is None:
            return 0
        return self.profile.encode_value(name, spec.default)

    def write_register(self, slave: int, address: int, value: int) -> None:
        self.write_registers(slave, address, [value])

    def write_registers(self, slave: int, address: int, values: Sequence[int]) -> None:
        with self._lock:
            ax = self._simulate_link(slave, writing=True)
            if self.busy_on_write:
                self.stats.exceptions += 1
                raise ModbusException(slave, 16, 6)
            for offset, value in enumerate(values):
                a = address + offset
                if a not in self._addr_param:
                    self.stats.exceptions += 1
                    raise ModbusException(slave, 16, 2)
                ax.params[a] = int(value) & 0xFFFF
                self._apply_param(ax, self._addr_param[a], int(value))
            self._tick()
            self.stats.record_ok(self.write_latency_ms)

    def _apply_param(self, ax: SimAxis, name: str, raw: int) -> None:
        """Побочные эффекты записи параметра — как у настоящего привода."""
        signed = raw - 0x10000 if raw >= 0x8000 else raw
        if name == "speed_preset_1":
            ax.target_rpm = float(signed)
        elif name == "accel_time":
            ax.accel_time_ms = max(1.0, float(raw))

    # ------------------------------------------------------------------ #
    #  Управление моделью (не часть протокола)
    # ------------------------------------------------------------------ #
    def set_enabled(self, slave: int, on: bool) -> None:
        """Разрешение привода приходит физическим входом SON, а не по Modbus,
        поэтому в симуляторе это отдельный метод, а не запись регистра."""
        ax = self.axes.get(slave)
        if ax is not None:
            ax.enabled = on
            if not on:
                ax.target_rpm = 0.0

    def inject_alarm(self, slave: int, code: int) -> None:
        if slave in self.axes:
            self.axes[slave].alarm = code

    def clear_alarms(self) -> None:
        for ax in self.axes.values():
            ax.alarm = 0
