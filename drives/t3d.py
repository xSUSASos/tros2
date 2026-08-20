"""Драйвер приводов HLTNC T3D.

Способ управления продиктован самим приводом: позиционный режим у T3D
работает только от импульсного входа PULS/SIGN и по Modbus недоступен.
Значит остаётся скоростной режим с уставкой из внутреннего регистра —
привод держит контур скорости, а контур положения замыкает хост.

    P-004 = 1   скоростной режим
    P-025 = 1   уставка берётся из P-137, а не с аналогового входа
    P-098 = 1   servo-on принудительно, без внешнего входа
    P-069/070   предел момента — единственная защита, не зависящая от софта
    P-137       пишется каждый цикл, читаются d-PoS, d-trq, d-Err

Разрешения приводов по Modbus здесь нет и не предполагается: питание снимает
физическая кнопка. Абсолютный энкодер переживает выключение, поэтому после
обратного включения отсчёты остаются верными и привязка не теряется.

Все адреса берутся из профиля (config/drive_t3d.yaml). Если профиль ещё не
заполнен разведкой, драйвер отказывается работать с железом — молча гонять
цикл по неизвестным адресам опаснее, чем не запуститься.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cdpr.config import ConfigError, DriveProfile, MachineConfig
from drives.base import (
    DriveGroup,
    DriveState,
    ModbusException,
    SigmaDeltaQuantizer,
    Transport,
    TransportError,
)
from drives.modbus_rtu import decode, encode

log = logging.getLogger(__name__)

#: Что читается КАЖДЫЙ цикл управления.
#
#  Набор выбран так, чтобы уложиться в две транзакции на ось: момент лежит
#  отдельным регистром (0x09), а код аварии и позиция — одним куском
#  (0x1A..0x20). На четырёх осях это восемь чтений и четыре записи за цикл, и
#  именно это число, а не желание, определяет достижимую частоту.
CYCLE_MONITORS = ("actual_position", "actual_torque", "alarm_code")

#: Фактическая скорость нужна для диагностики «команду дали, а вал стоит» —
#  самого коварного отказа, потому что выглядит он как «просто не едет». Но
#  каждый цикл её читать расточительно, поэтому оси опрашиваются по очереди:
#  одна лишняя транзакция на цикл вместо четырёх.
SPEED_MONITOR = "actual_speed"


# --------------------------------------------------------------------------- #
#  План чтения
# --------------------------------------------------------------------------- #
@dataclass
class ReadSpan:
    """Непрерывный кусок регистров, читаемый одной транзакцией."""

    address: int
    count: int
    fields: list[tuple[str, int]]  # (имя монитора, смещение внутри куска)


def plan_reads(profile: DriveProfile, names: Sequence[str], *, max_gap: int = 4,
               max_count: int | None = None) -> list[ReadSpan]:
    """Группирует мониторы в минимальное число транзакций.

    Одна транзакция стоит 1–5 мс, и на четырёх осях это прямо определяет
    частоту цикла. Соседние величины дешевле прочитать одним куском вместе
    с ненужными регистрами между ними, чем отдельными запросами — поэтому
    небольшие разрывы проглатываются.
    """
    limit = max_count or profile.limits.max_regs_per_read
    items = []
    for name in names:
        spec = profile.monitor(name)
        items.append((profile.monitor_address(name), spec.words, name))
    items.sort()

    spans: list[ReadSpan] = []
    for address, words, name in items:
        if spans:
            span = spans[-1]
            end = span.address + span.count
            gap = address - end
            new_count = address + words - span.address
            if 0 <= gap <= max_gap and new_count <= limit:
                span.fields.append((name, address - span.address))
                span.count = new_count
                continue
            if gap < 0:  # перекрытие: величина уже внутри куска
                span.fields.append((name, address - span.address))
                span.count = max(span.count, new_count)
                continue
        spans.append(ReadSpan(address=address, count=words, fields=[(name, 0)]))
    return spans


# --------------------------------------------------------------------------- #
#  Одна ось
# --------------------------------------------------------------------------- #
class T3DAxis:
    """Привод одной лебёдки: адрес на шине плюс квантователь уставки."""

    def __init__(self, index: int, anchor: str, transport: Transport, slave: int,
                 profile: DriveProfile, spans: list[ReadSpan]) -> None:
        self.index = index
        self.anchor = anchor
        self.transport = transport
        self.slave = slave
        self.profile = profile
        self.spans = spans
        self.dither = SigmaDeltaQuantizer()
        self.state = DriveState(axis=index)
        self.last_command_rpm = 0.0

    # ------------------------------------------------------------------ #
    def read_state(self) -> DriveState:
        raw: dict[str, int] = {}
        try:
            for span in self.spans:
                regs = self.transport.read_registers(
                    self.slave, span.address, span.count,
                    function=self.profile.monitor_function_code(),
                )
                for name, offset in span.fields:
                    spec = self.profile.monitor(name)
                    raw[name] = decode(
                        regs[offset : offset + spec.words], spec.type,
                        self.profile.addressing.word_order,
                    )
        except TransportError as exc:
            # Отсчёт сохраняем прежний, а не обнуляем: одна потерянная посылка
            # не должна выглядеть как прыжок платформы через всю раму. Признак
            # беды — online=False, по нему защиты и сработают.
            self.state = DriveState(
                axis=self.index, online=False, error=str(exc),
                position_counts=self.state.position_counts,
                torque_percent=self.state.torque_percent,
                enabled=self.state.enabled,
            )
            return self.state

        pos_spec = self.profile.monitor("actual_position")
        trq_spec = self.profile.monitor("actual_torque")
        self.state = DriveState(
            axis=self.index,
            online=True,
            position_counts=int(raw.get("actual_position", 0) * pos_spec.scale),
            torque_percent=raw.get("actual_torque", 0) * trq_spec.scale,
            # Скорость читается по кругу отдельным опросом, поэтому здесь
            # переносим прошлое значение, а не подставляем ноль: иначе панель
            # показывала бы «стоит» на трёх осях из четырёх.
            speed_rpm=self.state.speed_rpm,
            alarm=int(raw.get("alarm_code", 0)),
            enabled=self.state.enabled,
            stamp=time.perf_counter(),
        )
        return self.state

    # ------------------------------------------------------------------ #
    def write_speed(self, rpm: float, *, address: int) -> None:
        """Пишет уставку скорости с дизерингом.

        Уставка целочисленная, а один об/мин при барабане ⌀60 — это 1.6 мм/с
        троса, поэтому дробная часть переносится в следующий цикл. При нулевой
        команде остаток сбрасывается: иначе после остановки мотор дёрнулся бы
        на накопленную дробь.
        """
        if rpm == 0.0:
            self.dither.reset()
            value = 0
        else:
            value = self.dither(rpm)
        self.last_command_rpm = rpm
        self.transport.write_register(self.slave, address, encode(value, "i16")[0])

    def read_param(self, name: str) -> Any:
        spec = self.profile.param(name)
        regs = self.transport.read_registers(
            self.slave, self.profile.param_address(name), spec.words
        )
        return self.profile.decode_value(
            name, decode(regs, spec.type, self.profile.addressing.word_order)
        )

    def write_param(self, name: str, value: Any, *, ram: bool = False) -> None:
        spec = self.profile.param(name)
        raw = self.profile.encode_value(name, value)
        regs = encode(raw, spec.type, self.profile.addressing.word_order)
        address = self.profile.param_address(name, ram=ram)
        if len(regs) == 1:
            self.transport.write_register(self.slave, address, regs[0])
        else:
            self.transport.write_registers(self.slave, address, regs)


# --------------------------------------------------------------------------- #
#  Группа осей
# --------------------------------------------------------------------------- #
class T3DDriveGroup(DriveGroup):
    """Все лебёдки машины. Порядок осей совпадает с порядком якорей в конфиге."""

    def __init__(self, machine: MachineConfig, profile: DriveProfile,
                 transports: dict[str, Transport], *, simulated: bool = False) -> None:
        self.machine = machine
        self.profile = profile
        self.transports = transports
        self.simulated = simulated
        self.torque_limit_applied = False

        if not simulated:
            self._check_ready_for_hardware()

        spans = plan_reads(profile, CYCLE_MONITORS)
        log.info(
            "план опроса: %d транзакций на ось (%s)",
            len(spans),
            ", ".join(f"0x{s.address:04X}+{s.count}" for s in spans),
        )
        self.axes: list[T3DAxis] = []
        for i, anchor in enumerate(machine.geometry.anchors):
            self.axes.append(
                T3DAxis(i, anchor.id, transports[anchor.bus], anchor.slave, profile, spans)
            )
        self._speed_turn = 0
        self._hot_address = profile.param_address(
            profile.hot_register, ram=profile.addressing.param_ram_base is not None
        )

    # ------------------------------------------------------------------ #
    def _check_ready_for_hardware(self) -> None:
        """Отказывается стартовать на железе, пока это опасно."""
        if not self.profile.is_discovered:
            raise ConfigError(
                f"профиль {self.profile.name!r} не заполнен: адреса регистров неизвестны.\n"
                f"Для T3D обычно достаточно param_base: 0 и monitor_base: 0 — так задано\n"
                f"мануалом V3.3 (адрес FC03 равен номеру параметра).\n"
                f"Если привод отвечает иначе, снимите карту:\n"
                f"  python tools/reg_probe.py --port <порт> --slave 1 --apply"
            )
        if self.profile.eeprom_safe is False:
            raise ConfigError(
                f"профиль {self.profile.name!r} помечен eeprom_safe: false — уставка "
                f"скорости ложится в EEPROM.\nЦикл на "
                f"{self.machine.control.loop_hz:.0f} Гц исчерпает его ресурс примерно за "
                f"{100_000 / self.machine.control.loop_hz / 60:.0f} минут.\n"
                f"Нужен диапазон-зеркало без записи в EEPROM (addressing.param_ram_base)."
            )

    # ------------------------------------------------------------------ #
    @property
    def n_axes(self) -> int:
        return len(self.axes)

    def open(self) -> None:
        for t in self.transports.values():
            t.open()

    def close(self) -> None:
        for t in self.transports.values():
            t.close()

    def initialize(self) -> None:
        """Приводит все приводы к скоростному режиму по init_sequence профиля.

        Это запись параметров, а не уставок: делается один раз при старте,
        поэтому EEPROM ничего не грозит.
        """
        for axis in self.axes:
            for step in self.profile.init_sequence:
                try:
                    axis.write_param(step.param, step.value)
                except TransportError as exc:
                    raise TransportError(
                        f"ось {axis.index} ({axis.anchor}, адрес {axis.slave}): "
                        f"не удалось задать {step.param}={step.value} — {exc}"
                    ) from exc
            log.info("ось %d (%s): режим задан", axis.index, axis.anchor)
        self._verify_mode()
        # Предел момента ставится сразу и больше не трогается. Это
        # единственная защита, работающая независимо от цикла управления.
        self.set_torque_limit(self.machine.safety.drive_torque_limit_percent)

    def _verify_mode(self) -> None:
        """Проверяет, что привод действительно в скоростном режиме.

        Если он остался в позиционном, уставка скорости не подействует, мотор
        будет стоять, а контур управления начнёт наращивать команду — это
        худший из отказов, потому что выглядит как «просто не едет».
        """
        for axis in self.axes:
            try:
                mode = axis.read_param("control_mode")
                source = axis.read_param("speed_source")
            except TransportError as exc:
                log.warning("ось %d: режим не прочитался (%s)", axis.index, exc)
                continue
            if mode != "speed" or source != "internal":
                raise ConfigError(
                    f"ось {axis.index} ({axis.anchor}): режим {mode!r}, источник {source!r}, "
                    f"а нужно 'speed' и 'internal'. Уставка скорости иначе не подействует."
                )

    # ------------------------------------------------------------------ #
    def read_states(self) -> list[DriveState]:
        states = [axis.read_state() for axis in self.axes]
        self._poll_one_speed(states)
        return states

    def _poll_one_speed(self, states: list[DriveState]) -> None:
        """Читает фактическую скорость у одной оси за цикл, по кругу.

        Остальные оси показывают прошлое значение — для диагностики этого
        достаточно, а шина не нагружается лишними тремя транзакциями.
        """
        if not self.axes:
            return
        index = self._speed_turn % len(self.axes)
        self._speed_turn += 1
        axis = self.axes[index]
        if not states[index].online:
            return
        try:
            spec = self.profile.monitor(SPEED_MONITOR)
            regs = axis.transport.read_registers(
                axis.slave, self.profile.monitor_address(SPEED_MONITOR), spec.words,
                function=self.profile.monitor_function_code(),
            )
            value = decode(regs, spec.type, self.profile.addressing.word_order)
            states[index].speed_rpm = value * spec.scale
            axis.state.speed_rpm = states[index].speed_rpm
        except TransportError as exc:
            log.debug("ось %d: скорость не прочиталась (%s)", index, exc)
        else:
            pass

    def set_speeds(self, rpm: Sequence[float]) -> None:
        if len(rpm) != self.n_axes:
            raise ValueError(f"нужно {self.n_axes} уставок, передано {len(rpm)}")
        for axis, value in zip(self.axes, rpm, strict=True):
            try:
                axis.write_speed(value, address=self._hot_address)
            except TransportError as exc:
                # Отказ одной оси не должен ронять цикл: остальные надо
                # успеть остановить, а не бросить их с прежней уставкой.
                axis.state.online = False
                axis.state.error = str(exc)
                log.warning("ось %d (%s): уставка не прошла — %s", axis.index, axis.anchor, exc)

    def read_param(self, axis: int, name: str) -> Any:
        return self.axes[axis].read_param(name)

    def write_param(self, axis: int, name: str, value: Any) -> None:
        self.axes[axis].write_param(name, value)

    def reset_alarms(self) -> None:
        for transport in self.transports.values():
            clear = getattr(transport, "clear_alarms", None)
            if clear:
                clear()
        for axis in self.axes:
            try:
                axis.write_param("do1_function", axis.read_param("do1_function"))
            except (TransportError, ModbusException, ConfigError):
                pass
        log.info(
            "сброс аварий: у T3D он делается входом ACLR или с панели привода; "
            "по Modbus гарантированного способа в мануале нет"
        )

    def set_torque_limit(self, percent: float) -> bool:
        """Ограничение момента в самом приводе — жёсткий предел натяжения.

        Возвращает False, если параметр ограничения не найден: его номер в
        мануале отсутствует и подбирается разведкой. Пока не найден, предел
        натяжения держится только контуром управления, и это надо показывать
        в панели как незакрытый риск.
        """
        spec = self.profile.params.get("torque_limit_fwd")
        if spec is None or (spec.p is None and spec.address is None):
            log.warning(
                "параметр ограничения момента не найден: предел натяжения держится "
                "только софтом. Мотор способен выдать %.0f Н на тросе.",
                self.machine.ordered_winches()[0].torque_percent_to_force(100.0),
            )
            self.torque_limit_applied = False
            return False
        for axis in self.axes:
            axis.write_param("torque_limit_fwd", int(percent))
            if "torque_limit_rev" in self.profile.params:
                axis.write_param("torque_limit_rev", int(percent))
        self.torque_limit_applied = True
        winch = self.machine.ordered_winches()[0]
        log.info("предел момента в приводах: %.0f %% = %.1f Н на тросе",
                 percent, winch.torque_percent_to_force(percent))
        return True

    def stats(self) -> dict[str, Any]:
        return {name: t.stats.as_dict() for name, t in self.transports.items()}

    def describe(self) -> str:
        lines = [
            f"{'симулятор' if self.simulated else 'железо'}: {self.n_axes} осей, "
            f"профиль {self.profile.name!r}",
            f"предел момента в приводах: "
            f"{'выставлен' if self.torque_limit_applied else 'НЕ ВЫСТАВЛЕН, параметр не найден'}",
        ]
        for axis in self.axes:
            lines.append(
                f"  ось {axis.index} — якорь {axis.anchor}, "
                f"{axis.transport.describe()}, адрес {axis.slave}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Сборка
# --------------------------------------------------------------------------- #
def build_drive_group(
    machine: MachineConfig,
    profile: DriveProfile,
    *,
    simulated: bool = False,
    sim_options: dict[str, Any] | None = None,
) -> T3DDriveGroup:
    """Собирает группу осей по конфигу.

    Единственное место, где решается «железо или модель». Всё выше по стеку
    работает с DriveGroup и разницы не видит.
    """
    from drives.modbus_rtu import ModbusRtuTransport
    from drives.sim import SimTransport

    slaves_by_bus: dict[str, list[int]] = {}
    for anchor in machine.geometry.anchors:
        slaves_by_bus.setdefault(anchor.bus, []).append(anchor.slave)

    transports: dict[str, Transport] = {}
    effective_profile = profile

    for name, bus in machine.buses.items():
        slaves = sorted(slaves_by_bus.get(name, []))
        if not slaves:
            continue
        if simulated:
            winch = machine.ordered_winches()[0]
            transport = SimTransport(
                name, profile, slaves,
                counts_per_rev=winch.encoder_counts_per_rev,
                max_rpm=winch.max_rpm,
                baudrate=bus.baudrate, parity=bus.parity, stopbits=bus.stopbits,
                **(sim_options or {}),
            )
            effective_profile = transport.profile
            transports[name] = transport
        else:
            transports[name] = ModbusRtuTransport(name, bus)

    if not transports:
        raise ConfigError("ни одна шина не используется — проверьте bus у якорей")

    return T3DDriveGroup(machine, effective_profile, transports, simulated=simulated)
