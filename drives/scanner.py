"""Разведка железа: поиск приводов на шине и восстановление карты регистров.

Зачем это нужно. В имеющемся мануале на T3D есть номера параметров (P-181
адрес, P-182 скорость, P-137 уставка скорости и т.д.), но нет таблицы
соответствия этих номеров адресам Modbus — она была на диске, которого нет.
Без карты драйвер не может ни читать позицию, ни задавать скорость.

Карту можно не угадывать, а вывести. Ключ — то, что значения трёх параметров
связи нам известны точно: раз мы разговариваем с приводом по адресу N на
скорости B в формате F, то P-181=N, P-182=код(B), P-183=код(F). Три известных
числа подряд — это подпись, которую достаточно найти в адресном пространстве:
смещение найденного места и даёт param_base.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from cdpr.config import BusCfg, DriveProfile
from drives.base import ModbusException, Transport, TransportError

log = logging.getLogger(__name__)

#: скорости, которые поддерживает T3D (P-182)
BAUDRATES = [4800, 9600, 19200, 38400, 57600, 115200]

#: форматы кадра (P-183): (чётность, стоп-биты)
FORMATS: list[tuple[str, int]] = [("N", 1), ("E", 1), ("O", 1), ("N", 2), ("E", 2), ("O", 2)]


@dataclass
class FoundDrive:
    slave: int
    baudrate: int
    parity: str
    stopbits: int
    responded_to: str

    def __str__(self) -> str:
        return (
            f"привод {self.slave}: {self.baudrate} {self.parity}{self.stopbits} "
            f"(ответил на {self.responded_to})"
        )


@dataclass
class BusScanReport:
    port: str
    found: list[FoundDrive] = field(default_factory=list)
    tried_combinations: int = 0
    elapsed_s: float = 0.0

    @property
    def settings(self) -> tuple[int, str, int] | None:
        """Единые настройки линии, если все найденные приводы согласованы."""
        if not self.found:
            return None
        first = (self.found[0].baudrate, self.found[0].parity, self.found[0].stopbits)
        return first if all((d.baudrate, d.parity, d.stopbits) == first for d in self.found) else None

    def summary(self) -> str:
        if not self.found:
            return (
                f"На {self.port} не отозвался никто ({self.tried_combinations} сочетаний "
                f"за {self.elapsed_s:.0f} с).\n"
                "Проверьте: подана ли сила на приводы, не перепутаны ли A и B, "
                "есть ли терминаторы на концах линии, и не стоит ли P-181 = -1 "
                "(связь выключена) на панели привода."
            )
        lines = [f"На {self.port} найдено приводов: {len(self.found)}"]
        lines += ["  " + str(d) for d in sorted(self.found, key=lambda d: d.slave)]
        s = self.settings
        if s:
            lines.append(f"Общие настройки линии: {s[0]} {s[1]}{s[2]}")
            if s[0] != 115200:
                lines.append(
                    f"  ВНИМАНИЕ: линия на {s[0]} бод. Для цикла управления 50 Гц "
                    f"нужно 115200 — выставьте P-182 = 5 на каждом приводе с его панели."
                )
        else:
            lines.append("  ВНИМАНИЕ: приводы настроены по-разному — так шина работать не будет.")
        dupes = [d.slave for d in self.found]
        if len(set(dupes)) != len(dupes):
            lines.append("  ВНИМАНИЕ: адреса повторяются, задайте разные P-181.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Поиск приводов на линии
# --------------------------------------------------------------------------- #
def scan_bus(
    port: str,
    *,
    slaves: range | list[int] = range(1, 33),
    baudrates: list[int] | None = None,
    formats: list[tuple[str, int]] | None = None,
    probe_addresses: list[int] | None = None,
    timeout_ms: float = 60.0,
    stop_after: int | None = None,
    progress: Any = None,
) -> BusScanReport:
    """Перебирает настройки линии и адреса, пока кто-нибудь не ответит.

    Порядок перебора не случаен: сперва заводские настройки привода
    (19200 8E1) и целевые 115200, затем всё остальное — так типовой случай
    находится за секунды, а не за десять минут.
    """
    from drives.modbus_rtu import ModbusRtuTransport

    baudrates = baudrates or _ordered(BAUDRATES, [19200, 115200])
    formats = formats or _ordered(FORMATS, [("E", 1), ("N", 1)])
    probe_addresses = probe_addresses if probe_addresses is not None else [0, 1, 181]
    slave_list = list(slaves)

    report = BusScanReport(port=port)
    started = time.perf_counter()
    seen: set[int] = set()

    for baud in baudrates:
        for parity, stop in formats:
            cfg = BusCfg(
                port=port, baudrate=baud, parity=parity, stopbits=stop,
                timeout_ms=timeout_ms, retries=0, inter_frame_us=None,
            )
            transport = ModbusRtuTransport(f"scan@{baud}{parity}{stop}", cfg)
            try:
                transport.open()
            except TransportError as exc:
                log.error("%s", exc)
                return report
            try:
                for slave in slave_list:
                    report.tried_combinations += 1
                    if progress:
                        progress(baud, parity, stop, slave, report)
                    if slave in seen:
                        continue
                    hit = _probe_slave(transport, slave, probe_addresses)
                    if hit is not None:
                        seen.add(slave)
                        report.found.append(
                            FoundDrive(slave, baud, parity, stop, responded_to=hit)
                        )
                        log.info("найден %s", report.found[-1])
                        if stop_after and len(report.found) >= stop_after:
                            report.elapsed_s = time.perf_counter() - started
                            return report
            finally:
                transport.close()

    report.elapsed_s = time.perf_counter() - started
    return report


def _ordered(items: list, first: list) -> list:
    """Ставит вероятные варианты в начало, сохраняя остальные."""
    head = [i for i in first if i in items]
    return head + [i for i in items if i not in head]


def _probe_slave(transport: Transport, slave: int, addresses: list[int]) -> str | None:
    """Отвечает ли устройство хоть что-нибудь.

    Исключение Modbus — тоже ответ: устройство есть, просто регистр не тот.
    Это важно, потому что рабочих адресов регистров мы пока не знаем.
    """
    for addr in addresses:
        try:
            transport.read_registers(slave, addr, 1)
            return f"чтение регистра {addr}"
        except ModbusException as exc:
            return f"отказ на регистре {addr} (код {exc.code}) — устройство на линии"
        except TransportError:
            continue
    return None


# --------------------------------------------------------------------------- #
#  Снимок адресного пространства
# --------------------------------------------------------------------------- #
#: сначала пробуем «круглые» базы — у большинства приводов одна из них
LIKELY_BASES = [0x0000, 0x0100, 0x0200, 0x1000, 0x2000, 0x4000, 0x8000, 0x0400, 0x0800]


def dump_range(
    transport: Transport,
    slave: int,
    start: int,
    end: int,
    *,
    block: int = 16,
    function: int = 3,
    progress: Any = None,
) -> dict[int, int]:
    """Читает адреса [start, end) блоками. Нечитаемые участки просто пропускает.

    Привод отвечает отказом на несуществующие адреса, и это нормально —
    карта разрежена, и задача снимка в том, чтобы найти населённые островки.
    """
    values: dict[int, int] = {}

    def read_one(a: int) -> None:
        try:
            values[a] = transport.read_registers(slave, a, 1, function=function)[0]
        except ModbusException:
            pass
        except TransportError as exc:
            log.debug("снимок %s: адрес %d — %s", slave, a, exc)

    addr = start
    while addr < end:
        count = min(block, end - addr)
        # Сначала пробуем блоком: на населённом участке это в `block` раз
        # дешевле. Привод отвергает блок целиком, если внутри есть хоть один
        # несуществующий адрес, и тогда участок дочитывается поштучно —
        # рекурсивное деление здесь только удваивало бы число запросов,
        # потому что пустой участок всё равно пришлось бы обойти весь.
        try:
            regs = transport.read_registers(slave, addr, count, function=function)
            values.update({addr + i: v for i, v in enumerate(regs)})
        except ModbusException:
            for a in range(addr, addr + count):
                read_one(a)
        except TransportError as exc:
            log.debug("снимок %s: блок с %d — %s", slave, addr, exc)
        if progress:
            progress(addr, end, len(values))
        addr += count
    return values


# --------------------------------------------------------------------------- #
#  Поиск базы параметров по известной подписи
# --------------------------------------------------------------------------- #
@dataclass
class BaseCandidate:
    base: int
    matched: list[str]
    values: dict[str, int]

    @property
    def score(self) -> int:
        return len(self.matched)


def signature_for(profile: DriveProfile, slave: int, baudrate: int, parity: str, stopbits: int,
                  encoder_bits: int | None = 23) -> dict[str, int]:
    """Значения параметров, которые нам известны наверняка.

    Мы разговариваем с приводом — значит его параметры связи в точности те,
    на которых установлена наша сторона. Это и есть опорная подпись.
    """
    fmt = f"{8}{parity}{stopbits}"
    sig = {
        "slave_id": slave,
        "baudrate": profile.encode_value("baudrate", baudrate),
        "serial_format": profile.encode_value("serial_format", fmt),
    }
    if encoder_bits is not None:
        sig["encoder_bits"] = encoder_bits
    return sig


def find_param_base(
    values: dict[int, int],
    profile: DriveProfile,
    signature: dict[str, int],
    *,
    min_score: int = 3,
) -> list[BaseCandidate]:
    """Ищет смещение, при котором известные параметры оказываются на местах.

    Работает по снимку, поэтому тестируется без железа.
    """
    numbers = {}
    for name, expected in signature.items():
        spec = profile.params.get(name)
        if spec is None or spec.p is None:
            continue
        numbers[name] = (spec.p, expected)
    if not numbers:
        return []

    anchor_name, (anchor_p, anchor_value) = next(iter(numbers.items()))
    candidates: list[BaseCandidate] = []

    for addr, val in values.items():
        if val != anchor_value:
            continue
        base = addr - anchor_p
        matched, got = [], {}
        for name, (p, expected) in numbers.items():
            actual = values.get(base + p)
            got[name] = actual
            if actual == expected:
                matched.append(name)
        if len(matched) >= min_score:
            candidates.append(BaseCandidate(base=base, matched=matched, values=got))

    candidates.sort(key=lambda c: (-c.score, abs(c.base)))
    return candidates


def probe_param_base(
    transport: Transport,
    slave: int,
    profile: DriveProfile,
    signature: dict[str, int],
    *,
    quick_bases: list[int] | None = None,
    sweep: tuple[int, int] | None = (0x0000, 0x2000),
    progress: Any = None,
) -> list[BaseCandidate]:
    """Сначала проверяет вероятные базы точечно, потом при неудаче — сплошным снимком."""
    for base in (quick_bases if quick_bases is not None else LIKELY_BASES):
        window: dict[int, int] = {}
        try:
            regs = transport.read_registers(slave, base + 181, 4)
            window = {base + 181 + i: v for i, v in enumerate(regs)}
        except (ModbusException, TransportError):
            continue
        hits = find_param_base(window, profile, signature)
        if hits:
            log.info("база параметров найдена быстрой проверкой: 0x%04X", hits[0].base)
            return hits

    if sweep is None:
        return []
    log.info("быстрая проверка не помогла, снимаю адреса 0x%04X..0x%04X", *sweep)
    values = dump_range(transport, slave, sweep[0], sweep[1], progress=progress)
    log.info("прочитано регистров: %d", len(values))
    return find_param_base(values, profile, signature)


# --------------------------------------------------------------------------- #
#  Поиск регистра позиции по известному повороту вала
# --------------------------------------------------------------------------- #
@dataclass
class PositionCandidate:
    address: int
    word_order: str
    delta: int
    error_ratio: float

    def __str__(self) -> str:
        return (
            f"адрес {self.address} (0x{self.address:04X}), порядок слов {self.word_order}, "
            f"прирост {self.delta:+d} импульсов, расхождение {self.error_ratio * 100:.1f}%"
        )


def analyze_rotation(
    before: dict[int, int],
    after: dict[int, int],
    *,
    counts_per_rev: int,
    turns: float = 1.0,
    tolerance: float = 0.35,
) -> list[PositionCandidate]:
    """Находит 32-битный счётчик позиции по известному повороту вала.

    Идея: вал прокручен рукой примерно на `turns` оборотов, а сколько это в
    импульсах, мы знаем точно из разрядности энкодера. Значит нужно найти пару
    соседних регистров, которая как 32-битное число изменилась на эту величину.
    Заодно определяется порядок слов: неверный порядок даёт прирост, отличный
    на порядки, и просто не проходит по допуску.

    Функция чистая — проверяется на синтетических снимках без железа.
    """
    expected = counts_per_rev * turns
    changed = [a for a in before if a in after and before[a] != after[a]]
    out: list[PositionCandidate] = []

    for addr in sorted(set(changed) | {a - 1 for a in changed}):
        if addr not in before or addr + 1 not in before:
            continue
        if addr not in after or addr + 1 not in after:
            continue
        for order in ("lo_hi", "hi_lo"):
            delta = _i32(after, addr, order) - _i32(before, addr, order)
            if delta == 0:
                continue
            err = abs(abs(delta) - abs(expected)) / abs(expected)
            if err <= tolerance:
                out.append(PositionCandidate(addr, order, delta, err))

    out.sort(key=lambda c: c.error_ratio)
    return out


def _i32(values: dict[int, int], addr: int, order: str) -> int:
    lo, hi = (values[addr], values[addr + 1]) if order == "lo_hi" else (values[addr + 1], values[addr])
    raw = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)
    return raw - 0x100000000 if raw >= 0x80000000 else raw


def find_changed(before: dict[int, int], after: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Регистры, изменившиеся между снимками: адрес -> (было, стало)."""
    return {a: (before[a], after[a]) for a in before if a in after and before[a] != after[a]}


def find_stable_nonzero(dumps: list[dict[int, int]]) -> dict[int, int]:
    """Регистры, одинаковые во всех снимках и не равные нулю — обычно это
    константы вроде разрядности энкодера, напряжения шины и кодов модели."""
    if not dumps:
        return {}
    common = set(dumps[0])
    for d in dumps[1:]:
        common &= set(d)
    return {
        a: dumps[0][a]
        for a in sorted(common)
        if dumps[0][a] != 0 and all(d[a] == dumps[0][a] for d in dumps)
    }


# --------------------------------------------------------------------------- #
#  Проверка на износ EEPROM — самый важный вопрос ко всему проекту
# --------------------------------------------------------------------------- #
@dataclass
class EepromReport:
    """Итог проверки: можно ли писать уставку скорости каждый цикл.

    Если запись параметра ложится в EEPROM, то при 50 Гц ресурс ячейки
    (порядка 100 тысяч циклов) исчерпается примерно за полчаса работы, и
    привод выйдет из строя. Поэтому до этой проверки цикл управления на
    железо не выпускается.
    """

    address: int
    writes: int
    read_latency_ms: float
    write_latency_ms: float
    write_p95_ms: float
    slow_writes: int
    busy_responses: int
    verdict: str
    safe: bool | None

    def summary(self) -> str:
        lines = [
            f"Проверка записи в регистр {self.address} (0x{self.address:04X}), {self.writes} записей:",
            f"  чтение:  {self.read_latency_ms:6.2f} мс",
            f"  запись:  {self.write_latency_ms:6.2f} мс (95-й процентиль {self.write_p95_ms:.2f} мс)",
            f"  записей заметно медленнее чтения: {self.slow_writes}",
            f"  ответов «устройство занято»: {self.busy_responses}",
            "",
            f"Вывод: {self.verdict}",
        ]
        return "\n".join(lines)


#: во сколько раз запись должна быть медленнее чтения, чтобы заподозрить EEPROM
_EEPROM_LATENCY_FACTOR = 2.5


def probe_eeprom(
    transport: Transport,
    slave: int,
    address: int,
    *,
    writes: int = 60,
    value_a: int = 0,
    value_b: int = 1,
    read_address: int | None = None,
) -> EepromReport:
    """Сравнивает задержку записи с задержкой чтения.

    Запись в EEPROM у таких приводов занимает единицы-десятки миллисекунд,
    тогда как запись в оперативную ячейку сопоставима с чтением. Разница на
    порядок — надёжный признак. Дополнительно ловятся ответы «устройство
    занято» (код 5 или 6), которыми приводы прикрывают долгую запись.

    Значения подобраны безобидные: уставка скорости 0 и 1 об/мин, привод при
    этом не разрешён, так что вал не тронется.
    """
    read_at = read_address if read_address is not None else address

    read_times: list[float] = []
    for _ in range(min(writes, 20)):
        t0 = time.perf_counter()
        transport.read_registers(slave, read_at, 1)
        read_times.append((time.perf_counter() - t0) * 1000.0)
    read_ms = sorted(read_times)[len(read_times) // 2]

    write_times: list[float] = []
    busy = 0
    for i in range(writes):
        value = value_a if i % 2 == 0 else value_b
        t0 = time.perf_counter()
        try:
            transport.write_register(slave, address, value)
        except ModbusException as exc:
            if exc.code in (5, 6):
                busy += 1
            else:
                raise
        write_times.append((time.perf_counter() - t0) * 1000.0)

    write_times.sort()
    write_ms = write_times[len(write_times) // 2]
    p95 = write_times[min(len(write_times) - 1, int(0.95 * len(write_times)))]
    slow = sum(1 for t in write_times if t > read_ms * _EEPROM_LATENCY_FACTOR)

    ratio = write_ms / read_ms if read_ms > 0 else 1.0
    if busy:
        safe, verdict = False, (
            f"привод отвечает «занят» на {busy} записях — почти наверняка идёт запись в EEPROM. "
            "Писать уставку каждый цикл НЕЛЬЗЯ: нужен либо диапазон-зеркало без EEPROM, "
            "либо другой способ управления."
        )
    elif ratio > _EEPROM_LATENCY_FACTOR:
        safe, verdict = False, (
            f"запись медленнее чтения в {ratio:.1f} раза — похоже на запись в EEPROM. "
            "Ищите диапазон-зеркало (param_ram_base) прежде чем запускать цикл управления."
        )
    elif ratio > 1.5:
        safe, verdict = None, (
            f"запись медленнее чтения в {ratio:.1f} раза — однозначного ответа нет. "
            "Прогоните проверку с writes=1000 и посмотрите, не появится ли Err20 (ошибка EEPROM)."
        )
    else:
        safe, verdict = True, (
            f"запись сопоставима с чтением ({ratio:.2f}x) — похоже на оперативную ячейку. "
            "Частая запись уставки допустима."
        )

    return EepromReport(
        address=address, writes=writes, read_latency_ms=read_ms, write_latency_ms=write_ms,
        write_p95_ms=p95, slow_writes=slow, busy_responses=busy, verdict=verdict, safe=safe,
    )


def confirm_across_rotations(
    runs: list[list[PositionCandidate]],
) -> list[PositionCandidate]:
    """Оставляет только кандидатов, подтвердившихся во всех прогонах.

    Один поворот вала даёт несколько совпадений: соседние 32-битные величины
    перекрываются, и пара «старшее слово одной, младшее другой» иногда даёт
    правдоподобный прирост случайно. Второй прогон с другим числом оборотов
    случайные совпадения отсеивает: настоящий счётчик позиции масштабируется
    вместе с поворотом, а совпавший по случайности — нет.
    """
    if not runs:
        return []
    common = {(c.address, c.word_order) for c in runs[0]}
    for run in runs[1:]:
        common &= {(c.address, c.word_order) for c in run}

    best: dict[tuple[int, str], PositionCandidate] = {}
    for run in runs:
        for c in run:
            key = (c.address, c.word_order)
            if key in common and (key not in best or c.error_ratio > best[key].error_ratio):
                best[key] = c  # худшее расхождение из прогонов — консервативная оценка
    return sorted(best.values(), key=lambda c: c.error_ratio)


def refine_by_small_rotation(
    before: dict[int, int],
    after: dict[int, int],
    candidates: list[PositionCandidate],
) -> list[PositionCandidate]:
    """Отсеивает перекрывающиеся пары малым поворотом вала.

    При 23-битном энкодере поворот меньше 1/128 оборота меняет только младшее
    слово счётчика, старшее остаётся прежним. У настоящей пары (младшее,
    старшее) это видно сразу; а ложный кандидат, склеенный из старшего слова
    одной величины и младшего слова соседней, ведёт себя иначе — у него
    «младшим» оказывается регистр, который при малом повороте не менялся,
    либо «старшее» неожиданно поехало.
    """
    out = []
    for c in candidates:
        lo_addr = c.address if c.word_order == "lo_hi" else c.address + 1
        hi_addr = c.address + 1 if c.word_order == "lo_hi" else c.address
        if lo_addr not in before or hi_addr not in before:
            continue
        lo_moved = before[lo_addr] != after.get(lo_addr)
        hi_moved = before[hi_addr] != after.get(hi_addr)
        if lo_moved and not hi_moved:
            out.append(c)
    return out
