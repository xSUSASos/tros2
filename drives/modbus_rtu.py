"""Транспорт Modbus RTU поверх последовательного порта.

Реализован вручную, а не взят готовой библиотекой, по трём причинам:
цикл управления требует предсказуемой задержки и точного контроля
межкадровых пауз; нужна детальная статистика линии для панели; и нужен
разбор частично принятых кадров, потому что 40 м витой пары в гараже
рядом с силовыми кабелями приводов — это среда с помехами.
"""
from __future__ import annotations

import logging
import struct
import threading
import time
from collections.abc import Sequence
from typing import Any

import serial

from cdpr.config import BusCfg
from drives.base import (
    BusTimeout,
    CrcError,
    FramingError,
    ModbusException,
    Transport,
    TransportError,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  CRC16 (Modbus, полином 0xA001)
# --------------------------------------------------------------------------- #
def _build_crc_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        table.append(crc)
    return tuple(table)


_CRC_TABLE = _build_crc_table()


def crc16(data: bytes) -> int:
    """Контрольная сумма Modbus RTU. В кадр уходит младшим байтом вперёд."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


def append_crc(payload: bytes) -> bytes:
    return payload + struct.pack("<H", crc16(payload))


def check_crc(frame: bytes) -> bool:
    """У корректного кадра CRC от всего кадра вместе с суммой равна нулю."""
    return len(frame) >= 3 and crc16(frame) == 0


# --------------------------------------------------------------------------- #
#  Ожидаемая длина ответа
# --------------------------------------------------------------------------- #
_EXCEPTION_FRAME_LEN = 5  # адрес + функция|0x80 + код + CRC


def expected_response_len(function: int, count: int) -> int:
    if function in (3, 4):
        return 5 + 2 * count      # адрес + функция + N + данные + CRC
    if function in (6, 16):
        return 8                  # эхо адреса и значения/количества
    raise ValueError(f"функция {function} не поддерживается")


# --------------------------------------------------------------------------- #
#  Транспорт
# --------------------------------------------------------------------------- #
_PARITY = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN, "O": serial.PARITY_ODD}
_STOPBITS = {1: serial.STOPBITS_ONE, 2: serial.STOPBITS_TWO}


class ModbusRtuTransport(Transport):
    """Одна физическая шина RS-485.

    Транзакции сериализуются мьютексом: на полудуплексной линии два запроса
    одновременно означают мусор в эфире.
    """

    def __init__(self, name: str, cfg: BusCfg) -> None:
        super().__init__(name)
        self.cfg = cfg
        self._ser: serial.Serial | None = None
        self._lock = threading.RLock()
        self._last_activity = 0.0

    # ------------------------------------------------------------------ #
    #  Жизненный цикл
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                return
            cfg = self.cfg
            try:
                self._ser = serial.Serial(
                    port=cfg.port,
                    baudrate=cfg.baudrate,
                    bytesize=cfg.bytesize,
                    parity=_PARITY[cfg.parity],
                    stopbits=_STOPBITS[cfg.stopbits],
                    timeout=cfg.timeout_ms / 1000.0,
                    write_timeout=cfg.timeout_ms / 1000.0,
                )
            except serial.SerialException as exc:
                raise TransportError(
                    f"шина {self.name}: не открывается порт {cfg.port} — {exc}"
                ) from exc
            self._tune_latency()
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._last_activity = time.perf_counter()
            log.info(
                "шина %s открыта: %s %d %d%s%d, межкадровая пауза %.0f мкс",
                self.name, cfg.port, cfg.baudrate, cfg.bytesize, cfg.parity,
                cfg.stopbits, cfg.frame_gap_us,
            )

    def _tune_latency(self) -> None:
        """Снижает задержку USB-переходника.

        У FTDI по умолчанию latency timer 16 мс — это в разы больше самого
        обмена и убивает частоту цикла управления. На Windows задаётся в
        свойствах порта, на Linux — через sysfs; здесь пробуем то, что можно
        сделать программно, и честно сообщаем, если не вышло.
        """
        try:
            import os

            for path in (
                f"/sys/bus/usb-serial/devices/{self.cfg.port.split('/')[-1]}/latency_timer",
            ):
                if os.path.exists(path):
                    with open(path, "w") as fh:
                        fh.write("1")
                    log.info("шина %s: latency_timer выставлен в 1 мс", self.name)
                    return
        except Exception as exc:  # noqa: BLE001 — сугубо необязательная оптимизация
            log.debug("шина %s: не удалось настроить latency_timer (%s)", self.name, exc)

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ------------------------------------------------------------------ #
    #  Ядро транзакции
    # ------------------------------------------------------------------ #
    def _wait_frame_gap(self) -> None:
        """Выдерживает межкадровую паузу — без неё привод склеит два кадра."""
        gap = self.cfg.frame_gap_us / 1e6
        remaining = gap - (time.perf_counter() - self._last_activity)
        if remaining > 0:
            time.sleep(remaining)

    def _read_exact(self, n: int, deadline: float) -> bytes:
        """Читает ровно n байт или падает по таймауту."""
        assert self._ser is not None
        buf = bytearray()
        while len(buf) < n:
            if time.perf_counter() > deadline:
                raise BusTimeout(
                    f"шина {self.name}: получено {len(buf)} из {n} байт за "
                    f"{self.cfg.timeout_ms:.0f} мс"
                )
            chunk = self._ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    def _read_response(self, slave: int, function: int, count: int, deadline: float) -> bytes:
        """Читает ответ, разбирая длину по ходу — так исключение (5 байт)
        не заставляет ждать полный таймаут вместо длинного ответа."""
        head = self._read_exact(2, deadline)
        got_slave, got_fn = head[0], head[1]

        if got_fn & 0x80:
            rest = self._read_exact(_EXCEPTION_FRAME_LEN - 2, deadline)
            frame = head + rest
            if not check_crc(frame):
                raise CrcError(f"шина {self.name}: битая CRC в кадре исключения {frame.hex()}")
            raise ModbusException(got_slave, got_fn & 0x7F, frame[2])

        if got_slave != slave:
            raise FramingError(
                f"шина {self.name}: ответ от адреса {got_slave}, а спрашивали {slave} "
                f"(похоже на эхо чужого кадра или дубль адресов)"
            )
        if got_fn != function:
            raise FramingError(
                f"шина {self.name}, привод {slave}: ответ на функцию {got_fn}, а запрос был {function}"
            )

        if function in (3, 4):
            nbytes = self._read_exact(1, deadline)[0]
            if nbytes != 2 * count:
                raise FramingError(
                    f"шина {self.name}, привод {slave}: обещано {nbytes} байт данных, "
                    f"ожидалось {2 * count}"
                )
            body = self._read_exact(nbytes + 2, deadline)
            frame = head + bytes([nbytes]) + body
        else:
            frame = head + self._read_exact(expected_response_len(function, count) - 2, deadline)

        if not check_crc(frame):
            raise CrcError(f"шина {self.name}, привод {slave}: не сошлась CRC, кадр {frame.hex()}")
        return frame

    def _transact(self, slave: int, function: int, payload: bytes, count: int) -> bytes:
        """Один запрос-ответ с повторами. Возвращает кадр ответа целиком."""
        if not self.is_open:
            raise TransportError(f"шина {self.name} не открыта")

        request = append_crc(bytes([slave]) + payload)
        last_error: Exception | None = None

        with self._lock:
            for attempt in range(self.cfg.retries + 1):
                if attempt:
                    self.stats.retries += 1
                self.stats.requests += 1
                try:
                    assert self._ser is not None
                    self._wait_frame_gap()
                    self._ser.reset_input_buffer()
                    started = time.perf_counter()
                    self._ser.write(request)
                    deadline = started + self.cfg.timeout_ms / 1000.0
                    frame = self._read_response(slave, function, count, deadline)
                    self._last_activity = time.perf_counter()
                    self.stats.record_ok((self._last_activity - started) * 1000.0)
                    return frame
                except ModbusException:
                    self.stats.exceptions += 1
                    self._last_activity = time.perf_counter()
                    raise  # осмысленный отказ устройства — повторять бессмысленно
                except BusTimeout as exc:
                    self.stats.timeouts += 1
                    last_error = exc
                except CrcError as exc:
                    self.stats.crc_errors += 1
                    last_error = exc
                except FramingError as exc:
                    self.stats.framing_errors += 1
                    last_error = exc
                except serial.SerialException as exc:
                    last_error = TransportError(f"шина {self.name}: сбой порта — {exc}")
                self._last_activity = time.perf_counter()

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ #
    #  Публичные операции
    # ------------------------------------------------------------------ #
    def read_registers(self, slave: int, address: int, count: int, *, function: int = 3) -> list[int]:
        if function not in (3, 4):
            raise ValueError(f"чтение поддерживает только функции 3 и 4, а не {function}")
        if not 1 <= count <= 125:
            raise ValueError(f"за раз читается от 1 до 125 регистров, запрошено {count}")
        payload = struct.pack(">BHH", function, address, count)
        frame = self._transact(slave, function, payload, count)
        data = frame[3:-2]
        return list(struct.unpack(f">{count}H", data))

    def write_register(self, slave: int, address: int, value: int) -> None:
        payload = struct.pack(">BHH", 6, address, value & 0xFFFF)
        frame = self._transact(slave, 6, payload, 1)
        echo_addr, echo_val = struct.unpack(">HH", frame[2:6])
        if echo_addr != address or echo_val != (value & 0xFFFF):
            raise FramingError(
                f"шина {self.name}, привод {slave}: эхо записи не совпало — "
                f"отправили [{address}]={value & 0xFFFF}, вернулось [{echo_addr}]={echo_val}"
            )

    def write_registers(self, slave: int, address: int, values: Sequence[int]) -> None:
        n = len(values)
        if not 1 <= n <= 123:
            raise ValueError(f"за раз пишется от 1 до 123 регистров, передано {n}")
        body = struct.pack(">BHHB", 16, address, n, 2 * n) + struct.pack(
            f">{n}H", *(v & 0xFFFF for v in values)
        )
        frame = self._transact(slave, 16, body, n)
        echo_addr, echo_n = struct.unpack(">HH", frame[2:6])
        if echo_addr != address or echo_n != n:
            raise FramingError(
                f"шина {self.name}, привод {slave}: эхо блочной записи не совпало — "
                f"отправили {n} рег. с {address}, вернулось {echo_n} с {echo_addr}"
            )

    def describe(self) -> str:
        c = self.cfg
        return f"ModbusRTU({self.name} @ {c.port} {c.baudrate} {c.bytesize}{c.parity}{c.stopbits})"


# --------------------------------------------------------------------------- #
#  Кодирование значений
# --------------------------------------------------------------------------- #
def decode(regs: Sequence[int], kind: str, word_order: str = "lo_hi") -> int:
    """Сырые регистры -> число нужного типа.

    Порядок слов в 32-битных величинах у китайских приводов не стандартизован
    и определяется пробой (addressing.word_order в профиле).
    """
    if kind in ("u16", "i16"):
        raw = regs[0] & 0xFFFF
        return raw - 0x10000 if kind == "i16" and raw >= 0x8000 else raw
    if kind in ("u32", "i32"):
        lo, hi = (regs[0], regs[1]) if word_order == "lo_hi" else (regs[1], regs[0])
        raw = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)
        return raw - 0x100000000 if kind == "i32" and raw >= 0x80000000 else raw
    raise ValueError(f"неизвестный тип регистра {kind!r}")


def encode(value: int, kind: str, word_order: str = "lo_hi") -> list[int]:
    """Обратное к decode."""
    if kind in ("u16", "i16"):
        return [int(value) & 0xFFFF]
    if kind in ("u32", "i32"):
        raw = int(value) & 0xFFFFFFFF
        lo, hi = raw & 0xFFFF, (raw >> 16) & 0xFFFF
        return [lo, hi] if word_order == "lo_hi" else [hi, lo]
    raise ValueError(f"неизвестный тип регистра {kind!r}")
