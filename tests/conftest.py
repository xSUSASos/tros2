"""Общие приспособления для тестов."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdpr.config import load_machine, load_profile  # noqa: E402
from drives.modbus_rtu import append_crc, crc16  # noqa: E402


class FakeSerial:
    """Подставной последовательный порт с логикой ведомого Modbus.

    Позволяет прогнать НАСТОЯЩИЙ код транспорта — кадрирование, CRC, разбор
    ответа, повторы — не имея железа. Умеет по команде портить кадры, чтобы
    проверить поведение на помехах.
    """

    def __init__(self, registers: dict[int, int] | None = None, slave: int = 1) -> None:
        self.registers = dict(registers or {})
        self.slave = slave
        self.is_open = True
        self._rx = bytearray()   # то, что прочитает транспорт
        self.written: list[bytes] = []
        self.corrupt_crc = False
        self.drop_response = False
        self.exception_code: int | None = None
        self.wrong_slave = False

    # --- интерфейс pyserial ---
    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def reset_output_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False

    def read(self, n: int) -> bytes:
        if not self._rx:
            return b""
        take = min(n, len(self._rx))
        out = bytes(self._rx[:take])
        del self._rx[:take]
        return out

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        self._rx.extend(self._respond(bytes(data)))
        return len(data)

    # --- логика ведомого ---
    def _respond(self, request: bytes) -> bytes:
        if self.drop_response:
            return b""
        slave, function = request[0], request[1]
        reply_slave = slave + 1 if self.wrong_slave else slave

        if self.exception_code is not None:
            return append_crc(bytes([reply_slave, function | 0x80, self.exception_code]))

        if function in (3, 4):
            address, count = struct.unpack(">HH", request[2:6])
            values = [self.registers.get(address + i, 0) & 0xFFFF for i in range(count)]
            body = bytes([reply_slave, function, 2 * count]) + struct.pack(f">{count}H", *values)
        elif function == 6:
            address, value = struct.unpack(">HH", request[2:6])
            self.registers[address] = value
            body = request[:6]
            body = bytes([reply_slave]) + body[1:]
        elif function == 16:
            address, count = struct.unpack(">HH", request[2:6])
            for i in range(count):
                self.registers[address + i] = struct.unpack(">H", request[7 + 2 * i : 9 + 2 * i])[0]
            body = bytes([reply_slave]) + request[1:6]
        else:
            return append_crc(bytes([reply_slave, function | 0x80, 1]))

        frame = append_crc(body)
        if self.corrupt_crc:
            frame = frame[:-1] + bytes([frame[-1] ^ 0xFF])
        return frame


@pytest.fixture
def machine():
    return load_machine()


@pytest.fixture
def profile():
    return load_profile()


@pytest.fixture
def undiscovered_profile():
    """Профиль так, будто карта регистров ещё не снята с железа — для
    инструментов разведки (scanner/reg_probe), которые именно это и находят."""
    blank = load_profile().model_copy(deep=True)
    blank.addressing.param_base = None
    blank.addressing.param_ram_base = None
    blank.addressing.monitor_base = None
    blank.addressing.monitor_function = None
    blank.eeprom_safe = None
    return blank


@pytest.fixture
def fake_serial():
    return FakeSerial
