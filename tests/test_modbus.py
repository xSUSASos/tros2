"""Транспорт Modbus RTU: кадрирование, CRC, ошибки, повторы."""
from __future__ import annotations

import pytest

from cdpr.config import BusCfg
from drives import modbus_rtu
from drives.base import BusTimeout, CrcError, FramingError, ModbusException
from drives.modbus_rtu import ModbusRtuTransport, append_crc, check_crc, crc16, decode, encode


# --------------------------------------------------------------------------- #
#  CRC
# --------------------------------------------------------------------------- #
def test_crc_reference_vector():
    """Эталон из спецификации: CRC16/MODBUS от "123456789" равна 0x4B37."""
    assert crc16(b"123456789") == 0x4B37


def test_crc_appended_frame_self_validates():
    frame = append_crc(bytes.fromhex("010402FFFF"))
    assert frame.hex().upper() == "010402FFFFB880"
    assert check_crc(frame)


@pytest.mark.parametrize("bit", [0, 7, 15, 23])
def test_crc_detects_single_bit_flip(bit):
    frame = bytearray(append_crc(b"\x01\x03\x02\x12\x34"))
    frame[bit // 8] ^= 1 << (bit % 8)
    assert not check_crc(bytes(frame))


# --------------------------------------------------------------------------- #
#  Кодирование значений
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,value", [
    ("i16", -32768), ("i16", 32767), ("u16", 0), ("u16", 65535),
    ("i32", -2147483648), ("i32", 2147483647), ("u32", 4294967295),
])
@pytest.mark.parametrize("order", ["lo_hi", "hi_lo"])
def test_value_roundtrip(kind, value, order):
    assert decode(encode(value, kind, order), kind, order) == value


def test_word_order_actually_differs():
    """Если порядок слов угадан неверно, число выйдет другим — это ловушка,
    на которой легко потерять позицию энкодера."""
    assert encode(0x12345678, "i32", "lo_hi") != encode(0x12345678, "i32", "hi_lo")
    assert decode([0x5678, 0x1234], "u32", "lo_hi") == 0x12345678
    assert decode([0x5678, 0x1234], "u32", "hi_lo") == 0x56781234


# --------------------------------------------------------------------------- #
#  Транспорт на подставном порту
# --------------------------------------------------------------------------- #
@pytest.fixture
def bus():
    return BusCfg(port="FAKE", baudrate=115200, parity="E", timeout_ms=20, retries=2)


def _transport(monkeypatch, bus, fake):
    monkeypatch.setattr(modbus_rtu.serial, "Serial", lambda **kw: fake)
    t = ModbusRtuTransport("test", bus)
    t.open()
    # межкадровая пауза в тестах не нужна — она только замедляет прогон
    t.cfg = bus.model_copy(update={"inter_frame_us": 0.0})
    return t


def test_read_registers(monkeypatch, bus, fake_serial):
    fake = fake_serial({100: 0x1234, 101: 0xABCD})
    t = _transport(monkeypatch, bus, fake)
    assert t.read_registers(1, 100, 2) == [0x1234, 0xABCD]
    assert t.stats.ok == 1 and t.stats.error_rate == 0.0


def test_write_single_and_readback(monkeypatch, bus, fake_serial):
    fake = fake_serial({})
    t = _transport(monkeypatch, bus, fake)
    t.write_register(1, 137, 250)
    assert fake.registers[137] == 250
    assert t.read_registers(1, 137, 1) == [250]


def test_write_multiple(monkeypatch, bus, fake_serial):
    fake = fake_serial({})
    t = _transport(monkeypatch, bus, fake)
    t.write_registers(1, 200, [1, 2, 3])
    assert [fake.registers[200 + i] for i in range(3)] == [1, 2, 3]


def test_negative_value_is_written_as_twos_complement(monkeypatch, bus, fake_serial):
    """Отрицательная уставка скорости должна уходить дополнительным кодом."""
    fake = fake_serial({})
    t = _transport(monkeypatch, bus, fake)
    t.write_register(1, 137, -250)
    assert fake.registers[137] == 0x10000 - 250
    assert decode([fake.registers[137]], "i16") == -250


# --------------------------------------------------------------------------- #
#  Поведение на помехах — ради этого транспорт и написан вручную
# --------------------------------------------------------------------------- #
def test_modbus_exception_is_not_retried(monkeypatch, bus, fake_serial):
    """Отказ устройства осмысленный: повторять его — только терять время цикла."""
    fake = fake_serial({})
    fake.exception_code = 2
    t = _transport(monkeypatch, bus, fake)
    with pytest.raises(ModbusException) as info:
        t.read_registers(1, 999, 1)
    assert info.value.code == 2
    assert "недопустимый адрес" in str(info.value)
    assert t.stats.retries == 0
    assert t.stats.requests == 1


def test_crc_error_is_retried_then_raised(monkeypatch, bus, fake_serial):
    fake = fake_serial({100: 5})
    fake.corrupt_crc = True
    t = _transport(monkeypatch, bus, fake)
    with pytest.raises(CrcError):
        t.read_registers(1, 100, 1)
    assert t.stats.requests == bus.retries + 1
    assert t.stats.retries == bus.retries
    assert t.stats.crc_errors == bus.retries + 1


def test_recovers_after_transient_corruption(monkeypatch, bus, fake_serial):
    """Одиночная помеха не должна ронять цикл — на то и повторы."""
    fake = fake_serial({100: 42})
    fake.corrupt_crc = True
    t = _transport(monkeypatch, bus, fake)

    original_write = fake.write

    def write_once_bad(data):
        result = original_write(data)
        fake.corrupt_crc = False  # следующий ответ будет чистым
        return result

    fake.write = write_once_bad
    assert t.read_registers(1, 100, 1) == [42]
    assert t.stats.crc_errors == 1
    assert t.stats.ok == 1


def test_timeout_when_silent(monkeypatch, bus, fake_serial):
    fake = fake_serial({})
    fake.drop_response = True
    t = _transport(monkeypatch, bus, fake)
    with pytest.raises(BusTimeout):
        t.read_registers(1, 100, 1)
    assert t.stats.timeouts == bus.retries + 1


def test_response_from_wrong_slave_is_framing_error(monkeypatch, bus, fake_serial):
    """Так проявляются дублирующиеся адреса на шине — частая ошибка монтажа."""
    fake = fake_serial({100: 1})
    fake.wrong_slave = True
    t = _transport(monkeypatch, bus, fake)
    with pytest.raises(FramingError, match="дубль адресов"):
        t.read_registers(1, 100, 1)


def test_ping_counts_exception_as_alive(monkeypatch, bus, fake_serial):
    """Привод, ответивший отказом, на линии присутствует — сканер это учитывает."""
    fake = fake_serial({})
    fake.exception_code = 2
    t = _transport(monkeypatch, bus, fake)
    assert t.ping(1) is True

    fake.exception_code = None
    fake.drop_response = True
    assert t.ping(1) is False


def test_frame_gap_matches_modbus_spec():
    """Выше 19200 спецификация фиксирует паузу 1750 мкс, ниже — 3.5 символа."""
    slow = BusCfg(port="X", baudrate=9600, parity="E")
    fast = BusCfg(port="X", baudrate=115200, parity="E")
    assert slow.frame_gap_us == pytest.approx(3.5 * slow.char_time_us)
    assert fast.frame_gap_us == 1750.0
    assert BusCfg(port="X", baudrate=115200, inter_frame_us=200.0).frame_gap_us == 200.0


def test_read_count_limits(monkeypatch, bus, fake_serial):
    t = _transport(monkeypatch, bus, fake_serial({}))
    with pytest.raises(ValueError):
        t.read_registers(1, 0, 0)
    with pytest.raises(ValueError):
        t.read_registers(1, 0, 126)
    with pytest.raises(ValueError):
        t.read_registers(1, 0, 1, function=6)
