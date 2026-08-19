"""Симулятор приводов и дизеринг уставки."""
from __future__ import annotations

import time

import pytest

from drives.base import BusTimeout, ModbusException, SigmaDeltaQuantizer
from drives.modbus_rtu import decode
from drives.sim import SimTransport, make_sim_profile


# --------------------------------------------------------------------------- #
#  Дизеринг
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0.2, 1.3, 2.5, -0.7, -2.7, 9.99])
def test_dither_preserves_mean(value):
    """Уставка задаётся целыми об/мин; дизеринг возвращает среднее на месте."""
    q = SigmaDeltaQuantizer()
    n = 400
    seq = [q(value) for _ in range(n)]
    assert all(isinstance(v, int) for v in seq)
    assert sum(seq) / n == pytest.approx(value, abs=1.0 / n * 2)


def test_dither_output_is_adjacent_integers():
    """Выход не должен прыгать дальше соседних целых — иначе будут рывки."""
    q = SigmaDeltaQuantizer()
    seq = {q(1.3) for _ in range(200)}
    assert seq <= {1, 2}


def test_dither_reset_clears_residual():
    q = SigmaDeltaQuantizer()
    q(0.4)
    assert q.residual != 0.0
    q.reset()
    assert q.residual == 0.0


def test_dither_exact_integer_is_passed_through():
    q = SigmaDeltaQuantizer()
    assert {q(7.0) for _ in range(50)} == {7}


# --------------------------------------------------------------------------- #
#  Карта регистров симулятора
# --------------------------------------------------------------------------- #
def test_sim_profile_fills_known_addresses(profile):
    sim = make_sim_profile(profile)
    assert sim.is_discovered
    assert sim.eeprom_safe is True
    for name, spec in profile.params.items():
        if spec.p is not None:
            assert sim.param_address(name) is not None
    for name in profile.monitors:
        assert sim.monitor_address(name) is not None


def test_sim_does_not_invent_unknown_parameters(profile):
    """У параметров, номер которых в мануале не найден (ограничение момента),
    адреса быть не должно: модель не имеет права обещать возможность,
    которой на железе может не оказаться."""
    from cdpr.config import ConfigError

    sim = make_sim_profile(profile)
    assert profile.params["torque_limit_fwd"].p is None
    with pytest.raises(ConfigError):
        sim.param_address("torque_limit_fwd")


def test_sim_profile_monitor_addresses_do_not_overlap(profile):
    """32-битные мониторы занимают по два регистра — они не должны наезжать
    друг на друга, иначе позиция энкодера будет читаться мусором."""
    sim = make_sim_profile(profile)
    used: set[int] = set()
    for name, spec in sim.monitors.items():
        for w in range(spec.words):
            addr = spec.address + w
            assert addr not in used, f"адрес {addr} занят дважды ({name})"
            used.add(addr)


# --------------------------------------------------------------------------- #
#  Поведение модели
# --------------------------------------------------------------------------- #
@pytest.fixture
def sim(profile):
    t = SimTransport("test", profile, slaves=[1, 2], latency_ms=0.0)
    t.open()
    return t


def _read(sim, slave, monitor):
    spec = sim.profile.monitors[monitor]
    regs = sim.read_registers(slave, spec.address, spec.words)
    return decode(regs, spec.type)


def test_disabled_axis_does_not_move(sim):
    addr = sim.profile.param_address("speed_preset_1")
    sim.write_register(1, addr, 500)
    before = _read(sim, 1, "actual_position")
    time.sleep(0.05)
    after = _read(sim, 1, "actual_position")
    assert after == before, "без разрешения SON привод крутиться не должен"


def test_enabled_axis_follows_setpoint(sim):
    sim.set_enabled(1, True)
    sim.write_register(1, sim.profile.param_address("accel_time"), 1)
    sim.write_register(1, sim.profile.param_address("speed_preset_1"), 600)
    time.sleep(0.05)
    _read(sim, 1, "actual_position")
    assert _read(sim, 1, "actual_speed") == pytest.approx(600, abs=5)

    start = _read(sim, 1, "actual_position")
    time.sleep(0.1)
    moved = _read(sim, 1, "actual_position") - start
    expected = 600 / 60 * 0.1 * 8_388_608
    assert moved == pytest.approx(expected, rel=0.15)


def test_alarm_stops_motion(sim):
    sim.set_enabled(1, True)
    sim.write_register(1, sim.profile.param_address("speed_preset_1"), 300)
    time.sleep(0.03)
    sim.inject_alarm(1, 14)
    assert _read(sim, 1, "alarm_code") == 14
    time.sleep(0.15)
    assert _read(sim, 1, "actual_speed") == pytest.approx(0, abs=5)


def test_negative_setpoint_reverses(sim):
    sim.set_enabled(1, True)
    sim.write_register(1, sim.profile.param_address("accel_time"), 1)
    sim.write_register(1, sim.profile.param_address("speed_preset_1"), -300 & 0xFFFF)
    time.sleep(0.05)
    assert _read(sim, 1, "actual_speed") < 0


def test_absolute_encoder_does_not_start_at_zero(sim):
    """Энкодер многооборотный абсолютный: позиция переживает выключение,
    поэтому код не имеет права считать стартовый отсчёт нулём."""
    assert _read(sim, 1, "actual_position") != 0
    assert _read(sim, 1, "actual_position") != _read(sim, 2, "actual_position")


def test_unknown_slave_times_out(sim):
    with pytest.raises(BusTimeout):
        sim.read_registers(99, 0, 1)


def test_unknown_address_raises_modbus_exception(sim):
    with pytest.raises(ModbusException) as info:
        sim.read_registers(1, 0x7FFF, 1)
    assert info.value.code == 2


def test_loss_rate_produces_errors(profile):
    t = SimTransport("noisy", profile, slaves=[1], latency_ms=0.0, loss_rate=0.9, seed=1)
    t.open()
    errors = 0
    for _ in range(50):
        try:
            t.read_registers(1, t.profile.monitor_address("actual_speed"), 1)
        except Exception:
            errors += 1
    assert errors > 30, "модель помех должна реально ронять кадры"
