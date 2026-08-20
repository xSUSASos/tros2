"""Драйвер T3D: план опроса, защита от опасного пуска, поведение при отказах."""
from __future__ import annotations

import time

import pytest

from cdpr.config import ConfigError
from drives.base import BusTimeout
from drives.t3d import CYCLE_MONITORS, build_drive_group, plan_reads
from drives.sim import make_sim_profile


# --------------------------------------------------------------------------- #
#  План опроса
# --------------------------------------------------------------------------- #
def test_plan_reads_merges_neighbours(profile):
    """Транзакция стоит миллисекунды, и на четырёх осях это прямо определяет
    частоту цикла — соседние величины обязаны читаться одним куском."""
    sim = make_sim_profile(profile)
    spans = plan_reads(sim, CYCLE_MONITORS)
    assert len(spans) < len(CYCLE_MONITORS)
    covered = {name for s in spans for name, _ in s.fields}
    assert covered == set(CYCLE_MONITORS)


def test_plan_reads_respects_register_limit(profile):
    sim = make_sim_profile(profile)
    spans = plan_reads(sim, CYCLE_MONITORS, max_count=2)
    assert all(s.count <= 2 for s in spans)


def test_plan_reads_does_not_bridge_large_gaps(profile):
    sim = make_sim_profile(profile)
    tight = plan_reads(sim, CYCLE_MONITORS, max_gap=0)
    loose = plan_reads(sim, CYCLE_MONITORS, max_gap=64)
    assert len(loose) <= len(tight)


def test_plan_reads_offsets_decode_correctly(profile):
    """Смещение внутри куска должно указывать ровно на нужные слова."""
    sim = make_sim_profile(profile)
    for span in plan_reads(sim, CYCLE_MONITORS):
        for name, offset in span.fields:
            spec = sim.monitor(name)
            assert span.address + offset == spec.address
            assert offset + spec.words <= span.count


# --------------------------------------------------------------------------- #
#  Защита от опасного пуска
# --------------------------------------------------------------------------- #
def test_refuses_hardware_without_register_map(machine, profile):
    undiscovered = profile.model_copy(deep=True)
    undiscovered.addressing.param_base = None
    undiscovered.addressing.monitor_base = None
    with pytest.raises(ConfigError, match="reg_probe"):
        build_drive_group(machine, undiscovered, simulated=False)


def test_unknown_eeprom_behaviour_does_not_block(machine, profile):
    """Мануал V3.3 говорит прямо: FC06 пишет в RAM, в EEPROM значение попадает
    только явной командой FC41. Значит гонять уставку скорости каждый цикл
    безопасно, и запрещать выход на железо из-за непроверенного поля незачем."""
    discovered = make_sim_profile(profile)
    discovered.eeprom_safe = None
    group = build_drive_group(machine, discovered, simulated=False)
    assert group.n_axes == machine.n_cables


def test_eeprom_unsafe_profile_is_refused(machine, profile):
    """А вот если разведка ДОКАЗАЛА, что уставка ложится в EEPROM, работать
    нельзя: цикл управления исчерпает его ресурс за десятки минут."""
    discovered = make_sim_profile(profile)
    discovered.eeprom_safe = False
    with pytest.raises(ConfigError, match="EEPROM"):
        build_drive_group(machine, discovered, simulated=False)


def test_simulated_group_needs_no_probing(machine, profile):
    group = build_drive_group(machine, profile, simulated=True)
    assert group.simulated
    assert group.n_axes == 4


# --------------------------------------------------------------------------- #
#  Работа группы
# --------------------------------------------------------------------------- #
@pytest.fixture
def group(machine, profile):
    g = build_drive_group(machine, profile, simulated=True,
                          sim_options={"latency_ms": 0.0})
    g.open()
    g.initialize()
    return g


def test_initialize_puts_drives_in_speed_mode(group):
    """Если привод остался в позиционном режиме, уставка скорости не подействует,
    мотор будет стоять, а контур начнёт наращивать команду — худший отказ."""
    for axis in range(group.n_axes):
        assert group.read_param(axis, "control_mode") == "speed"
        assert group.read_param(axis, "speed_source") == "internal"


def test_initialize_raises_if_mode_did_not_stick(group, monkeypatch):
    monkeypatch.setattr(type(group.axes[0]), "read_param",
                        lambda self, name: "position" if name == "control_mode" else "internal")
    with pytest.raises(ConfigError, match="не подействует"):
        group.initialize()


def test_group_moves_on_command(group):
    """Разрешения приводов по Modbus у T3D нет: P-098 держит servo-on всегда,
    а питание снимает физическая кнопка. Значит уставка действует сразу."""
    group.set_speeds([120.0, -120.0, 60.0, 0.0])
    time.sleep(0.1)
    # Фактическая скорость читается по кругу — по одной оси за цикл, чтобы не
    # тратить на диагностику четыре транзакции из двенадцати. Значит и
    # проверять её надо после полного оборота опроса.
    for _ in range(group.n_axes):
        states = group.read_states()
    assert states[0].speed_rpm > 50
    assert states[1].speed_rpm < -50
    assert abs(states[3].speed_rpm) < 5


def test_stop_zeroes_every_setpoint(group):
    """Остановка обязана дойти до всех осей, а не до первых попавшихся."""
    group.set_speeds([200.0] * group.n_axes)
    time.sleep(0.05)
    group.stop()
    time.sleep(0.15)
    assert all(abs(s.speed_rpm) < 5 for s in group.read_states())
    assert all(a.last_command_rpm == 0.0 for a in group.axes)


def test_fractional_speed_is_dithered(group):
    """Уставка целочисленная, а один об/мин при барабане ⌀60 — это 3.1 мм/с
    троса. Без дизеринга тонкое позиционирование невозможно."""
    axis = group.axes[0]
    sent = []
    original = axis.transport.write_register

    def spy(slave, address, value):
        if slave == axis.slave:
            sent.append(value)
        return original(slave, address, value)

    axis.transport.write_register = spy
    for _ in range(200):
        group.set_speeds([2.3, 0, 0, 0])
    values = [v if v < 0x8000 else v - 0x10000 for v in sent]
    assert set(values) <= {2, 3}
    assert sum(values) / len(values) == pytest.approx(2.3, abs=0.05)


def test_axis_failure_does_not_break_the_cycle(group):
    """Одна отвалившаяся ось не должна ронять опрос: остальные надо успеть
    остановить, а не бросить с прежней уставкой."""
    broken = group.axes[1]
    broken.transport = _DeadTransport()
    states = group.read_states()
    assert states[1].online is False and states[1].error
    assert all(states[i].online for i in (0, 2, 3))
    group.set_speeds([10.0] * group.n_axes)  # не должно бросить исключение


class _DeadTransport:
    stats = None

    def read_registers(self, *a, **kw):
        raise BusTimeout("ось отвалилась")

    def write_register(self, *a, **kw):
        raise BusTimeout("ось отвалилась")

    def describe(self):
        return "мёртвая"


def test_torque_limit_is_written_to_the_drives(group, machine):
    """Предел момента в приводе — единственная защита, не зависящая от софта.
    Он выставляется при инициализации и больше не трогается."""
    assert group.set_torque_limit(20.0) is True
    assert group.torque_limit_applied
    for axis in group.axes:
        assert axis.read_param("torque_limit_fwd") == 20
        # P-069 действует только в положительную сторону вращения. При
        # direction: -1 привод крутится в минус, и активен один P-070 —
        # поэтому писать надо оба, иначе предела фактически нет.
        assert axis.read_param("torque_limit_rev") == 20


def test_torque_limit_reports_when_parameter_unknown(group):
    """Если номер параметра неизвестен, драйвер обязан честно сказать, что
    жёсткого предела натяжения нет, а не сделать вид, что всё хорошо."""
    group.profile.params["torque_limit_fwd"].p = None
    group.profile.params["torque_limit_fwd"].address = None
    assert group.set_torque_limit(50.0) is False
    assert group.torque_limit_applied is False
