"""Разведка железа: восстановление карты регистров без документации."""
from __future__ import annotations

import pytest

from drives import scanner
from drives.sim import SimTransport

CPR = 8_388_608


# --------------------------------------------------------------------------- #
#  Поиск базы параметров по подписи
# --------------------------------------------------------------------------- #
def test_signature_uses_known_link_settings(profile):
    sig = scanner.signature_for(profile, slave=7, baudrate=115200, parity="E", stopbits=1)
    assert sig == {"slave_id": 7, "baudrate": 5, "serial_format": 1, "encoder_bits": 23}


@pytest.mark.parametrize("base", [0x0000, 0x0200, 0x1000, 0x4000])
def test_find_param_base_on_synthetic_dump(profile, base):
    sig = scanner.signature_for(profile, slave=3, baudrate=19200, parity="N", stopbits=1)
    values = {base + 181: 3, base + 182: 2, base + 183: 0, base + 184: 23}
    values.update({base + i: 0 for i in range(0, 180)})  # шум вокруг
    hits = scanner.find_param_base(values, profile, sig)
    assert hits and hits[0].base == base
    assert hits[0].score == 4


def test_find_param_base_rejects_wrong_signature(profile):
    """Если подпись не та, база не должна «находиться» — лучше ничего,
    чем неверная карта регистров."""
    sig = scanner.signature_for(profile, slave=3, baudrate=115200, parity="E", stopbits=1)
    values = {181: 99, 182: 99, 183: 99, 184: 99}
    assert scanner.find_param_base(values, profile, sig) == []


def test_find_param_base_needs_min_score(profile):
    """Одного совпавшего числа мало — таких в дампе сотни."""
    sig = scanner.signature_for(profile, slave=3, baudrate=115200, parity="E", stopbits=1)
    values = {181: 3, 182: 0, 183: 0, 184: 0}
    assert scanner.find_param_base(values, profile, sig) == []


# --------------------------------------------------------------------------- #
#  Поиск регистра позиции по повороту вала
# --------------------------------------------------------------------------- #
def _pos_dump(addr: int, value: int, order: str = "lo_hi") -> dict[int, int]:
    raw = value & 0xFFFFFFFF
    lo, hi = raw & 0xFFFF, (raw >> 16) & 0xFFFF
    return {addr: lo, addr + 1: hi} if order == "lo_hi" else {addr: hi, addr + 1: lo}


@pytest.mark.parametrize("order", ["lo_hi", "hi_lo"])
def test_analyze_rotation_finds_counter_and_word_order(order):
    before = _pos_dump(0x20, 1_000_000, order)
    after = _pos_dump(0x20, 1_000_000 + CPR, order)
    hits = scanner.analyze_rotation(before, after, counts_per_rev=CPR, turns=1.0)
    assert hits
    assert (hits[0].address, hits[0].word_order) == (0x20, order)
    assert hits[0].delta == CPR


def test_analyze_rotation_ignores_unchanged_registers():
    before = {0x20: 5, 0x21: 0, 0x22: 7, 0x23: 0}
    hits = scanner.analyze_rotation(before, dict(before), counts_per_rev=CPR, turns=1.0)
    assert hits == []


def test_analyze_rotation_rejects_implausible_delta():
    """Регистр скорости меняется, но не на величину поворота — он не должен
    попасть в кандидаты на позицию."""
    before = _pos_dump(0x20, 0)
    after = _pos_dump(0x20, 1234)
    assert scanner.analyze_rotation(before, after, counts_per_rev=CPR, turns=1.0) == []


def test_confirm_across_rotations_drops_inconsistent():
    good = scanner.PositionCandidate(0x20, "lo_hi", CPR, 0.0)
    fluke = scanner.PositionCandidate(0x99, "hi_lo", CPR, 0.1)
    assert scanner.confirm_across_rotations([[good, fluke], [good]]) == [good]


def test_refine_by_small_rotation_keeps_only_true_pair():
    """Малый поворот меняет лишь младшее слово — этим отсекаются пары,
    склеенные из соседних величин."""
    before = {0x20: 100, 0x21: 5, 0x22: 200, 0x23: 9}
    after = {0x20: 900, 0x21: 5, 0x22: 200, 0x23: 9}
    cands = [
        scanner.PositionCandidate(0x20, "lo_hi", CPR, 0.0),   # настоящая пара
        scanner.PositionCandidate(0x21, "hi_lo", CPR, 0.0),   # перекрытие
    ]
    kept = scanner.refine_by_small_rotation(before, after, cands)
    assert [(c.address, c.word_order) for c in kept] == [(0x20, "lo_hi")]


# --------------------------------------------------------------------------- #
#  Снимок разреженной карты
# --------------------------------------------------------------------------- #
def test_dump_range_finds_islands_in_sparse_map(profile):
    """Привод отвергает блок целиком, если внутри есть несуществующий адрес.
    Снимок обязан всё равно достать населённые участки."""
    sim = SimTransport("t", profile, slaves=[1], latency_ms=0.0)
    sim.open()
    values = scanner.dump_range(sim, 1, 0, 300, block=16)
    assert sim.profile.param_address("speed_preset_1") in values
    assert sim.profile.param_address("slave_id") in values
    assert 250 not in values, "несуществующие адреса не должны попадать в снимок"


def test_dump_range_is_cheap_on_populated_region(undiscovered_profile):
    """На населённом участке блочное чтение экономит на порядок."""
    sim = SimTransport("t", undiscovered_profile, slaves=[1], latency_ms=0.0)
    sim.open()
    start = sim.profile.monitor_address("actual_speed")
    n = 16  # внутри блока мониторов, без выхода за населённый участок
    scanner.dump_range(sim, 1, start, start + n, block=16)
    assert sim.stats.requests == 1


def test_dump_range_overhead_bounded_on_sparse_region(profile):
    """На разреженном участке дочитывание поштучно не должно превращаться
    в кратный перерасход запросов."""
    sim = SimTransport("t", profile, slaves=[1], latency_ms=0.0)
    sim.open()
    n = 256
    scanner.dump_range(sim, 1, 0, n, block=16)
    assert sim.stats.requests <= n * 1.2


# --------------------------------------------------------------------------- #
#  Полный проход разведки против модели
# --------------------------------------------------------------------------- #
def test_full_discovery_pipeline(undiscovered_profile):
    """От «ничего не известно» до адреса позиции и порядка слов."""
    sim = SimTransport("t", undiscovered_profile, slaves=[2], latency_ms=0.0,
                       baudrate=115200, parity="E", stopbits=1)
    sim.open()
    truth = sim.profile

    sig = scanner.signature_for(truth, slave=2, baudrate=115200, parity="E", stopbits=1)
    bases = scanner.probe_param_base(sim, 2, truth, sig)
    assert bases and bases[0].base == truth.addressing.param_base

    runs = []
    for turns in (1.0, 2.5):
        before = scanner.dump_range(sim, 2, 0x1000, 0x1040)
        sim.axes[2].position_counts += CPR * turns
        after = scanner.dump_range(sim, 2, 0x1000, 0x1040)
        runs.append(scanner.analyze_rotation(before, after, counts_per_rev=CPR, turns=turns))

    cands = scanner.confirm_across_rotations(runs)
    before = scanner.dump_range(sim, 2, 0x1000, 0x1040)
    sim.axes[2].position_counts += 20_000
    after = scanner.dump_range(sim, 2, 0x1000, 0x1040)
    final = scanner.refine_by_small_rotation(before, after, cands)

    assert len(final) == 1, f"разведка должна дать однозначный ответ, а дала {final}"
    assert final[0].address == truth.monitor_address("actual_position")
    assert final[0].word_order == truth.addressing.word_order


# --------------------------------------------------------------------------- #
#  Проверка на EEPROM
# --------------------------------------------------------------------------- #
def test_eeprom_probe_passes_for_fast_writes(profile):
    sim = SimTransport("t", profile, slaves=[1], latency_ms=1.0)
    sim.open()
    rep = scanner.probe_eeprom(sim, 1, sim.profile.param_address("speed_preset_1"), writes=30)
    assert rep.safe is True
    assert "оперативную ячейку" in rep.verdict


def test_eeprom_probe_flags_slow_writes(profile):
    """Запись на порядок медленнее чтения — признак EEPROM. Пропустить это
    означает сжечь привод примерно за полчаса работы цикла на 50 Гц."""
    sim = SimTransport("t", profile, slaves=[1], latency_ms=1.0, write_latency_ms=8.0)
    sim.open()
    rep = scanner.probe_eeprom(sim, 1, sim.profile.param_address("speed_preset_1"), writes=30)
    assert rep.safe is False
    assert "EEPROM" in rep.verdict
    assert rep.slow_writes > 0


def test_eeprom_probe_flags_busy_responses(profile):
    sim = SimTransport("t", profile, slaves=[1], latency_ms=0.5, busy_on_write=True)
    sim.open()
    rep = scanner.probe_eeprom(sim, 1, sim.profile.param_address("speed_preset_1"), writes=20)
    assert rep.safe is False
    assert rep.busy_responses == 20
    assert "занят" in rep.verdict
