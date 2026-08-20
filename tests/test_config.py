"""Конфигурация: производные величины, проверки целостности, правка файла."""
from __future__ import annotations

import io
import math
import shutil

import pytest

from cdpr.config import (
    ConfigError,
    MachineConfig,
    load_machine,
    load_profile,
    patch_yaml,
    save_calibration,
)


# --------------------------------------------------------------------------- #
#  Лебёдка
# --------------------------------------------------------------------------- #
def test_winch_derived_values(machine):
    w = machine.ordered_winches()[0]
    assert w.drum_radius_mm == 30.0
    assert w.first_layer_radius_mm == 30.0 + w.line_diameter_mm / 2
    assert w.turns_per_layer == int(w.drum_width_mm // w.line_diameter_mm)
    expected = 2 * math.pi * w.first_layer_radius_mm / w.counts_per_drum_rev
    assert w.nominal_mm_per_count == pytest.approx(expected)


def test_torque_force_roundtrip(machine):
    w = machine.ordered_winches()[0]
    for force in (5.0, 30.0, 120.0):
        assert w.torque_percent_to_force(w.force_to_torque_percent(force)) == pytest.approx(force)


def test_full_torque_can_exceed_configured_tension_limit(machine):
    """Мотор физически сильнее заданного предела натяжения — поэтому предел
    обязан дублироваться ограничением момента в самом приводе."""
    w = machine.ordered_winches()[0]
    assert w.torque_percent_to_force(100.0) > machine.tension.max_n


def test_uncalibrated_by_default(machine):
    assert not machine.is_calibrated
    assert set(machine.uncalibrated()) == {a.id for a in machine.geometry.anchors}


# --------------------------------------------------------------------------- #
#  Проверки целостности
# --------------------------------------------------------------------------- #
def _raw(machine: MachineConfig) -> dict:
    return machine.model_dump()


def test_duplicate_slave_on_bus_rejected(machine):
    data = _raw(machine)
    data["geometry"]["anchors"][1]["slave"] = data["geometry"]["anchors"][0]["slave"]
    with pytest.raises(ValueError, match="занят дважды"):
        MachineConfig.model_validate(data)


def test_winch_pointing_at_unknown_anchor_rejected(machine):
    data = _raw(machine)
    data["winches"][0]["anchor"] = "НЕТ_ТАКОГО"
    with pytest.raises(ValueError, match="неизвестный якорь"):
        MachineConfig.model_validate(data)


def test_attachment_count_must_match_cables(machine):
    data = _raw(machine)
    data["platform"]["attachments"] = data["platform"]["attachments"][:2]
    with pytest.raises(ValueError, match="точек крепления"):
        MachineConfig.model_validate(data)


def test_too_few_cables_for_dof_rejected(machine):
    """Для управляемости нужно минимум dof+1 троса: тросы только тянут."""
    data = _raw(machine)
    data["geometry"]["anchors"] = data["geometry"]["anchors"][:3]
    data["winches"] = data["winches"][:3]
    data["platform"]["attachments"] = data["platform"]["attachments"][:3]
    with pytest.raises(ValueError, match="для управляемости"):
        MachineConfig.model_validate(data)


def test_tension_order_enforced(machine):
    data = _raw(machine)
    data["tension"]["min_n"] = data["tension"]["max_n"] + 1
    with pytest.raises(ValueError, match="min_n <= target_n <= max_n"):
        MachineConfig.model_validate(data)


def test_unknown_bus_rejected(machine):
    data = _raw(machine)
    data["geometry"]["anchors"][0]["bus"] = "нет_такой_шины"
    with pytest.raises(ValueError, match="не описана в buses"):
        MachineConfig.model_validate(data)


def test_winch_defaults_are_merged(machine):
    """В YAML лебёдка задана одной строкой — остальное приходит из умолчаний."""
    first = machine.winches[0]
    for w in machine.winches:
        assert w.drum_diameter_mm == first.drum_diameter_mm
        assert w.encoder_counts_per_rev == first.encoder_counts_per_rev
        assert w.rated_torque_nm == first.rated_torque_nm
        assert w.direction == first.direction


# --------------------------------------------------------------------------- #
#  Профиль привода
# --------------------------------------------------------------------------- #
def test_profile_reports_addresses_unknown(profile):
    """Пока карта регистров не снята с железа, профиль честно об этом говорит."""
    undiscovered = profile.model_copy(deep=True)
    undiscovered.addressing.param_base = None
    undiscovered.addressing.monitor_base = None
    assert not undiscovered.is_discovered
    with pytest.raises(ConfigError, match="reg_probe"):
        undiscovered.param_address("speed_preset_1")
    with pytest.raises(ConfigError, match="reg_probe"):
        undiscovered.monitor_address("encoder_bits_now")


def test_profile_addresses_confirmed_by_manual(profile):
    """Карта регистров подтверждена V3.3 (FC03 addr = P-xxx, таблица FC04) —
    дефолтный профиль больше не требует reg_probe для выхода на железо."""
    assert profile.is_discovered
    assert profile.param_address("speed_preset_1") == 137
    # Предел момента — единственная защита, не зависящая от софта; его номер
    # обязан быть известен, иначе привод способен порвать трос.
    assert profile.param_address("torque_limit_fwd") == 69
    assert profile.param_address("torque_limit_rev") == 70
    # Слежение идёт по АБСОЛЮТНОЙ позиции (0x1F/0x20), а не по счёту от
    # включения питания (0x05/0x06): энкодер многооборотный, и его показания
    # переживают снятие питания кнопкой аварийного стопа.
    assert profile.monitor_address("actual_position") == 31
    assert profile.monitor_address("position_from_power_on") == 5


def test_profile_enum_encoding(profile):
    assert profile.encode_value("control_mode", "speed") == 1
    assert profile.encode_value("speed_source", "internal") == 1
    assert profile.encode_value("baudrate", 115200) == 5
    assert profile.encode_value("serial_format", "8E1") == 1
    assert profile.decode_value("baudrate", 5) == "115200"


def test_profile_rejects_bad_enum_value(profile):
    with pytest.raises(ConfigError, match="допустимо только"):
        profile.encode_value("control_mode", "телепатия")


def test_profile_init_sequence_is_consistent(profile):
    """Все шаги инициализации должны ссылаться на существующие параметры —
    иначе ошибка вылезет уже на живом приводе."""
    for step in profile.init_sequence:
        assert step.param in profile.params
    assert profile.hot_register in profile.params


def test_hot_register_is_the_speed_setpoint(profile):
    """Каждый цикл пишется именно уставка скорости: позиционный режим у T3D
    работает только от импульсного входа и по Modbus недоступен."""
    assert profile.hot_register == "speed_preset_1"
    assert profile.params["speed_preset_1"].p == 137


def test_eeprom_safety_unknown_until_probed(profile):
    """Пока не проверено, что горячий регистр не пишется в EEPROM,
    выходить на железо с циклом 50 Гц нельзя."""
    unprobed = profile.model_copy(deep=True)
    unprobed.eeprom_safe = None
    assert unprobed.eeprom_safe is None


def test_eeprom_safety_confirmed_by_manual(profile):
    """V3.3: FC06 пишет только в RAM, EEPROM требует явного FC41 — частая
    запись hot_register ресурс EEPROM не расходует."""
    assert profile.eeprom_safe is True


def test_alarm_text(profile):
    assert profile.alarm_text(0) == "нет аварии"
    assert "EEPROM" in profile.alarm_text(20)
    assert "неизвестный" in profile.alarm_text(9999)


# --------------------------------------------------------------------------- #
#  Правка конфига (так панель сохраняет настройки)
# --------------------------------------------------------------------------- #
def test_patch_yaml_preserves_comments(tmp_path):
    src = load_machine.__globals__["DEFAULT_MACHINE"]
    dst = tmp_path / "machine.yaml"
    shutil.copy2(src, dst)
    before = io.open(dst, encoding="utf-8").read()
    assert "# ---" in before

    patch_yaml(dst, {"tension.target_n": 11.5}, backup=False)
    save_calibration({0: {"count_ref": 123456, "length_at_ref_mm": 7100.0}}, dst)

    after = io.open(dst, encoding="utf-8").read()
    assert "# ---" in after, "комментарии не должны теряться при сохранении из панели"
    assert "заполняется хомингом" in after

    cfg = load_machine(dst)
    assert cfg.tension.target_n == 11.5
    assert cfg.winches[0].count_ref == 123456
    assert cfg.winches[0].length_at_ref_mm == 7100.0
    assert cfg.winches[1].count_ref is None


def test_save_calibration_refuses_non_calibration_fields(tmp_path):
    """Калибровка не должна иметь права трогать геометрию или лимиты."""
    src = load_machine.__globals__["DEFAULT_MACHINE"]
    dst = tmp_path / "machine.yaml"
    shutil.copy2(src, dst)
    with pytest.raises(ConfigError, match="не калибровочные поля"):
        save_calibration({0: {"max_rpm": 9000}}, dst)


def test_patch_yaml_rejects_typos(tmp_path):
    src = load_machine.__globals__["DEFAULT_MACHINE"]
    dst = tmp_path / "machine.yaml"
    shutil.copy2(src, dst)
    with pytest.raises(ConfigError, match="опечатка"):
        patch_yaml(dst, {"tension.targt_n": 1.0}, backup=False)
