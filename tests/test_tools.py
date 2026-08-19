"""Утилиты разведки должны работать хотя бы в режиме репетиции."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import bus_scan, reg_probe


def test_bus_scan_sim_finds_all_drives(capsys):
    assert bus_scan.main(["--sim", "--slaves", "1-4"]) == 0
    out = capsys.readouterr().out
    assert "найдено приводов: 4" in out


def test_bus_scan_parse_slaves():
    assert bus_scan.parse_slaves("1-4") == [1, 2, 3, 4]
    assert bus_scan.parse_slaves("1,3,7") == [1, 3, 7]
    assert bus_scan.parse_slaves("1-2,9") == [1, 2, 9]


def test_reg_probe_sim_recovers_full_map(capsys):
    """Полная репетиция: от связи до вердикта по EEPROM, без записи в профиль."""
    assert reg_probe.main(["--sim", "--slave", "1", "--writes", "10"]) == 0
    out = capsys.readouterr().out
    assert "совпало 4 из 4" in out
    assert "порядок слов lo_hi" in out
    assert "3000 об/мин" in out
    assert "eeprom_safe" in out
    assert "Ничего не записано" in out


def test_reg_probe_does_not_touch_profile_without_apply():
    """Без --apply профиль обязан остаться нетронутым."""
    from cdpr.config import DEFAULT_PROFILE

    before = DEFAULT_PROFILE.read_bytes()
    reg_probe.main(["--sim", "--slave", "1", "--writes", "5"])
    assert DEFAULT_PROFILE.read_bytes() == before
