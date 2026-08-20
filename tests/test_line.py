"""Модель намотки: слои и упругость."""
from __future__ import annotations

import math

import pytest

from cdpr.config import ConfigError
from cdpr.line import TWO_PI, LineModel


@pytest.fixture
def winch(machine):
    return machine.ordered_winches()[0].model_copy(
        update={"count_ref": 0, "length_at_ref_mm": 12000.0}
    )


@pytest.fixture
def line(winch):
    return LineModel(winch)


def test_first_layer_matches_simple_circumference(line):
    """Пока намотка в один слой, модель обязана совпадать с 2*pi*R."""
    one_turn = TWO_PI * line.layer_radius(0)
    assert line.wound_length(1) == pytest.approx(one_turn)
    assert line.wound_length(10) == pytest.approx(10 * one_turn)


def test_radius_grows_by_line_diameter_per_layer(winch):
    """Послойная модель нужна другой механике: на этой машине трос лежит в
    один слой, поэтому проверяем её на явно многослойной лебёдке."""
    layered = LineModel(winch.model_copy(update={"winding": "multi_layer"}))
    assert layered.layer_radius(1) - layered.layer_radius(0) == pytest.approx(winch.line_diameter_mm)
    assert layered.layer_radius(3) - layered.layer_radius(0) == pytest.approx(3 * winch.line_diameter_mm)


def test_layer_transition_is_continuous(line):
    """На стыке слоёв длина не должна прыгать."""
    n = line.per_layer
    before = line.wound_length(n - 1e-6)
    after = line.wound_length(n + 1e-6)
    assert after - before < 1e-3


def test_wound_length_roundtrip(line):
    for turns in (0.0, 3.7, 109.0, 110.0, 250.0, 700.0):
        assert line.turns_for_wound(line.wound_length(turns)) == pytest.approx(turns, abs=1e-6)


def test_counts_length_roundtrip(line):
    for length in (300.0, 2500.0, 8000.0, 11800.0):
        counts = line.counts_from_length(length)
        assert line.length_from_counts(counts) == pytest.approx(length, abs=0.01)


def test_ignoring_layers_would_be_a_large_error(winch):
    """Ради этого модель и послойная: на нескольких слоях масштаб уходит на
    проценты, а это десятки сантиметров на длинном тросе. На капроне 0.3 мм
    слоёв не будет, но механика может смениться."""
    thick = LineModel(winch.model_copy(update={"winding": "multi_layer",
                                               "line_diameter_mm": 0.5}))
    error = thick.radius_at(3 * thick.per_layer) / thick.radius_at(0) - 1.0
    assert error > 0.04


def test_single_layer_mode_is_linear(winch):
    line = LineModel(winch.model_copy(update={"winding": "single_layer"}))
    assert line.radius_at(0) == line.radius_at(10_000)
    assert line.wound_length(1000) == pytest.approx(1000 * TWO_PI * line.layer_radius(0))


def test_one_layer_holds_a_lot_of_line(line):
    """При леске 0.3 мм и барабане шириной 55 мм в слой входит больше 30 м, а
    самый длинный трос на этой раме около 4.5 м. Второй слой не начинается
    никогда, поэтому «мм на импульс» — честная константа."""
    assert line.wound_length(line.per_layer) / 1000.0 > 20.0


def test_uncalibrated_winch_refuses_to_convert(machine):
    line = LineModel(machine.ordered_winches()[0])
    with pytest.raises(ConfigError, match="не привязана"):
        line.length_from_counts(0)


def test_speed_conversion_roundtrip(line):
    count = line.counts_from_length(5000.0)
    for speed in (-300.0, -5.0, 12.5, 250.0):
        rpm = line.rpm_for_line_speed(speed, count)
        assert line.line_speed_for_rpm(rpm, count) == pytest.approx(speed)


def test_winding_direction_is_respected(winch):
    count = 1_000_000
    forward = LineModel(winch.model_copy(update={"direction": 1}))
    reverse = LineModel(winch.model_copy(update={"direction": -1}))
    assert forward.rpm_for_line_speed(100.0, count) == pytest.approx(
        -reverse.rpm_for_line_speed(100.0, count)
    )


def test_elasticity_is_the_dominant_error_when_uncompensated(winch):
    """Плетёнка 50 кг на 8 м под 100 Н тянется на сантиметры — без компенсации
    о миллиметровой точности говорить бессмысленно."""
    line = LineModel(winch.model_copy(update={"ea_n": 4300.0}))
    assert line.elongation(8000.0, 100.0) > 100.0
    assert line.stretched(8000.0, 100.0) > 8000.0
    assert line.unstretched(line.stretched(8000.0, 100.0), 100.0) == pytest.approx(8000.0)


def test_no_ea_means_no_compensation(winch):
    """Пока жёсткость не задана, поправки нет — и это честнее, чем гадать."""
    bare = LineModel(winch.model_copy(update={"ea_n": None}))
    assert bare.stretched(8000.0, 100.0) == 8000.0
    assert bare.elongation(8000.0, 100.0) == 0.0
