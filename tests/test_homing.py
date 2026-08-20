"""Привязка системы: геометрия модулей и калибровка лебёдок по замерам."""
from __future__ import annotations

import numpy as np
import pytest

from cdpr.calibration import RangeStation, identify_from_ranges
from cdpr.geometry_fit import PAIRS, fit_modules
from cdpr.line import LineModel
from cdpr.modes.autohoming import AutoHoming, default_deltas
from cdpr.runtime import build_runtime

TRUTH = np.array([[0., 0., 3000.], [5980., 40., 2975.],
                  [6040., 4950., 3060.], [-30., 5010., 3015.]])


def _distances(A):
    return [float(np.linalg.norm(A[i] - A[j])) for i, j in PAIRS]


def _canon(A):
    A = A - A[0]
    x = A[1].copy(); x[2] = 0.0; x /= np.linalg.norm(x)
    y = np.array([-x[1], x[0], 0.0])
    return A @ np.column_stack([x, y, [0, 0, 1]])


# --------------------------------------------------------------------------- #
#  Геометрия из расстояний и высот
# --------------------------------------------------------------------------- #
def test_exact_measurements_give_exact_geometry():
    fit = fit_modules(_distances(TRUTH), TRUTH[:, 2])
    assert fit.residual_rms_mm < 1e-6
    assert np.allclose(_canon(fit.positions), _canon(TRUTH), atol=1e-6)


def test_heights_are_taken_as_given_not_fitted():
    """Высоты — самое слабое место любой автоматики (модули почти в одной
    плоскости), поэтому они измеряются, а не подгоняются."""
    heights = TRUTH[:, 2] + np.array([0.0, 5.0, -5.0, 3.0])
    fit = fit_modules(_distances(TRUTH), heights)
    assert np.allclose(fit.positions[:, 2], heights)


def test_noisy_measurements_stay_within_a_few_millimetres():
    rng = np.random.default_rng(4)
    errors = []
    for _ in range(10):
        d = np.array(_distances(TRUTH)) + rng.normal(0, 2.0, 6)
        h = TRUTH[:, 2] + rng.normal(0, 3.0, 4)
        fit = fit_modules(d, h)
        errors.append(np.linalg.norm(_canon(fit.positions) - _canon(TRUTH), axis=1).max())
    assert np.mean(errors) < 10.0


def test_wrong_distance_is_caught():
    """Перепутанная пара обязана быть замечена, а не молча испортить геометрию."""
    d = _distances(TRUTH)
    d[2] *= 0.6
    fit = fit_modules(d, TRUTH[:, 2])
    assert not fit.ok
    assert any("не сходятся" in w or "треугольник" in w for w in fit.warnings)


def test_impossible_triangle_is_rejected():
    d = _distances(TRUTH)
    d[0] = 1.0   # два модуля якобы в одной точке, хотя до третьего далеко
    fit = fit_modules(d, TRUTH[:, 2])
    assert any("треугольник" in w for w in fit.warnings)


def test_wrong_number_of_measurements():
    with pytest.raises(ValueError, match="расстояний"):
        fit_modules([1000.0] * 5, TRUTH[:, 2])
    with pytest.raises(ValueError, match="высот"):
        fit_modules(_distances(TRUTH), [3000.0, 3000.0])


def test_negative_distance_rejected():
    d = _distances(TRUTH)
    d[1] = -100.0
    with pytest.raises(ValueError, match="положительными"):
        fit_modules(d, TRUTH[:, 2])


# --------------------------------------------------------------------------- #
#  Калибровка лебёдок по замерам дальномером
# --------------------------------------------------------------------------- #
@pytest.fixture
def truth_lines(machine):
    lengths = [11800.0, 12050.0, 11930.0, 12210.0]
    empties = [1_000_000, -500_000, 2_000_000, 700_000]
    stiffness = [4300.0, 4500.0, 4100.0, 4400.0]
    return [
        LineModel(w.model_copy(update={"count_empty": empties[i],
                                       "length_at_empty_mm": lengths[i],
                                       "ea_n": stiffness[i]}))
        for i, w in enumerate(machine.ordered_winches())
    ], stiffness


def _station(machine, lines, stiffness, pose, tension, rng, noise=44_000.0):
    from cdpr.kinematics import CDPRKinematics

    kin = CDPRKinematics.from_config(machine)
    distances = kin.inverse(np.asarray(pose, float))
    counts = np.array([
        lines[i].counts_from_length(distances[i] / (1 + tension / stiffness[i]))
        for i in range(4)
    ]) + rng.normal(0, noise, 4)
    return RangeStation(distances_mm=distances, counts=counts,
                        tensions_n=np.full(4, tension))


def test_ranges_recover_stiffness_and_offsets(machine, truth_lines):
    lines, stiffness = truth_lines
    rng = np.random.default_rng(7)
    stations = [
        _station(machine, lines, stiffness, [3000, 2500, 2100], 70.0, rng),
        _station(machine, lines, stiffness, [2400, 2000, 1200], 35.0, rng),
        _station(machine, lines, stiffness, [3600, 3100, 600], 26.0, rng),
    ]
    result = identify_from_ranges(machine, stations)
    assert result.residual_rms_mm < 5.0
    assert result.ea_n is not None
    for got, want in zip(result.ea_n, stiffness, strict=True):
        assert abs(got / want - 1) < 0.4


def test_same_height_stations_cannot_give_stiffness(machine, truth_lines):
    """Если все стоянки на одном уровне, натяжение одинаковое, и отличить
    вытяжку от постоянного смещения нечем. Система обязана это сказать."""
    lines, stiffness = truth_lines
    rng = np.random.default_rng(8)
    stations = [
        _station(machine, lines, stiffness, [2600, 2200, 1200], 30.0, rng),
        _station(machine, lines, stiffness, [3400, 2800, 1200], 30.0, rng),
        _station(machine, lines, stiffness, [3000, 2500, 1200], 30.0, rng),
    ]
    result = identify_from_ranges(machine, stations)
    assert any("разнесите стоянки по высоте" in w.lower() for w in result.warnings)


def test_ranges_need_at_least_two_stations(machine):
    with pytest.raises(ValueError, match="две стоянки"):
        identify_from_ranges(machine, [RangeStation(np.ones(4), np.zeros(4))])


def test_calibration_from_ranges_restores_position(machine, truth_lines):
    """Проверка по назначению: после привязки положение должно считаться
    с точностью порядка точности дальномера."""
    from cdpr.kinematics import CDPRKinematics

    lines, stiffness = truth_lines
    rng = np.random.default_rng(12)
    kin = CDPRKinematics.from_config(machine)
    stations = [
        _station(machine, lines, stiffness, [3000, 2500, 2100], 70.0, rng),
        _station(machine, lines, stiffness, [2400, 2000, 1200], 35.0, rng),
        _station(machine, lines, stiffness, [3600, 3100, 600], 26.0, rng),
    ]
    result = identify_from_ranges(machine, stations)
    calibrated = [
        LineModel(w.model_copy(update={"count_empty": result.count_empty[i],
                                       "length_at_empty_mm": result.length_at_empty_mm[i],
                                       "ea_n": result.ea_n[i]}))
        for i, w in enumerate(machine.ordered_winches())
    ]
    low = np.array([-2000.0, -2000.0, 50.0])
    high = np.array([9000.0, 8000.0, 2900.0])
    for check in ([2800, 2300, 1500], [3400, 2700, 900]):
        pose = np.array(check, float)
        distances = kin.inverse(pose)
        counts = [lines[i].counts_from_length(distances[i] / (1 + 30.0 / stiffness[i]))
                  for i in range(4)]
        measured = np.array([
            calibrated[i].stretched(calibrated[i].length_from_counts(int(counts[i])), 30.0)
            for i in range(4)
        ])
        estimated, _ = kin.forward(measured, guess=pose + 120.0, bounds=(low, high))
        assert np.linalg.norm(estimated - pose) < 15.0


# --------------------------------------------------------------------------- #
#  Автоцикл объезда стоянок
# --------------------------------------------------------------------------- #
@pytest.fixture
def rt():
    runtime = build_runtime(simulated=True, virtual_clock=True,
                            sim_options={"latency_ms": 0.0})
    runtime.drives.open()
    runtime.drives.initialize()
    runtime.controller.enable(True)
    runtime.step(0.02)
    yield runtime
    runtime.drives.close()


def _drive_until_waiting(rt, mode, limit=4000):
    for _ in range(limit):
        rt.step(0.02)
        if mode.waiting or mode.phase == "готово":
            return True
    return False


def test_autohoming_visits_stations_at_different_heights(rt):
    """Стоянки берутся на разной высоте намеренно: жёсткость троса видна
    только там, где меняется натяжение."""
    mode = AutoHoming(default_deltas(300.0), feed_mms=60.0, settle_s=0.3)
    rt.controller.set_mode(mode)
    tensions = []
    for _ in range(len(mode.plan)):
        assert _drive_until_waiting(rt, mode), f"не доехали до стоянки {mode.index + 1}"
        pose = rt.controller.state.pose_mm
        distances = [float(np.linalg.norm(pose - a)) for a in rt.controller.kinematics.anchors]
        station = mode.confirm(distances, rt.controller)
        tensions.append(float(station.tensions_n.mean()))
    assert len(mode.stations) == len(mode.plan)
    assert max(tensions) - min(tensions) > 8.0, f"натяжение почти не изменилось: {tensions}"


def test_autohoming_does_not_need_the_platform_position(rt):
    """До привязки положение платформы неизвестно, поэтому цикл задан
    приращениями длин тросов, а не координатами. Режим обязан работать,
    когда положение не вычисляется вовсе."""
    mode = AutoHoming(default_deltas(200.0), feed_mms=60.0, settle_s=0.2)
    rt.controller.set_mode(mode)
    rt.step(0.02)
    assert mode.requires_homing is False

    rt.controller.state.pose_mm = None
    rt.controller.state.lengths_mm = None
    output = mode.update(rt.controller, 0.02)
    assert output.cable_velocity_mms is not None
    assert output.target_pose is None


def test_confirm_rejected_while_moving(rt):
    mode = AutoHoming(default_deltas(400.0), feed_mms=20.0, settle_s=0.5)
    rt.controller.set_mode(mode)
    rt.step(0.02)
    for _ in range(20):
        rt.step(0.02)
    if not mode.waiting:
        with pytest.raises(RuntimeError, match="не время"):
            mode.confirm([1000.0] * 4, rt.controller)


def test_confirm_validates_input(rt):
    mode = AutoHoming(default_deltas(150.0), feed_mms=80.0, settle_s=0.2)
    rt.controller.set_mode(mode)
    assert _drive_until_waiting(rt, mode)
    with pytest.raises(ValueError, match="нужно 4"):
        mode.confirm([1000.0, 2000.0], rt.controller)
    with pytest.raises(ValueError, match="положительными"):
        mode.confirm([1000.0, -5.0, 1000.0, 1000.0], rt.controller)


def test_autohoming_can_be_aborted(rt):
    mode = AutoHoming(default_deltas(600.0), feed_mms=20.0, settle_s=0.3)
    rt.controller.set_mode(mode)
    rt.step(0.02)
    mode.abort()
    state = rt.step(0.02)
    assert "прервана" in " ".join(state.messages)
