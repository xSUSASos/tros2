"""Кинематика и распределение натяжений."""
from __future__ import annotations

import numpy as np
import pytest

from cdpr import tension as T
from cdpr.kinematics import CDPRKinematics, KinematicsError


@pytest.fixture
def kin(machine):
    return CDPRKinematics.from_config(machine)


@pytest.fixture
def centre(kin):
    c = kin.anchors.mean(axis=0)
    return np.array([c[0], c[1], 1000.0])


# --------------------------------------------------------------------------- #
#  Кинематика
# --------------------------------------------------------------------------- #
def test_symmetric_pose_gives_equal_lengths(kin, centre):
    lengths = kin.inverse(centre)
    assert np.allclose(lengths, lengths[0])


def test_forward_inverse_roundtrip(kin):
    for pose in ([3000, 2500, 1000], [1500, 1200, 1800], [4500, 3800, 700]):
        pose = np.array(pose, dtype=float)
        recovered, rms = kin.forward(kin.inverse(pose), guess=pose + 250)
        assert np.allclose(recovered, pose, atol=1e-6)
        assert rms < 1e-6


def test_forward_residual_grows_with_inconsistent_lengths(kin, centre):
    """Невязка — рабочий признак того, что трос провис или проскользнул."""
    lengths = kin.inverse(centre)
    _, clean = kin.forward(lengths, guess=centre)
    lengths[1] += 60.0
    _, dirty = kin.forward(lengths, guess=centre)
    assert dirty > clean + 1.0


def test_unit_vectors_point_from_platform_to_anchor(kin, centre):
    u = kin.unit_vectors(centre)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    for i, anchor in enumerate(kin.anchors):
        assert np.dot(u[i], anchor - centre) > 0


def test_jacobian_matches_numeric_derivative(kin, centre):
    """Знак якобиана — источник трудноуловимых ошибок, проверяем численно."""
    analytic = kin.jacobian(centre)
    h = 0.01
    numeric = np.zeros_like(analytic)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        numeric[:, axis] = (kin.inverse(centre + step) - kin.inverse(centre - step)) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_moving_toward_anchor_winds_that_cable_in(kin, centre):
    """Положительная скорость наматывания означает укорочение троса."""
    target = kin.anchors[1]
    direction = (target - centre) / np.linalg.norm(target - centre)
    rates = kin.winding_rates(centre, direction * 100.0)
    assert rates[1] == pytest.approx(100.0, abs=1e-6)
    assert rates[1] > max(rates[0], rates[2], rates[3])


def test_condition_number_degrades_near_anchor_plane(kin):
    """У плоскости якорей тросы становятся горизонтальными и держать вес нечем."""
    low = kin.condition_number(np.array([3000.0, 2500.0, 500.0]))
    high = kin.condition_number(np.array([3000.0, 2500.0, 2900.0]))
    assert high > low * 5


def test_platform_at_anchor_is_rejected(kin):
    with pytest.raises(KinematicsError, match="нулевая"):
        kin.unit_vectors(kin.anchors[0].copy())


def test_wrong_anchor_shape_rejected():
    with pytest.raises(KinematicsError, match=r"\(m, 3\)"):
        CDPRKinematics(np.zeros((4, 2)))


# --------------------------------------------------------------------------- #
#  Натяжения
# --------------------------------------------------------------------------- #
def test_equilibrium_is_actually_satisfied(kin, machine, centre):
    """Проверка знака: W f = -w_внеш. С плюсом получается правдоподобный,
    но ровно неверный набор натяжений, и в симметричной раме это незаметно."""
    W = kin.structure_matrix(centre)
    w_ext = T.gravity_wrench(machine.platform.mass_kg)
    result = T.distribute(W, w_ext, f_min=machine.tension.min_n,
                          f_max=machine.tension.max_n, f_target=machine.tension.target_n)
    assert result.feasible
    assert np.allclose(W @ result.forces + w_ext, 0.0, atol=1e-9)


def test_wrong_sign_would_not_balance(kin, machine, centre):
    """Явная фиксация ловушки: с обратным знаком равновесия не будет."""
    W = kin.structure_matrix(centre)
    w_ext = T.gravity_wrench(machine.platform.mass_kg)
    wrong = T.distribute_closed_form(W, -w_ext, np.full(4, machine.tension.target_n))
    assert not np.allclose(W @ wrong + w_ext, 0.0, atol=1e-6)


def test_all_tensions_within_limits(kin, machine, centre):
    W = kin.structure_matrix(centre)
    result = T.distribute(W, T.gravity_wrench(machine.platform.mass_kg),
                          f_min=machine.tension.min_n, f_max=machine.tension.max_n,
                          f_target=machine.tension.target_n)
    assert result.min_force >= machine.tension.min_n - 1e-9
    assert result.max_force <= machine.tension.max_n + 1e-9


def test_symmetric_pose_gives_symmetric_tensions(kin, machine, centre):
    W = kin.structure_matrix(centre)
    forces = T.distribute(W, T.gravity_wrench(machine.platform.mass_kg),
                          f_min=machine.tension.min_n, f_max=machine.tension.max_n,
                          f_target=machine.tension.target_n).forces
    assert np.allclose(forces, forces[0], atol=1e-6)


def test_uniform_preload_is_not_a_free_parameter(kin, centre):
    """Физика, которую важно понимать при настройке: при 4 тросах и 3
    координатах равновесие задаёт общий уровень натяжения однозначно.
    Свободна ровно одна комбинация — перетяжка диагональных пар."""
    W = kin.structure_matrix(centre)
    basis = T.null_space(W)
    assert basis.shape[1] == 1, "избыточность должна быть равна единице"
    n = basis[:, 0] / np.abs(basis[:, 0]).max()
    assert np.allclose(np.abs(n), 1.0)
    assert np.sign(n[0]) != np.sign(n[1]) and np.sign(n[1]) != np.sign(n[2])
    assert not np.allclose(W @ np.ones(4), 0.0)


def test_target_tension_sets_the_weakest_cable(kin, machine, centre):
    """Целевое натяжение — цель для самого ненагруженного троса: именно он
    рискует провиснуть, и именно его имеет смысл держать подальше от нуля."""
    W = kin.structure_matrix(centre)
    w = T.gravity_wrench(machine.platform.mass_kg)
    result = T.distribute(W, w, f_min=8.0, f_max=120.0, f_target=15.0)
    assert result.min_force == pytest.approx(15.0, abs=1e-3)
    assert np.allclose(W @ result.forces + w, 0.0, atol=1e-8)


def test_unreachable_target_is_reported_honestly(kin, machine, centre):
    """Если геометрия не даёт поднять слабый трос до цели, система должна
    сказать об этом, а не молча выдать другое значение."""
    W = kin.structure_matrix(centre)
    w = T.gravity_wrench(machine.platform.mass_kg)
    result = T.distribute(W, w, f_min=8.0, f_max=120.0, f_target=100.0)
    assert result.feasible
    assert "предел геометрии" in result.message
    assert result.min_force < 100.0


def test_distribution_is_fast_enough_for_the_control_loop(kin, machine, centre):
    """Расчёт идёт каждый цикл, поэтому он обязан укладываться в доли
    миллисекунды — иначе частота цикла упрётся в математику, а не в шину."""
    import time

    W = kin.structure_matrix(centre)
    w = T.gravity_wrench(machine.platform.mass_kg)
    started = time.perf_counter()
    for _ in range(300):
        T.distribute(W, w, f_min=8.0, f_max=120.0, f_target=25.0)
    per_call_ms = (time.perf_counter() - started) / 300 * 1000
    assert per_call_ms < 2.0, f"{per_call_ms:.2f} мс на расчёт — слишком долго"


def test_near_edge_solution_still_balances(kin, machine):
    """У края рабочей зоны цель недостижима, но равновесие обязано сойтись
    и все тросы остаться в пределах."""
    pose = np.array([900.0, 800.0, 1000.0])
    W = kin.structure_matrix(pose)
    w = T.gravity_wrench(machine.platform.mass_kg)
    result = T.distribute(W, w, f_min=8.0, f_max=120.0, f_target=60.0)
    assert result.feasible
    assert np.allclose(W @ result.forces + w, 0, atol=1e-8)
    assert result.min_force >= 8.0 - 1e-6
    assert result.max_force <= 120.0 + 1e-6


def test_anchor_plane_is_unreachable(kin, machine):
    """У плоскости якорей вертикальной составляющей нет, вес держать нечем —
    это физика, а не настройка."""
    pose = np.array([3000.0, 2500.0, 2950.0])
    W = kin.structure_matrix(pose)
    w = T.gravity_wrench(machine.platform.mass_kg)
    assert not T.wrench_feasible(W, w, f_min=8.0, f_max=120.0)
    result = T.distribute(W, w, f_min=8.0, f_max=120.0, f_target=30.0)
    assert not result.feasible


def test_capacity_margin_is_highest_at_centre(kin, machine):
    w = T.gravity_wrench(machine.platform.mass_kg)
    kw = dict(f_min=machine.tension.min_n, f_max=machine.tension.max_n)
    centre = T.capacity_margin(kin.structure_matrix(np.array([3000., 2500., 1000.])), w, **kw)
    edge = T.capacity_margin(kin.structure_matrix(np.array([800., 2500., 1000.])), w, **kw)
    assert centre > edge * 3


def test_external_wrench_recovered_from_forces(kin, machine, centre):
    """На этом стоит перемещение «за руку»: по натяжениям восстанавливается
    приложенная извне сила."""
    W = kin.structure_matrix(centre)
    applied = np.array([0.0, 0.0, -machine.platform.mass_kg * T.GRAVITY])
    forces = T.distribute(W, applied, f_min=8.0, f_max=120.0, f_target=30.0).forces
    assert np.allclose(T.external_wrench_from_forces(W, forces), applied, atol=1e-8)


def test_torque_force_conversion_roundtrip(machine, kin, centre):
    winches = machine.ordered_winches()
    radii = [w.first_layer_radius_mm for w in winches]
    forces = np.array([10.0, 30.0, 55.0, 90.0])
    percent = T.forces_to_torque_percent(forces, winches, radii)
    assert np.allclose(T.torque_percent_to_forces(percent, winches, radii), forces)
