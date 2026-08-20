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
def centre(machine, kin):
    """Центр рамы в рабочей плоскости — там, где машина и работает."""
    c = kin.anchors.mean(axis=0)
    z = machine.geometry.plane_z_mm
    if z is None:
        z = 0.5 * (machine.workspace.z_min_mm + machine.workspace.z_max_mm)
    return np.array([c[0], c[1], float(z)])


@pytest.fixture
def plane_z(machine):
    z = machine.geometry.plane_z_mm
    return float(z if z is not None else machine.workspace.z_min_mm)


def _uniform_tension(kin, pose, machine) -> float:
    """Натяжение решения, в котором все тросы натянуты одинаково.

    Это и есть НАИБОЛЬШЕЕ достижимое натяжение самого слабого троса: любое
    отклонение по нуль-пространству два троса поднимает, а два опускает.
    """
    unit = kin.unit_vectors(pose)
    return float(machine.platform.mass_kg * T.GRAVITY / unit[:, 2].sum())


# --------------------------------------------------------------------------- #
#  Кинематика
# --------------------------------------------------------------------------- #
def test_symmetric_pose_gives_equal_lengths(kin, centre):
    lengths = kin.inverse(centre)
    assert np.allclose(lengths, lengths[0])


def test_forward_inverse_roundtrip(kin, plane_z):
    for xy in ([1200, 700], [2600, 1100], [1990, 960]):
        pose = np.array([xy[0], xy[1], plane_z], dtype=float)
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


def test_condition_number_degrades_near_anchor_plane(kin, machine, centre):
    """У плоскости якорей тросы становятся горизонтальными и держать вес нечем.

    Отсюда и требование к провису: подвесить коробку почти вплотную к
    плоскости модулей — значит потребовать натяжений, которых трос не выдержит.
    """
    working = kin.condition_number(centre)
    near_plane = centre.copy()
    near_plane[2] = machine.geometry.anchor_z_mm - 5.0
    assert kin.condition_number(near_plane) > working * 5


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
    """В центре прямоугольника все четыре троса одинаковой длины, поэтому
    решение обязано быть симметричным относительно диагоналей. Свободна ровно
    одна комбинация — перетяжка диагональных пар, — и именно по ней решение и
    расходится, если целевое натяжение ниже равновесного."""
    W = kin.structure_matrix(centre)
    forces = T.distribute(W, T.gravity_wrench(machine.platform.mass_kg),
                          f_min=machine.tension.min_n, f_max=machine.tension.max_n,
                          f_target=machine.tension.target_n).forces
    assert forces[0] == pytest.approx(forces[2], abs=1e-6)
    assert forces[1] == pytest.approx(forces[3], abs=1e-6)


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
    uniform = _uniform_tension(kin, centre, machine)
    modest = 0.5 * uniform
    result = T.distribute(W, w, f_min=0.5 * modest, f_max=1e4, f_target=modest)
    assert result.min_force == pytest.approx(modest, abs=1e-3)
    assert np.allclose(W @ result.forces + w, 0.0, atol=1e-8)


def test_target_above_equilibrium_cannot_be_reached(kin, machine, centre):
    """Ползунок натяжения не всесилен, и это не баг, а физика. Свободна ровно
    одна комбинация — перетяжка диагональных пар: она ДВА троса поднимает и
    ДВА опускает. Значит поднять самый слабый выше равновесного уровня нельзя
    вовсе, и лучшее, что бывает, — равномерное решение. Общий уровень
    натяжения выбирается провисом, а не алгоритмом."""
    W = kin.structure_matrix(centre)
    w = T.gravity_wrench(machine.platform.mass_kg)
    uniform = _uniform_tension(kin, centre, machine)
    greedy = T.distribute(W, w, f_min=1.0, f_max=1e4, f_target=10.0 * uniform)
    assert greedy.feasible
    assert greedy.min_force <= uniform + 1e-6
    assert greedy.min_force == pytest.approx(uniform, rel=0.05)


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


def test_near_edge_solution_still_balances(kin, machine, plane_z):
    """У края рабочей зоны цель недостижима, но равновесие обязано сойтись
    и все тросы остаться в пределах."""
    low = kin.anchors.min(axis=0)
    pose = np.array([low[0] + 700.0, low[1] + 500.0, plane_z])
    W = kin.structure_matrix(pose)
    w = T.gravity_wrench(machine.platform.mass_kg)
    limits = machine.tension
    result = T.distribute(W, w, f_min=limits.min_n, f_max=limits.max_n,
                          f_target=limits.max_n)
    assert result.feasible
    assert np.allclose(W @ result.forces + w, 0, atol=1e-8)
    assert result.min_force >= limits.min_n - 1e-6
    assert result.max_force <= limits.max_n + 1e-6


def test_anchor_plane_is_unreachable(kin, machine, centre):
    """У самой плоскости якорей вертикальной составляющей нет, вес держать
    нечем — это физика, а не настройка. Отсюда и нижняя граница на провис."""
    pose = centre.copy()
    pose[2] = machine.geometry.anchor_z_mm - 0.5
    W = kin.structure_matrix(pose)
    w = T.gravity_wrench(machine.platform.mass_kg)
    limits = machine.tension
    assert not T.wrench_feasible(W, w, f_min=limits.min_n, f_max=limits.max_n)
    result = T.distribute(W, w, f_min=limits.min_n, f_max=limits.max_n,
                          f_target=limits.target_n)
    assert not result.feasible


def test_capacity_margin_is_highest_at_centre(kin, machine, centre, plane_z):
    w = T.gravity_wrench(machine.platform.mass_kg)
    kw = dict(f_min=machine.tension.min_n, f_max=machine.tension.max_n)
    low = kin.anchors.min(axis=0)
    edge_pose = np.array([low[0] + 350.0, centre[1], plane_z])
    middle = T.capacity_margin(kin.structure_matrix(centre), w, **kw)
    edge = T.capacity_margin(kin.structure_matrix(edge_pose), w, **kw)
    assert middle > edge * 3


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
