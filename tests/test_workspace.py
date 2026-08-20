"""Рабочая зона считается по силам, а не по отступам от стен."""
from __future__ import annotations

import numpy as np
import pytest

from cdpr.kinematics import CDPRKinematics
from cdpr.workspace import best_height, box_limits, check_pose, compute_map


@pytest.fixture
def kin(machine):
    return CDPRKinematics.from_config(machine)


@pytest.fixture(scope="module")
def coarse_map():
    from cdpr.config import load_machine

    m = load_machine()
    z = m.geometry.plane_z_mm
    if z is None:
        z = 0.5 * (m.workspace.z_min_mm + m.workspace.z_max_mm)
    return compute_map(m, z_mm=float(z), step_mm=300.0, directions=8)


@pytest.fixture
def plane_z(machine):
    z = machine.geometry.plane_z_mm
    return float(z if z is not None else machine.workspace.z_min_mm)


@pytest.fixture
def centre(kin, plane_z):
    c = kin.anchors.mean(axis=0)
    return np.array([c[0], c[1], plane_z])


def test_centre_is_stronger_than_edges(coarse_map):
    m = coarse_map.margin_n
    cy, cx = m.shape[0] // 2, m.shape[1] // 2
    assert m[cy, cx] > m[cy, 1] * 2
    assert m[cy, cx] > m[1, cx] * 2


def test_frame_boundary_is_unusable(coarse_map):
    """На линии якорей тросы коллинеарны и держать нечем — это не настройка."""
    assert coarse_map.margin_n[0, :].max() == pytest.approx(0.0, abs=1e-6)
    assert coarse_map.margin_n[:, 0].max() == pytest.approx(0.0, abs=1e-6)


def test_working_area_is_much_smaller_than_the_frame(coarse_map):
    """Ради этого зона и считается: формальный отступ 20 см дал бы почти всю
    раму, а реально управляемая часть заметно меньше."""
    assert 0.0 < coarse_map.area_fraction < 0.6


def test_margin_interpolation_matches_grid(coarse_map):
    ix, iy = 2, 2
    x, y = coarse_map.xs[ix], coarse_map.ys[iy]
    assert coarse_map.margin_at(x, y) == pytest.approx(coarse_map.margin_n[iy, ix])


def test_margin_outside_grid_is_zero(coarse_map):
    assert coarse_map.margin_at(-500.0, 900.0) == 0.0
    assert coarse_map.margin_at(99_000.0, 900.0) == 0.0


def test_contains_respects_height_limits(coarse_map, machine, kin):
    low, high = box_limits(machine, kin)
    below = np.array([0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1]), low[2] - 50.0])
    assert not coarse_map.contains(below)


def test_check_pose_accepts_centre(machine, kin, centre):
    ok, margin, why = check_pose(machine, kin, centre)
    assert ok and margin > machine.workspace.feasibility_margin_n
    assert why == "ок"


def test_check_pose_rejects_outside_box(machine, kin, centre):
    far = centre + np.array([9000.0, 0.0, 0.0])
    ok, _, why = check_pose(machine, kin, far)
    assert not ok and "габарит" in why


def test_check_pose_rejects_unholdable_pose(machine, kin, plane_z):
    """У самого угла держать нечем: три троса тянут прочь, и поперёк этого
    направления запас усилия падает до нуля. Это геометрия, а не настройка."""
    low, _ = box_limits(machine, kin)
    corner = np.array([low[0] + 1.0, low[1] + 1.0, plane_z])
    ok, _, why = check_pose(machine, kin, corner)
    assert not ok
    assert "удержать" in why or "запас" in why


def test_check_pose_explains_low_margin(machine, kin, plane_z):
    """Отказ должен объяснять причину, а не просто запрещать."""
    low, high = box_limits(machine, kin)
    near_edge = np.array([low[0] + 0.15 * (high[0] - low[0]),
                          low[1] + 0.15 * (high[1] - low[1]), plane_z])
    ok, margin, why = check_pose(machine, kin, near_edge)
    if not ok:
        assert "запас" in why or "удержать" in why


def test_box_limits_apply_inset_and_height(machine, kin):
    low, high = box_limits(machine, kin)
    assert low[0] == pytest.approx(machine.workspace.inset_mm)
    if machine.geometry.is_planar:
        # Высота не диапазон, а одно число: коробка всегда в рабочей плоскости.
        assert low[2] == pytest.approx(machine.geometry.plane_z_mm)
        assert high[2] == pytest.approx(machine.geometry.plane_z_mm)
    else:
        assert high[2] == pytest.approx(machine.workspace.z_max_mm)
        assert low[2] == pytest.approx(machine.workspace.z_min_mm)


def test_best_height_lies_below_the_anchors(machine, kin):
    """Высота рабочей плоскости — это и есть настройка натяжения: чем меньше
    провис, тем сильнее натянуты тросы и тем лучше держится горизонтальное
    возмущение, но тем ближе предел прочности. Оптимум лежит между этими
    двумя стенками, и панель обязана его показывать, а не заставлять
    подбирать вслепую."""
    z, margin = best_height(machine, kin, samples=13)
    assert margin > machine.workspace.feasibility_margin_n
    assert machine.workspace.z_min_mm <= z <= machine.workspace.z_max_mm
    assert z < machine.geometry.anchor_z_mm


def test_payload_shrinks_the_working_area(machine, kin):
    """Полезная нагрузка должна сужать зону — иначе легко выехать за предел."""
    light = check_pose(machine, kin, np.array([1400.0, 1300.0, 1000.0]), payload_kg=0.0)
    heavy = check_pose(machine, kin, np.array([1400.0, 1300.0, 1000.0]), payload_kg=15.0)
    assert heavy[1] <= light[1]


def test_map_serialises_for_the_panel(coarse_map):
    data = coarse_map.as_dict()
    assert set(data) >= {"xs", "ys", "margin_n", "required_n", "area_fraction"}
    assert len(data["margin_n"]) == len(data["ys"])
    assert len(data["margin_n"][0]) == len(data["xs"])
