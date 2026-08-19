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
def coarse_map(request):
    from cdpr.config import load_machine

    m = load_machine()
    return compute_map(m, z_mm=1000.0, step_mm=750.0, directions=8)


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
    ix, iy = 3, 3
    x, y = coarse_map.xs[ix], coarse_map.ys[iy]
    assert coarse_map.margin_at(x, y) == pytest.approx(coarse_map.margin_n[iy, ix])


def test_margin_outside_grid_is_zero(coarse_map):
    assert coarse_map.margin_at(-500.0, 2500.0) == 0.0
    assert coarse_map.margin_at(99_000.0, 2500.0) == 0.0


def test_contains_respects_height_limits(coarse_map, machine):
    centre = np.array([3000.0, 2500.0, machine.workspace.z_min_mm - 50.0])
    assert not coarse_map.contains(centre)


def test_check_pose_accepts_centre(machine, kin):
    ok, margin, why = check_pose(machine, kin, np.array([3000.0, 2500.0, 1000.0]))
    assert ok and margin > machine.workspace.feasibility_margin_n
    assert why == "ок"


def test_check_pose_rejects_outside_box(machine, kin):
    ok, _, why = check_pose(machine, kin, np.array([3000.0, 2500.0, 9000.0]))
    assert not ok and "габарит" in why


def test_check_pose_rejects_unholdable_pose(machine, kin):
    ok, _, why = check_pose(machine, kin, np.array([400.0, 400.0, 1000.0]))
    assert not ok
    assert "удержать" in why or "запас" in why


def test_check_pose_explains_low_margin(machine, kin):
    """Отказ должен объяснять причину, а не просто запрещать."""
    ok, margin, why = check_pose(machine, kin, np.array([1100.0, 1000.0, 1000.0]))
    if not ok:
        assert "запас" in why or "удержать" in why


def test_box_limits_apply_inset_and_height(machine, kin):
    low, high = box_limits(machine, kin)
    assert low[0] == pytest.approx(machine.workspace.inset_mm)
    assert high[2] == pytest.approx(machine.workspace.z_max_mm)
    assert low[2] == pytest.approx(machine.workspace.z_min_mm)


def test_best_height_is_above_the_middle(machine, kin):
    """У подвеса оптимум смещён вверх: внизу тросы почти вертикальны и
    горизонтального усилия не создают."""
    z, margin = best_height(machine, kin, samples=9)
    assert margin > 50.0
    assert z > 0.5 * (machine.workspace.z_min_mm + machine.workspace.z_max_mm)


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
