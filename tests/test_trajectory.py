"""Планировщик траекторий и разбор G-code."""
from __future__ import annotations

import numpy as np
import pytest

from cdpr.gcode import Move as _Move  # noqa: F401 — проверяем, что тип общий
from cdpr.gcode import parse
from cdpr.kinematics import CDPRKinematics
from cdpr.trajectory import Move, TrajectoryPlanner, max_feed_for_direction


@pytest.fixture
def planner(machine):
    return TrajectoryPlanner(machine, CDPRKinematics.from_config(machine))


#: Подача для проб: заведомо в пределах машины, иначе планировщик её
#: обрежет и тест начнёт проверять не то, что задумано.
FEED = 150.0


def _square(z=350.0, feed=FEED):
    pts = [(1000, 600, z), (2600, 600, z), (2600, 1300, z), (1000, 1300, z), (1000, 600, z)]
    return [Move(np.array(pts[i], float), np.array(pts[i + 1], float), feed, line=i + 1)
            for i in range(len(pts) - 1)]


# --------------------------------------------------------------------------- #
def test_straight_line_keeps_full_speed_through_junction(planner):
    moves = [
        Move(np.array([1200., 800., 350.]), np.array([2200., 800., 350.]), FEED),
        Move(np.array([2200., 800., 350.]), np.array([3200., 800., 350.]), FEED),
    ]
    planned = planner.plan(moves)
    assert planned[0].exit_mms == pytest.approx(FEED)


def test_reversal_requires_full_stop(planner):
    moves = [
        Move(np.array([1200., 800., 350.]), np.array([2200., 800., 350.]), FEED),
        Move(np.array([2200., 800., 350.]), np.array([1200., 800., 350.]), FEED),
    ]
    assert planner.plan(moves)[0].exit_mms == pytest.approx(0.0)


def test_right_angle_slows_but_does_not_stop(planner):
    planned = planner.plan(_square())
    assert 0.0 < planned[0].exit_mms < 100.0


def test_profile_never_exceeds_requested_feed(planner):
    for move in planner.plan(_square(feed=180.0)):
        assert move.peak_mms <= move.feed_mms + 1e-9


def test_feed_limited_by_cable_speed_not_axis(machine):
    """Подача ограничена тросами: в одном направлении барабану приходится
    крутиться быстрее, чем в другом, при той же скорости платформы."""
    kin = CDPRKinematics.from_config(machine)
    winches = machine.ordered_winches()
    speeds = np.array([w.max_line_speed_mms for w in winches])
    pose = np.array([3000.0, 2500.0, 1000.0])
    along = max_feed_for_direction(kin, pose, np.array([1.0, 0.0, 0.0]), speeds)
    down = max_feed_for_direction(kin, pose, np.array([0.0, 0.0, -1.0]), speeds)
    assert along > 0 and down > 0
    assert not np.isclose(along, down)


def test_sampling_is_continuous(planner):
    planner.plan(_square())
    times = np.linspace(0, planner.total_time_s, 1500)
    points = np.array([planner.sample(t)[0] for t in times])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dt = times[1] - times[0]
    assert steps.max() < 300.0 * dt * 1.5


def test_sampling_hits_endpoints(planner):
    moves = _square()
    planner.plan(moves)
    start, _, _ = planner.sample(0.0)
    end, _, _ = planner.sample(planner.total_time_s)
    assert np.allclose(start, moves[0].start)
    assert np.allclose(end, moves[-1].end)


def test_speed_is_zero_at_both_ends(planner):
    planner.plan(_square())
    assert planner.sample(0.0)[1] == pytest.approx(0.0)
    assert planner.sample(planner.total_time_s)[1] == pytest.approx(0.0)


def test_zero_length_moves_are_dropped(planner):
    same = np.array([1200., 800., 350.])
    planned = planner.plan([Move(same, same.copy(), 100.0)])
    assert planned == []


# --------------------------------------------------------------------------- #
#  G-code
# --------------------------------------------------------------------------- #
def _parse(text, start=(3000, 2500, 1000), feed=100.0):
    return parse(text, start_pose=np.array(start, float), default_feed_mms=feed,
                 rapid_feed_mms=300.0)


def test_absolute_and_relative_modes():
    program = _parse("G90\nG1 X1000 Y1000\nG91\nG1 X500\nG90\nG1 X2000")
    ends = [m.end.tolist() for m in program.moves]
    assert ends[0][:2] == [1000.0, 1000.0]
    assert ends[1][:2] == [1500.0, 1000.0]
    assert ends[2][:2] == [2000.0, 1000.0]


def test_feed_is_converted_from_per_minute():
    """В G-code подача задаётся на минуту, внутри системы всё в секундах."""
    program = _parse("F3000\nG1 X1000")
    assert program.moves[0].feed_mms == pytest.approx(50.0)


def test_rapid_uses_its_own_feed():
    program = _parse("F600\nG0 X1000\nG1 Y1000")
    assert program.moves[0].feed_mms == pytest.approx(300.0)
    assert program.moves[1].feed_mms == pytest.approx(10.0)


def test_inch_mode():
    program = _parse("G20\nG90\nG1 X10")
    assert program.moves[0].end[0] == pytest.approx(254.0)


def test_g92_shifts_origin():
    program = _parse("G90\nG1 X1000\nG92 X0\nG1 X500")
    assert program.moves[1].end[0] == pytest.approx(1500.0)


def test_arc_is_split_into_chords():
    program = _parse("G90\nG1 X4000 Y3500\nG3 X3500 Y4000 I-500 J0")
    arc_moves = [m for m in program.moves if m.line == 3]
    assert len(arc_moves) > 5
    centre = np.array([3500.0, 3500.0])
    for move in arc_moves:
        assert np.linalg.norm(move.end[:2] - centre) == pytest.approx(500.0, abs=1.0)


def test_arc_direction_matters():
    cw = _parse("G90\nG1 X4000 Y3500\nG2 X3500 Y4000 I-500 J0")
    ccw = _parse("G90\nG1 X4000 Y3500\nG3 X3500 Y4000 I-500 J0")
    assert cw.path_length_mm > ccw.path_length_mm


def test_inconsistent_arc_is_rejected():
    program = _parse("G90\nG1 X4000 Y3500\nG2 X3000 Y4000 I-500 J0")
    assert not program.ok
    assert "радиус" in program.issues[0].message


def test_unsupported_codes_are_reported_not_ignored():
    """Молча пропустить непонятый код на машине с киловаттными моторами нельзя."""
    program = _parse("G1 X100\nG33 X5\nM99")
    assert not program.ok
    messages = " ".join(i.message for i in program.issues)
    assert "G33" in messages and "M99" in messages


def test_dwell_and_pause_are_operations():
    from cdpr.gcode import Dwell, Pause

    program = _parse("G1 X100\nG4 P2.5\nM0")
    assert any(isinstance(op, Dwell) and op.seconds == 2.5 for op in program.operations)
    assert any(isinstance(op, Pause) for op in program.operations)


def test_comments_are_ignored():
    program = _parse("; комментарий\nG1 X100 (тоже комментарий)\n")
    assert program.ok and len(program.moves) == 1


def test_home_request_is_detected():
    assert _parse("G28").home_requested


def test_program_bounds_and_summary():
    program = _parse("G90\nG1 X1000 Y1000\nG1 X4000 Y3000")
    low, high = program.bounds
    assert low[0] == pytest.approx(1000.0)
    assert high[1] == pytest.approx(3000.0)
    assert "перемещений 2" in program.summary()
