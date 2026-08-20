"""Контур управления, режимы и безопасность — на полной модели машины."""
from __future__ import annotations

import numpy as np
import pytest

from cdpr.calibration import CalibrationPoint, identify
from cdpr.modes.admittance import AdmittanceMode
from cdpr.modes.base import IdleMode
from cdpr.modes.homing import LandingProbe
from cdpr.modes.manual import JogMode, MdiMode
from cdpr.modes.tensioning import AutoTensionMode
from cdpr.runtime import build_runtime
from cdpr.state import Health


@pytest.fixture
def rt():
    # Виртуальные часы: время модели двигает тест, а не системный таймер.
    # Иначе результат зависел бы от того, насколько загружена машина.
    runtime = build_runtime(simulated=True, virtual_clock=True,
                            sim_options={"latency_ms": 0.0})
    runtime.drives.open()
    runtime.drives.initialize()
    runtime.controller.enable(True)
    runtime.step(0.02)
    yield runtime
    runtime.drives.close()


def spin(rt, cycles=400, dt=0.02):
    """Прогоняет заданное МОДЕЛЬНОЕ время."""
    state = None
    for _ in range(cycles):
        state = rt.step(dt)
    return state


# --------------------------------------------------------------------------- #
#  Измерение
# --------------------------------------------------------------------------- #
def test_platform_hangs_in_equilibrium(rt):
    """Модель обязана держать вес: сумма вертикальных проекций натяжений
    равна весу платформы. Если это не так, все расчёты выше бессмысленны."""
    st = rt.step(0.02)
    kin = rt.controller.kinematics
    unit = kin.unit_vectors(st.pose_mm)
    vertical = float(np.sum(st.tensions_n * unit[:, 2]))
    assert vertical == pytest.approx(rt.machine.platform.mass_kg * 9.80665, rel=0.02)


def test_forward_kinematics_stays_below_anchors(rt):
    """У подвеса есть зеркальное решение выше плоскости якорей, физически
    невозможное. Оно не должно выбираться никогда."""
    st = rt.step(0.02)
    assert st.pose_mm[2] < min(a.pos[2] for a in rt.machine.geometry.anchors)
    assert st.fk_residual_mm < 1.0


# --------------------------------------------------------------------------- #
#  Режимы
# --------------------------------------------------------------------------- #
def test_mdi_reaches_target(rt):
    start = rt.step(0.02).pose_mm.copy()
    target = start + np.array([500.0, -400.0, 150.0])
    rt.controller.set_mode(MdiMode(target, feed_mms=250.0))
    st = spin(rt, 600)
    assert np.linalg.norm(st.pose_mm - target) < 5.0


def test_jog_accumulates_steps(rt):
    start = rt.step(0.02).pose_mm.copy()
    jog = JogMode(feed_mms=200.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    for _ in range(3):
        jog.step([100.0, 0.0, 0.0])
    st = spin(rt, 500)
    assert st.pose_mm[0] - start[0] == pytest.approx(300.0, abs=8.0)


def test_jog_target_is_clamped_to_workspace(rt):
    jog = JogMode(feed_mms=200.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    jog.step([99_000.0, 0.0, 0.0])
    st = spin(rt, 100)
    assert st.target_mm[0] <= rt.controller.box_high[0] + 1e-6


def test_idle_still_holds_tension(rt):
    """Стоять на месте и дать тросам провиснуть — значит потерять
    управляемость там, где её труднее всего вернуть."""
    rt.controller.set_mode(IdleMode())
    rt.controller.set_target_tension(18.0)
    st = spin(rt, 500, dt=0.01)
    assert st.tensions_n.min() > rt.machine.tension.min_n


def test_target_tension_changes_the_weakest_cable(rt):
    """Ползунок натяжения обязан реально что-то менять: свободна ровно одна
    комбинация — перетяжка диагональных пар, ею и пользуемся."""
    rt.controller.set_mode(IdleMode())
    rt.controller.set_target_tension(15.0)
    low = spin(rt, 500).tensions_n.copy()
    assert low.max() - low.min() > 8.0, "перетяжка диагональных пар должна появиться"
    assert low.min() < 24.0, "самый слабый трос должен пойти к цели"


def test_tension_choice_does_not_chatter(rt):
    """В симметричной раме цель достижима двумя способами. Выбирать между
    ними случайно нельзя: решение заскачет, и приводы будут дёргаться."""
    rt.controller.set_mode(IdleMode())
    rt.controller.set_target_tension(20.0)
    spin(rt, 200)
    pattern = []
    commands = []
    for _ in range(80):
        st = rt.step(0.02)
        target = rt.controller.state.target_tensions_n
        pattern.append(np.sign(target[0] - target[1]))
        commands.append(st.commands_rpm[0])
    assert len(set(pattern)) == 1, "выбор перетягиваемой диагонали не должен скакать"
    assert max(abs(c) for c in commands) < 0.5, "после схождения приводы должны стоять"


def test_autotension_stops_when_reached(rt):
    mode = AutoTensionMode(target_n=20.0, feed_mms=20.0)
    rt.controller.set_mode(mode)
    spin(rt, 300)
    assert rt.controller.state.tensions_n.min() > 5.0


def test_admittance_measures_external_force(rt):
    """Без внешнего усилия остаток должен быть около нуля: иначе платформа
    поедет сама, и мёртвая зона окажется бессмысленной."""
    mode = AdmittanceMode()
    rt.controller.set_mode(mode)
    rt.step(0.02)
    force = mode.external_force(rt.controller)
    assert np.linalg.norm(force) < mode.deadband


def test_admittance_does_not_drift_in_deadband(rt):
    rt.controller.set_mode(AdmittanceMode())
    start = spin(rt, 50).pose_mm.copy()
    end = spin(rt, 400).pose_mm
    assert np.linalg.norm(end - start) < 15.0


# --------------------------------------------------------------------------- #
#  Калибровка от начала до конца
# --------------------------------------------------------------------------- #
def test_landing_is_detected_by_tension_collapse(rt):
    """Признак касания — падение натяжения СРАЗУ НА ВСЕХ тросах. Это гораздо
    чётче, чем скачок момента при боковом упоре: скачок легко спутать с
    рывком или трением, а одновременную разгрузку четырёх тросов — нет."""
    probe = LandingProbe(feed_mms=120.0, settle_s=0.2, timeout_s=40.0)
    rt.controller.set_mode(probe)
    for _ in range(40):
        rt.step(0.02)
    assert probe.baseline is not None
    assert probe.phase == "спуск"

    # опускаем платформу до пола: модель считает касание за счёт того, что
    # тросы стравливаются и натяжение уходит
    for _ in range(2000):
        rt.step(0.02)
        if probe.landed_counts is not None:
            break
    assert probe.landed_counts is not None, "касание не зафиксировано"
    assert probe.landed_tensions.max() < probe.baseline.max()


def test_calibration_recovers_position_accuracy(machine):
    """Проверка по назначению: после калибровки по точкам посадки положение должно
    считаться с точностью, сравнимой с точностью разметки этих точек посадки."""
    import numpy as np

    from cdpr.kinematics import CDPRKinematics
    from cdpr.line import LineModel

    rng = np.random.default_rng(11)
    kin = CDPRKinematics.from_config(machine)
    truth = [
        LineModel(w.model_copy(update={
            "count_empty": 1_000_000 * (i + 1), "length_at_empty_mm": 11800.0 + 120.0 * i,
            "ea_n": 4300.0 + 100.0 * i}))
        for i, w in enumerate(machine.ordered_winches())
    ]

    points = []
    for target in ([1200, 1000, 60], [4800, 1000, 60], [4800, 4000, 60],
                   [1200, 4000, 60], [3000, 2500, 60]):
        position = np.array(target, float)
        distance = kin.inverse(position)
        tensions = np.full(4, 25.0)
        counts = np.array([
            truth[i].counts_from_length(distance[i] / (1 + tensions[i] / truth[i].winch.ea_n))
            for i in range(4)
        ]) + rng.normal(0, 44_000, 4)   # шум около миллиметра троса
        points.append(CalibrationPoint(position, counts, tensions))

    result = identify(machine, points, kinematics=kin)
    assert result.residual_rms_mm < 5.0

    calibrated = [
        LineModel(w.model_copy(update={
            "count_empty": result.count_empty[i],
            "length_at_empty_mm": result.length_at_empty_mm[i],
            "ea_n": result.ea_n[i] if result.ea_n else None}))
        for i, w in enumerate(machine.ordered_winches())
    ]
    for check in ([2000, 2000, 800], [4000, 3000, 1200]):
        position = np.array(check, float)
        distance = kin.inverse(position)
        counts = [truth[i].counts_from_length(distance[i] / (1 + 25.0 / truth[i].winch.ea_n))
                  for i in range(4)]
        lengths = np.array([
            calibrated[i].stretched(calibrated[i].length_from_counts(int(counts[i])), 25.0)
            for i in range(4)
        ])
        estimated, _ = kin.forward(lengths, guess=position + 100.0)
        assert np.linalg.norm(estimated - position) < 8.0


def test_calibration_needs_enough_points(machine):
    import numpy as np

    with pytest.raises(ValueError, match="минимум две"):
        identify(machine, [CalibrationPoint(np.zeros(3), np.zeros(4))])


def test_calibration_warns_when_degenerate(machine):
    """Если барабан не выходит за первый слой, разделить параметры нельзя.
    Система обязана сказать об этом, а не выдать красивые, но пустые числа."""
    import numpy as np

    from cdpr.kinematics import CDPRKinematics
    from cdpr.line import LineModel

    kin = CDPRKinematics.from_config(machine)
    truth = [LineModel(w.model_copy(update={"count_empty": 0, "length_at_empty_mm": 12000.0}))
             for w in machine.ordered_winches()]
    points = []
    for target in ([2000, 2000, 60], [4000, 3000, 60], [3000, 2000, 60]):
        position = np.array(target, float)
        distance = kin.inverse(position)
        counts = np.array([truth[i].counts_from_length(distance[i]) for i in range(4)])
        points.append(CalibrationPoint(position, counts))
    result = identify(machine, points, fit_elasticity=False, kinematics=kin)
    assert any("вырождена" in w or "натяжения" in w for w in result.warnings)
