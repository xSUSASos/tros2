"""Контур управления, режимы и привязка — на полной модели машины."""
from __future__ import annotations

import numpy as np
import pytest

from cdpr.calibration import park_pose, solve_from_corners
from cdpr.kinematics import CDPRKinematics
from cdpr.line import LineModel
from cdpr.modes.admittance import AdmittanceMode
from cdpr.modes.base import IdleMode
from cdpr.modes.cable import CableMode
from cdpr.modes.homing import CornerHoming
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
    runtime.controller.allow_motion(True)
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
    равна весу коробки. Если это не так, все расчёты выше бессмысленны."""
    rt.step(0.02)
    kin = rt.controller.kinematics
    # Берём натяжения самой модели, а не восстановленные из момента: момент
    # квантован целыми процентами и включает трение, поэтому проверять по нему
    # равновесие — значит проверять точность датчика, а не физику.
    unit = kin.unit_vectors(rt.platform.pose)
    vertical = float(np.sum(rt.platform.tensions * unit[:, 2]))
    assert vertical == pytest.approx(rt.machine.platform.mass_kg * 9.80665, rel=0.05)


def test_tension_matches_the_sag(rt):
    """Натяжение задаётся геометрией, а не настройкой. При провисе 350 мм и
    коробке 600 г в центре рамы выходит около 9 Н на трос — если модель даёт
    заметно другое, врут либо провис, либо масса."""
    st = spin(rt, 50)
    assert 5.0 < float(np.mean(st.tensions_n)) < 15.0


def test_pose_lies_in_the_working_plane(rt):
    """Z не управляется: прямая задача обязана возвращать рабочую плоскость,
    а не гадать высоту."""
    st = rt.step(0.02)
    assert st.pose_mm[2] == pytest.approx(rt.machine.geometry.plane_z_mm)
    assert st.pose_mm[2] < min(a.pos[2] for a in rt.machine.geometry.anchors)
    assert st.fk_residual_mm < 10.0


def test_closed_form_agrees_with_the_solver(rt):
    """Замкнутая форма и общий решатель обязаны давать одно и то же.
    Замкнутая при этом не нуждается ни в начальном приближении, ни в границах,
    и не имеет зеркального решения выше плоскости якорей."""
    kin = rt.controller.kinematics
    plane = rt.machine.geometry.plane_z_mm
    for target in ([700.0, 500.0], [1990.0, 960.0], [3400.0, 1500.0]):
        truth = np.array([target[0], target[1], plane])
        lengths = kin.inverse(truth)
        fast, residual = kin.forward_planar(lengths, plane)
        assert np.linalg.norm(fast - truth) < 0.01
        assert residual < 0.01


def test_inconsistent_lengths_show_up_as_residual(rt):
    """Невязка — единственный признак того, что трос провис или проскользнул.
    Она обязана расти, а не прятаться."""
    kin = rt.controller.kinematics
    plane = rt.machine.geometry.plane_z_mm
    lengths = kin.inverse(np.array([1990.0, 960.0, plane]))
    _, clean = kin.forward_planar(lengths, plane)
    lengths[2] += 40.0
    _, dirty = kin.forward_planar(lengths, plane)
    assert clean < 0.01
    assert dirty > 5.0


# --------------------------------------------------------------------------- #
#  Режимы
# --------------------------------------------------------------------------- #
def test_mdi_reaches_target(rt):
    start = rt.step(0.02).pose_mm.copy()
    target = start + np.array([500.0, -300.0, 0.0])
    rt.controller.set_mode(MdiMode(target, feed_mms=150.0, tolerance_mm=8.0))
    st = spin(rt, 800)
    assert np.linalg.norm(st.pose_mm - target) < 15.0


def test_jog_accumulates_steps(rt):
    start = rt.step(0.02).pose_mm.copy()
    jog = JogMode(feed_mms=150.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    for _ in range(3):
        jog.step([100.0, 0.0, 0.0])
    st = spin(rt, 800)
    assert st.pose_mm[0] - start[0] == pytest.approx(300.0, abs=20.0)


def test_jog_target_is_clamped_to_workspace(rt):
    jog = JogMode(feed_mms=150.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    jog.step([99_000.0, 0.0, 0.0])
    st = spin(rt, 100)
    assert st.target_mm[0] <= rt.controller.box_high[0] + 1e-6


def test_z_is_never_commanded(rt):
    """Коробка ходит в одной плоскости. Шаг по Z обязан не иметь последствий,
    а не тихо увести её вверх или вниз."""
    plane = rt.machine.geometry.plane_z_mm
    jog = JogMode(feed_mms=150.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    jog.step([0.0, 0.0, 500.0])
    st = spin(rt, 200)
    assert st.target_mm[2] == pytest.approx(plane)
    assert st.pose_mm[2] == pytest.approx(plane)


def test_idle_still_holds_tension(rt):
    """Стоять на месте и дать тросам провиснуть — значит потерять
    управляемость там, где её труднее всего вернуть."""
    rt.controller.set_mode(IdleMode())
    st = spin(rt, 500, dt=0.01)
    assert st.tensions_n.min() > rt.machine.tension.min_n


def test_idle_does_not_chatter(rt):
    """После схождения приводы обязаны стоять: дёрганье вокруг цели изнашивает
    механику и раскачивает коробку на мягком тросе."""
    rt.controller.set_mode(IdleMode())
    spin(rt, 400)
    commands = [abs(rt.step(0.02).commands_rpm).max() for _ in range(60)]
    assert max(commands) < 2.0


def test_autotension_stops_when_reached(rt):
    mode = AutoTensionMode(target_n=12.0, feed_mms=15.0)
    rt.controller.set_mode(mode)
    spin(rt, 400)
    assert rt.controller.state.tensions_n.min() > rt.machine.tension.min_n


def test_cable_mode_needs_no_calibration(rt):
    """Первое, что должно работать на новой машине: покрутить барабан кнопкой.
    Ни привязки, ни геометрии, ни рабочей зоны для этого не нужно."""
    mode = CableMode(rt.drives.n_axes)
    assert mode.requires_homing is False
    rt.controller.set_mode(mode)
    rt.step(0.02)
    mode.set_speed(1, -20.0)
    st = spin(rt, 20)
    assert st.commands_rpm[1] != 0.0
    assert abs(st.commands_rpm[0]) < 1e-9
    mode.stop_all()
    st = spin(rt, 5)
    assert abs(st.commands_rpm).max() < 1e-9


def test_admittance_measures_external_force(rt):
    """Без внешнего усилия остаток должен быть около нуля: иначе платформа
    поедет сама, и мёртвая зона окажется бессмысленной."""
    mode = AdmittanceMode()
    rt.controller.set_mode(mode)
    rt.step(0.02)
    force = mode.external_force(rt.controller)
    assert np.linalg.norm(force) < mode.deadband


# --------------------------------------------------------------------------- #
#  Защиты
# --------------------------------------------------------------------------- #
def test_motion_gate_blocks_commands(rt):
    """Программное разрешение — предохранитель от случайной команды. Пока оно
    снято, на шину уходят нули, что бы ни насчитал режим."""
    jog = JogMode(feed_mms=150.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    jog.step([300.0, 0.0, 0.0])
    rt.controller.allow_motion(False)
    st = spin(rt, 40)
    assert abs(st.commands_rpm).max() == 0.0
    assert any("не разрешено" in m for m in st.messages)


def test_estop_zeroes_commands(rt):
    jog = JogMode(feed_mms=150.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    jog.step([300.0, 0.0, 0.0])
    rt.controller.estop("проверка")
    st = spin(rt, 10)
    assert st.health is Health.ESTOP
    assert abs(st.commands_rpm).max() == 0.0


def test_bus_silence_zeroes_commands(rt):
    """Шина замолчала — считать не по чему, и продолжать выдавать прежние
    уставки нельзя: мотор крутился бы вслепую."""
    jog = JogMode(feed_mms=150.0)
    rt.controller.set_mode(jog)
    rt.step(0.02)
    jog.step([300.0, 0.0, 0.0])
    spin(rt, 20)
    rt.drives.close()          # шина легла
    st = spin(rt, 40)
    assert abs(st.commands_rpm).max() == 0.0
    assert st.health in (Health.FAULT, Health.ESTOP)
    rt.drives.open()


# --------------------------------------------------------------------------- #
#  Привязка от начала до конца
# --------------------------------------------------------------------------- #
def test_homing_pulls_the_box_into_the_corner(rt):
    """Хоминг работает напрямую скоростями тросов и положением не пользуется —
    иначе привязаться было бы нечем. Признак упора: натяжение ведущего троса
    выросло выше порога."""
    start = rt.step(0.02).pose_mm.copy()
    mode = CornerHoming([0])
    rt.controller.set_mode(mode)
    # Ход через всю раму на скорости выборки троса — это минуты модельного
    # времени, а не секунды.
    for _ in range(12000):
        rt.step(0.02)
        if mode.records:
            break
    assert mode.records, "коробка так и не упёрлась в модуль"

    record = mode.records[0]
    assert record.corner == 0
    assert record.tensions_n[0] > record.tensions_n[1:].max()

    end = rt.controller.state.pose_mm
    anchor = rt.controller.kinematics.anchors[0]
    assert np.linalg.norm(end[:2] - anchor[:2]) < np.linalg.norm(start[:2] - anchor[:2])


def test_one_corner_is_enough_to_restore_position(machine):
    """Проверка по назначению: после привязки по одному углу положение должно
    считаться с точностью, сравнимой с точностью замера отступа от модуля."""
    kin = CDPRKinematics.from_config(machine)
    plane = machine.geometry.plane_z_mm
    ea = 200.0

    truth = [
        LineModel(w.model_copy(update={
            "count_ref": 1_000_000 * (i + 1),
            "length_at_ref_mm": 6000.0 + 130.0 * i,
            "ea_n": ea,
        }))
        for i, w in enumerate(machine.ordered_winches())
    ]

    def counts_at(pose, tension_n):
        distance = kin.inverse(pose)
        return np.array([
            truth[i].counts_from_length(float(distance[i]) / (1.0 + tension_n / ea))
            for i in range(4)
        ], dtype=float)

    corner = 0
    park = park_pose(machine, corner, kin)
    record_tension = np.full(4, 9.0)
    from cdpr.calibration import CornerRecord

    result = solve_from_corners(
        machine,
        [CornerRecord(corner, counts_at(park, 9.0), record_tension)],
        ea_n=ea, kinematics=kin,
    )
    assert result.ok

    calibrated = [
        LineModel(w.model_copy(update={
            "count_ref": result.count_ref[i],
            "length_at_ref_mm": result.length_at_ref_mm[i],
            "ea_n": ea,
        }))
        for i, w in enumerate(machine.ordered_winches())
    ]

    for check in ([1200.0, 700.0], [3000.0, 1400.0]):
        pose = np.array([check[0], check[1], plane])
        counts = counts_at(pose, 9.0)
        lengths = np.array([
            calibrated[i].stretched(calibrated[i].length_from_counts(int(counts[i])), 9.0)
            for i in range(4)
        ])
        estimated, residual = kin.forward_planar(lengths, plane)
        assert np.linalg.norm(estimated - pose) < 5.0


def test_extra_corners_reveal_a_wrong_frame(machine):
    """Лишние углы — это проверка. Если стороны рамы введены неверно, отсчёты
    в остальных углах не сойдутся, и система обязана сказать об этом, а не
    выдать красивые, но неверные числа."""
    from cdpr.calibration import CornerRecord

    kin = CDPRKinematics.from_config(machine)
    ea = 200.0
    truth = [
        LineModel(w.model_copy(update={
            "count_ref": 500_000 * (i + 1), "length_at_ref_mm": 6000.0, "ea_n": ea,
        }))
        for i, w in enumerate(machine.ordered_winches())
    ]

    # «Настоящая» рама на 150 мм шире той, что записана в конфиге.
    wrong = CDPRKinematics(kin.anchors + np.array([[0, 0, 0], [150.0, 0, 0],
                                                   [150.0, 0, 0], [0, 0, 0]]))

    records = []
    for corner in (0, 1, 3):
        park = park_pose(machine, corner, kin)
        distance = wrong.inverse(park)
        counts = np.array([
            truth[i].counts_from_length(float(distance[i]) / (1.0 + 9.0 / ea))
            for i in range(4)
        ], dtype=float)
        records.append(CornerRecord(corner, counts, np.full(4, 9.0)))

    result = solve_from_corners(machine, records, ea_n=ea, kinematics=kin)
    assert result.residual_rms_mm > 8.0
    assert result.warnings, "расхождение обязано быть названо вслух"


def test_calibration_needs_at_least_one_corner(machine):
    with pytest.raises(ValueError, match="хотя бы один"):
        solve_from_corners(machine, [])
