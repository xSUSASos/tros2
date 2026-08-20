"""Веб-интерфейс: команды, настройки, защита от неверных запросов."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from cdpr.runtime import build_runtime


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Панель правит machine.yaml, поэтому тест обязан работать с КОПИЕЙ:
    # иначе прогон тестов молча меняет настройки реальной машины.
    import shutil

    from cdpr.config import DEFAULT_MACHINE

    sandbox = tmp_path_factory.mktemp("config") / "machine.yaml"
    shutil.copy2(DEFAULT_MACHINE, sandbox)
    runtime = build_runtime(sandbox, simulated=True, virtual_clock=True,
                            sim_options={"latency_ms": 0.0})
    runtime.drives.open()
    runtime.drives.initialize()
    runtime.step(0.02)
    app = create_app(runtime)
    with TestClient(app) as test_client:
        test_client.runtime = runtime
        yield test_client
    runtime.drives.close()


def test_info_describes_the_machine(client):
    data = client.get("/api/info").json()
    assert data["n_cables"] == 4
    assert data["simulated"] is True
    assert len(data["anchors"]) == 4


def test_state_has_everything_the_panel_needs(client):
    data = client.get("/api/state").json()
    for key in ("mode", "health", "pose_mm", "tensions_n", "lengths_mm", "online", "alarms"):
        assert key in data


def test_enable_is_a_software_gate_only(client):
    """Разрешения приводов по Modbus у T3D нет: servo-on держит P-098, а
    питание снимает физическая кнопка. Кнопка в панели — программный
    предохранитель, и панель обязана говорить об этом прямо, а не изображать
    силовое разрешение."""
    data = client.post("/api/enable", json={"on": True}).json()
    assert data["enabled"] is True
    assert "питание" in data["note"]


def test_estop_and_recovery(client):
    assert client.post("/api/estop", json={"reason": "тест"}).json()["estop"] is True
    client.runtime.step(0.02)
    assert client.get("/api/state").json()["health"] == "estop"
    assert client.post("/api/estop/clear", json={}).json()["estop"] is False
    client.runtime.step(0.02)
    assert client.get("/api/state").json()["health"] != "estop"


def test_enable_refused_while_estopped(client):
    client.post("/api/estop", json={"reason": "тест"})
    response = client.post("/api/enable", json={"on": True})
    assert response.status_code == 409
    assert "аварийный стоп" in response.json()["detail"]
    client.post("/api/estop/clear", json={})


def test_mdi_rejects_unreachable_target(client):
    """Ехать в точку, где коробку не удержать, нельзя — и отказ должен
    объяснять причину, а не просто возвращать ошибку."""
    low = client.runtime.controller.box_low
    response = client.post("/api/mdi", json={"x": low[0] + 1, "y": low[1] + 1, "z": 0})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "удержать" in detail or "запас" in detail or "габарит" in detail


def test_mdi_accepts_reachable_target(client):
    client.post("/api/enable", json={"on": True})
    centre = 0.5 * (client.runtime.controller.box_low + client.runtime.controller.box_high)
    response = client.post("/api/mdi",
                           json={"x": centre[0], "y": centre[1], "z": 0, "feed_mms": 100})
    assert response.status_code == 200
    assert response.json()["margin_n"] > 0


def test_mdi_ignores_z_on_a_planar_machine(client):
    """Z не управляется: что бы ни прислала панель, коробка остаётся в
    рабочей плоскости."""
    machine = client.runtime.machine
    if not machine.geometry.is_planar:
        pytest.skip("машина не плоская")
    centre = 0.5 * (client.runtime.controller.box_low + client.runtime.controller.box_high)
    data = client.post("/api/mdi",
                       json={"x": centre[0], "y": centre[1], "z": 99_000}).json()
    assert data["target"][2] == pytest.approx(machine.geometry.plane_z_mm)


def test_jog_and_tension(client):
    assert client.post("/api/jog/step", json={"dx": 50}).json()["ok"]
    data = client.post("/api/tension/target", json={"target_n": 18}).json()
    assert data["target_n"] == 18.0


def test_gcode_rejects_broken_program(client):
    response = client.post("/api/gcode/load", json={"text": "G1 X100\nG33 X5\n"}).json()
    assert response["ok"] is False
    assert any("G33" in issue for issue in response["issues"])


def test_gcode_rejects_program_outside_workspace(client):
    response = client.post("/api/gcode/load",
                           json={"text": "G90\nG1 X200 Y200 Z1000\n"}).json()
    assert response["ok"] is False
    assert response["workspace_problems"]


def test_gcode_accepts_good_program_and_plans_it(client):
    text = "G21 G90\nF3000\nG1 X1400 Y800\nG1 X2400\nG1 Y1100\n"
    response = client.post("/api/gcode/load", json={"text": text}).json()
    assert response["ok"] is True
    assert response["duration_s"] > 0
    assert len(response["path"]) == response["moves"]


def test_gcode_run_requires_loaded_program(client):
    client.post("/api/gcode/load", json={"text": "G33 X1"})
    assert client.post("/api/gcode/run").status_code == 409


def test_config_exposes_derived_numbers(client):
    """В настройках должны быть видны и производные величины — например,
    какое усилие мотор способен дать на тросе при полном моменте."""
    config = client.get("/api/config").json()
    winch = config["winches"][0]
    assert winch["force_at_full_torque_n"] > 0
    assert winch["turns_per_layer"] > 0
    assert "max_line_speed_mms" in winch


def test_config_patch_applies_live_settings(client):
    response = client.post("/api/config", json={"updates": {"tension.target_n": 24.0}}).json()
    assert response["ok"] and response["applied_now"]
    assert client.get("/api/config").json()["tension"]["target_n"] == 24.0


def test_config_patch_rejects_typos(client):
    response = client.post("/api/config", json={"updates": {"tension.targt_n": 1}})
    assert response.status_code == 400
    assert "опечатка" in response.json()["detail"]


def test_homing_solve_requires_a_visited_corner(client):
    """Привязку не из чего считать, пока коробка не побывала ни в одном углу."""
    client.post("/api/homing/abort")
    response = client.post("/api/homing/solve", json={"apply": False})
    assert response.status_code == 400
    assert "угол" in response.json()["detail"]


def test_homing_starts_and_reports_progress(client):
    """Хоминг обязан идти без привязки: он её и добывает."""
    started = client.post("/api/homing/start", json={"corners": [0]}).json()
    assert started["ok"] and started["corners"] == [0]
    client.runtime.step(0.02)
    status = client.get("/api/homing/status").json()
    assert status["running"] and status["corner"] == 0
    assert client.post("/api/homing/abort").json()["ok"]


def test_cable_mode_works_without_calibration(client):
    """Первое, что должно работать на новой машине: покрутить барабан кнопкой."""
    data = client.post("/api/cable/speed", json={"index": 2, "speed_mms": -15.0}).json()
    assert data["velocity_mms"][2] == pytest.approx(-15.0)
    assert client.post("/api/cable/stop").json()["ok"]


def test_workspace_map_is_served(client):
    data = client.get("/api/workspace?step=400").json()
    assert len(data["margin_n"]) == len(data["ys"])
    assert 0.0 <= data["area_fraction"] <= 1.0


def test_best_height_is_reported(client):
    data = client.get("/api/workspace/best_height").json()
    assert data["z_mm"] > 0 and data["margin_n"] > 0
