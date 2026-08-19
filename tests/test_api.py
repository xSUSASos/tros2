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


def test_enable_reports_how_permission_works(client):
    """Панель обязана показывать, может ли софт снять разрешение сам:
    от этого зависит, что на самом деле делает кнопка аварийного стопа."""
    data = client.post("/api/enable", json={"on": True}).json()
    assert data["enabled"] is True
    assert "automatic" in data and "note" in data


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
    """Ехать в точку, где платформу не удержать, нельзя — и отказ должен
    объяснять причину, а не просто возвращать ошибку."""
    response = client.post("/api/mdi", json={"x": 300, "y": 300, "z": 1000})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "удержать" in detail or "запас" in detail or "габарит" in detail


def test_mdi_accepts_reachable_target(client):
    client.post("/api/enable", json={"on": True})
    response = client.post("/api/mdi", json={"x": 3000, "y": 2500, "z": 1100, "feed_mms": 150})
    assert response.status_code == 200
    assert response.json()["margin_n"] > 0


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
    text = "G21 G90\nF3000\nG1 X2600 Y2100 Z1100\nG1 X3400\nG1 Y2900\n"
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


def test_calibration_requires_points(client):
    client.post("/api/calibration/clear")
    response = client.post("/api/calibration/solve", json={"apply": False})
    assert response.status_code == 400
    assert "две точки" in response.json()["detail"]


def test_workspace_map_is_served(client):
    data = client.get("/api/workspace?z=1000&step=600").json()
    assert len(data["margin_n"]) == len(data["ys"])
    assert 0.0 <= data["area_fraction"] <= 1.0


def test_best_height_is_reported(client):
    data = client.get("/api/workspace/best_height").json()
    assert data["z_mm"] > 0 and data["margin_n"] > 0
