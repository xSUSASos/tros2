"""Веб-сервер панели управления.

REST для команд и настроек, WebSocket для телеметрии. Состояние машины
рассылается тем, кто подключён, с частотой заметно ниже цикла управления:
человеку хватает двадцати кадров в секунду, а грузить шину лишними задачами
незачем.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import schemas
from cdpr import gcode as gcode_module
from cdpr.calibration import CalibrationPoint, identify
from cdpr.config import ConfigError, load_machine, patch_yaml, save_calibration
from cdpr.modes.admittance import AdmittanceMode
from cdpr.modes.base import IdleMode
from cdpr.modes.homing import LandingProbe
from cdpr.modes.manual import JogMode, MdiMode
from cdpr.modes.program import GcodeMode, check_program_fits
from cdpr.modes.tensioning import AutoTensionMode
from cdpr.runtime import Runtime
from cdpr.trajectory import TrajectoryPlanner
from cdpr.workspace import best_height, compute_map

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class TelemetryHub:
    """Рассылка состояния подключённым панелям."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.latest: dict[str, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, state) -> None:
        """Вызывается из потока управления, поэтому кладём в цикл событий."""
        self.latest = state.as_dict()

    async def pump(self, interval: float = 0.05) -> None:
        while True:
            await asyncio.sleep(interval)
            if not self.clients or not self.latest:
                continue
            message = json.dumps({"type": "state", "data": self.latest}, ensure_ascii=False)
            for client in list(self.clients):
                try:
                    await client.send_text(message)
                except Exception:  # noqa: BLE001 — отвалившийся клиент просто убирается
                    self.clients.discard(client)


def create_app(runtime: Runtime) -> FastAPI:
    hub = TelemetryHub()
    runtime.controller.add_listener(hub.publish)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        hub.bind(asyncio.get_running_loop())
        task = asyncio.create_task(hub.pump())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Тросовая система", docs_url="/api/docs", lifespan=lifespan)

    state: dict[str, Any] = {
        "jog": None, "gcode": None, "program": None,
        "probe_points": [], "last_calibration": None, "workspace_cache": {},
    }

    def controller():
        return runtime.controller

    # ------------------------------------------------------------------ #
    #  Телеметрия
    # ------------------------------------------------------------------ #
    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        try:
            await ws.send_text(json.dumps(
                {"type": "hello", "data": _describe(runtime)}, ensure_ascii=False))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return controller().state.as_dict()

    @app.get("/api/info")
    def get_info() -> dict[str, Any]:
        return _describe(runtime)

    @app.get("/api/drives")
    def get_drives() -> dict[str, Any]:
        return {"stats": runtime.drives.stats(), "description": runtime.drives.describe()}

    # ------------------------------------------------------------------ #
    #  Безопасность и разрешение
    # ------------------------------------------------------------------ #
    @app.post("/api/estop")
    def estop(request: schemas.EstopRequest) -> dict[str, Any]:
        controller().estop(request.reason)
        return {"ok": True, "estop": True}

    @app.post("/api/estop/clear")
    def clear_estop() -> dict[str, Any]:
        controller().clear_estop()
        return {"ok": True, "estop": False}

    @app.post("/api/enable")
    def enable(request: schemas.EnableRequest) -> dict[str, Any]:
        try:
            controller().enable(request.on)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "enabled": request.on,
                "note": runtime.drives.enabler.describe(),
                "automatic": runtime.drives.enabler.is_automatic}

    @app.post("/api/mode/idle")
    def go_idle() -> dict[str, Any]:
        controller().set_mode(IdleMode())
        state["jog"] = None
        return {"ok": True, "mode": "idle"}

    # ------------------------------------------------------------------ #
    #  Ручное управление
    # ------------------------------------------------------------------ #
    def _ensure_jog() -> JogMode:
        mode = controller().mode
        if isinstance(mode, JogMode):
            return mode
        jog = JogMode(feed_mms=runtime.machine.motion.jog_feed_mms)
        controller().set_mode(jog)
        state["jog"] = jog
        return jog

    @app.post("/api/jog/step")
    def jog_step(request: schemas.JogStepRequest) -> dict[str, Any]:
        jog = _ensure_jog()
        if request.feed_mms:
            jog.feed = request.feed_mms
        jog.step([request.dx, request.dy, request.dz])
        return {"ok": True}

    @app.post("/api/jog/continuous")
    def jog_continuous(request: schemas.JogContinuousRequest) -> dict[str, Any]:
        jog = _ensure_jog()
        jog.set_continuous([request.vx, request.vy, request.vz])
        return {"ok": True}

    @app.post("/api/mdi")
    def mdi(request: schemas.MdiRequest) -> dict[str, Any]:
        target = np.array([request.x, request.y, request.z], dtype=float)
        from cdpr.workspace import check_pose

        ok, margin, why = check_pose(runtime.machine, controller().kinematics, target)
        if not ok:
            raise HTTPException(status_code=400, detail=why)
        controller().set_mode(MdiMode(target, feed_mms=request.feed_mms))
        return {"ok": True, "target": target.tolist(), "margin_n": round(margin, 1)}

    # ------------------------------------------------------------------ #
    #  Натяжение
    # ------------------------------------------------------------------ #
    @app.post("/api/tension/target")
    def set_tension(request: schemas.TensionRequest) -> dict[str, Any]:
        controller().set_target_tension(request.target_n)
        return {"ok": True, "target_n": controller().target_tension_n}

    @app.post("/api/tension/auto")
    def auto_tension(request: schemas.AutoTensionRequest) -> dict[str, Any]:
        controller().set_mode(AutoTensionMode(request.target_n, feed_mms=request.feed_mms))
        return {"ok": True}

    @app.post("/api/handguide")
    def hand_guide(request: schemas.HandGuideRequest) -> dict[str, Any]:
        if not request.on:
            controller().set_mode(IdleMode())
            return {"ok": True, "on": False}
        controller().set_mode(AdmittanceMode(
            gain_mms_per_n=request.gain_mms_per_n, deadband_n=request.deadband_n))
        return {"ok": True, "on": True,
                "note": ("частота цикла упирается в шину Modbus, поэтому отклик мягкий; "
                         "разбиение гирлянды на две шины его заметно улучшает")}

    # ------------------------------------------------------------------ #
    #  Рабочая зона
    # ------------------------------------------------------------------ #
    @app.get("/api/workspace")
    def workspace(z: float | None = None, step: float = 300.0, payload_kg: float = 0.0) -> dict[str, Any]:
        machine = runtime.machine
        height = z if z is not None else (
            controller().state.pose_mm[2] if controller().state.pose_mm is not None
            else 0.5 * (machine.workspace.z_min_mm + machine.workspace.z_max_mm)
        )
        key = (round(height, 1), step, payload_kg, machine.tension.min_n, machine.tension.max_n)
        cached = state["workspace_cache"].get(key)
        if cached is None:
            cached = compute_map(machine, controller().kinematics, z_mm=height,
                                 step_mm=step, payload_kg=payload_kg).as_dict()
            state["workspace_cache"] = {key: cached}
        return cached

    @app.get("/api/workspace/best_height")
    def workspace_best_height(payload_kg: float = 0.0) -> dict[str, Any]:
        z, margin = best_height(runtime.machine, controller().kinematics, payload_kg=payload_kg)
        return {"z_mm": round(z, 1), "margin_n": round(margin, 1)}

    # ------------------------------------------------------------------ #
    #  G-code
    # ------------------------------------------------------------------ #
    @app.post("/api/gcode/load")
    def gcode_load(request: schemas.GcodeLoadRequest) -> dict[str, Any]:
        pose = controller().state.pose_mm
        if pose is None:
            pose = 0.5 * (controller().box_low + controller().box_high)
        program = gcode_module.parse(
            request.text,
            start_pose=pose,
            default_feed_mms=runtime.machine.motion.jog_feed_mms,
            rapid_feed_mms=runtime.machine.motion.max_velocity_mms,
        )
        response: dict[str, Any] = {
            "ok": program.ok,
            "summary": program.summary(),
            "issues": [str(i) for i in program.issues],
            "moves": len(program.moves),
            "path": [
                {"from": m.start.tolist(), "to": m.end.tolist(), "line": m.line}
                for m in program.moves[:5000]
            ],
        }
        if not program.ok:
            state["program"] = None
            return response

        if request.check_workspace:
            problems = check_program_fits(program, runtime.machine, controller().kinematics)
            response["workspace_problems"] = problems
            if problems:
                response["ok"] = False
                state["program"] = None
                return response

        planner = TrajectoryPlanner(runtime.machine, controller().kinematics)
        planner.plan(program.moves)
        response["trajectory"] = planner.summary()
        response["duration_s"] = round(planner.total_time_s, 1)
        state["program"] = (program, planner)
        return response

    @app.post("/api/gcode/run")
    def gcode_run() -> dict[str, Any]:
        loaded = state.get("program")
        if not loaded:
            raise HTTPException(status_code=409, detail="программа не загружена или содержит ошибки")
        program, planner = loaded
        mode = GcodeMode(program, planner)
        state["gcode"] = mode
        controller().set_mode(mode)
        return {"ok": True, "duration_s": round(planner.total_time_s, 1)}

    @app.post("/api/gcode/pause")
    def gcode_pause() -> dict[str, Any]:
        mode = state.get("gcode")
        if isinstance(mode, GcodeMode):
            mode.pause()
        return {"ok": True}

    @app.post("/api/gcode/resume")
    def gcode_resume() -> dict[str, Any]:
        mode = state.get("gcode")
        if isinstance(mode, GcodeMode):
            mode.resume()
        return {"ok": True}

    @app.post("/api/gcode/stop")
    def gcode_stop() -> dict[str, Any]:
        mode = state.get("gcode")
        if isinstance(mode, GcodeMode):
            mode.stop()
        controller().set_mode(IdleMode())
        return {"ok": True}

    @app.post("/api/gcode/feed_override")
    def gcode_feed_override(request: schemas.FeedOverrideRequest) -> dict[str, Any]:
        mode = state.get("gcode")
        if isinstance(mode, GcodeMode):
            mode.set_feed_override(request.value)
        return {"ok": True, "value": request.value}

    @app.get("/api/gcode/progress")
    def gcode_progress() -> dict[str, Any]:
        mode = state.get("gcode")
        if not isinstance(mode, GcodeMode):
            return {"running": False}
        return {"running": True, "progress": round(mode.progress, 4),
                "feed_override": mode.feed_override}

    # ------------------------------------------------------------------ #
    #  Калибровка
    # ------------------------------------------------------------------ #
    @app.post("/api/calibration/probe")
    def calibration_probe(request: schemas.ProbeRequest) -> dict[str, Any]:
        probe = LandingProbe(label=request.label or f"{request.x:.0f},{request.y:.0f}")
        state["pending_probe"] = (probe, np.array([request.x, request.y, request.z], float))
        controller().set_mode(probe)
        return {"ok": True, "note": "платформа опускается до касания; следите за натяжениями"}

    @app.post("/api/calibration/accept")
    def calibration_accept() -> dict[str, Any]:
        pending = state.get("pending_probe")
        if not pending:
            raise HTTPException(status_code=409, detail="посадка не выполнялась")
        probe, position = pending
        if probe.landed_counts is None:
            raise HTTPException(status_code=409, detail="касание не зафиксировано")
        point = CalibrationPoint(position, probe.landed_counts, probe.landed_tensions, probe.label)
        state["probe_points"].append(point)
        state["pending_probe"] = None
        return {"ok": True, "points": len(state["probe_points"])}

    @app.get("/api/calibration/points")
    def calibration_points() -> dict[str, Any]:
        return {"points": [
            {"label": p.label, "position": p.position_mm.tolist(), "counts": p.counts.tolist()}
            for p in state["probe_points"]
        ]}

    @app.post("/api/calibration/clear")
    def calibration_clear() -> dict[str, Any]:
        state["probe_points"] = []
        return {"ok": True}

    @app.post("/api/calibration/solve")
    def calibration_solve(request: schemas.CalibrationSolveRequest) -> dict[str, Any]:
        points = state["probe_points"]
        if len(points) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"нужно минимум две точки, снято {len(points)}. "
                       f"Практически берите четыре-шесть, разнесённых по площади",
            )
        result = identify(runtime.machine, points, fit_elasticity=request.fit_elasticity,
                          kinematics=controller().kinematics)
        state["last_calibration"] = result
        payload = {
            "ok": result.ok,
            "summary": result.summary(),
            "residual_rms_mm": round(result.residual_rms_mm, 3),
            "residual_max_mm": round(result.residual_max_mm, 3),
            "warnings": result.warnings,
            "updates": result.as_updates(),
        }
        if request.apply and result.ok:
            save_calibration(result.as_updates(), runtime.machine_path)
            payload["applied"] = True
            payload["note"] = "записано в machine.yaml; перезапустите сервер, чтобы применить"
        return payload

    # ------------------------------------------------------------------ #
    #  Настройки
    # ------------------------------------------------------------------ #
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        machine = runtime.machine
        return {
            "name": machine.name,
            "anchors": [
                {"id": a.id, "pos": list(a.pos), "bus": a.bus, "slave": a.slave}
                for a in machine.geometry.anchors
            ],
            "platform": {"mass_kg": machine.platform.mass_kg,
                         "landing_height_mm": machine.platform.landing_height_mm},
            "winches": [
                {
                    "anchor": w.anchor,
                    "drum_diameter_mm": w.drum_diameter_mm,
                    "drum_width_mm": w.drum_width_mm,
                    "line_diameter_mm": w.line_diameter_mm,
                    "winding": w.winding,
                    "max_rpm": w.max_rpm,
                    "rated_torque_nm": w.rated_torque_nm,
                    "direction": w.direction,
                    "count_empty": w.count_empty,
                    "length_at_empty_mm": w.length_at_empty_mm,
                    "ea_n": w.ea_n,
                    "calibrated": w.is_calibrated,
                    "turns_per_layer": w.turns_per_layer,
                    "max_line_speed_mms": round(w.max_line_speed_mms, 1),
                    "force_at_full_torque_n": round(w.torque_percent_to_force(100.0), 1),
                }
                for w in machine.ordered_winches()
            ],
            "tension": machine.tension.model_dump(),
            "workspace": machine.workspace.model_dump(),
            "motion": machine.motion.model_dump(),
            "control": machine.control.model_dump(),
            "admittance": machine.admittance.model_dump(),
            "safety": machine.safety.model_dump(),
            "buses": {k: v.model_dump() for k, v in machine.buses.items()},
            "calibrated": machine.is_calibrated,
            "uncalibrated": machine.uncalibrated(),
        }

    @app.post("/api/config")
    def patch_config(request: schemas.ConfigPatchRequest) -> dict[str, Any]:
        """Правит machine.yaml, сохраняя комментарии.

        Часть настроек применяется на лету, часть требует перезапуска —
        ответ говорит об этом прямо, а не оставляет гадать.
        """
        try:
            patch_yaml(runtime.machine_path, request.updates)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            fresh = load_machine(runtime.machine_path)
        except ConfigError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"после правки конфиг не читается: {exc}. Файл сохранён, "
                       f"исправьте значение или восстановите из machine.yaml.bak",
            ) from exc

        live = {"tension", "motion", "control", "admittance", "workspace"}
        touched = {key.split(".")[0] for key in request.updates}
        hot = touched <= live
        if hot:
            machine = runtime.machine
            machine.tension = fresh.tension
            machine.motion = fresh.motion
            machine.control = fresh.control
            machine.admittance = fresh.admittance
            machine.workspace = fresh.workspace
            controller().set_target_tension(fresh.tension.target_n)
            state["workspace_cache"] = {}
        return {
            "ok": True,
            "applied_now": hot,
            "note": "" if hot else "изменения записаны, но применятся после перезапуска сервера",
        }

    register_homing(app, runtime, state, controller)

    # ------------------------------------------------------------------ #
    #  Статика панели
    # ------------------------------------------------------------------ #
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:
        @app.get("/")
        def index_missing() -> JSONResponse:
            return JSONResponse({"error": f"нет каталога панели {WEB_DIR}"}, status_code=500)

    return app


def _describe(runtime: Runtime) -> dict[str, Any]:
    machine = runtime.machine
    anchors = np.array([a.pos for a in machine.geometry.anchors], dtype=float)
    return {
        "name": machine.name,
        "simulated": runtime.drives.simulated,
        "n_cables": machine.n_cables,
        "anchors": anchors.tolist(),
        "anchor_ids": [a.id for a in machine.geometry.anchors],
        "calibrated": machine.is_calibrated,
        "enable_backend": runtime.drives.enabler.describe(),
        "enable_automatic": runtime.drives.enabler.is_automatic,
        "tension": machine.tension.model_dump(),
        "workspace": machine.workspace.model_dump(),
        "motion": machine.motion.model_dump(),
        "profile": runtime.profile.name,
        "eeprom_safe": runtime.profile.eeprom_safe,
    }


def register_homing(app: FastAPI, runtime: Runtime, state: dict[str, Any], controller) -> None:
    """Привязка системы: геометрия модулей и калибровка лебёдок.

    Отдельным блоком, потому что это самостоятельная процедура со своим
    порядком действий, а не одна кнопка.
    """
    from cdpr.calibration import identify_from_ranges
    from cdpr.geometry_fit import PAIRS, fit_modules
    from cdpr.modes.autohoming import AutoHoming, default_deltas

    @app.post("/api/geometry/fit")
    def geometry_fit(request: schemas.GeometryFitRequest) -> dict[str, Any]:
        n = runtime.machine.n_cables
        if len(request.heights_mm) != n:
            raise HTTPException(status_code=400,
                                detail=f"нужно {n} высот модулей, передано {len(request.heights_mm)}")
        try:
            fit = fit_modules(request.distances_mm, request.heights_mm, n=n)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ids = [a.id for a in runtime.machine.geometry.anchors]
        payload = {
            "ok": fit.ok,
            "summary": fit.summary(ids),
            "residual_rms_mm": round(fit.residual_rms_mm, 2),
            "warnings": fit.warnings,
            "positions": [[round(float(v), 1) for v in p] for p in fit.positions],
            "pairs": [list(p) for p in PAIRS],
        }
        if request.apply:
            if not fit.ok:
                raise HTTPException(
                    status_code=400,
                    detail="замеры не сходятся между собой, записывать такую геометрию нельзя: "
                           + "; ".join(fit.warnings))
            updates = {}
            for i, position in enumerate(fit.positions):
                updates[f"geometry.anchors.{i}.pos"] = [round(float(v), 1) for v in position]
            patch_yaml(runtime.machine_path, updates)
            payload["applied"] = True
            payload["note"] = ("координаты модулей записаны в machine.yaml; "
                               "перезапустите сервер, затем пройдите привязку лебёдок")
        return payload

    @app.post("/api/homing/start")
    def homing_start(request: schemas.HomingStartRequest) -> dict[str, Any]:
        mode = AutoHoming(default_deltas(request.step_mm), feed_mms=request.feed_mms)
        state["homing"] = mode
        controller().set_mode(mode)
        return {"ok": True, "stations": len(mode.plan),
                "note": "платформа объедет стоянки; на каждой измерьте расстояния дальномером"}

    @app.get("/api/homing/status")
    def homing_status() -> dict[str, Any]:
        mode = state.get("homing")
        if not isinstance(mode, AutoHoming):
            return {"running": False, "stations": 0}
        return {
            "running": controller().mode is mode,
            "phase": mode.phase,
            "waiting": mode.waiting,
            "index": mode.index,
            "total": len(mode.plan),
            "label": mode.current_label,
            "stations": len(mode.stations),
            "progress": round(mode.progress, 3),
        }

    @app.post("/api/homing/confirm")
    def homing_confirm(request: schemas.HomingConfirmRequest) -> dict[str, Any]:
        mode = state.get("homing")
        if not isinstance(mode, AutoHoming):
            raise HTTPException(status_code=409, detail="привязка не запущена")
        try:
            station = mode.confirm(request.distances_mm, controller())
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "stations": len(mode.stations), "label": station.label,
                "tensions_n": None if station.tensions_n is None
                else [round(float(v), 1) for v in station.tensions_n]}

    @app.post("/api/homing/abort")
    def homing_abort() -> dict[str, Any]:
        mode = state.get("homing")
        if isinstance(mode, AutoHoming):
            mode.abort()
        controller().set_mode(IdleMode())
        return {"ok": True}

    @app.post("/api/homing/solve")
    def homing_solve(request: schemas.HomingSolveRequest) -> dict[str, Any]:
        mode = state.get("homing")
        if not isinstance(mode, AutoHoming) or len(mode.stations) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"нужно минимум две стоянки, снято "
                       f"{0 if not isinstance(mode, AutoHoming) else len(mode.stations)}")
        result = identify_from_ranges(runtime.machine, mode.stations,
                                      fit_elasticity=request.fit_elasticity)
        payload = {
            "ok": result.ok,
            "summary": result.summary(),
            "residual_rms_mm": round(result.residual_rms_mm, 3),
            "warnings": result.warnings,
            "updates": result.as_updates(),
        }
        if request.apply and result.ok:
            save_calibration(result.as_updates(), runtime.machine_path)
            payload["applied"] = True
            payload["note"] = "записано в machine.yaml; перезапустите сервер"
        return payload
