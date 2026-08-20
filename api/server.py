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
from cdpr.calibration import park_pose, solve_from_corners
from cdpr.config import ConfigError, load_machine, patch_yaml, save_calibration
from cdpr.modes.admittance import AdmittanceMode
from cdpr.modes.base import IdleMode
from cdpr.modes.cable import CableMode
from cdpr.modes.homing import CornerHoming
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
        "homing": None, "last_calibration": None, "workspace_cache": {},
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
        """Программное разрешение движения.

        Приводы включены всегда (P-098 = 1), силового разрешения по Modbus у
        T3D нет. Это предохранитель от случайной команды: пока он снят, на
        шину уходят нули, что бы ни насчитал режим. Настоящий стоп — кнопка,
        снимающая питание.
        """
        try:
            controller().allow_motion(request.on)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "enabled": request.on,
                "note": "приводы включены всегда; питание снимает физическая кнопка"}

    @app.post("/api/mode/idle")
    def go_idle() -> dict[str, Any]:
        controller().set_mode(IdleMode())
        state["jog"] = None
        return {"ok": True, "mode": "idle"}

    # ------------------------------------------------------------------ #
    #  Ручное вращение барабанов — работает без всякой привязки
    # ------------------------------------------------------------------ #
    def _ensure_cable() -> CableMode:
        mode = controller().mode
        if isinstance(mode, CableMode):
            return mode
        cable = CableMode(runtime.drives.n_axes)
        controller().set_mode(cable)
        state["cable"] = cable
        return cable

    @app.post("/api/cable/speed")
    def cable_speed(request: schemas.CableSpeedRequest) -> dict[str, Any]:
        cable = _ensure_cable()
        try:
            cable.set_speed(request.index, request.speed_mms)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "velocity_mms": cable.velocity_mms.tolist()}

    @app.post("/api/cable/stop")
    def cable_stop() -> dict[str, Any]:
        mode = controller().mode
        if isinstance(mode, CableMode):
            mode.stop_all()
        return {"ok": True}

    # ------------------------------------------------------------------ #
    #  Ручное управление в координатах
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
        # На плоской машине Z не задаётся: коробка всегда в рабочей плоскости.
        plane = runtime.machine.geometry.plane_z_mm
        z = plane if plane is not None else request.z
        target = np.array([request.x, request.y, z], dtype=float)
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
            "platform": {"mass_kg": machine.platform.mass_kg},
            "geometry": {"plane_z_mm": machine.geometry.plane_z_mm,
                         "anchor_z_mm": machine.geometry.anchor_z_mm,
                         "planar": machine.geometry.is_planar},
            "homing": machine.homing.model_dump(),
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
                    "count_ref": w.count_ref,
                    "length_at_ref_mm": w.length_at_ref_mm,
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
        "planar": machine.geometry.is_planar,
        "plane_z_mm": machine.geometry.plane_z_mm,
        "torque_limit_percent": machine.safety.drive_torque_limit_percent,
        "torque_limit_applied": getattr(runtime.drives, "torque_limit_applied", False),
        "tension": machine.tension.model_dump(),
        "workspace": machine.workspace.model_dump(),
        "motion": machine.motion.model_dump(),
        "profile": runtime.profile.name,
        "eeprom_safe": runtime.profile.eeprom_safe,
    }


def register_homing(app: FastAPI, runtime: Runtime, state: dict[str, Any], controller) -> None:
    """Привязка: подтягивание коробки в углы и запись отсчётов.

    Отдельным блоком, потому что это самостоятельная процедура со своим
    порядком действий, а не одна кнопка.
    """

    def homing_mode() -> CornerHoming | None:
        mode = state.get("homing")
        return mode if isinstance(mode, CornerHoming) else None

    @app.post("/api/homing/start")
    def homing_start(request: schemas.HomingStartRequest) -> dict[str, Any]:
        machine = runtime.machine
        corners = request.corners if request.corners is not None else list(machine.homing.corners)
        for index in corners:
            if not 0 <= index < machine.n_cables:
                raise HTTPException(
                    status_code=400,
                    detail=f"нет угла {index}, модулей {machine.n_cables}")
        mode = CornerHoming(corners, feed_mms=request.feed_mms)
        state["homing"] = mode
        controller().set_mode(mode)
        return {
            "ok": True,
            "corners": corners,
            "note": ("коробка подтягивается к каждому модулю по очереди; "
                     "положение для этого не нужно, работа идёт скоростями тросов"),
        }

    @app.get("/api/homing/status")
    def homing_status() -> dict[str, Any]:
        mode = homing_mode()
        if mode is None:
            return {"running": False, "recorded": 0}
        return {
            "running": controller().mode is mode,
            "phase": mode.phase,
            "corner": mode.current_corner,
            "index": mode.index,
            "total": len(mode.corners or []),
            "recorded": len(mode.records),
            "progress": round(mode.progress, 3),
        }

    @app.post("/api/homing/abort")
    def homing_abort() -> dict[str, Any]:
        mode = homing_mode()
        if mode is not None:
            mode.abort()
        controller().set_mode(IdleMode())
        return {"ok": True}

    @app.post("/api/homing/solve")
    def homing_solve(request: schemas.HomingSolveRequest) -> dict[str, Any]:
        mode = homing_mode()
        if mode is None or not mode.records:
            raise HTTPException(
                status_code=400,
                detail="ни один угол не пройден: сначала запустите хоминг")
        result = solve_from_corners(
            runtime.machine, mode.records, kinematics=controller().kinematics,
        )
        state["last_calibration"] = result
        payload = {
            "ok": result.ok,
            "summary": result.summary(),
            "residual_rms_mm": round(result.residual_rms_mm, 2),
            "residual_max_mm": round(result.residual_max_mm, 2),
            "warnings": result.warnings,
            "updates": result.as_updates(),
            "park_poses": {
                str(r.corner): [round(float(v), 1)
                                for v in park_pose(runtime.machine, r.corner,
                                                   controller().kinematics)]
                for r in mode.records
            },
        }
        if request.apply and result.ok:
            save_calibration(result.as_updates(), runtime.machine_path)
            payload["applied"] = True
            payload["note"] = "записано в machine.yaml; перезапустите сервер, чтобы применить"
        return payload
