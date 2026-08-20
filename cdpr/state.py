"""Снимок состояния машины — общий язык контура управления и панели."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ModeName(str, Enum):
    IDLE = "idle"
    JOG = "jog"
    MDI = "mdi"
    GCODE = "gcode"
    HOMING = "homing"
    ADMITTANCE = "admittance"
    AUTOTENSION = "autotension"
    CABLE = "cable"        # ручное вращение одного барабана, без всякой привязки


class Health(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAULT = "fault"
    ESTOP = "estop"


@dataclass
class MachineState:
    """Всё, что известно о машине на текущий цикл."""

    stamp: float = field(default_factory=time.time)
    cycle: int = 0
    loop_hz_actual: float = 0.0
    mode: ModeName = ModeName.IDLE
    health: Health = Health.OK
    messages: list[str] = field(default_factory=list)

    enabled: bool = False   # программное разрешение движения (не SON)
    estop: bool = False
    homed: bool = False     # привязка пройдена: отсчёты переводятся в длины
    arrived: bool = False   # цель достигнута в пределах допуска

    # положение
    pose_mm: np.ndarray | None = None
    target_mm: np.ndarray | None = None
    fk_residual_mm: float = 0.0

    # тросы: свободная длина — что стравлено с барабана (меряет энкодер),
    # lengths_mm — расстояние до платформы, то есть свободная плюс вытяжка
    free_lengths_mm: np.ndarray | None = None
    lengths_mm: np.ndarray | None = None
    target_lengths_mm: np.ndarray | None = None
    tensions_n: np.ndarray | None = None
    target_tensions_n: np.ndarray | None = None
    speeds_rpm: np.ndarray | None = None
    commands_rpm: np.ndarray | None = None

    # приводы
    online: list[bool] = field(default_factory=list)
    alarms: list[int] = field(default_factory=list)

    # запас
    margin_n: float = 0.0
    tension_min_n: float = 0.0
    tension_max_n: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        def arr(a: np.ndarray | None, digits: int = 2) -> list[float] | None:
            return None if a is None else [round(float(v), digits) for v in np.asarray(a).ravel()]

        return {
            "stamp": self.stamp,
            "cycle": self.cycle,
            "loop_hz_actual": round(self.loop_hz_actual, 1),
            "mode": self.mode.value,
            "health": self.health.value,
            "messages": list(self.messages),
            "enabled": self.enabled,
            "estop": self.estop,
            "homed": self.homed,
            "arrived": self.arrived,
            "pose_mm": arr(self.pose_mm, 1),
            "target_mm": arr(self.target_mm, 1),
            "fk_residual_mm": round(self.fk_residual_mm, 3),
            "free_lengths_mm": arr(self.free_lengths_mm, 1),
            "lengths_mm": arr(self.lengths_mm, 1),
            "target_lengths_mm": arr(self.target_lengths_mm, 1),
            "tensions_n": arr(self.tensions_n, 1),
            "target_tensions_n": arr(self.target_tensions_n, 1),
            "speeds_rpm": arr(self.speeds_rpm, 1),
            "commands_rpm": arr(self.commands_rpm, 2),
            "online": list(self.online),
            "alarms": list(self.alarms),
            "margin_n": round(self.margin_n, 1),
            "tension_min_n": round(self.tension_min_n, 1),
            "tension_max_n": round(self.tension_max_n, 1),
        }
