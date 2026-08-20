"""Формы запросов панели."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EnableRequest(BaseModel):
    on: bool


class EstopRequest(BaseModel):
    reason: str = "кнопка в панели"


class JogStepRequest(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    feed_mms: float | None = None


class JogContinuousRequest(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


class MdiRequest(BaseModel):
    x: float
    y: float
    z: float
    feed_mms: float | None = Field(default=None, gt=0)


class TensionRequest(BaseModel):
    target_n: float = Field(gt=0)


class AutoTensionRequest(BaseModel):
    target_n: float | None = Field(default=None, gt=0)
    feed_mms: float = Field(default=15.0, gt=0)


class HandGuideRequest(BaseModel):
    on: bool
    gain_mms_per_n: float | None = None
    deadband_n: float | None = None


class GcodeLoadRequest(BaseModel):
    text: str
    check_workspace: bool = True


class FeedOverrideRequest(BaseModel):
    value: float = Field(gt=0.05, le=2.0)


class ConfigPatchRequest(BaseModel):
    """Точечная правка конфига: путь через точку -> значение."""

    updates: dict[str, Any]


class ProbeRequest(BaseModel):
    """Посадка на точка посадки с известными координатами."""

    x: float
    y: float
    z: float = 0.0
    label: str = ""


class CalibrationSolveRequest(BaseModel):
    fit_elasticity: bool = True
    apply: bool = False


class GeometryFitRequest(BaseModel):
    """Восстановление расположения модулей по простым замерам.

    Расстояния — шесть штук, в порядке пар (1-2, 1-3, 1-4, 2-3, 2-4, 3-4),
    от точки схода троса до точки схода. Высоты — четыре, над полом.
    """

    distances_mm: list[float] = Field(min_length=6, max_length=6)
    heights_mm: list[float] = Field(min_length=3, max_length=8)
    apply: bool = False


class HomingStartRequest(BaseModel):
    step_mm: float = Field(default=400.0, gt=50.0, le=2000.0)
    feed_mms: float = Field(default=25.0, gt=0, le=100.0)


class HomingConfirmRequest(BaseModel):
    """Замеры от каждого модуля до платформы на текущей стоянке."""

    distances_mm: list[float] = Field(min_length=3, max_length=8)


class HomingSolveRequest(BaseModel):
    fit_elasticity: bool = True
    apply: bool = False
