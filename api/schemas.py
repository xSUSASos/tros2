"""Формы запросов панели."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EnableRequest(BaseModel):
    """Программное разрешение движения. Это не SON: приводы включены всегда,
    питание с них снимает физическая кнопка."""

    on: bool


class CableSpeedRequest(BaseModel):
    """Ручное вращение одного барабана. Привязка для этого не нужна."""

    index: int = Field(ge=0)
    speed_mms: float


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


class HomingStartRequest(BaseModel):
    """Хоминг: коробка подтягивается в углы и там записываются отсчёты.

    corners — индексы модулей. Одного достаточно для запуска; остальные
    работают проверкой и дают жёсткость троса.
    """

    corners: list[int] | None = None
    feed_mms: float | None = Field(default=None, gt=0, le=100.0)


class HomingSolveRequest(BaseModel):
    apply: bool = False
