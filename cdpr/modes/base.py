"""Общий интерфейс режимов работы.

Режим отвечает на один вопрос: куда система должна двигаться прямо сейчас.
Как именно это исполнить — забота контура управления, поэтому добавить новый
режим значит написать один класс с методом update, ничего больше не трогая.

Режимы бывают двух видов. Одни задают ЦЕЛЕВОЕ ПОЛОЖЕНИЕ платформы (джог,
ручной ввод координат, G-code, перемещение за руку) — контур сам считает
длины тросов и следит за ними. Другие работают напрямую СКОРОСТЯМИ ТРОСОВ
(выборка слабины, калибровка), потому что положение платформы в этот момент
либо неизвестно, либо неважно.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cdpr.state import ModeName


@dataclass
class ModeOutput:
    """Чего режим хочет от контура в этом цикле."""

    target_pose: np.ndarray | None = None
    feed_mms: float | None = None
    cable_velocity_mms: np.ndarray | None = None
    hold: bool = False          # стоять на месте, но держать натяжение
    done: bool = False          # режим отработал, можно вернуться в ожидание
    message: str = ""

    @staticmethod
    def idle(message: str = "") -> "ModeOutput":
        return ModeOutput(hold=True, message=message)


class Mode(ABC):
    """Базовый режим."""

    name: ModeName = ModeName.IDLE

    def enter(self, ctx) -> None:  # noqa: ANN001 — ctx это Controller
        """Вызывается при переключении в режим."""

    def exit(self, ctx) -> None:  # noqa: ANN001
        """Вызывается при выходе из режима."""

    @abstractmethod
    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        """Один шаг режима."""

    @property
    def requires_homing(self) -> bool:
        """Нужна ли калибровка. Движение в координатах без неё бессмысленно:
        отсчёт энкодера не перевести в длину троса."""
        return True

    def describe(self) -> str:
        return self.name.value


class IdleMode(Mode):
    """Ожидание: платформа держится на месте."""

    name = ModeName.IDLE

    @property
    def requires_homing(self) -> bool:
        return False

    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        return ModeOutput.idle()
