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
        """Нужна ли привязка. Движение в координатах без неё бессмысленно:
        отсчёт энкодера не перевести в длину троса."""
        return True

    @property
    def tension_ceiling_n(self) -> float | None:
        """Свой потолок натяжения, если он в этом режиме другой.

        Хоминг намеренно тянет трос до упора, поэтому обычный программный
        предел ему мешал бы: он сработал бы раньше, чем привод дойдёт до
        своего, и упор так и не был бы пойман. Взамен на время хоминга
        снижается аппаратный предел момента — трос защищён им, а не софтом.
        """
        return None

    @property
    def tolerates_slack(self) -> bool:
        """Ожидается ли провис троса в этом режиме.

        Обычно провис — авария: платформа теряет управляемость. Но выборка
        слабины, хоминг и ручное вращение барабанов именно с провисом и
        работают, и останавливать их за это — значит сделать невозможным
        первый запуск машины. Перетяг при этом продолжает проверяться: он
        рвёт трос в любом режиме.
        """
        return False

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
