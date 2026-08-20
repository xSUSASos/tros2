"""Ручное вращение отдельных барабанов.

Первое, что должно работать на новой машине, и единственный режим, которому
не нужно вообще ничего: ни привязки, ни геометрии, ни рабочей зоны. Кнопка —
скорость троса, и всё.

Нужен он не для работы, а для пусконаладки: проверить, что мотор вообще
крутится, что он крутится в ту сторону, что энкодер считает туда же, и что
за десять оборотов сматывается столько троса, сколько говорит формула. Без
этого любые разговоры о точности преждевременны.

Защиты продолжают работать: перетяг и авария привода останавливают машину так
же, как в любом другом режиме.
"""
from __future__ import annotations

import threading

import numpy as np

from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName


class CableMode(Mode):
    """Скорости тросов задаются прямо, по одному."""

    name = ModeName.CABLE

    def __init__(self, n_axes: int | None = None) -> None:
        self._velocity = np.zeros(n_axes or 0)
        self._lock = threading.RLock()

    @property
    def requires_homing(self) -> bool:
        return False

    @property
    def tolerates_slack(self) -> bool:
        return True

    def enter(self, ctx) -> None:  # noqa: ANN001
        with self._lock:
            if len(self._velocity) != ctx.drives.n_axes:
                self._velocity = np.zeros(ctx.drives.n_axes)

    def exit(self, ctx) -> None:  # noqa: ANN001
        self.stop_all()

    # ------------------------------------------------------------------ #
    def set_speed(self, index: int, speed_mms: float) -> None:
        """Скорость выборки троса: положительная — наматывать, отрицательная — стравливать."""
        with self._lock:
            if not 0 <= index < len(self._velocity):
                raise ValueError(f"нет троса {index}, всего {len(self._velocity)}")
            self._velocity[index] = float(speed_mms)

    def stop_all(self) -> None:
        with self._lock:
            self._velocity = np.zeros(len(self._velocity))

    @property
    def velocity_mms(self) -> np.ndarray:
        with self._lock:
            return self._velocity.copy()

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        with self._lock:
            velocity = self._velocity.copy()
        limit = ctx.machine.motion.jog_feed_mms
        velocity = np.clip(velocity, -limit, limit)
        if not np.any(velocity):
            return ModeOutput(cable_velocity_mms=np.zeros(len(velocity)),
                              message="ручной режим: тросы стоят")
        active = [f"{i}: {v:+.0f}" for i, v in enumerate(velocity) if v]
        return ModeOutput(cable_velocity_mms=velocity,
                          message="ручной режим, мм/с — " + ", ".join(active))

    def describe(self) -> str:
        return "ручное вращение барабанов"
