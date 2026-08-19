"""Посадка на реперную точку — основа калибровки.

Здесь курица и яйцо: чтобы знать, где платформа, нужна калибровка, а чтобы
калиброваться, надо привести платформу в известную точку. Разрывается это
тем, что режим работает НАПРЯМУЮ скоростями тросов и не пользуется
положением вовсе.

Признак касания — одновременное падение натяжения на ВСЕХ тросах. Когда
платформа встала на пол, вес принимает пол, и тросы разгружаются все сразу.
Это гораздо более чёткий и повторяемый сигнал, чем скачок момента при
боковом упоре: скачок легко спутать с рывком, трением или качанием, а
падение сразу на четырёх тросах ни с чем не спутаешь.
"""
from __future__ import annotations

import numpy as np

from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName


class LandingProbe(Mode):
    """Медленно опускает платформу до касания и запоминает отсчёты."""

    name = ModeName.HOMING

    def __init__(self, feed_mms: float | None = None, *, release_ratio: float = 0.35,
                 settle_s: float = 1.0, timeout_s: float = 120.0,
                 backoff_mm: float = 15.0, label: str = "") -> None:
        self.feed_mms = feed_mms
        self.release_ratio = release_ratio
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self.backoff_mm = backoff_mm
        self.label = label

        self.phase = "оседание"
        self.baseline: np.ndarray | None = None
        self.landed_counts: np.ndarray | None = None
        self.landed_tensions: np.ndarray | None = None
        self._elapsed = 0.0
        self._backoff_left = 0.0

    @property
    def requires_homing(self) -> bool:
        return False

    def enter(self, ctx) -> None:  # noqa: ANN001
        if self.feed_mms is None:
            self.feed_mms = ctx.machine.motion.homing_feed_mms
        self.phase = "оседание"
        self.baseline = None
        self.landed_counts = None
        self._elapsed = 0.0

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        tensions = ctx.state.tensions_n
        n = ctx.drives.n_axes
        if tensions is None or len(tensions) != n:
            return ModeOutput.idle("нет данных о натяжении")
        self._elapsed += dt

        if self.phase == "оседание":
            if self._elapsed < self.settle_s:
                return ModeOutput(cable_velocity_mms=np.zeros(n),
                                  message="жду, пока платформа успокоится")
            self.baseline = tensions.copy()
            self.phase = "спуск"
            return ModeOutput(cable_velocity_mms=np.zeros(n),
                              message=f"опорные натяжения {np.round(self.baseline, 1).tolist()} Н")

        if self.phase == "спуск":
            if self._is_landed(tensions):
                self.landed_counts = self._counts(ctx)
                self.landed_tensions = tensions.copy()
                self.phase = "отход"
                self._backoff_left = self.backoff_mm
                return ModeOutput(cable_velocity_mms=np.zeros(n),
                                  message=f"касание: натяжения {np.round(tensions, 1).tolist()} Н")
            if self._elapsed > self.timeout_s:
                return ModeOutput(
                    cable_velocity_mms=np.zeros(n), done=True,
                    message=(
                        f"касание не поймано за {self.timeout_s:.0f} с. Проверьте, "
                        f"что платформа действительно идёт вниз и что натяжение "
                        f"измеряется (натяжения сейчас {np.round(tensions, 1).tolist()} Н)"
                    ),
                )
            # стравливаем все тросы: отрицательная скорость наматывания
            return ModeOutput(cable_velocity_mms=np.full(n, -self.feed_mms),
                              message=f"опускаю, натяжения {np.round(tensions, 1).tolist()} Н")

        if self.phase == "отход":
            step = self.feed_mms * dt
            self._backoff_left -= step
            if self._backoff_left <= 0.0:
                return ModeOutput(cable_velocity_mms=np.zeros(n), done=True,
                                  message="посадка выполнена, отсчёты записаны")
            return ModeOutput(cable_velocity_mms=np.full(n, self.feed_mms * 0.5),
                              message="отхожу от точки касания")

        return ModeOutput.idle()

    # ------------------------------------------------------------------ #
    def _is_landed(self, tensions: np.ndarray) -> bool:
        """Все тросы разгрузились одновременно — платформа стоит на опоре."""
        if self.baseline is None:
            return False
        threshold = np.maximum(self.baseline * self.release_ratio, 1.0)
        return bool(np.all(tensions < threshold))

    @staticmethod
    def _counts(ctx) -> np.ndarray:  # noqa: ANN001
        return np.array([axis.state.position_counts for axis in ctx.drives.axes], dtype=float)

    def describe(self) -> str:
        suffix = f" ({self.label})" if self.label else ""
        return f"посадка на репер{suffix}: {self.phase}"
