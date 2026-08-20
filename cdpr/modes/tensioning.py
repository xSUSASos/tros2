"""Выборка слабины: подтянуть тросы до заданного натяжения."""
from __future__ import annotations

import numpy as np

from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName


class AutoTensionMode(Mode):
    """Медленно выбирает каждый трос, пока не появится нужное натяжение.

    Работает НАПРЯМУЮ скоростями тросов, а не через положение платформы: до
    выборки слабины положение попросту не определено — провисший трос выпадает
    из геометрии, и решать по нему обратную задачу бессмысленно.

    Каждый трос тянется независимо и только на выборку. Стравливать здесь
    нечего: если трос уже перетянут, он просто останавливается, а разгружать
    его движением платформы — задача обычных режимов.
    """

    name = ModeName.AUTOTENSION

    def __init__(self, target_n: float | None = None, feed_mms: float = 15.0,
                 tolerance_n: float = 2.0, timeout_s: float = 60.0) -> None:
        self.target_n = target_n
        self.feed_mms = feed_mms
        self.tolerance_n = tolerance_n
        self.timeout_s = timeout_s
        self._elapsed = 0.0

    @property
    def requires_homing(self) -> bool:
        return False

    @property
    def tolerates_slack(self) -> bool:
        return True

    def enter(self, ctx) -> None:  # noqa: ANN001
        self._elapsed = 0.0
        if self.target_n is None:
            self.target_n = ctx.target_tension_n

    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        self._elapsed += dt
        tensions = ctx.state.tensions_n
        if tensions is None or not len(tensions):
            return ModeOutput.idle("нет данных о натяжении")

        target = float(self.target_n)
        limits = ctx.machine.tension
        velocity = np.zeros(len(tensions))
        for i, value in enumerate(tensions):
            if value < target - self.tolerance_n:
                # чем ближе к цели, тем медленнее — иначе проскочим и дёрнем
                scale = min(1.0, (target - value) / max(target, 1.0))
                velocity[i] = self.feed_mms * max(0.15, scale)
            elif value > limits.max_n:
                velocity[i] = -self.feed_mms * 0.5

        if not np.any(velocity):
            return ModeOutput(cable_velocity_mms=np.zeros(len(tensions)), done=True,
                              message=f"натяжение выбрано: {np.round(tensions, 1).tolist()} Н")

        if self._elapsed > self.timeout_s:
            return ModeOutput(
                cable_velocity_mms=np.zeros(len(tensions)), done=True,
                message=(
                    f"выборка слабины прервана по времени ({self.timeout_s:.0f} с). "
                    f"Натяжение {np.round(tensions, 1).tolist()} Н при цели {target:.0f} Н — "
                    f"проверьте, не кончился ли трос и не мешает ли что-то платформе"
                ),
            )
        return ModeOutput(cable_velocity_mms=velocity,
                          message=f"выбираю слабину, {np.round(tensions, 1).tolist()} Н")

    def describe(self) -> str:
        return f"выборка слабины до {self.target_n} Н"
