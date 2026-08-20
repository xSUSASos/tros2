"""Ручные режимы: ввод координат и джог."""
from __future__ import annotations

import threading

import numpy as np

from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName


class MdiMode(Mode):
    """Переезд в заданные координаты.

    Так работает поле ручного ввода в панели: вводите X, Y, Z и подачу,
    платформа едет туда и останавливается.
    """

    name = ModeName.MDI

    def __init__(self, target_mm, feed_mms: float | None = None, tolerance_mm: float = 2.0) -> None:
        self.target = np.asarray(target_mm, dtype=float)
        self.feed = feed_mms
        self.tolerance = tolerance_mm

    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        pose = ctx.state.pose_mm
        if pose is None:
            return ModeOutput.idle("положение неизвестно — нужна калибровка")
        distance = float(np.linalg.norm(self.target - pose))
        if distance <= self.tolerance:
            return ModeOutput(target_pose=self.target, feed_mms=self.feed, done=True,
                              message=f"пришли, расхождение {distance:.1f} мм")
        return ModeOutput(target_pose=self.target, feed_mms=self.feed)

    def describe(self) -> str:
        return f"переезд в {np.round(self.target).astype(int).tolist()} мм"


class JogMode(Mode):
    """Ручное перемещение шагами и удержанием — крестовина в панели.

    Цель накапливается: нажатия складываются, и платформа догоняет её с
    заданной подачей. Так же работает джог в станках с ЧПУ, и это заметно
    удобнее, чем «пока держу — еду», потому что не зависит от задержек связи.
    """

    name = ModeName.JOG

    def __init__(self, feed_mms: float | None = None) -> None:
        self.feed = feed_mms
        self._target: np.ndarray | None = None
        self._pending = np.zeros(3)
        self._continuous = np.zeros(3)
        self._lock = threading.RLock()

    def enter(self, ctx) -> None:  # noqa: ANN001
        pose = ctx.state.pose_mm
        with self._lock:
            self._target = None if pose is None else pose + self._pending
            if pose is not None:
                self._pending = np.zeros(3)

    def step(self, delta_mm) -> None:
        """Шаг по осям, например +10 мм по X.

        Режим включается панелью и тем же запросом получает первый шаг, а
        `enter()` отработает только в начале следующего цикла. Поэтому шаги,
        пришедшие до входа в режим, копятся отдельно, иначе первое нажатие
        крестовины пропадало бы впустую.
        """
        with self._lock:
            if self._target is None:
                self._pending = self._pending + np.asarray(delta_mm, dtype=float)
            else:
                self._target = self._target + np.asarray(delta_mm, dtype=float)

    def set_continuous(self, velocity_mms) -> None:
        """Непрерывное движение, пока кнопка нажата."""
        with self._lock:
            self._continuous = np.asarray(velocity_mms, dtype=float)

    def stop_continuous(self) -> None:
        self.set_continuous(np.zeros(3))

    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        pose = ctx.state.pose_mm
        if pose is None:
            return ModeOutput.idle("положение неизвестно — нужна калибровка")
        with self._lock:
            if self._target is None:
                self._target = pose + self._pending
                self._pending = np.zeros(3)
            if np.any(self._continuous):
                self._target = self._target + self._continuous * dt
            target = np.clip(self._target, ctx.box_low, ctx.box_high)
            self._target = target
        return ModeOutput(target_pose=target, feed_mms=self.feed)

    def describe(self) -> str:
        return "джог"
