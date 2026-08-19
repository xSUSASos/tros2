"""Ручное перемещение «за руку» (admittance control).

Как в ролике IPAnema: берёшь платформу рукой и ведёшь, а система идёт следом.

Как это работает. В равновесии тросы уравновешивают вес: W f = -w_вес. Если
кто-то толкает платформу, натяжения перестают сходиться с одним лишь весом, и
разница — это ровно приложенное усилие:

    F_рука = -W f + (0, 0, m g)

Дальше усилие превращается в скорость платформы, и цель едет туда, куда её
ведут. Мёртвая зона нужна обязательно: момент привода измеряется грубо, в
барабане и подшипниках есть трение, и без зоны нечувствительности платформа
поползёт сама.

Честное ограничение. Настоящие тактильные стенды считают контур на килогерцах,
здесь же частота упирается в шину Modbus — реально 30–80 Гц. Ощущение будет
мягким и слегка запаздывающим. Разбиение гирлянды на две шины по два привода
удваивает частоту и заметно улучшает отклик.
"""
from __future__ import annotations

import numpy as np

from cdpr import tension as T
from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName


class AdmittanceMode(Mode):
    """Платформа следует за усилием руки."""

    name = ModeName.ADMITTANCE

    def __init__(self, gain_mms_per_n: float | None = None, deadband_n: float | None = None,
                 max_velocity_mms: float | None = None, filter_hz: float = 3.0) -> None:
        self.gain = gain_mms_per_n
        self.deadband = deadband_n
        self.max_velocity = max_velocity_mms
        self.filter_hz = filter_hz
        self._target: np.ndarray | None = None
        self._force = np.zeros(3)

    def enter(self, ctx) -> None:  # noqa: ANN001
        cfg = ctx.machine.admittance
        if self.gain is None:
            self.gain = cfg.gain_mms_per_n
        if self.deadband is None:
            self.deadband = cfg.deadband_n
        if self.max_velocity is None:
            self.max_velocity = cfg.max_velocity_mms
        pose = ctx.state.pose_mm
        self._target = None if pose is None else pose.copy()
        self._force = np.zeros(3)

    def external_force(self, ctx) -> np.ndarray:  # noqa: ANN001
        """Усилие, приложенное к платформе извне, Н."""
        pose, tensions = ctx.state.pose_mm, ctx.state.tensions_n
        if pose is None or tensions is None:
            return np.zeros(3)
        W = ctx.kinematics.structure_matrix(pose)
        weight = T.gravity_wrench(ctx.machine.platform.mass_kg)
        return T.external_wrench_from_forces(W, tensions) - weight

    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        pose = ctx.state.pose_mm
        if pose is None:
            return ModeOutput.idle("положение неизвестно — нужна калибровка")
        if self._target is None:
            self._target = pose.copy()

        raw = self.external_force(ctx)
        # сглаживание: момент измеряется грубо и дёргается, без фильтра
        # платформа отзывалась бы рывками
        alpha = min(1.0, 2.0 * np.pi * self.filter_hz * dt) if dt > 0 else 1.0
        self._force += alpha * (raw - self._force)

        magnitude = float(np.linalg.norm(self._force))
        if magnitude <= self.deadband:
            self._target = pose.copy()
            return ModeOutput(target_pose=self._target, feed_mms=self.max_velocity,
                              message=f"усилие {magnitude:.1f} Н — в мёртвой зоне")

        direction = self._force / magnitude
        speed = min(self.max_velocity, self.gain * (magnitude - self.deadband))
        self._target = np.clip(self._target + direction * speed * dt, ctx.box_low, ctx.box_high)
        return ModeOutput(
            target_pose=self._target, feed_mms=self.max_velocity,
            message=f"веду за руку: {magnitude:.1f} Н, {speed:.0f} мм/с",
        )

    def describe(self) -> str:
        return "перемещение за руку"
