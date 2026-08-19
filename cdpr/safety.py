"""Проверки, останавливающие движение.

Отдельный модуль, а не разбросанные по коду условия: список причин остановки
должен быть виден целиком, и каждая обязана объяснять себя человеку. В панели
показывается ровно то, что здесь написано.

Про аварийный стоп. Настоящий стоп — это снятие разрешения SON аппаратной
цепью; софт может только обнулить уставки и попросить реле разомкнуться.
Если цепь заведена на ручной тумблер, кнопка в панели физически не может
обесточить приводы, и это состояние показывается явно, а не замалчивается.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cdpr.config import MachineConfig
from cdpr.state import Health


@dataclass
class SafetyVerdict:
    stop: bool
    disable: bool
    health: Health
    reasons: list[str] = field(default_factory=list)


class SafetyMonitor:
    """Оценивает состояние и решает, можно ли продолжать движение."""

    def __init__(self, machine: MachineConfig) -> None:
        self.machine = machine
        self.estop = False
        self.estop_reason = ""
        self._last_ok = time.perf_counter()

    # ------------------------------------------------------------------ #
    def trigger_estop(self, reason: str = "кнопка в панели") -> None:
        self.estop = True
        self.estop_reason = reason

    def clear_estop(self) -> None:
        self.estop = False
        self.estop_reason = ""

    # ------------------------------------------------------------------ #
    def check(
        self,
        *,
        states,
        pose: np.ndarray | None,
        tensions: np.ndarray | None,
        fk_residual_mm: float,
        moving: bool,
    ) -> SafetyVerdict:
        reasons: list[str] = []
        stop = False
        disable = False
        health = Health.OK

        if self.estop:
            return SafetyVerdict(True, True, Health.ESTOP, [f"аварийный стоп: {self.estop_reason}"])

        now = time.perf_counter()
        max_age = self.machine.safety.max_state_age_ms / 1000.0

        for s in states:
            if not s.online:
                stop = True
                reasons.append(f"ось {s.axis}: нет связи ({s.error or 'нет ответа'})")
            elif s.age_s(now) > max_age:
                stop = True
                reasons.append(
                    f"ось {s.axis}: данные устарели на {s.age_s(now) * 1000:.0f} мс "
                    f"при пределе {self.machine.safety.max_state_age_ms:.0f} мс"
                )
            if s.alarm:
                stop = True
                disable = True
                reasons.append(f"ось {s.axis}: авария привода, код {s.alarm}")

        if tensions is not None and len(tensions):
            limits = self.machine.tension
            slack = np.where(tensions < limits.min_n * 0.5)[0]
            if slack.size and moving:
                stop = True
                reasons.append(
                    f"тросы {list(slack)} провисли (натяжение ниже {limits.min_n * 0.5:.1f} Н) — "
                    f"платформа неуправляема"
                )
            over = np.where(tensions > limits.max_n)[0]
            if over.size:
                stop = True
                reasons.append(
                    f"тросы {list(over)} перетянуты: {np.round(tensions[over], 1)} Н "
                    f"при пределе {limits.max_n:.0f} Н"
                )

        if pose is None and moving:
            stop = True
            reasons.append("положение платформы неизвестно — движение запрещено")

        if fk_residual_mm > 25.0:
            health = Health.WARNING
            reasons.append(
                f"длины тросов не сходятся между собой на {fk_residual_mm:.0f} мм — "
                f"проскользнул трос, провис или уехала калибровка"
            )
        if fk_residual_mm > 100.0:
            stop = True

        if stop and health is Health.OK:
            health = Health.FAULT
        elif reasons and health is Health.OK:
            health = Health.WARNING
        return SafetyVerdict(stop, disable, health, reasons)
