"""Автоматический объезд стоянок для привязки системы.

Здесь есть тонкость, из-за которой обычный «переезд в координаты» не годится.
До привязки положение платформы неизвестно — отсчёт энкодера ещё не переведён
в длину троса, — а значит и ехать в заданную точку не во что.

Но ПРИРАЩЕНИЕ длины известно точно и без всякой калибровки: сколько импульсов
намотал барабан, столько троса и выбрал. Поэтому цикл задан не координатами, а
изменениями длин: «подобрать все тросы на столько-то», «стравить на столько-то».
Куда именно платформа при этом приедет, неважно — на каждой стоянке её всё
равно измеряют дальномером.

Стоянки берутся на РАЗНОЙ ВЫСОТЕ намеренно. Жёсткость троса видна только там,
где меняется натяжение: если все стоянки на одном уровне, отличить вытяжку от
постоянного смещения нечем, и ошибка положения вырастает с единиц миллиметров
до десятков.
"""
from __future__ import annotations

import threading

import numpy as np

from cdpr.calibration import RangeStation
from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName


def default_deltas(step_mm: float = 400.0) -> list[tuple[list[float], str]]:
    """Три стоянки: как есть, выше, ниже. Разброс по высоте даёт разброс
    натяжения, без которого жёсткость троса не определяется."""
    n = 4
    return [
        ([0.0] * n, "как есть"),
        ([+step_mm] * n, f"выше: подобрать все тросы на {step_mm:.0f} мм"),
        ([-2.0 * step_mm] * n, f"ниже: стравить все тросы на {2 * step_mm:.0f} мм"),
    ]


class AutoHoming(Mode):
    """Объезжает стоянки и на каждой ждёт, пока их измерят."""

    name = ModeName.HOMING

    def __init__(self, deltas: list[tuple[list[float], str]] | None = None, *,
                 feed_mms: float = 25.0, settle_s: float = 2.0,
                 tolerance_mm: float = 3.0) -> None:
        self.plan = deltas if deltas is not None else default_deltas()
        self.feed_mms = feed_mms
        self.settle_s = settle_s
        self.tolerance_mm = tolerance_mm

        self.index = 0
        self.phase = "переезд"
        self.stations: list[RangeStation] = []
        self.message = ""
        self._lock = threading.RLock()
        self._start_counts: np.ndarray | None = None
        self._settled = 0.0
        self._abort = False
        self._scale: np.ndarray | None = None

    @property
    def requires_homing(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    def enter(self, ctx) -> None:  # noqa: ANN001
        # Масштаб первого слоя: точной калибровки ещё нет, но для приращения
        # в несколько сотен миллиметров двух процентов погрешности достаточно —
        # фактическое положение всё равно измеряется дальномером.
        self._scale = np.array([w.nominal_mm_per_count for w in ctx.winches])
        self.index = 0
        self.phase = "переезд"
        self.stations = []
        self._start_counts = None
        self._abort = False

    @property
    def waiting(self) -> bool:
        return self.phase == "ждём замеров"

    @property
    def progress(self) -> float:
        return len(self.stations) / max(1, len(self.plan))

    @property
    def current_label(self) -> str:
        return self.plan[self.index][1] if self.index < len(self.plan) else "готово"

    # ------------------------------------------------------------------ #
    def confirm(self, distances_mm, ctx) -> RangeStation:  # noqa: ANN001
        """Оператор ввёл замеры от каждого модуля до платформы."""
        with self._lock:
            if not self.waiting:
                raise RuntimeError(
                    f"сейчас не время вводить замеры: {self.phase}. "
                    f"Дождитесь, пока платформа встанет"
                )
            distances = np.asarray(distances_mm, dtype=float)
            if distances.shape != (ctx.drives.n_axes,):
                raise ValueError(f"нужно {ctx.drives.n_axes} замеров, передано {distances.size}")
            if np.any(distances <= 0):
                raise ValueError("расстояния должны быть положительными")

            station = RangeStation(
                distances_mm=distances,
                counts=np.array([a.state.position_counts for a in ctx.drives.axes], dtype=float),
                tensions_n=(None if ctx.state.tensions_n is None else ctx.state.tensions_n.copy()),
                label=self.current_label,
            )
            self.stations.append(station)
            self.index += 1
            self.phase = "готово" if self.index >= len(self.plan) else "переезд"
            self._start_counts = None
            return station

    def abort(self) -> None:
        with self._lock:
            self._abort = True

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        n = ctx.drives.n_axes
        with self._lock:
            if self._abort:
                return ModeOutput(cable_velocity_mms=np.zeros(n), done=True,
                                  message="привязка прервана")
            if self.phase == "готово":
                return ModeOutput(cable_velocity_mms=np.zeros(n), done=True,
                                  message=f"объезд закончен, стоянок снято {len(self.stations)}")
            if self.phase == "ждём замеров":
                return ModeOutput(
                    cable_velocity_mms=np.zeros(n),
                    message=(f"стоянка {self.index + 1} из {len(self.plan)} — "
                             f"измерьте дальномером расстояние от каждого модуля до платформы"),
                )

            target = np.asarray(self.plan[self.index][0], dtype=float)
            counts = np.array([a.state.position_counts for a in ctx.drives.axes], dtype=float)
            if self._start_counts is None:
                self._start_counts = counts.copy()
                self._settled = 0.0

            moved = (counts - self._start_counts) * self._scale * np.array(
                [w.direction for w in ctx.winches]
            )
            remaining = target - moved

            if np.all(np.abs(remaining) <= self.tolerance_mm):
                self._settled += dt
                if self._settled >= self.settle_s:
                    self.phase = "ждём замеров"
                return ModeOutput(cable_velocity_mms=np.zeros(n),
                                  message="платформа успокаивается")

            self._settled = 0.0
            velocity = np.clip(remaining * 2.0, -self.feed_mms, self.feed_mms)
            velocity[np.abs(remaining) <= self.tolerance_mm] = 0.0
            return ModeOutput(
                cable_velocity_mms=velocity,
                message=(f"переезд к стоянке {self.index + 1}: осталось "
                         f"{np.round(remaining).astype(int).tolist()} мм по тросам"),
            )

    def describe(self) -> str:
        return f"привязка: стоянка {self.index + 1} из {len(self.plan)} ({self.phase})"
