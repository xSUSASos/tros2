"""Калибровка: определение параметров лебёдок по реперным точкам.

Что калибруется и почему именно это.

Координаты якорей ЗАМЕРЯЮТСЯ дальномером, а не вычисляются. Проверка на
модели показала, что попытка вывести все двенадцать координат из одних лишь
длин тросов плохо обусловлена: при шуме измерения в 1 мм ошибка положения
выходит 40-70 мм. Если же якоря замерены с точностью 5 мм, ошибка
позиционирования получается около 10 мм. Вывод простой: калибровка не может
быть точнее рулетки, и незачем делать вид, что может.

Зато калибровкой прекрасно определяется то, что руками не измерить:
сколько троса намотано на барабане (а значит, как отсчёт энкодера связан с
длиной) и жёсткость троса на растяжение.

Реперная точка — место с известными координатами, куда платформа приводится
и где записываются отсчёты всех энкодеров. На подвесе это посадка на пол в
размеченную точку: касание видно по одновременному падению натяжения на всех
тросах, и сигнал этот куда чётче, чем скачок момента при боковом упоре.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from cdpr.config import MachineConfig, WinchCfg
from cdpr.kinematics import CDPRKinematics
from cdpr.line import LineModel

log = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """Замер в точке с известными координатами."""

    position_mm: np.ndarray
    counts: np.ndarray
    tensions_n: np.ndarray | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.position_mm = np.asarray(self.position_mm, dtype=float)
        self.counts = np.asarray(self.counts, dtype=float)
        if self.tensions_n is not None:
            self.tensions_n = np.asarray(self.tensions_n, dtype=float)


@dataclass
class CalibrationResult:
    count_empty: list[int]
    length_at_empty_mm: list[float]
    ea_n: list[float] | None
    residual_rms_mm: float
    residual_max_mm: float
    conditioning: float
    warnings: list[str] = field(default_factory=list)
    points_used: int = 0

    @property
    def ok(self) -> bool:
        return self.residual_rms_mm < 10.0 and not any("не сошлось" in w for w in self.warnings)

    def as_updates(self) -> dict[int, dict[str, float]]:
        """В формате, который принимает save_calibration."""
        updates: dict[int, dict[str, float]] = {}
        for i, (empty, length) in enumerate(zip(self.count_empty, self.length_at_empty_mm, strict=True)):
            entry = {"count_empty": int(empty), "length_at_empty_mm": round(float(length), 3)}
            if self.ea_n is not None:
                entry["ea_n"] = round(float(self.ea_n[i]), 1)
            updates[i] = entry
        return updates

    def summary(self) -> str:
        lines = [
            f"Калибровка по {self.points_used} точкам:",
            f"  расхождение: среднеквадратичное {self.residual_rms_mm:.2f} мм, "
            f"наибольшее {self.residual_max_mm:.2f} мм",
            f"  обусловленность задачи: {self.conditioning:.1f}",
        ]
        for i, length in enumerate(self.length_at_empty_mm):
            ea = f", EA {self.ea_n[i]:.0f} Н" if self.ea_n else ""
            lines.append(f"  лебёдка {i}: троса при пустом барабане {length / 1000:.3f} м{ea}")
        lines += ["  " + w for w in self.warnings]
        return "\n".join(lines)


def _predicted_counts(
    winch: WinchCfg, line: LineModel, distance_mm: float, tension_n: float,
    count_empty: float, length_at_empty: float, ea_n: float | None,
) -> float:
    """Какой отсчёт энкодера должен быть, если трос дотянут до этой точки."""
    free = distance_mm / (1.0 + tension_n / ea_n) if ea_n else distance_mm
    wound = max(0.0, length_at_empty - free)
    turns = line.turns_for_wound(wound)
    return count_empty + winch.direction * turns * winch.counts_per_drum_rev


def identify(
    machine: MachineConfig,
    points: list[CalibrationPoint],
    *,
    fit_elasticity: bool = True,
    kinematics: CDPRKinematics | None = None,
) -> CalibrationResult:
    """Определяет параметры лебёдок по замерам в известных точках.

    Невязка считается в МИЛЛИМЕТРАХ троса, а не в импульсах: так число сразу
    понятно человеку и сравнимо с точностью разметки реперов.
    """
    if len(points) < 2:
        raise ValueError(
            f"нужно минимум две реперные точки, передана {len(points)}. "
            f"Практически стоит взять четыре-шесть, разнесённых по всей площади."
        )

    kin = kinematics or CDPRKinematics.from_config(machine)
    winches = machine.ordered_winches()
    lines = [LineModel(w) for w in winches]
    n_winches = len(winches)
    warnings: list[str] = []

    distances = np.array([kin.inverse(p.position_mm) for p in points])   # (точки, тросы)
    counts = np.array([p.counts for p in points])
    if any(p.tensions_n is None for p in points):
        tensions = np.full_like(distances, machine.tension.target_n)
        if fit_elasticity:
            warnings.append(
                "натяжения в точках не записаны — жёсткость троса определить нельзя, "
                "вытяжка учтена приближённо по целевому преднатягу"
            )
            fit_elasticity = False
    else:
        tensions = np.array([p.tensions_n for p in points])

    # начальные приближения: барабан пуст при нулевом отсчёте, троса намотано
    # столько, сколько нужно для самой дальней точки плюс запас
    guess_length = float(distances.max()) * 1.3
    x0 = np.concatenate([
        np.zeros(n_winches),                       # count_empty, в оборотах барабана
        np.full(n_winches, guess_length),          # length_at_empty, мм
    ])
    if fit_elasticity:
        x0 = np.concatenate([x0, np.full(n_winches, 5000.0)])

    scale = np.array([w.counts_per_drum_rev for w in winches])

    def unpack(x: np.ndarray):
        empty = x[:n_winches] * scale
        length = x[n_winches:2 * n_winches]
        ea = x[2 * n_winches:3 * n_winches] if fit_elasticity else [None] * n_winches
        return empty, length, ea

    def residual(x: np.ndarray) -> np.ndarray:
        empty, length, ea = unpack(x)
        out = np.zeros(len(points) * n_winches)
        k = 0
        for j, point in enumerate(points):
            for i in range(n_winches):
                predicted = _predicted_counts(
                    winches[i], lines[i], float(distances[j, i]), float(tensions[j, i]),
                    float(empty[i]), float(length[i]), None if ea[i] is None else float(ea[i]),
                )
                # Переводим невязку в миллиметры троса — так она читаема и
                # сравнима с точностью разметки реперов. Масштаб берём
                # номинальный, по первому слою: он нужен лишь как весовой
                # коэффициент, и его нескольких процентов погрешности хватает.
                out[k] = (predicted - counts[j, i]) * winches[i].nominal_mm_per_count
                k += 1
        return out

    bounds_low = np.concatenate([np.full(n_winches, -1e6), np.full(n_winches, 1.0)])
    bounds_high = np.concatenate([np.full(n_winches, 1e6), np.full(n_winches, 1e6)])
    if fit_elasticity:
        bounds_low = np.concatenate([bounds_low, np.full(n_winches, 200.0)])
        bounds_high = np.concatenate([bounds_high, np.full(n_winches, 1e6)])

    solution = least_squares(residual, x0, bounds=(bounds_low, bounds_high), max_nfev=20000)
    if not solution.success:
        warnings.append(f"решение не сошлось: {solution.message}")

    empty, length, ea = unpack(solution.x)
    residuals = solution.fun
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    worst = float(np.max(np.abs(residuals)))

    singular = np.linalg.svd(solution.jac, compute_uv=False)
    conditioning = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if conditioning > 1e6:
        warnings.append(
            "задача вырождена: по этим точкам параметры разделить нельзя. "
            "Обычно так бывает, когда барабан не выходит за первый слой — "
            "тогда результат верен в рабочем диапазоне, но за его пределами не годится"
        )
    if rms > 5.0:
        warnings.append(
            f"расхождение {rms:.1f} мм великовато. Проверьте разметку реперов, "
            f"не проскальзывает ли трос на барабане и верны ли координаты якорей"
        )

    return CalibrationResult(
        count_empty=[int(round(v)) for v in empty],
        length_at_empty_mm=[float(v) for v in length],
        ea_n=[float(v) for v in ea] if fit_elasticity else None,
        residual_rms_mm=rms, residual_max_mm=worst, conditioning=conditioning,
        warnings=warnings, points_used=len(points),
    )
