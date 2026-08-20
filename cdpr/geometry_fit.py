"""Восстановление расположения модулей по простым замерам.

Зачем это нужно. Модули переставляют, и после каждой перестановки координаты
всех точек схода приходится задавать заново. Мерить двенадцать координат в
общей системе отсчёта — работа кропотливая и ненадёжная: нужно откуда-то
взять начало координат, выставить оси, ничего не перепутать.

Здесь то же самое получается из замеров, которые делаются без всякой системы
координат:

* **шесть расстояний между модулями** — от точки схода до точки схода,
  дальномером, в любом порядке;
* **четыре высоты модулей над полом** — рулеткой вниз.

Почему именно так, а не иначе. Проверка на модели показала, что высоты — самое
слабое место любой автоматической процедуры: модули стоят почти в одной
плоскости, тетраэдр получается почти плоским, и взаимные расстояния определяют
высоту четвёртой точки крайне неуверенно. Зато высота — это единственное, что
меряется совсем просто, рулеткой вниз до пола. Поэтому высоты измеряются, а
горизонталь считается из расстояний, где обусловленность хорошая.

Точность такого гибрида при дальномере 2 мм и высотах 3 мм — около 5 мм по
положению платформы, то есть не хуже, чем при аккуратном замере всех
двенадцати координат.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

#: порядок пар при вводе шести расстояний
PAIRS: list[tuple[int, int]] = list(itertools.combinations(range(4), 2))


@dataclass
class GeometryFit:
    positions: np.ndarray            # координаты точек схода, мм
    distance_residuals_mm: np.ndarray
    residual_rms_mm: float
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.residual_rms_mm < 15.0 and not self.warnings

    def summary(self, ids: list[str] | None = None) -> str:
        names = ids or [f"M{i}" for i in range(len(self.positions))]
        lines = [f"Расхождение по расстояниям: {self.residual_rms_mm:.1f} мм "
                 f"(наибольшее {np.abs(self.distance_residuals_mm).max():.1f} мм)", ""]
        for name, p in zip(names, self.positions, strict=True):
            lines.append(f"  {name}: {p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f} мм")
        lines += ["  " + w for w in self.warnings]
        return "\n".join(lines)


def fit_modules(
    distances_mm: dict[tuple[int, int], float] | list[float],
    heights_mm: list[float] | np.ndarray,
    *,
    n: int = 4,
    prior_xy: np.ndarray | None = None,
) -> GeometryFit:
    """Считает координаты точек схода по взаимным расстояниям и высотам.

    Базис задаётся самой процедурой, потому что снаружи его взять неоткуда:
    первый модуль кладётся в начало координат, второй — на ось X, третий —
    в положительную полуплоскость Y. Абсолютная привязка к комнате не нужна:
    системе важно только взаимное расположение.

    Высоты входят как измеренные и не подгоняются — они и так самое надёжное
    из того, что можно померить, и одновременно самое слабое место расчёта.
    """
    heights = np.asarray(heights_mm, dtype=float)
    if heights.shape != (n,):
        raise ValueError(f"нужно {n} высот модулей, передано {heights.shape[0]}")

    if isinstance(distances_mm, dict):
        try:
            d = np.array([float(distances_mm[pair]) for pair in PAIRS])
        except KeyError as exc:
            raise ValueError(
                f"не хватает расстояния между модулями {exc}. "
                f"Нужны все шесть пар: {PAIRS}"
            ) from None
    else:
        d = np.asarray(distances_mm, dtype=float)
        if d.shape != (len(PAIRS),):
            raise ValueError(f"нужно {len(PAIRS)} расстояний, передано {d.shape[0]}")

    if np.any(d <= 0):
        raise ValueError("расстояния между модулями должны быть положительными")

    warnings: list[str] = []
    # Треугольник должен существовать: сумма двух сторон больше третьей.
    for a, b, c in itertools.combinations(range(4), 3):
        sides = [d[PAIRS.index(tuple(sorted(p)))] for p in ((a, b), (b, c), (a, c))]
        sides.sort()
        if sides[0] + sides[1] <= sides[2] + 1.0:
            warnings.append(
                f"расстояния между модулями {a}, {b}, {c} не складываются в треугольник "
                f"({sides[0]:.0f} + {sides[1]:.0f} <= {sides[2]:.0f} мм) — где-то ошибка замера"
            )

    guess = _initial_xy(d) if prior_xy is None else np.asarray(prior_xy, dtype=float)

    def build(x: np.ndarray) -> np.ndarray:
        positions = np.zeros((n, 3))
        positions[1, 0] = x[0]
        positions[2, 0], positions[2, 1] = x[1], x[2]
        positions[3, 0], positions[3, 1] = x[3], x[4]
        positions[:, 2] = heights
        return positions

    def residual(x: np.ndarray) -> np.ndarray:
        positions = build(x)
        model = np.array([np.linalg.norm(positions[i] - positions[j]) for i, j in PAIRS])
        return model - d

    solution = least_squares(residual, guess, method="lm", xtol=1e-14, ftol=1e-14,
                             max_nfev=100000)
    positions = build(solution.x)
    residuals = residual(solution.x)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    if positions[2, 1] < 0:  # третий модуль обязан лежать в положительной полуплоскости
        positions[:, 1] *= -1.0

    if rms > 5.0:
        warnings.append(
            f"расстояния не сходятся между собой на {rms:.1f} мм. Проверьте замеры: "
            f"скорее всего перепутаны пары или дальномер бил не в точку схода троса"
        )

    a = positions[1, :2] - positions[0, :2]
    b = positions[2, :2] - positions[0, :2]
    area = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
    span = float(np.max(d))
    if area < 0.05 * span ** 2:
        warnings.append("модули стоят почти на одной линии — рабочей зоны у такой рамы нет")

    return GeometryFit(positions, residuals, rms, warnings)


def _initial_xy(d: np.ndarray) -> np.ndarray:
    """Начальное приближение из плоской задачи: раскладываем четырёхугольник
    по расстояниям, считая все модули на одной высоте."""
    d01, d02, d03, d12, d13, d23 = d
    x1 = d01
    x2 = (d01 ** 2 + d02 ** 2 - d12 ** 2) / (2 * d01)
    y2 = np.sqrt(max(1.0, d02 ** 2 - x2 ** 2))
    x3 = (d01 ** 2 + d03 ** 2 - d13 ** 2) / (2 * d01)
    y3sq = d03 ** 2 - x3 ** 2
    y3 = np.sqrt(max(1.0, y3sq))
    # четвёртый модуль обычно напротив третьего — выбираем ту же полуплоскость
    if abs(np.hypot(x3 - x2, y3 - y2) - d23) > abs(np.hypot(x3 - x2, -y3 - y2) - d23):
        y3 = -y3
    return np.array([x1, x2, y2, x3, y3])
