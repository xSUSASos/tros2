"""Распределение натяжений по тросам.

Тросов больше, чем степеней свободы, поэтому набор натяжений, удерживающих
платформу, не единственный — и этой свободой надо пользоваться, а не гасить
её произвольным выбором. Запас нужен на две вещи сразу: ни один трос не
должен провиснуть (провисший выпадает из модели, и платформа теряет
управляемость), и ни один не должен превысить предел самого слабого звена.

Основной способ — замкнутая формула: берём желаемый преднатяг и добавляем
минимальную поправку, приводящую сумму сил к равновесию. Считается за
микросекунды, что важно для цикла на 50 Гц. Если результат вылез за пределы,
включается линейное программирование, которое ищет решение с наибольшим
запасом до границ.

Знак: `W f = -w_внеш`. С плюсом получается правдоподобный, но ровно неверный
набор натяжений, и в симметричной раме ошибка незаметна — поэтому она здесь
зафиксирована в одном месте и закрыта тестами.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

GRAVITY = 9.80665  # м/с^2


@dataclass
class TensionResult:
    forces: np.ndarray          # натяжения по тросам, Н
    feasible: bool              # удалось ли удержать платформу в пределах
    method: str                 # каким способом получено
    margin_n: float             # минимальный зазор до ближайшей границы, Н
    message: str = ""

    @property
    def min_force(self) -> float:
        return float(np.min(self.forces)) if self.forces.size else 0.0

    @property
    def max_force(self) -> float:
        return float(np.max(self.forces)) if self.forces.size else 0.0


def gravity_wrench(mass_kg: float) -> np.ndarray:
    """Внешняя нагрузка от веса платформы, Н. Направлена вниз."""
    return np.array([0.0, 0.0, -mass_kg * GRAVITY])


def distribute_closed_form(
    W: np.ndarray,
    wrench_ext: np.ndarray,
    f_pref: np.ndarray,
) -> np.ndarray:
    """Минимальная поправка к желаемому преднатягу, дающая равновесие.

    f = f_pref + W^+ (-w_внеш - W f_pref)

    Псевдообратная матрица даёт поправку наименьшей нормы, то есть решение
    держится как можно ближе к заданному преднатягу. Это и есть смысл
    настройки «целевое натяжение» в панели.
    """
    residual = -np.asarray(wrench_ext, dtype=float) - W @ f_pref
    return f_pref + np.linalg.pinv(W) @ residual


def distribute_lp(
    W: np.ndarray,
    wrench_ext: np.ndarray,
    f_min: float,
    f_max: float,
) -> tuple[np.ndarray | None, float]:
    """Решение с наибольшим запасом до границ.

    Максимизируем t при f_min + t <= f_i <= f_max - t и W f = -w_внеш.
    Положительное t означает, что до провисания и до предела прочности
    остаётся именно столько ньютонов.
    """
    m = W.shape[1]
    # переменные: [f_1..f_m, t]; максимизируем t -> минимизируем -t
    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_eq = np.hstack([W, np.zeros((W.shape[0], 1))])
    b_eq = -np.asarray(wrench_ext, dtype=float)

    # f_i - t >= f_min  ->  -f_i + t <= -f_min
    # f_i + t <= f_max
    A_ub = np.vstack([
        np.hstack([-np.eye(m), np.ones((m, 1))]),
        np.hstack([np.eye(m), np.ones((m, 1))]),
    ])
    b_ub = np.concatenate([np.full(m, -f_min), np.full(m, f_max)])

    bounds = [(f_min, f_max)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None, float("-inf")
    return np.asarray(res.x[:m]), float(res.x[m])


def null_space(W: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Базис нуль-пространства структурной матрицы, форма (m, r).

    Это в точности те комбинации натяжений, которые НЕ двигают платформу, а
    значит ими и можно распоряжаться. Для четырёх тросов и трёх степеней
    свободы базис один и выглядит как (1, -1, 1, -1): перетяжка диагональных
    пар друг против друга.
    """
    _, singular, vt = np.linalg.svd(np.asarray(W, dtype=float))
    rank = int(np.sum(singular > tol * max(1.0, singular[0])))
    return vt[rank:].T


def feasible_lambda_range(
    f_particular: np.ndarray, direction: np.ndarray, f_min: float, f_max: float
) -> tuple[float, float]:
    """Диапазон свободного параметра, при котором все тросы в пределах."""
    lo, hi = -np.inf, np.inf
    for f0, n in zip(f_particular, direction, strict=True):
        if abs(n) < 1e-12:
            if not (f_min - 1e-9 <= f0 <= f_max + 1e-9):
                return 1.0, -1.0  # пустой диапазон
            continue
        a, b = (f_min - f0) / n, (f_max - f0) / n
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return lo, hi


def distribute(
    W: np.ndarray,
    wrench_ext: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    f_target: float | None = None,
    f_prefer: np.ndarray | None = None,
) -> TensionResult:
    """Натяжения для удержания платформы.

    Важное свойство, которое стоит понимать при настройке. Общий уровень
    натяжения НЕ является свободным параметром: при четырёх тросах и трёх
    степенях свободы равновесие задаёт его однозначно, и «поднять натяжение
    во всех тросах» алгоритмом нельзя — вектор (1,1,1,1) не лежит в
    нуль-пространстве. Свободна ровно одна комбинация — перетяжка
    диагональных пар. Поэтому целевое натяжение здесь означает цель для
    САМОГО НЕНАГРУЖЕННОГО троса: именно он рискует провиснуть, и именно его
    имеет смысл держать подальше от нуля.

    Для случая с одной степенью свободы (4 троса, 3 координаты) задача
    одномерная и решается аналитически за микросекунды. Для более
    избыточных систем включается линейное программирование.
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[1]
    target = f_target if f_target is not None else 0.5 * (f_min + f_max)

    rhs = -np.asarray(wrench_ext, dtype=float)
    u, singular, vt = np.linalg.svd(W)
    rank = int(np.sum(singular > 1e-9 * max(1.0, singular[0])))
    # решение наименьшей нормы и базис нуль-пространства — из одного разложения
    inv_s = np.zeros_like(singular)
    inv_s[:rank] = 1.0 / singular[:rank]
    f_particular = vt[:len(singular)].T @ (inv_s * (u[:, :len(singular)].T @ rhs))
    basis = vt[rank:].T

    if basis.shape[1] == 1:
        return _distribute_1dof(W, f_particular, basis[:, 0], f_min, f_max, target, f_prefer)

    forces_lp, margin = distribute_lp(W, wrench_ext, f_min, f_max)
    if forces_lp is None:
        return TensionResult(
            f_particular, False, "нет решения", -np.inf,
            f"в этой позе удержать платформу нельзя: нужны натяжения "
            f"{f_particular.min():.1f}..{f_particular.max():.1f} Н "
            f"при пределах {f_min:.1f}..{f_max:.1f} Н",
        )
    return TensionResult(forces_lp, True, "линейное программирование", margin)


def _distribute_1dof(
    W: np.ndarray,
    f_particular: np.ndarray,
    direction: np.ndarray,
    f_min: float,
    f_max: float,
    target: float,
    f_prefer: np.ndarray | None = None,
) -> TensionResult:
    """Одна свободная степень: подбираем её так, чтобы самый слабый трос
    оказался как можно ближе к целевому натяжению.

    Натяжение каждого троса линейно по свободному параметру, поэтому минимум
    по тросам — вогнутая ломаная, и её максимум лежит ровно в точке излома,
    то есть там, где две прямые пересекаются. Точек излома при четырёх тросах
    всего шесть, так что ответ берётся точно и без итераций — это важно,
    потому что расчёт идёт в каждом цикле управления.
    """
    lo, hi = feasible_lambda_range(f_particular, direction, f_min, f_max)
    if lo > hi:
        return TensionResult(
            f_particular, False, "нет решения", -np.inf,
            f"в этой позе удержать платформу нельзя: нужны натяжения "
            f"{f_particular.min():.1f}..{f_particular.max():.1f} Н "
            f"при пределах {f_min:.1f}..{f_max:.1f} Н",
        )

    # точки излома: пересечения пар прямых f_i(lam) = f_j(lam)
    df = f_particular[:, None] - f_particular[None, :]
    dn = direction[None, :] - direction[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = np.where(np.abs(dn) > 1e-12, df / dn, np.nan)
    lambdas = np.concatenate([[lo, hi], crossings[np.isfinite(crossings)]])
    lambdas = lambdas[(lambdas >= lo - 1e-9) & (lambdas <= hi + 1e-9)]
    lambdas = np.clip(lambdas, lo, hi)

    forces = f_particular[None, :] + lambdas[:, None] * direction[None, :]
    weakest = forces.min(axis=1)
    best = int(np.argmax(weakest))

    if weakest[best] <= target + 1e-9:
        chosen = forces[best]
        margin = float(min(chosen.min() - f_min, f_max - chosen.max()))
        note = ""
        if weakest[best] < target - 1e-6:
            note = (
                f"самый слабый трос удаётся держать только на {weakest[best]:.1f} Н "
                f"вместо желаемых {target:.1f} Н — это предел геометрии, а не настройки"
            )
        return TensionResult(chosen, True, "аналитически, 1 степень свободы", margin, note)

    # цель достижима: она берётся там, где какой-то трос выходит ровно на неё
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = np.where(np.abs(direction) > 1e-12,
                        (target - f_particular) / np.where(np.abs(direction) > 1e-12, direction, 1.0),
                        np.nan)
    hits = hits[np.isfinite(hits)]
    hits = hits[(hits >= lo - 1e-9) & (hits <= hi + 1e-9)]
    if hits.size:
        candidate_forces = f_particular[None, :] + np.clip(hits, lo, hi)[:, None] * direction[None, :]
        exact = np.abs(candidate_forces.min(axis=1) - target) < 1e-6
        if np.any(exact):
            candidate_forces = candidate_forces[exact]
        chosen = _pick(candidate_forces, f_prefer)
    else:
        chosen = forces[best]

    margin = float(min(chosen.min() - f_min, f_max - chosen.max()))
    return TensionResult(chosen, True, "аналитически, 1 степень свободы", margin)


def _pick(candidates: np.ndarray, f_prefer: np.ndarray | None) -> np.ndarray:
    """Выбирает один вариант из равнозначных.

    В симметричной раме цель достижима двумя способами: перетянуть одну
    диагональ или другую. Оба одинаково хороши, но выбирать между ними
    случайно нельзя — решение будет скакать от цикла к циклу, и приводы
    начнут дёргаться вместо того, чтобы подтягивать тросы. Поэтому при
    прочих равных берётся вариант, ближайший к текущему состоянию.
    """
    if f_prefer is not None and len(f_prefer) == candidates.shape[1]:
        distance = np.linalg.norm(candidates - np.asarray(f_prefer, dtype=float), axis=1)
        best = float(distance.min())
        # Ничья случается ровно тогда, когда состояние симметрично, а это
        # обычное дело в прямоугольной раме. Разрывать её случайно нельзя:
        # решение заскачет между вариантами. Берём детерминированный признак.
        tied = np.flatnonzero(distance <= best + 1e-6)
        if tied.size == 1:
            return candidates[int(tied[0])]
        return candidates[int(tied[np.argmax(candidates[tied, 0])])]
    lowest = candidates.max(axis=1)
    best = float(lowest.min())
    tied = np.flatnonzero(lowest <= best + 1e-6)
    return candidates[int(tied[np.argmax(candidates[tied, 0])])]


def capacity_margin(
    W: np.ndarray,
    wrench_ext: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    directions: int = 24,
    plane_only: bool = True,
) -> float:
    """Наибольшее возмущение, которое платформа удержит в ХУДШЕМ направлении, Н.

    Это и есть честная мера «хорошести» позы: обусловленность матрицы про
    неё говорит косвенно, а здесь сразу ответ в ньютонах, который можно
    сравнить с реальными помехами — ветром, рывком при разгоне, трением.

    plane_only считает только горизонтальные возмущения: для подвеса именно
    они ограничивают рабочую зону, вертикаль держит вес.
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[1]
    b_eq = -np.asarray(wrench_ext, dtype=float)
    worst = float("inf")

    for theta in np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False):
        if plane_only:
            u = np.array([np.cos(theta), np.sin(theta), 0.0])
        else:
            u = np.array([np.cos(theta), 0.0, np.sin(theta)])
        # максимизируем R при W f = -w_внеш + R*u
        c = np.zeros(m + 1)
        c[-1] = -1.0
        A_eq = np.hstack([W, -u.reshape(-1, 1)])
        res = linprog(
            c, A_eq=A_eq, b_eq=b_eq,
            bounds=[(f_min, f_max)] * m + [(0.0, None)], method="highs",
        )
        worst = min(worst, float(res.x[m]) if res.success else 0.0)
        if worst <= 0.0:
            return 0.0
    return worst


def wrench_feasible(
    W: np.ndarray,
    wrench_ext: np.ndarray,
    *,
    f_min: float,
    f_max: float,
) -> bool:
    """Существует ли вообще допустимый набор натяжений в этой позе."""
    forces, _ = distribute_lp(W, wrench_ext, f_min, f_max)
    return forces is not None


def forces_to_torque_percent(forces: np.ndarray, winches, radii_mm) -> np.ndarray:
    """Натяжения -> моменты моторов в процентах (для проверки и телеметрии)."""
    return np.array([
        w.force_to_torque_percent(float(f), r)
        for w, f, r in zip(winches, forces, radii_mm, strict=True)
    ])


def torque_percent_to_forces(percent, winches, radii_mm) -> np.ndarray:
    """Моменты моторов -> натяжения тросов. Так система «чувствует» нагрузку."""
    return np.array([
        w.torque_percent_to_force(float(p), r)
        for w, p, r in zip(winches, percent, radii_mm, strict=True)
    ])


def external_wrench_from_forces(W: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Какая внешняя сила соответствует измеренным натяжениям.

    На этом стоит ручное перемещение «за руку»: вычитая из измеренного
    вес платформы, получаем усилие руки, и по нему задаём скорость.
    """
    return -(np.asarray(W, dtype=float) @ np.asarray(forces, dtype=float))
