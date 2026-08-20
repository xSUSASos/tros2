"""Кинематика тросовой системы.

Соглашения о знаках, на которых легко ошибиться и потом долго искать:

* `u_i` — единичный вектор от платформы К ТОЧКЕ СХОДА троса. Именно в эту
  сторону трос может тянуть, и только тянуть.
* Длина троса `l_i = |A_i - P|`, поэтому её производная по положению равна
  `-u_i`: если двигаться в сторону якоря, трос укорачивается.
* Скорость наматывания положительна, когда трос укорачивается, то есть
  `winding_i = +u_i . v`. Это согласовано с `direction` в конфиге лебёдки.
* Структурная матрица `W = [u_1 ... u_m]`, равновесие `W f = -w_внеш`.
  Знак минус — самая частая ошибка в расчётах тросовых систем: с плюсом
  получается правдоподобный, но ровно неверный набор натяжений.

Число тросов и размерность берутся из конфига, поэтому один и тот же код
работает и для плоской системы, и для подвеса.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from cdpr.config import MachineConfig


class KinematicsError(RuntimeError):
    pass


class CDPRKinematics:
    """Прямая и обратная задача для платформы, подвешенной на m тросах."""

    def __init__(self, anchors: np.ndarray, attachments: np.ndarray | None = None) -> None:
        self.anchors = np.asarray(anchors, dtype=float)
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 3:
            raise KinematicsError(f"якоря должны быть массивом (m, 3), получено {self.anchors.shape}")
        self.m = self.anchors.shape[0]
        if attachments is None:
            self.attachments = np.zeros_like(self.anchors)
        else:
            self.attachments = np.asarray(attachments, dtype=float)
            if self.attachments.shape != self.anchors.shape:
                raise KinematicsError(
                    f"точек крепления {self.attachments.shape[0]}, а якорей {self.m}"
                )

    @classmethod
    def from_config(cls, machine: MachineConfig) -> "CDPRKinematics":
        anchors = np.array([a.pos for a in machine.geometry.anchors], dtype=float)
        attachments = np.array(machine.platform.attachments, dtype=float)
        return cls(anchors, attachments)

    # ------------------------------------------------------------------ #
    #  Обратная задача
    # ------------------------------------------------------------------ #
    def cable_vectors(self, pose: np.ndarray) -> np.ndarray:
        """Векторы от точек крепления к якорям, форма (m, 3)."""
        return self.anchors - (np.asarray(pose, dtype=float) + self.attachments)

    def inverse(self, pose: np.ndarray) -> np.ndarray:
        """Положение платформы -> длины тросов, форма (m,)."""
        return np.linalg.norm(self.cable_vectors(pose), axis=1)

    def unit_vectors(self, pose: np.ndarray) -> np.ndarray:
        """Единичные векторы платформа -> якорь, форма (m, 3)."""
        vectors = self.cable_vectors(pose)
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths < 1e-9):
            raise KinematicsError("платформа совпала с точкой схода троса — длина нулевая")
        return vectors / lengths[:, None]

    def jacobian(self, pose: np.ndarray) -> np.ndarray:
        """dl/dP, форма (m, 3). Равна -u, потому что движение к якорю укорачивает трос."""
        return -self.unit_vectors(pose)

    def length_rates(self, pose: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Скорости изменения длин при скорости платформы `velocity`.
        Положительное значение — трос удлиняется (стравливается)."""
        return self.jacobian(pose) @ np.asarray(velocity, dtype=float)

    def winding_rates(self, pose: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Скорости наматывания: положительная — трос выбирается."""
        return -self.length_rates(pose, velocity)

    # ------------------------------------------------------------------ #
    #  Прямая задача
    # ------------------------------------------------------------------ #
    def forward(
        self,
        lengths: np.ndarray,
        guess: np.ndarray | None = None,
        *,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Длины тросов -> положение платформы. Возвращает (положение, невязка).

        Задача переопределена (тросов больше, чем степеней свободы), поэтому
        решается методом наименьших квадратов: лишние измерения гасят шум, а
        невязка сама по себе полезна — её резкий рост означает, что трос
        провис, проскользнул или калибровка уехала.
        """
        lengths = np.asarray(lengths, dtype=float)
        if lengths.shape != (self.m,):
            raise KinematicsError(f"нужно {self.m} длин, передано {lengths.shape}")

        start = np.asarray(guess, dtype=float) if guess is not None else self.anchors.mean(axis=0)

        def residual(p: np.ndarray) -> np.ndarray:
            return self.inverse(p) - lengths

        kwargs = {}
        if bounds is not None:
            # Границы нужны не для красоты: у подвеса задача имеет ЗЕРКАЛЬНОЕ
            # решение выше плоскости якорей, физически невозможное. Без границ
            # решатель уходит туда, стоит начальному приближению оказаться
            # рядом с плоскостью.
            low, high = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
            kwargs["bounds"] = (low, high)
            start = np.clip(start, low + 1e-6, high - 1e-6)
        solution = least_squares(residual, start, **kwargs)
        rms = float(np.sqrt(np.mean(solution.fun ** 2)))
        return solution.x, rms

    # ------------------------------------------------------------------ #
    #  Прямая задача для плоской машины — в замкнутой форме
    # ------------------------------------------------------------------ #
    def forward_planar(self, lengths: np.ndarray, plane_z: float) -> tuple[np.ndarray, float]:
        """Длины тросов -> положение, когда высота платформы известна заранее.

        На плоской машине Z не управляется: платформа висит на постоянном
        отдалении от плоскости якорей. Это убирает из задачи одну неизвестную,
        и оставшиеся две находятся без всякого решателя.

            r_i^2 = L_i^2 - (z_i - z)^2            плоское расстояние до якоря
            (x - ax_i)^2 + (y - ay_i)^2 = r_i^2

        Вычитая первое уравнение из остальных, квадраты сокращаются и остаётся
        линейная система на (x, y). Три уравнения на две неизвестные решаются
        наименьшими квадратами — лишнее измерение гасит шум.

        Ни итераций, ни начального приближения, ни границ, ни зеркального
        решения выше плоскости якорей: его здесь просто неоткуда взять.
        """
        lengths = np.asarray(lengths, dtype=float)
        if lengths.shape != (self.m,):
            raise KinematicsError(f"нужно {self.m} длин, передано {lengths.shape}")
        if np.any(np.abs(self.attachments) > 1e-9):
            raise KinematicsError(
                "замкнутая форма выведена для точечной платформы; при ненулевых "
                "точках крепления пользуйтесь forward()"
            )

        drop = self.anchors[:, 2] - float(plane_z)
        flat = lengths ** 2 - drop ** 2
        # Трос короче собственного вертикального перепада — так не бывает.
        # Обычно это значит, что калибровка уехала или трос провис.
        impossible = flat < 0.0
        flat = np.clip(flat, 0.0, None)
        radii = np.sqrt(flat)

        xy = self.anchors[:, :2]
        base = xy[0]
        matrix = 2.0 * (xy[1:] - base)
        rhs = (
            (xy[1:] ** 2).sum(axis=1) - (base ** 2).sum()
            - (radii[1:] ** 2 - radii[0] ** 2)
        )
        try:
            solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise KinematicsError(f"якоря вырождены, положение не определяется: {exc}") from exc

        pose = np.array([solution[0], solution[1], float(plane_z)])
        residual = float(np.sqrt(np.mean((self.inverse(pose) - lengths) ** 2)))
        if impossible.any():
            # Невязку в этом случае занижать нельзя: она единственный признак
            # того, что данные противоречивы, и по ней срабатывает защита.
            residual = max(residual, float(np.max(np.sqrt(-(lengths ** 2 - drop ** 2)[impossible]))))
        return pose, residual

    # ------------------------------------------------------------------ #
    #  Структурная матрица
    # ------------------------------------------------------------------ #
    def structure_matrix(self, pose: np.ndarray) -> np.ndarray:
        """W, форма (3, m). Столбцы — направления, в которых тросы могут тянуть."""
        return self.unit_vectors(pose).T

    def condition_number(self, pose: np.ndarray) -> float:
        """Обусловленность W: насколько «вырождена» поза.

        Большое значение означает, что тросы почти коллинеарны и удержание
        платформы поперёк этого направления требует огромных натяжений.
        """
        return float(np.linalg.cond(self.structure_matrix(pose)))


def workspace_bounds(machine: MachineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Габаритная коробка по якорям с учётом отступа и пределов по высоте.

    На плоской машине высота не диапазон, а одно число: платформа всегда в
    рабочей плоскости, поэтому нижняя и верхняя границы по Z совпадают.
    """
    anchors = np.array([a.pos for a in machine.geometry.anchors], dtype=float)
    inset = machine.workspace.inset_mm
    low = anchors.min(axis=0) + inset
    high = anchors.max(axis=0) - inset
    if machine.geometry.is_planar:
        low[2] = high[2] = machine.geometry.plane_z_mm
    else:
        low[2] = machine.workspace.z_min_mm
        high[2] = machine.workspace.z_max_mm
    return low, high
