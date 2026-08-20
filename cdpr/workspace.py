"""Рабочая зона: где платформа действительно управляема.

Задавать рабочую область отступом от стен неправильно. Граница определяется
не расстоянием, а тем, могут ли тросы вообще создать усилие в нужную сторону.
Для подвеса 6x5 м на высоте 3 м запас горизонтального усилия в центре около
43 Н, а в метре от стены — уже 3 Н, хотя формально «отступ целый метр».

Поэтому граница считается: в каждой точке ищется наибольшее возмущение,
которое платформа удержит в худшем направлении, и рабочей считается область,
где этот запас не меньше заданного. Пользовательский отступ добавляется
сверху, как дополнительная страховка.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from cdpr.config import MachineConfig
from cdpr.kinematics import CDPRKinematics
from cdpr.tension import capacity_margin, gravity_wrench, wrench_feasible

log = logging.getLogger(__name__)


@dataclass
class WorkspaceMap:
    """Карта запаса усилия на одной высоте."""

    z_mm: float
    xs: np.ndarray                      # узлы по X, мм
    ys: np.ndarray                      # узлы по Y, мм
    margin_n: np.ndarray                # запас в каждом узле, форма (len(ys), len(xs))
    required_n: float                   # порог, по которому область считается рабочей
    inset_mm: float
    box_low: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_high: np.ndarray = field(default_factory=lambda: np.zeros(3))
    elapsed_s: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def mask(self) -> np.ndarray:
        """Булева маска рабочей области."""
        inside = np.zeros_like(self.margin_n, dtype=bool)
        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                inside[iy, ix] = (
                    self.box_low[0] <= x <= self.box_high[0]
                    and self.box_low[1] <= y <= self.box_high[1]
                )
        return inside & (self.margin_n >= self.required_n)

    @property
    def area_fraction(self) -> float:
        return float(self.mask.mean())

    def margin_at(self, x: float, y: float) -> float:
        """Запас в произвольной точке — билинейной интерполяцией по сетке."""
        if x < self.xs[0] or x > self.xs[-1] or y < self.ys[0] or y > self.ys[-1]:
            return 0.0
        ix = np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2)
        iy = np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 2)
        tx = (x - self.xs[ix]) / (self.xs[ix + 1] - self.xs[ix])
        ty = (y - self.ys[iy]) / (self.ys[iy + 1] - self.ys[iy])
        m = self.margin_n
        return float(
            m[iy, ix] * (1 - tx) * (1 - ty) + m[iy, ix + 1] * tx * (1 - ty)
            + m[iy + 1, ix] * (1 - tx) * ty + m[iy + 1, ix + 1] * tx * ty
        )

    def contains(self, pose: np.ndarray) -> bool:
        x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
        if not (self.box_low[2] <= z <= self.box_high[2]):
            return False
        if not (self.box_low[0] <= x <= self.box_high[0]):
            return False
        if not (self.box_low[1] <= y <= self.box_high[1]):
            return False
        return self.margin_at(x, y) >= self.required_n

    def as_dict(self) -> dict:
        return {
            "z_mm": self.z_mm,
            "xs": self.xs.tolist(),
            "ys": self.ys.tolist(),
            "margin_n": np.round(self.margin_n, 2).tolist(),
            "required_n": self.required_n,
            "area_fraction": round(self.area_fraction, 4),
            "box_low": self.box_low.tolist(),
            "box_high": self.box_high.tolist(),
        }


# --------------------------------------------------------------------------- #
#  Расчёт карты
# --------------------------------------------------------------------------- #
def compute_map(
    machine: MachineConfig,
    kinematics: CDPRKinematics | None = None,
    *,
    z_mm: float | None = None,
    step_mm: float = 250.0,
    directions: int = 12,
    payload_kg: float = 0.0,
) -> WorkspaceMap:
    """Считает запас усилия по сетке на заданной высоте.

    Шаг сетки и число направлений — компромисс между точностью и временем:
    в каждом узле решается по одной задаче линейного программирования на
    направление. Для панели этого достаточно, а точечная проверка перед
    движением делается отдельно и с полной точностью.
    """
    kin = kinematics or CDPRKinematics.from_config(machine)
    anchors = kin.anchors
    z = z_mm if z_mm is not None else 0.5 * (machine.workspace.z_min_mm + machine.workspace.z_max_mm)

    x0, x1 = float(anchors[:, 0].min()), float(anchors[:, 0].max())
    y0, y1 = float(anchors[:, 1].min()), float(anchors[:, 1].max())
    xs = np.arange(x0, x1 + step_mm * 0.5, step_mm)
    ys = np.arange(y0, y1 + step_mm * 0.5, step_mm)

    wrench = gravity_wrench(machine.platform.mass_kg + payload_kg)
    f_min, f_max = machine.tension.min_n, machine.tension.max_n

    margin = np.zeros((len(ys), len(xs)))
    started = time.perf_counter()
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            pose = np.array([x, y, z])
            try:
                W = kin.structure_matrix(pose)
            except Exception:  # noqa: BLE001 — платформа совпала с якорем
                continue
            margin[iy, ix] = capacity_margin(
                W, wrench, f_min=f_min, f_max=f_max, directions=directions
            )
    elapsed = time.perf_counter() - started

    inset = machine.workspace.inset_mm
    # Границы берутся из общей функции, а не собираются заново: иначе карта и
    # проверка позы разошлись бы в том, что считается рабочей зоной, и панель
    # показывала бы одно, а контур запрещал другое.
    low, high = box_limits(machine, kin)

    result = WorkspaceMap(
        z_mm=z, xs=xs, ys=ys, margin_n=margin,
        required_n=machine.workspace.feasibility_margin_n,
        inset_mm=inset, box_low=low, box_high=high, elapsed_s=elapsed,
    )
    log.info(
        "карта на z=%.0f мм: %dx%d узлов за %.1f с, рабочей области %.0f%%",
        z, len(xs), len(ys), elapsed, result.area_fraction * 100,
    )
    return result


def check_pose(
    machine: MachineConfig,
    kinematics: CDPRKinematics,
    pose: np.ndarray,
    *,
    payload_kg: float = 0.0,
    directions: int = 24,
) -> tuple[bool, float, str]:
    """Точная проверка одной позы. Возвращает (можно, запас в Н, причина)."""
    pose = np.asarray(pose, dtype=float)
    low, high = box_limits(machine, kinematics)
    if np.any(pose < low) or np.any(pose > high):
        return False, 0.0, (
            f"вне габаритов рабочей зоны: {np.round(pose).astype(int)} "
            f"не входит в {np.round(low).astype(int)}..{np.round(high).astype(int)}"
        )
    try:
        W = kinematics.structure_matrix(pose)
    except Exception as exc:  # noqa: BLE001
        return False, 0.0, str(exc)

    wrench = gravity_wrench(machine.platform.mass_kg + payload_kg)
    f_min, f_max = machine.tension.min_n, machine.tension.max_n
    if not wrench_feasible(W, wrench, f_min=f_min, f_max=f_max):
        return False, 0.0, (
            "платформу здесь не удержать: нужные натяжения выходят за пределы "
            f"{f_min:.0f}..{f_max:.0f} Н"
        )
    margin = capacity_margin(W, wrench, f_min=f_min, f_max=f_max, directions=directions)
    required = machine.workspace.feasibility_margin_n
    if margin < required:
        return False, margin, (
            f"мало запаса: {margin:.1f} Н при требуемых {required:.1f} Н — "
            f"платформу здесь легко сбить с траектории"
        )
    return True, margin, "ок"


def box_limits(machine: MachineConfig, kinematics: CDPRKinematics) -> tuple[np.ndarray, np.ndarray]:
    """Габаритные пределы с учётом отступа и границ по высоте."""
    anchors = kinematics.anchors
    inset = machine.workspace.inset_mm
    low = anchors.min(axis=0) + inset
    high = anchors.max(axis=0) - inset
    if machine.geometry.is_planar:
        # На плоской машине высота не диапазон, а одно число: коробка всегда в
        # рабочей плоскости. Границы по Z совпадают, и любая цель по Z сама
        # прижимается к ней обрезкой — отдельной проверки не нужно.
        low[2] = high[2] = machine.geometry.plane_z_mm
    else:
        low[2] = machine.workspace.z_min_mm
        high[2] = machine.workspace.z_max_mm
    return low, high


def best_height(
    machine: MachineConfig,
    kinematics: CDPRKinematics | None = None,
    *,
    samples: int = 15,
    payload_kg: float = 0.0,
) -> tuple[float, float]:
    """Высота с наибольшим запасом усилия в центре и сам запас.

    У подвеса есть оптимум: внизу тросы почти вертикальны и горизонтального
    усилия создать не могут, вверху упираются в предел натяжения. Панель
    показывает эту высоту, чтобы не подбирать её вслепую.
    """
    kin = kinematics or CDPRKinematics.from_config(machine)
    centre = kin.anchors.mean(axis=0)
    wrench = gravity_wrench(machine.platform.mass_kg + payload_kg)
    best_z, best_margin = machine.workspace.z_min_mm, -1.0
    for z in np.linspace(machine.workspace.z_min_mm, machine.workspace.z_max_mm, samples):
        pose = np.array([centre[0], centre[1], z])
        margin = capacity_margin(
            kin.structure_matrix(pose), wrench,
            f_min=machine.tension.min_n, f_max=machine.tension.max_n, directions=12,
        )
        if margin > best_margin:
            best_z, best_margin = float(z), float(margin)
    return best_z, best_margin
