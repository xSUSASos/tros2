"""Привязка: перевод отсчётов энкодера в длину троса.

Нужно узнать всего два числа на лебёдку — отсчёт энкодера в известной точке и
длину троса в этот момент. Приращение длины дальше известно точно и безо
всякой калибровки: сколько импульсов намотал барабан, столько троса и выбрал.
Неизвестно только начало отсчёта, и вся привязка состоит в том, чтобы его
получить.

Известная точка — угол. Коробка подтягивается вплотную к модулю и упирается;
повторяемость тут механическая, а не программная, и потому надёжная. Одного
угла достаточно, и одним углом стоит и ограничиться.

Про лишние углы стоит сказать прямо, потому что соблазн ими проверить себя
велик. У угла натянут ровно ОДИН трос — тот, что тянет; остальные три
следящих держатся у нижнего края, около двух ньютонов, и провисают. А
провисший трос о геометрии не говорит ничего: его отсчёт показывает, сколько
стравлено с барабана, а не расстояние до платформы. Поэтому проверка по
второму и третьему углу опирается на один-два числа и слаба. Настоящие
проверки другие и делаются на верстаке: проворот барабана рукой ловит ошибку
масштаба, рулетка — ошибку в размерах рамы.

Чего здесь СОЗНАТЕЛЬНО нет.

*Восстановления геометрии рамы.* Соблазн вывести стороны прямоугольника из
разности отсчётов между углами велик, но чувствительность к тому, насколько
коробка не доходит до модуля, выходит примерно двукратной: знание этой
величины с точностью 20 мм даёт сторону с точностью 36 мм. Рулетка точнее.
Поэтому стороны вводятся руками, а лишние углы работают ПРОВЕРКОЙ: по
привязке из первого угла предсказываются отсчёты в остальных, и расхождение
сразу показывает, врут ли размеры рамы или масштаб «мм на импульс».

*Подгонки координат модулей по позам.* Проверено численно и отброшено: модули
почти в одной плоскости, и смещения тросов размениваются на положение модулей.
Решатель уходит от истины на сотни миллиметров, получая при этом МЕНЬШУЮ
невязку. Координаты якорей замеряются, а не вычисляются.

*Подгонки жёсткости троса.* Тот же капкан. В углу натяжение ведущего троса
на порядок больше, чем у остальных, поэтому вытяжка становится очень сильной
ручкой: решатель охотно объясняет ею любую ошибку в том, где именно встала
коробка, и уезжает к границе допустимого. Жёсткость меряется прямо и за две
минуты: подвесить на отрезок лески известной длины известный груз и замерить
удлинение. EA = груз * длина / удлинение.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from cdpr.config import MachineConfig
from cdpr.kinematics import CDPRKinematics
from cdpr.line import LineModel

log = logging.getLogger(__name__)


@dataclass
class CornerRecord:
    """Отсчёты энкодеров в момент, когда коробка упёрлась в модуль."""

    corner: int
    counts: np.ndarray
    tensions_n: np.ndarray | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=float)
        if self.tensions_n is not None:
            self.tensions_n = np.asarray(self.tensions_n, dtype=float)


@dataclass
class CalibrationResult:
    count_ref: list[int]
    length_at_ref_mm: list[float]
    ea_n: list[float] | None
    residual_rms_mm: float
    residual_max_mm: float
    warnings: list[str] = field(default_factory=list)
    points_used: int = 0
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not any(w.startswith("!") for w in self.warnings)

    def as_updates(self) -> dict[int, dict[str, float]]:
        """В формате, который принимает save_calibration."""
        updates: dict[int, dict[str, float]] = {}
        for i, (ref, length) in enumerate(zip(self.count_ref, self.length_at_ref_mm, strict=True)):
            entry: dict[str, float] = {
                "count_ref": int(ref),
                "length_at_ref_mm": round(float(length), 3),
            }
            if self.ea_n is not None:
                entry["ea_n"] = round(float(self.ea_n[i]), 1)
            updates[i] = entry
        return updates

    def summary(self) -> str:
        lines = [f"Привязка по углу, проверок {self.checked}:"]
        for i, length in enumerate(self.length_at_ref_mm):
            ea = f", жёсткость троса {self.ea_n[i]:.0f} Н" if self.ea_n else ""
            lines.append(f"  трос {i}: в опорной точке {length:.1f} мм{ea}")
        if self.checked and self.residual_rms_mm:
            lines.append(
                f"  расхождение на проверочных углах (по натянутым тросам): "
                f"среднеквадратичное {self.residual_rms_mm:.1f} мм, "
                f"наибольшее {self.residual_max_mm:.1f} мм"
            )
        lines += ["  " + w.lstrip("!") for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Где стоит коробка, упершись в модуль
# --------------------------------------------------------------------------- #
def park_pose(machine: MachineConfig, corner: int,
              kinematics: CDPRKinematics | None = None) -> np.ndarray:
    """Положение центра коробки, когда она подтянута к модулю `corner`.

    Коробка приходит к модулю по диагонали, со стороны центра рамы, поэтому
    смещение откладывается вдоль этого направления. Обе величины —
    горизонтальный отступ и высота — меряются линейкой один раз.

    Ошибка в них сдвигает всю систему координат целиком, но движение не
    искажает: она входит во все длины одинаково.
    """
    kin = kinematics or CDPRKinematics.from_config(machine)
    anchors = kin.anchors
    if not 0 <= corner < len(anchors):
        raise ValueError(f"нет угла {corner}, якорей {len(anchors)}")

    centre = anchors[:, :2].mean(axis=0)
    inward = centre - anchors[corner, :2]
    norm = float(np.linalg.norm(inward))
    if norm < 1e-6:
        raise ValueError(f"якорь {corner} совпал с центром рамы — направление не определить")

    xy = anchors[corner, :2] + inward / norm * machine.homing.corner_inset_mm
    z = machine.homing.corner_z_mm
    if z is None:
        z = machine.geometry.plane_z_mm
    if z is None:
        z = float(anchors[:, 2].mean())
    return np.array([xy[0], xy[1], float(z)])


# --------------------------------------------------------------------------- #
#  Собственно привязка
# --------------------------------------------------------------------------- #
def solve_from_corners(
    machine: MachineConfig,
    records: list[CornerRecord],
    *,
    ea_n: float | None = None,
    kinematics: CDPRKinematics | None = None,
) -> CalibrationResult:
    """Привязка по первому углу; остальные углы — проверка.

    Вытяжка троса не игнорируется: геометрия «видит» растянутый трос, а
    энкодер меряет свободную длину, и разница при капроне доходит до сотни
    миллиметров. Жёсткость берётся из конфига (`winch.ea_n`) и здесь не
    подбирается — см. пояснение в шапке модуля.
    """
    if not records:
        raise ValueError("нужен хотя бы один угол")

    kin = kinematics or CDPRKinematics.from_config(machine)
    winches = machine.ordered_winches()
    lines = [LineModel(w) for w in winches]
    n = len(winches)
    warnings: list[str] = []

    for record in records:
        if record.counts.shape != (n,):
            raise ValueError(f"нужно {n} отсчётов, в записи {record.counts.shape}")

    have_tension = all(r.tensions_n is not None for r in records)
    ea_value: float | None = ea_n if ea_n is not None else winches[0].ea_n
    if ea_value and not have_tension:
        warnings.append(
            "натяжения в углах не записаны — вытяжка не учтена. При мягком тросе это "
            "десятки миллиметров ошибки"
        )
        ea_value = None
    if not ea_value:
        warnings.append(
            "жёсткость троса не задана (winch.ea_n) — вытяжка не учитывается. Померьте её "
            "грузом: подвесьте известный вес на отрезок лески известной длины и замерьте "
            "удлинение, EA = вес * длина / удлинение"
        )

    poses = [park_pose(machine, r.corner, kin) for r in records]
    geometric = [kin.inverse(p) for p in poses]

    def calibrate(ea: float | None) -> tuple[np.ndarray, np.ndarray]:
        """Опорные числа по первой записи при данной жёсткости троса."""
        reference, g = records[0], geometric[0]
        tension = reference.tensions_n if reference.tensions_n is not None else np.zeros(n)
        free = np.array([
            float(g[i]) / (1.0 + float(tension[i]) / ea) if ea else float(g[i])
            for i in range(n)
        ])
        return reference.counts.astype(float), free

    # Провисший трос в проверке не участвует: его отсчёт говорит, сколько
    # стравлено с барабана, а вовсе не расстояние до коробки. Считать по нему
    # расхождение — значит мерить величину слабины и называть её ошибкой
    # геометрии.
    taut_floor = machine.tension.min_n
    usable = [0]

    def check_residual(ea: float | None) -> np.ndarray:
        """Расхождение на проверочных углах, в миллиметрах троса."""
        count_ref, length_ref = calibrate(ea)
        out: list[float] = []
        usable[0] = 0
        for record, g in zip(records[1:], geometric[1:], strict=True):
            tension = record.tensions_n if record.tensions_n is not None else np.zeros(n)
            for i in range(n):
                if float(tension[i]) < taut_floor:
                    continue
                usable[0] += 1
                free_expected = (
                    float(g[i]) / (1.0 + float(tension[i]) / ea) if ea else float(g[i])
                )
                wound = length_ref[i] - free_expected
                turns = lines[i].turns_for_wound(wound)
                predicted = count_ref[i] + winches[i].direction * turns * winches[i].counts_per_drum_rev
                out.append((predicted - record.counts[i]) * winches[i].nominal_mm_per_count)
        return np.array(out) if out else np.zeros(0)

    count_ref, length_ref = calibrate(ea_value)
    residuals = check_residual(ea_value)
    rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    worst = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    if residuals.size and usable[0] < 2:
        warnings.append(
            f"проверка по лишним углам не набрала данных: натянутых тросов оказалось "
            f"{usable[0]}. Так и должно быть — у угла тянет один трос, остальные провисают. "
            f"Опирайтесь на проворот барабана и рулетку, а не на это число"
        )

    if residuals.size and rms > 60.0:
        warnings.append(
            f"!расхождение {rms:.0f} мм между углами — это много. Проверьте по порядку: "
            f"совпадает ли encoder_counts_per_rev с проворотом барабана рукой, верны ли "
            f"стороны рамы в geometry.anchors, и не проскальзывает ли трос"
        )
    elif residuals.size and rms > 15.0:
        warnings.append(
            f"расхождение {rms:.0f} мм между углами. Чаще всего это значит, что коробка "
            f"встаёт не там, где считает homing.corner_inset_mm — померьте линейкой, "
            f"где она на самом деле останавливается"
        )

    return CalibrationResult(
        count_ref=[int(round(v)) for v in count_ref],
        length_at_ref_mm=[float(v) for v in length_ref],
        ea_n=None,   # жёсткость здесь не подбирается и перезаписывать её нельзя
        residual_rms_mm=rms,
        residual_max_mm=worst,
        warnings=warnings,
        points_used=1,
        checked=max(0, len(records) - 1),
    )
