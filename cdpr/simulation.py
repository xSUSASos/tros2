"""Физика платформы для симулятора.

Симулятор приводов сам по себе крутит моторы, но не знает, что тросы связаны
одной платформой. Эта модель замыкает круг: по намотанным длинам считается,
где платформа висит, и какие при этом натяжения — а натяжения возвращаются в
приводы как момент. После этого весь верхний уровень можно гонять целиком, не
выходя к железу.

Модель тросов упругая, и это важно. С нерастяжимыми тросами длины пришлось бы
согласовывать точно, иначе задача не имеет решения; в реальности же лишняя
длина даёт провисание, а нехватка — рост натяжения. Именно так ведёт себя
настоящая система, и именно на этом работает управление натяжением.

Положение платформы ищется минимизацией энергии: упругая энергия тросов плюс
потенциальная энергия веса. В минимуме сумма натяжений уравновешивает вес, то
есть выполняется то самое W f = -w, на котором стоит весь расчёт.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from cdpr.config import MachineConfig
from cdpr.kinematics import CDPRKinematics
from cdpr.line import build_line_models
from cdpr.tension import GRAVITY

log = logging.getLogger(__name__)

#: жёсткость троса по умолчанию, Н.
#  Капрон ⌀0.3 мм: E около 3 ГПа, сечение 0.07 мм^2 -> EA примерно 200 Н.
#  Число намеренно маленькое: этот трос очень мягкий, и модель должна это
#  показывать, иначе на симуляторе всё выглядит точнее, чем будет на деле.
DEFAULT_EA_N = 200.0


class PlatformSimulator:
    """Платформа на упругих тросах."""

    def __init__(self, machine: MachineConfig, *, ea_n: float | None = None,
                 friction_percent: float = 1.5, floor_z_mm: float | None = None,
                 stop_distance_mm: float | None = None) -> None:
        self.machine = machine
        self.kinematics = CDPRKinematics.from_config(machine)
        self.winches = machine.ordered_winches()
        self.lines = build_line_models(self.winches)
        self.ea_n = ea_n if ea_n is not None else (self.winches[0].ea_n or DEFAULT_EA_N)
        self.friction_percent = friction_percent
        self.weight_n = machine.platform.mass_kg * GRAVITY
        # Пол: ниже него платформа не опустится.
        self.floor_z_mm = 0.0 if floor_z_mm is None else floor_z_mm

        # Упор в модуле. Ближе этого расстояния к точке схода коробка подойти
        # не может — упирается в корпус. Без упора хоминг не на чем проверить:
        # трос наматывался бы вечно, а натяжение так и не выросло бы, потому
        # что почти вертикальный трос держит только вес.
        if stop_distance_mm is not None:
            self.stop_distance_mm = float(stop_distance_mm)
        else:
            inset = machine.homing.corner_inset_mm
            sag = machine.geometry.sag_mm if machine.geometry.is_planar else 0.0
            self.stop_distance_mm = float(np.hypot(inset, sag))
        # Жёсткость упора. Корпус модуля жёстче троса, но не бесконечно:
        # коробка, кронштейн и сам модуль немного пружинят. Слишком жёсткая
        # модель даёт скачок натяжения на десятки ньютонов за один цикл, чего
        # на железе не бывает, и хоминг выглядел бы хуже, чем он есть.
        self.stop_stiffness_n_per_mm = 5.0

        self.pose = self.kinematics.anchors.mean(axis=0) - np.array([0.0, 0.0, 1500.0])
        self.tensions = np.zeros(len(self.winches))
        self._transport = None
        self._counts: dict[int, float] = {}
        self._cache_key: tuple | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    def attach(self, transport, slaves: list[int]) -> None:
        """Подключает модель к симулятору приводов."""
        self._transport = transport
        self._slaves = list(slaves)
        transport.load_model = self._torque_for

    # ------------------------------------------------------------------ #
    def free_lengths(self, counts: np.ndarray) -> np.ndarray:
        """Ненагруженные длины тросов по отсчётам энкодеров."""
        return np.array([
            line.length_from_counts(int(c)) for line, c in zip(self.lines, counts, strict=True)
        ])

    def solve(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Положение платформы и натяжения при данных длинах."""
        free = np.maximum(self.free_lengths(counts), 1.0)
        stiffness = self.ea_n / free  # Н на мм

        # Единицы всюду ньютоны и миллиметры: жёсткость Н/мм, растяжение мм,
        # энергия Н*мм. Смешать здесь Н*м с Н*мм — значит ослабить вес в тысячу
        # раз, и платформа повиснет почти без натяжения.
        stop = self.stop_distance_mm

        def energy(p: np.ndarray) -> float:
            distance = np.linalg.norm(self.kinematics.anchors - p, axis=1)
            stretch = np.maximum(0.0, distance - free)
            squeeze = np.maximum(0.0, stop - distance)   # упёрлись в корпус модуля
            return float(
                0.5 * np.sum(stiffness * stretch ** 2)
                + 0.5 * self.stop_stiffness_n_per_mm * np.sum(squeeze ** 2)
                + self.weight_n * p[2]
            )

        def gradient(p: np.ndarray) -> np.ndarray:
            vectors = self.kinematics.anchors - p
            distance = np.linalg.norm(vectors, axis=1)
            unit = vectors / np.maximum(distance, 1e-9)[:, None]
            forces = stiffness * np.maximum(0.0, distance - free)
            forces = forces - self.stop_stiffness_n_per_mm * np.maximum(0.0, stop - distance)
            grad = -(unit * forces[:, None]).sum(axis=0)
            grad[2] += self.weight_n
            return grad

        start = self.pose.copy()
        start[2] = max(start[2], self.floor_z_mm)
        result = minimize(energy, start, jac=gradient, method="L-BFGS-B",
                          bounds=[(None, None), (None, None), (self.floor_z_mm, None)],
                          options={"maxiter": 200, "ftol": 1e-12})
        self.pose = result.x

        vectors = self.kinematics.anchors - self.pose
        distance = np.linalg.norm(vectors, axis=1)
        self.tensions = stiffness * np.maximum(0.0, distance - free)
        return self.pose, self.tensions

    # ------------------------------------------------------------------ #
    def _torque_for(self, slave: int, position_counts: float, speed_rpm: float) -> float:
        """Момент, который симулятор приводов покажет как d-trq."""
        self._counts[slave] = position_counts
        if len(self._counts) < len(self._slaves):
            return 0.0
        counts = np.array([self._counts.get(s, 0.0) for s in self._slaves])

        # Равновесие решается один раз на согласованное состояние, а не по
        # разу на каждую ось: иначе одна и та же поза считалась бы четырежды.
        key = tuple(np.round(counts, 3))
        if key == self._cache_key and self._cache is not None:
            tensions = self._cache[1]
        else:
            try:
                _, tensions = self.solve(counts)
            except Exception as exc:  # noqa: BLE001 — до калибровки длины неизвестны
                log.debug("модель платформы: %s", exc)
                return 0.0
            self._cache_key, self._cache = key, (self.pose.copy(), tensions.copy())
        index = self._slaves.index(slave)
        winch, line = self.winches[index], self.lines[index]
        radius = line.radius_at(line.turns_at(int(counts[index])))
        percent = winch.force_to_torque_percent(float(tensions[index]), radius)
        friction = self.friction_percent * np.sign(speed_rpm)
        return float(percent + friction)

    # ------------------------------------------------------------------ #
    def place_at(self, pose, counts_reference: int = 0) -> np.ndarray:
        """Отсчёты энкодеров, при которых платформа висит в заданной точке
        с рабочим преднатягом. Нужно, чтобы стартовать модель не наугад."""
        pose = np.asarray(pose, dtype=float)
        distance = self.kinematics.inverse(pose)
        preload = self.machine.tension.target_n
        free = distance / (1.0 + preload / self.ea_n)
        counts = np.array([
            line.counts_from_length(float(length))
            for line, length in zip(self.lines, free, strict=True)
        ])
        self.pose = pose.copy()
        return counts
