"""Планировщик траекторий.

Отличие от обычного станка в одном месте, но принципиальное: подача
ограничена не «осями», а ТРОСАМИ. Скорость троса зависит от направления
движения платформы, и в углу рабочей зоны та же подача требует от одного
барабана вдвое больших оборотов, чем в центре. Поэтому предел подачи
считается для каждого отрезка отдельно, из геометрии.

Профиль скорости трапецеидальный, со сглаживанием углов: скорость на стыке
отрезков берётся тем меньше, чем круче поворот. Это убирает рывки, из-за
которых на тросах возникают колебания — а тросовая система, в отличие от
жёсткой, гасит их плохо.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cdpr.config import MachineConfig
from cdpr.kinematics import CDPRKinematics


@dataclass
class Move:
    """Один прямолинейный отрезок пути."""

    start: np.ndarray
    end: np.ndarray
    feed_mms: float
    line: int | None = None          # строка исходной программы
    source: str = ""

    @property
    def delta(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.delta))

    @property
    def direction(self) -> np.ndarray:
        length = self.length_mm
        return self.delta / length if length > 1e-9 else np.zeros(3)


@dataclass
class PlannedMove(Move):
    """Отрезок с рассчитанным профилем скорости."""

    entry_mms: float = 0.0
    exit_mms: float = 0.0
    peak_mms: float = 0.0
    duration_s: float = 0.0
    t_start: float = 0.0


def max_feed_for_direction(
    kinematics: CDPRKinematics,
    pose: np.ndarray,
    direction: np.ndarray,
    max_line_speeds: np.ndarray,
) -> float:
    """Предельная подача в данном направлении из данной точки.

    Скорость выборки троса равна проекции скорости платформы на его
    направление, поэтому ограничение — минимальное по тросам отношение
    предельной скорости троса к этой проекции.
    """
    rates = np.abs(kinematics.winding_rates(pose, direction))
    active = rates > 1e-9
    if not np.any(active):
        return float("inf")
    return float(np.min(max_line_speeds[active] / rates[active]))


class TrajectoryPlanner:
    """Превращает список отрезков в положение платформы как функцию времени."""

    def __init__(self, machine: MachineConfig, kinematics: CDPRKinematics) -> None:
        self.machine = machine
        self.kinematics = kinematics
        self.accel = machine.motion.max_acceleration_mms2
        self.max_feed = machine.motion.max_velocity_mms
        winches = machine.ordered_winches()
        self.max_line_speeds = np.array([w.max_line_speed_mms for w in winches])
        self.moves: list[PlannedMove] = []
        self.total_time_s = 0.0

    # ------------------------------------------------------------------ #
    def plan(self, moves: list[Move]) -> list[PlannedMove]:
        planned = [
            PlannedMove(
                start=m.start, end=m.end, feed_mms=self._limit_feed(m),
                line=m.line, source=m.source,
            )
            for m in moves
            if m.length_mm > 1e-6
        ]
        if not planned:
            self.moves, self.total_time_s = [], 0.0
            return []

        self._plan_junctions(planned)
        self._plan_profiles(planned)

        clock = 0.0
        for move in planned:
            move.t_start = clock
            clock += move.duration_s
        self.moves, self.total_time_s = planned, clock
        return planned

    def _limit_feed(self, move: Move) -> float:
        """Ограничивает заданную подачу тем, что физически вытянут тросы."""
        feed = min(move.feed_mms, self.max_feed)
        for point in (move.start, 0.5 * (move.start + move.end), move.end):
            try:
                limit = max_feed_for_direction(
                    self.kinematics, point, move.direction, self.max_line_speeds
                )
            except Exception:  # noqa: BLE001 — вырожденная поза, ограничим общим пределом
                continue
            feed = min(feed, limit)
        return max(feed, 1.0)

    def _plan_junctions(self, moves: list[PlannedMove]) -> None:
        """Скорость на стыке: чем круче поворот, тем медленнее его проходим."""
        for i, move in enumerate(moves):
            if i == 0:
                move.entry_mms = 0.0
                continue
            previous = moves[i - 1]
            # Классическая оценка через допустимое отклонение от угла: чем
            # круче поворот, тем меньше скорость, на развороте — ноль, а на
            # прямой ограничения нет вовсе.
            cos_theta = -float(np.clip(previous.direction @ move.direction, -1.0, 1.0))
            sin_half = math.sqrt(max(0.0, (1.0 - cos_theta) / 2.0))
            if sin_half >= 1.0 - 1e-9:
                junction = min(previous.feed_mms, move.feed_mms)
            elif sin_half <= 1e-9:
                junction = 0.0
            else:
                deviation = self.machine.motion.junction_deviation_mm
                junction = math.sqrt(self.accel * deviation * sin_half / (1.0 - sin_half))
            junction = min(junction, previous.feed_mms, move.feed_mms)
            previous.exit_mms = junction
            move.entry_mms = junction
        moves[-1].exit_mms = 0.0

    def _plan_profiles(self, moves: list[PlannedMove]) -> None:
        """Трапеция скорости с проходами вперёд и назад — чтобы торможение
        всегда успевало, даже на коротких отрезках."""
        for i in range(len(moves) - 1, 0, -1):
            move = moves[i]
            reachable = math.sqrt(move.exit_mms ** 2 + 2 * self.accel * move.length_mm)
            if move.entry_mms > reachable:
                move.entry_mms = reachable
                moves[i - 1].exit_mms = reachable

        for move in moves:
            reachable = math.sqrt(move.entry_mms ** 2 + 2 * self.accel * move.length_mm)
            move.exit_mms = min(move.exit_mms, reachable)
            move.peak_mms = min(
                move.feed_mms,
                math.sqrt(
                    max(0.0, (2 * self.accel * move.length_mm + move.entry_mms ** 2
                              + move.exit_mms ** 2) / 2.0)
                ),
            )
            move.duration_s = self._duration(move)

    def _duration(self, move: PlannedMove) -> float:
        a = self.accel
        accel_len = max(0.0, (move.peak_mms ** 2 - move.entry_mms ** 2) / (2 * a))
        decel_len = max(0.0, (move.peak_mms ** 2 - move.exit_mms ** 2) / (2 * a))
        cruise_len = max(0.0, move.length_mm - accel_len - decel_len)
        t_a = (move.peak_mms - move.entry_mms) / a if a > 0 else 0.0
        t_d = (move.peak_mms - move.exit_mms) / a if a > 0 else 0.0
        t_c = cruise_len / move.peak_mms if move.peak_mms > 1e-9 else 0.0
        return t_a + t_c + t_d

    # ------------------------------------------------------------------ #
    def sample(self, t: float) -> tuple[np.ndarray, float, int]:
        """Положение, подача и номер отрезка в момент времени t."""
        if not self.moves:
            return np.zeros(3), 0.0, -1
        if t <= 0.0:
            return self.moves[0].start.copy(), 0.0, 0
        if t >= self.total_time_s:
            return self.moves[-1].end.copy(), 0.0, len(self.moves) - 1

        index = 0
        for i, move in enumerate(self.moves):
            if t < move.t_start + move.duration_s:
                index = i
                break
        else:
            index = len(self.moves) - 1

        move = self.moves[index]
        local = t - move.t_start
        distance, speed = self._advance(move, local)
        position = move.start + move.direction * min(distance, move.length_mm)
        return position, speed, index

    def _advance(self, move: PlannedMove, t: float) -> tuple[float, float]:
        """Пройденный путь и текущая скорость внутри отрезка."""
        a = self.accel
        t_a = (move.peak_mms - move.entry_mms) / a if a > 0 else 0.0
        d_a = move.entry_mms * t_a + 0.5 * a * t_a ** 2
        t_d = (move.peak_mms - move.exit_mms) / a if a > 0 else 0.0
        d_d = move.peak_mms * t_d - 0.5 * a * t_d ** 2
        d_c = max(0.0, move.length_mm - d_a - d_d)
        t_c = d_c / move.peak_mms if move.peak_mms > 1e-9 else 0.0

        if t < t_a:
            return move.entry_mms * t + 0.5 * a * t ** 2, move.entry_mms + a * t
        if t < t_a + t_c:
            return d_a + move.peak_mms * (t - t_a), move.peak_mms
        td = min(t - t_a - t_c, t_d)
        return d_a + d_c + move.peak_mms * td - 0.5 * a * td ** 2, move.peak_mms - a * td

    @property
    def path_length_mm(self) -> float:
        return sum(m.length_mm for m in self.moves)

    def summary(self) -> str:
        if not self.moves:
            return "траектория пуста"
        feeds = [m.feed_mms for m in self.moves]
        return (
            f"отрезков {len(self.moves)}, путь {self.path_length_mm / 1000:.2f} м, "
            f"время {self.total_time_s:.1f} с, подача {min(feeds):.0f}..{max(feeds):.0f} мм/с"
        )
