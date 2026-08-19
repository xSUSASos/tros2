"""Выполнение программы G-code."""
from __future__ import annotations

import threading

import numpy as np

from cdpr.gcode import Dwell, Pause, Program
from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName
from cdpr.trajectory import Move, TrajectoryPlanner


class GcodeMode(Mode):
    """Проигрывает разобранную программу.

    Траектория считается заранее целиком: так известны и время выполнения, и
    габарит пути, и можно заранее сказать, выходит ли программа за рабочую
    зону — до того, как платформа поедет. Проверять это на ходу поздно.
    """

    name = ModeName.GCODE

    def __init__(self, program: Program, planner: TrajectoryPlanner,
                 *, feed_override: float = 1.0) -> None:
        if not program.ok:
            raise ValueError(
                "программа с ошибками не запускается: "
                + "; ".join(str(i) for i in program.issues[:3])
            )
        self.program = program
        self.planner = planner
        self.feed_override = feed_override
        self._lock = threading.RLock()
        self._clock = 0.0
        self._paused = False
        self._stopped = False
        self._pause_message = ""
        self._dwell_left = 0.0
        self._segment_index = 0
        self._pending: list = list(program.operations)
        self._planned = False

    # ------------------------------------------------------------------ #
    def enter(self, ctx) -> None:  # noqa: ANN001
        moves = [op for op in self.program.operations if isinstance(op, Move)]
        self.planner.plan(moves)
        self._planned = True
        self._clock = 0.0

    # ------------------------------------------------------------------ #
    def pause(self, message: str = "пауза") -> None:
        with self._lock:
            self._paused = True
            self._pause_message = message

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._pause_message = ""

    def stop(self) -> None:
        with self._lock:
            self._stopped = True

    def set_feed_override(self, value: float) -> None:
        with self._lock:
            self.feed_override = float(np.clip(value, 0.05, 2.0))

    @property
    def progress(self) -> float:
        total = self.planner.total_time_s
        return 0.0 if total <= 0 else min(1.0, self._clock / total)

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        with self._lock:
            if self._stopped:
                return ModeOutput(hold=True, done=True, message="программа остановлена")
            if self._dwell_left > 0.0:
                self._dwell_left -= dt
                return ModeOutput(hold=True, message=f"выдержка {self._dwell_left:.1f} с")
            if self._paused:
                return ModeOutput(hold=True, message=self._pause_message)
            self._clock += dt * self.feed_override

        position, speed, index = self.planner.sample(self._clock)
        self._segment_index = index

        if self._clock >= self.planner.total_time_s:
            return ModeOutput(target_pose=position, feed_mms=speed, done=True,
                              message="программа выполнена")

        move = self.planner.moves[index] if 0 <= index < len(self.planner.moves) else None
        line = move.line if move else None
        return ModeOutput(
            target_pose=position,
            feed_mms=max(speed, 1.0),
            message=f"строка {line}, {self.progress * 100:.0f}%",
        )

    def describe(self) -> str:
        return f"G-code: {self.planner.summary()}"


def check_program_fits(program: Program, machine, kinematics) -> list[str]:
    """Проверяет программу на попадание в рабочую зону ДО пуска.

    Возвращает список проблем. Пустой список означает, что весь путь лежит
    внутри области, где платформа управляема.
    """
    from cdpr.workspace import check_pose

    problems: list[str] = []
    seen: set[tuple[int, int, int]] = set()
    for move in program.moves:
        for point in (move.start, move.end):
            key = (int(point[0] // 50), int(point[1] // 50), int(point[2] // 50))
            if key in seen:
                continue
            seen.add(key)
            ok, margin, why = check_pose(machine, kinematics, point, directions=8)
            if not ok:
                problems.append(
                    f"строка {move.line}: точка {np.round(point).astype(int).tolist()} — {why}"
                )
                if len(problems) >= 10:
                    problems.append("...дальше не проверяю, ошибок уже достаточно")
                    return problems
    return problems
