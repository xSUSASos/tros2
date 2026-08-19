"""Разбор и интерпретация G-code.

Поддержан разумный минимум для тросовой системы: перемещения, дуги, паузы,
переключение абсолютных и относительных координат, смена нуля и подача.
Дуги разбиваются на хорды прямо при разборе — тросовой платформе всё равно
негде выполнять дугу «аппаратно», а контроль стрелки прогиба делает
результат предсказуемым.

Ошибки не глотаются: строка, которую не удалось понять, попадает в список
проблем вместе со своим номером и текстом. Программа с ошибками не
запускается — на машине с четырьмя тросами и мотором на киловатт «выполним
что поняли» не годится.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cdpr.trajectory import Move

WORD = re.compile(r"([A-Za-z])\s*([-+]?\d*\.?\d+)")
COMMENT_PAREN = re.compile(r"\([^)]*\)")


@dataclass
class Dwell:
    seconds: float
    line: int


@dataclass
class Pause:
    line: int
    message: str = ""


@dataclass
class ProgramIssue:
    line: int
    text: str
    message: str

    def __str__(self) -> str:
        return f"строка {self.line}: {self.message}  ->  {self.text.strip()!r}"


@dataclass
class Program:
    """Разобранная программа."""

    operations: list[Any] = field(default_factory=list)
    issues: list[ProgramIssue] = field(default_factory=list)
    home_requested: bool = False
    start_pose: np.ndarray | None = None
    source_lines: int = 0

    @property
    def moves(self) -> list[Move]:
        return [op for op in self.operations if isinstance(op, Move)]

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        points = []
        for move in self.moves:
            points.extend([move.start, move.end])
        if not points:
            return None
        stacked = np.vstack(points)
        return stacked.min(axis=0), stacked.max(axis=0)

    @property
    def path_length_mm(self) -> float:
        return sum(m.length_mm for m in self.moves)

    def summary(self) -> str:
        if not self.ok:
            return f"программа с ошибками: {len(self.issues)} проблем(ы)"
        bounds = self.bounds
        extent = "" if bounds is None else (
            f", габарит {np.round(bounds[0]).astype(int).tolist()}"
            f"..{np.round(bounds[1]).astype(int).tolist()} мм"
        )
        return (
            f"строк {self.source_lines}, перемещений {len(self.moves)}, "
            f"путь {self.path_length_mm / 1000:.2f} м{extent}"
        )


class Interpreter:
    """Состояние интерпретатора: где мы, в каких координатах и с какой подачей."""

    def __init__(self, start_pose: np.ndarray, default_feed_mms: float,
                 rapid_feed_mms: float, arc_tolerance_mm: float = 0.2) -> None:
        self.position = np.asarray(start_pose, dtype=float).copy()
        self.feed = float(default_feed_mms)
        self.rapid_feed = float(rapid_feed_mms)
        self.absolute = True
        self.units_scale = 1.0      # 25.4 после G20
        self.offset = np.zeros(3)   # смещение нуля от G92
        self.arc_tolerance = arc_tolerance_mm

    def resolve(self, words: dict[str, float]) -> np.ndarray:
        """Целевая точка по словам X/Y/Z с учётом режима и единиц."""
        target = self.position.copy()
        for i, letter in enumerate("XYZ"):
            if letter in words:
                value = words[letter] * self.units_scale
                target[i] = value + self.offset[i] if self.absolute else self.position[i] + value
        return target


def _strip(line: str) -> str:
    line = COMMENT_PAREN.sub(" ", line)
    for marker in (";", "%"):
        if marker in line:
            line = line.split(marker, 1)[0]
    return line.strip()


def _words(line: str) -> dict[str, float]:
    return {letter.upper(): float(value) for letter, value in WORD.findall(line)}


def _arc_points(start: np.ndarray, end: np.ndarray, centre: np.ndarray,
                clockwise: bool, tolerance: float) -> list[np.ndarray]:
    """Разбивает дугу в плоскости XY на хорды с заданной стрелкой прогиба."""
    v0 = start[:2] - centre[:2]
    v1 = end[:2] - centre[:2]
    radius = float(np.linalg.norm(v0))
    if radius < 1e-6:
        return [end]

    a0 = math.atan2(v0[1], v0[0])
    a1 = math.atan2(v1[1], v1[0])
    sweep = a1 - a0
    if clockwise:
        while sweep >= 0:
            sweep -= 2 * math.pi
    else:
        while sweep <= 0:
            sweep += 2 * math.pi
    if abs(sweep) < 1e-9:
        sweep = -2 * math.pi if clockwise else 2 * math.pi

    # угловой шаг из допустимой стрелки прогиба хорды
    ratio = max(-1.0, min(1.0, 1.0 - tolerance / radius))
    step = 2.0 * math.acos(ratio) if ratio < 1.0 else abs(sweep)
    count = max(2, int(math.ceil(abs(sweep) / max(step, 1e-3))))

    points = []
    for i in range(1, count + 1):
        angle = a0 + sweep * i / count
        fraction = i / count
        points.append(np.array([
            centre[0] + radius * math.cos(angle),
            centre[1] + radius * math.sin(angle),
            start[2] + (end[2] - start[2]) * fraction,
        ]))
    points[-1] = end.copy()
    return points


#: что понимаем; всё остальное отмечается как проблема, а не игнорируется молча
SUPPORTED_G = {0, 1, 2, 3, 4, 20, 21, 28, 90, 91, 92}
SUPPORTED_M = {0, 1, 2, 30, 112}


def parse(
    text: str,
    *,
    start_pose,
    default_feed_mms: float,
    rapid_feed_mms: float | None = None,
    arc_tolerance_mm: float = 0.2,
) -> Program:
    """Разбирает текст программы."""
    interp = Interpreter(
        np.asarray(start_pose, dtype=float),
        default_feed_mms,
        rapid_feed_mms if rapid_feed_mms is not None else default_feed_mms,
        arc_tolerance_mm,
    )
    program = Program(start_pose=interp.position.copy())
    lines = text.splitlines()
    program.source_lines = len(lines)

    for number, raw in enumerate(lines, start=1):
        clean = _strip(raw)
        if not clean:
            continue
        words = _words(clean)
        if not words:
            program.issues.append(ProgramIssue(number, raw, "не разобрать строку"))
            continue

        if "F" in words:
            feed = words["F"] * interp.units_scale / 60.0  # G-code задаёт подачу в минуту
            if feed <= 0:
                program.issues.append(ProgramIssue(number, raw, "подача должна быть положительной"))
            else:
                interp.feed = feed

        if "M" in words:
            code = int(round(words["M"]))
            if code not in SUPPORTED_M:
                program.issues.append(ProgramIssue(number, raw, f"код M{code} не поддержан"))
            elif code in (0, 1):
                program.operations.append(Pause(number, "пауза по M{}".format(code)))
            elif code == 112:
                program.operations.append(Pause(number, "аварийный стоп по M112"))
            if "G" not in words:
                continue

        if "G" not in words:
            if any(letter in words for letter in "XYZ"):
                target = interp.resolve(words)
                _emit_line(program, interp, target, interp.feed, number, raw)
            continue

        code = int(round(words["G"]))
        if code not in SUPPORTED_G:
            program.issues.append(ProgramIssue(number, raw, f"код G{code} не поддержан"))
            continue

        if code == 20:
            interp.units_scale = 25.4
        elif code == 21:
            interp.units_scale = 1.0
        elif code == 90:
            interp.absolute = True
        elif code == 91:
            interp.absolute = False
        elif code == 92:
            for i, letter in enumerate("XYZ"):
                if letter in words:
                    interp.offset[i] = interp.position[i] - words[letter] * interp.units_scale
        elif code == 28:
            program.home_requested = True
        elif code == 4:
            seconds = words.get("P", words.get("S", 0.0))
            if "P" in words and seconds > 1000:
                seconds /= 1000.0  # P в миллисекундах у части диалектов
            program.operations.append(Dwell(max(0.0, seconds), number))
        elif code in (0, 1):
            target = interp.resolve(words)
            feed = interp.rapid_feed if code == 0 else interp.feed
            _emit_line(program, interp, target, feed, number, raw)
        elif code in (2, 3):
            _emit_arc(program, interp, words, code == 2, number, raw)

    return program


def _emit_line(program: Program, interp: Interpreter, target: np.ndarray,
               feed: float, number: int, raw: str) -> None:
    if np.allclose(target, interp.position, atol=1e-9):
        return
    program.operations.append(
        Move(start=interp.position.copy(), end=target.copy(), feed_mms=feed,
             line=number, source=raw.strip())
    )
    interp.position = target


def _emit_arc(program: Program, interp: Interpreter, words: dict[str, float],
              clockwise: bool, number: int, raw: str) -> None:
    target = interp.resolve(words)
    if "R" in words:
        program.issues.append(ProgramIssue(
            number, raw, "дуга задана радиусом R — поддержан только формат с I/J"))
        return
    if "I" not in words and "J" not in words:
        program.issues.append(ProgramIssue(number, raw, "у дуги нет смещения центра I/J"))
        return

    centre = interp.position.copy()
    centre[0] += words.get("I", 0.0) * interp.units_scale
    centre[1] += words.get("J", 0.0) * interp.units_scale

    r_start = float(np.linalg.norm(interp.position[:2] - centre[:2]))
    r_end = float(np.linalg.norm(target[:2] - centre[:2]))
    if abs(r_start - r_end) > max(0.5, 0.01 * r_start):
        program.issues.append(ProgramIssue(
            number, raw,
            f"дуга несогласована: радиус в начале {r_start:.1f} мм, в конце {r_end:.1f} мм"))
        return

    for point in _arc_points(interp.position, target, centre, clockwise, interp.arc_tolerance):
        program.operations.append(
            Move(start=interp.position.copy(), end=point.copy(), feed_mms=interp.feed,
                 line=number, source=raw.strip())
        )
        interp.position = point
