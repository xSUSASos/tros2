"""Границы модулей нижнего уровня.

Три слоя, каждый заменяется независимо:

    Transport   — как байты ходят до устройства (Modbus RTU / симулятор / ...)
    Drive       — что означают регистры конкретной модели привода (из YAML-профиля)
    DriveGroup  — как опросить все оси за один цикл управления

Верхние слои (контур управления, режимы, панель) знают только про DriveGroup
и DriveState, поэтому подмена железа на симулятор — это одна строка в конфиге.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
#  Ошибки
# --------------------------------------------------------------------------- #
class TransportError(RuntimeError):
    """Базовая ошибка обмена."""


class BusTimeout(TransportError):
    """Устройство не ответило за отведённое время."""


class CrcError(TransportError):
    """Ответ пришёл, но контрольная сумма не сошлась."""


class FramingError(TransportError):
    """Ответ структурно неверен: не тот адрес, функция или длина."""


class ModbusException(TransportError):
    """Устройство вернуло код исключения Modbus."""

    #: расшифровка стандартных кодов
    TEXT = {
        1: "недопустимая функция",
        2: "недопустимый адрес регистра",
        3: "недопустимое значение",
        4: "отказ устройства",
        5: "запрос принят, выполняется",
        6: "устройство занято",
        8: "ошибка чётности в памяти",
    }

    def __init__(self, slave: int, function: int, code: int) -> None:
        self.slave, self.function, self.code = slave, function, code
        super().__init__(
            f"привод {slave}: функция {function} отклонена, код {code} "
            f"({self.TEXT.get(code, 'неизвестный код')})"
        )


# --------------------------------------------------------------------------- #
#  Статистика шины
# --------------------------------------------------------------------------- #
@dataclass
class TransportStats:
    """Счётчики обмена. Выводятся в панель — по ним видно здоровье линии."""

    requests: int = 0
    ok: int = 0
    timeouts: int = 0
    crc_errors: int = 0
    framing_errors: int = 0
    exceptions: int = 0
    retries: int = 0
    latency_last_ms: float = 0.0
    latency_max_ms: float = 0.0
    _latency_sum: float = field(default=0.0, repr=False)
    _latency_n: int = field(default=0, repr=False)

    def record_ok(self, latency_ms: float) -> None:
        self.ok += 1
        self.latency_last_ms = latency_ms
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)
        self._latency_sum += latency_ms
        self._latency_n += 1

    @property
    def latency_avg_ms(self) -> float:
        return self._latency_sum / self._latency_n if self._latency_n else 0.0

    @property
    def error_rate(self) -> float:
        return 0.0 if not self.requests else 1.0 - self.ok / self.requests

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "ok": self.ok,
            "timeouts": self.timeouts,
            "crc_errors": self.crc_errors,
            "framing_errors": self.framing_errors,
            "exceptions": self.exceptions,
            "retries": self.retries,
            "error_rate": round(self.error_rate, 5),
            "latency_last_ms": round(self.latency_last_ms, 2),
            "latency_avg_ms": round(self.latency_avg_ms, 2),
            "latency_max_ms": round(self.latency_max_ms, 2),
        }


# --------------------------------------------------------------------------- #
#  Транспорт
# --------------------------------------------------------------------------- #
class Transport(ABC):
    """Канал к устройствам Modbus: чтение и запись регистров.

    Реализация обязана быть потокобезопасной на уровне транзакции: цикл
    управления и веб-панель ходят на одну шину из разных потоков.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.stats = TransportStats()

    # --- жизненный цикл ---
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    def __enter__(self) -> "Transport":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- обмен ---
    @abstractmethod
    def read_registers(self, slave: int, address: int, count: int, *, function: int = 3) -> list[int]:
        """Читает `count` 16-битных регистров. Возвращает сырые значения без знака."""

    @abstractmethod
    def write_register(self, slave: int, address: int, value: int) -> None:
        """Записывает один регистр (функция 6)."""

    @abstractmethod
    def write_registers(self, slave: int, address: int, values: Sequence[int]) -> None:
        """Записывает несколько подряд идущих регистров (функция 16)."""

    def ping(self, slave: int, address: int = 0, *, function: int = 3) -> bool:
        """Отвечает ли устройство. Исключение Modbus считается ответом:
        устройство есть, просто регистр не тот."""
        try:
            self.read_registers(slave, address, 1, function=function)
            return True
        except ModbusException:
            return True
        except TransportError:
            return False

    def describe(self) -> str:
        return f"{type(self).__name__}({self.name})"


# --------------------------------------------------------------------------- #
#  Состояние оси
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DriveState:
    """Мгновенный снимок одной оси. Всё, что читается за цикл управления."""

    axis: int
    online: bool = False
    position_counts: int = 0
    speed_rpm: float = 0.0
    torque_percent: float = 0.0
    alarm: int = 0
    enabled: bool = False
    stamp: float = field(default_factory=time.perf_counter)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.online and self.alarm == 0 and self.error is None

    def age_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.perf_counter()) - self.stamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "online": self.online,
            "position_counts": self.position_counts,
            "speed_rpm": round(self.speed_rpm, 2),
            "torque_percent": round(self.torque_percent, 2),
            "alarm": self.alarm,
            "enabled": self.enabled,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
#  Дизеринг дробной уставки скорости
# --------------------------------------------------------------------------- #
class SigmaDeltaQuantizer:
    """Превращает дробную уставку в целые значения, сохраняя среднее.

    Уставка скорости привода задаётся целыми об/мин, а один об/мин при
    барабане D60 — это ~1.6 мм/с троса. Для джога и hand-guide на единицах
    мм/с такой квант неприемлем.

    Классическая обратная связь по ошибке квантования: остаток переносится
    в следующий цикл, поэтому среднее за несколько циклов равно заданному.
    Механика на 50 Гц сама сглаживает переключения.

        q = SigmaDeltaQuantizer()
        [q(1.3) for _ in range(10)]  ->  1, 2, 1, 1, 2, 1, 1, 2, 1, 1  (среднее 1.3)
    """

    __slots__ = ("_err", "step")

    def __init__(self, step: float = 1.0) -> None:
        if step <= 0:
            raise ValueError("шаг квантования должен быть положительным")
        self.step = step
        self._err = 0.0

    def __call__(self, value: float) -> int:
        u = value + self._err
        out = round(u / self.step) * self.step
        self._err = u - out
        return int(out)

    def reset(self) -> None:
        """Сбрасывает накопленный остаток — при останове и смене режима."""
        self._err = 0.0

    @property
    def residual(self) -> float:
        return self._err


# --------------------------------------------------------------------------- #
#  Группа осей
# --------------------------------------------------------------------------- #
class DriveGroup(ABC):
    """Все оси машины как единое целое.

    Говорит в единицах привода: об/мин и импульсы энкодера. Перевод в
    миллиметры троса — забота слоя cdpr (line.py + kinematics.py), потому что
    он зависит от геометрии барабана, а не от протокола.

    Верхний слой обязан считать, что любой вызов может частично не пройти:
    оси, с которыми не удалось поговорить, возвращаются с online=False, а не
    роняют весь цикл.
    """

    @property
    @abstractmethod
    def n_axes(self) -> int: ...

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def initialize(self) -> None:
        """Приводит параметры привода к рабочему режиму (init_sequence профиля)."""

    @abstractmethod
    def enable(self, on: bool) -> None: ...

    @abstractmethod
    def read_states(self) -> list[DriveState]:
        """Один опрос всех осей. Вызывается каждый цикл управления."""

    @abstractmethod
    def set_speeds(self, rpm: Sequence[float]) -> None:
        """Уставки скорости в об/мин; дробные значения допустимы (дизеринг внутри)."""

    @abstractmethod
    def read_param(self, axis: int, name: str) -> Any: ...

    @abstractmethod
    def write_param(self, axis: int, name: str, value: Any) -> None: ...

    @abstractmethod
    def reset_alarms(self) -> None: ...

    def stop(self) -> None:
        """Немедленный останов: нулевая скорость на все оси."""
        self.set_speeds([0.0] * self.n_axes)

    def stats(self) -> dict[str, Any]:
        return {}

    def __enter__(self) -> "DriveGroup":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.stop()
        finally:
            self.close()
