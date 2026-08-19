"""Разрешение приводов (SON).

У T3D разрешение приходит физическим входом DI, а не по Modbus, и это к
лучшему: цепь SON всех приводов, заведённая через одно реле, работает как
аппаратный аварийный стоп, который не зависит от того, жив ли софт.

Здесь описан только способ этим реле управлять. Вариантов три, выбирается
в конфиге: щёлкает человек, щёлкает Raspberry Pi ножкой GPIO, либо это
симулятор.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class EnableBackend(ABC):
    """Управление цепью SON."""

    def __init__(self) -> None:
        self._state = False

    @property
    def state(self) -> bool:
        return self._state

    @abstractmethod
    def _apply(self, on: bool) -> None: ...

    def set(self, on: bool) -> None:
        self._apply(on)
        self._state = on

    @abstractmethod
    def describe(self) -> str: ...

    @property
    def is_automatic(self) -> bool:
        """Может ли софт снять разрешение сам. Если нет, аварийный стоп
        сводится к обнулению скорости, и это надо честно показывать в панели."""
        return True


class ManualEnable(EnableBackend):
    """Реле щёлкает человек. Софт только просит и ведёт учёт."""

    def _apply(self, on: bool) -> None:
        log.warning(
            "ВНИМАНИЕ: разрешение приводов управляется вручную — %s тумблер SON",
            "включите" if on else "выключите",
        )

    def describe(self) -> str:
        return "ручной тумблер SON (софт не может снять разрешение сам)"

    @property
    def is_automatic(self) -> bool:
        return False


class SimEnable(EnableBackend):
    """Разрешение в симуляторе."""

    def __init__(self, transports, slaves_by_transport) -> None:
        super().__init__()
        self._transports = transports
        self._slaves = slaves_by_transport

    def _apply(self, on: bool) -> None:
        for name, transport in self._transports.items():
            for slave in self._slaves.get(name, []):
                setter = getattr(transport, "set_enabled", None)
                if setter:
                    setter(slave, on)

    def describe(self) -> str:
        return "симулятор"


class GpioEnable(EnableBackend):
    """Ножка GPIO Raspberry Pi через реле в цепи SON."""

    def __init__(self, pin: int, active_high: bool = True) -> None:
        super().__init__()
        self.pin, self.active_high = pin, active_high
        self._line = None
        try:
            from gpiozero import DigitalOutputDevice

            self._line = DigitalOutputDevice(pin, active_high=active_high, initial_value=False)
        except Exception as exc:  # noqa: BLE001 — на ноутбуке GPIO нет, это нормально
            log.warning("GPIO %d недоступен (%s) — разрешение приводов работать не будет", pin, exc)

    def _apply(self, on: bool) -> None:
        if self._line is None:
            raise RuntimeError(
                f"GPIO {self.pin} недоступен: разрешить приводы отсюда нельзя. "
                f"Либо запускайте на Raspberry Pi, либо переключите "
                f"safety.enable_backend на manual."
            )
        self._line.value = bool(on)

    def describe(self) -> str:
        return f"реле на GPIO {self.pin}"


def make_enable_backend(kind: str, *, pin: int | None = None, transports=None,
                        slaves_by_transport=None) -> EnableBackend:
    if kind == "manual":
        return ManualEnable()
    if kind == "sim":
        return SimEnable(transports or {}, slaves_by_transport or {})
    if kind == "gpio":
        if pin is None:
            raise ValueError("для safety.enable_backend=gpio нужен safety.enable_gpio_pin")
        return GpioEnable(pin)
    raise ValueError(f"неизвестный способ разрешения приводов {kind!r}: manual, gpio или sim")
