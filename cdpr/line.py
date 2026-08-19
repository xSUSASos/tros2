"""Перевод отсчётов энкодера в длину троса и обратно.

Два эффекта, без которых числа не сходятся.

**Слои намотки.** Леска ложится на барабан слоями, и радиус растёт на её
диаметр за слой. При леске 0.5 мм на барабане ⌀60 это 1.7 % масштаба за
слой, то есть примерно каждые 10 метров троса. Модель «столько-то
миллиметров на импульс» здесь просто неверна, поэтому калибруются два
физических параметра — отсчёт энкодера при пустом барабане и длина троса
в этот момент, — а длина считается послойно и точна на любом слое.

**Упругость.** Плетёная леска 50 кг на пролёте 8 м при 100 Н удлиняется
примерно на 5 см. Это на порядок больше всех прочих источников ошибки, но
оно измеримо: натяжение известно из момента мотора, а жёсткость EA
определяется калибровкой. Компенсация возвращает точность к миллиметрам.
"""
from __future__ import annotations

import math

from cdpr.config import ConfigError, WinchCfg

TWO_PI = 2.0 * math.pi


class LineModel:
    """Геометрия намотки одной лебёдки."""

    def __init__(self, winch: WinchCfg) -> None:
        self.winch = winch
        self.r0 = winch.drum_radius_mm
        self.d = winch.line_diameter_mm
        self.per_layer = winch.turns_per_layer
        self.single_layer = winch.winding == "single_layer"
        self.counts_per_drum_rev = winch.counts_per_drum_rev

    # ------------------------------------------------------------------ #
    #  Радиус и длина по числу витков
    # ------------------------------------------------------------------ #
    def layer_radius(self, layer: int) -> float:
        """Радиус по оси лески в слое `layer` (0 — первый слой)."""
        if self.single_layer:
            layer = 0
        return self.r0 + self.d * (layer + 0.5)

    def layer_of(self, turns: float) -> int:
        if self.single_layer or turns <= 0:
            return 0
        return int(turns // self.per_layer)

    def radius_at(self, turns: float) -> float:
        """Действующий радиус при данном числе намотанных витков."""
        return self.layer_radius(self.layer_of(turns))

    def _full_layers_length(self, layers: int) -> float:
        """Длина троса в `layers` полностью намотанных слоях.

        Сумма радиусов слоёв берётся в замкнутом виде, а не циклом:
        sum(r0 + d*(j+0.5)) по j<L равна L*r0 + d*L^2/2.
        """
        if layers <= 0:
            return 0.0
        if self.single_layer:
            return TWO_PI * self.layer_radius(0) * layers * self.per_layer
        return TWO_PI * self.per_layer * (layers * self.r0 + self.d * layers * layers / 2.0)

    def wound_length(self, turns: float) -> float:
        """Сколько троса намотано на барабан при `turns` витках."""
        if turns <= 0:
            return 0.0
        layer = self.layer_of(turns)
        rest = turns - layer * self.per_layer
        return self._full_layers_length(layer) + TWO_PI * self.layer_radius(layer) * rest

    def turns_for_wound(self, wound: float) -> float:
        """Обратное к wound_length."""
        if wound <= 0:
            return 0.0
        if self.single_layer:
            return wound / (TWO_PI * self.layer_radius(0))
        layer = 0
        while self._full_layers_length(layer + 1) <= wound:
            layer += 1
            if layer > 10_000:
                raise ConfigError("не сходится число слоёв — проверьте геометрию барабана")
        rest = wound - self._full_layers_length(layer)
        return layer * self.per_layer + rest / (TWO_PI * self.layer_radius(layer))

    # ------------------------------------------------------------------ #
    #  Отсчёты энкодера <-> длина
    # ------------------------------------------------------------------ #
    def _require_calibration(self) -> tuple[int, float]:
        w = self.winch
        if w.count_empty is None or w.length_at_empty_mm is None:
            raise ConfigError(
                f"лебёдка {w.anchor} не откалибрована: неизвестны count_empty и "
                f"length_at_empty_mm. Пройдите калибровку — без неё отсчёт энкодера "
                f"нельзя перевести в длину троса."
            )
        return w.count_empty, w.length_at_empty_mm

    def turns_at(self, count: int) -> float:
        """Сколько витков намотано при данном отсчёте энкодера."""
        count_empty, _ = self._require_calibration()
        turns = self.winch.direction * (count - count_empty) / self.counts_per_drum_rev
        return max(0.0, turns)

    def length_from_counts(self, count: int) -> float:
        """Свободная (ненагруженная) длина троса от схода до платформы, мм."""
        _, length_at_empty = self._require_calibration()
        return length_at_empty - self.wound_length(self.turns_at(count))

    def counts_from_length(self, length_mm: float) -> int:
        """Обратное к length_from_counts."""
        count_empty, length_at_empty = self._require_calibration()
        wound = length_at_empty - length_mm
        turns = self.turns_for_wound(max(0.0, wound))
        return int(round(count_empty + self.winch.direction * turns * self.counts_per_drum_rev))

    def counts_per_mm(self, count: int) -> float:
        """Локальный масштаб при данном отсчёте — нужен контуру управления."""
        return self.counts_per_drum_rev / (TWO_PI * self.radius_at(self.turns_at(count)))

    # ------------------------------------------------------------------ #
    #  Скорость
    # ------------------------------------------------------------------ #
    def rpm_for_line_speed(self, speed_mms: float, count: int) -> float:
        """Обороты мотора, дающие заданную скорость выборки троса.

        Знак: положительная скорость означает наматывание, то есть укорочение
        троса. Направление вращения задаётся `direction` в конфиге.
        """
        radius = self.radius_at(self.turns_at(count))
        drum_rpm = speed_mms * 60.0 / (TWO_PI * radius)
        return self.winch.direction * drum_rpm * self.winch.gear_ratio

    def line_speed_for_rpm(self, rpm: float, count: int) -> float:
        radius = self.radius_at(self.turns_at(count))
        drum_rpm = rpm / self.winch.gear_ratio * self.winch.direction
        return drum_rpm * TWO_PI * radius / 60.0

    def max_line_speed(self, count: int) -> float:
        return abs(self.line_speed_for_rpm(self.winch.max_rpm, count))

    # ------------------------------------------------------------------ #
    #  Упругость
    # ------------------------------------------------------------------ #
    def stretched(self, free_length_mm: float, tension_n: float) -> float:
        """Длина под нагрузкой: именно её «видит» геометрия платформы."""
        ea = self.winch.ea_n
        if not ea:
            return free_length_mm
        return free_length_mm * (1.0 + tension_n / ea)

    def unstretched(self, distance_mm: float, tension_n: float) -> float:
        """Сколько троса надо стравить, чтобы под нагрузкой получить `distance_mm`."""
        ea = self.winch.ea_n
        if not ea:
            return distance_mm
        return distance_mm / (1.0 + tension_n / ea)

    def elongation(self, free_length_mm: float, tension_n: float) -> float:
        ea = self.winch.ea_n
        return 0.0 if not ea else free_length_mm * tension_n / ea


def build_line_models(winches: list[WinchCfg]) -> list[LineModel]:
    return [LineModel(w) for w in winches]
