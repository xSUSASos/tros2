"""Модели конфигурации машины и профиля привода.

Единицы во всём проекте: длина — мм, сила — Н, масса — кг, скорость — мм/с,
угловая скорость — об/мин, момент — % от номинального (как отдаёт привод).

Конфиг читается из YAML, валидируется pydantic и сохраняется обратно через
ruamel.yaml с сохранением комментариев — панель правит значения, а пояснения
в файле остаются на месте.
"""
from __future__ import annotations

import io
import math
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)


class ConfigError(RuntimeError):
    """Ошибка в конфигурации, которую должен исправить человек."""


# --------------------------------------------------------------------------- #
#  Геометрия и платформа
# --------------------------------------------------------------------------- #
class AnchorCfg(BaseModel):
    """Точка схода троса — ролик/глазок, откуда трос уходит к платформе.

    Это НЕ корпус лебёдки. Точность этих координат определяет точность всей
    системы: ошибка позиционирования выходит примерно вдвое больше ошибки
    измерения якорей, и калибровка её не исправит.
    """

    model_config = _STRICT
    id: str
    pos: Vec3
    bus: str
    slave: int = Field(ge=1, le=32)


class GeometryCfg(BaseModel):
    model_config = _STRICT
    dof: Literal[2, 3] = 3
    anchors: list[AnchorCfg] = Field(min_length=3)


class PlatformCfg(BaseModel):
    model_config = _STRICT
    mass_kg: float = Field(gt=0)
    attachments: list[Vec3]
    landing_height_mm: float = 0.0

    @property
    def weight_n(self) -> float:
        return self.mass_kg * 9.80665


# --------------------------------------------------------------------------- #
#  Лебёдка
# --------------------------------------------------------------------------- #
class WinchCfg(BaseModel):
    """Параметры одной лебёдки.

    Про намотку. При многослойной укладке эффективный радиус барабана растёт
    на диаметр лески за слой, поэтому «мм на импульс» — не константа: при
    леске 0.5 мм на барабане D60 каждый слой добавляет 1.7 % к масштабу.
    Поэтому калибруются не k и c, а два физических параметра — отсчёт
    энкодера при пустом барабане и длина троса в этот момент, — из которых
    длина считается точно на любом слое (см. cdpr/line.py).
    """

    model_config = _STRICT

    anchor: str
    drum_diameter_mm: float = Field(gt=0)
    drum_width_mm: float = Field(gt=0)
    line_diameter_mm: float = Field(gt=0)
    winding: Literal["single_layer", "multi_layer"] = "multi_layer"
    gear_ratio: float = Field(default=1.0, gt=0, description="оборотов мотора на оборот барабана")
    encoder_counts_per_rev: int = Field(gt=0, description="импульсов на оборот МОТОРА")
    max_rpm: float = Field(gt=0)
    rated_torque_nm: float = Field(gt=0)
    direction: Literal[-1, 1] = 1

    # --- заполняется калибровкой ---
    count_empty: int | None = None
    length_at_empty_mm: float | None = None
    ea_n: float | None = None

    @property
    def drum_radius_mm(self) -> float:
        return self.drum_diameter_mm / 2.0

    @property
    def counts_per_drum_rev(self) -> float:
        return self.encoder_counts_per_rev * self.gear_ratio

    @property
    def turns_per_layer(self) -> int:
        """Сколько витков помещается в один слой по ширине барабана."""
        return max(1, int(self.drum_width_mm // self.line_diameter_mm))

    @property
    def first_layer_radius_mm(self) -> float:
        """Радиус по оси лески в первом слое."""
        return self.drum_radius_mm + self.line_diameter_mm / 2.0

    @property
    def nominal_mm_per_count(self) -> float:
        """Масштаб на первом слое — ориентир для лимитов и начального приближения."""
        return 2.0 * math.pi * self.first_layer_radius_mm / self.counts_per_drum_rev

    @property
    def max_line_speed_mms(self) -> float:
        """Предельная скорость выборки троса на первом слое."""
        return 2.0 * math.pi * self.first_layer_radius_mm * self.max_rpm / (60.0 * self.gear_ratio)

    @property
    def is_calibrated(self) -> bool:
        return self.count_empty is not None and self.length_at_empty_mm is not None

    def force_to_torque_percent(self, force_n: float, radius_mm: float | None = None) -> float:
        """Натяжение троса (Н) -> момент мотора (% от номинала)."""
        r_m = (radius_mm if radius_mm is not None else self.first_layer_radius_mm) / 1000.0
        return 100.0 * force_n * r_m / (self.rated_torque_nm * self.gear_ratio)

    def torque_percent_to_force(self, percent: float, radius_mm: float | None = None) -> float:
        """Момент мотора (%) -> натяжение троса (Н). Обратное к force_to_torque_percent."""
        r_m = (radius_mm if radius_mm is not None else self.first_layer_radius_mm) / 1000.0
        if r_m <= 0:
            raise ConfigError("нулевой радиус барабана")
        return percent * self.rated_torque_nm * self.gear_ratio / (100.0 * r_m)


# --------------------------------------------------------------------------- #
#  Режимы, лимиты, шина
# --------------------------------------------------------------------------- #
class TensionCfg(BaseModel):
    model_config = _STRICT
    min_n: float = Field(gt=0, description="ниже — трос провисает и выпадает из модели")
    target_n: float = Field(gt=0, description="рабочий преднатяг")
    max_n: float = Field(gt=0, description="предел по самому слабому звену, не по разрывной лески")

    @model_validator(mode="after")
    def _ordered(self) -> "TensionCfg":
        if not (self.min_n <= self.target_n <= self.max_n):
            raise ValueError(f"нужно min_n <= target_n <= max_n, получено {self.min_n}/{self.target_n}/{self.max_n}")
        return self


class WorkspaceCfg(BaseModel):
    model_config = _STRICT
    z_min_mm: float
    z_max_mm: float
    inset_mm: float = Field(ge=0, description="доп. отступ поверх расчётной границы")
    feasibility_margin_n: float = Field(ge=0, description="требуемый запас по горизонтальному возмущению")

    @model_validator(mode="after")
    def _ordered(self) -> "WorkspaceCfg":
        if self.z_min_mm >= self.z_max_mm:
            raise ValueError("z_min_mm должен быть меньше z_max_mm")
        return self


class MotionCfg(BaseModel):
    model_config = _STRICT
    max_velocity_mms: float = Field(gt=0)
    max_acceleration_mms2: float = Field(gt=0)
    jog_feed_mms: float = Field(gt=0)
    homing_feed_mms: float = Field(gt=0)
    # На сколько платформе разрешено срезать угол на стыке отрезков.
    # Больше — быстрее и мягче, но траектория скругляется сильнее.
    junction_deviation_mm: float = Field(default=1.0, gt=0)


class ControlCfg(BaseModel):
    model_config = _STRICT
    loop_hz: float = Field(gt=0, le=1000)
    position_kp: float = Field(ge=0, description="1/с: скорость троса = kp * ошибка длины")
    tension_kp: float = Field(ge=0, description="мм/с на Н")
    # Ниже этой ошибки натяжение не правится: иначе приводы будут бесконечно
    # дёргаться вокруг цели, изнашивая механику без всякой пользы.
    tension_deadband_n: float = Field(default=1.5, ge=0)
    watchdog_ms: float = Field(gt=0)

    @property
    def dt(self) -> float:
        return 1.0 / self.loop_hz


class AdmittanceCfg(BaseModel):
    """Ручное перемещение «за руку»: усилие руки -> скорость платформы."""

    model_config = _STRICT
    gain_mms_per_n: float = Field(gt=0)
    deadband_n: float = Field(ge=0, description="трение + погрешность измерения момента")
    max_velocity_mms: float = Field(gt=0)


class SafetyCfg(BaseModel):
    """Аварийная цепь и границы, за которые софт не должен выходить."""

    model_config = _STRICT
    # Разрешение приводов (SON) приходит физическим входом, поэтому софт может
    # им управлять только через реле. manual — щёлкает человек, gpio — Pi,
    # sim — симулятор.
    enable_backend: Literal["manual", "gpio", "sim"] = "manual"
    enable_gpio_pin: int | None = None
    # Останавливаться, если ось не отвечает дольше этого времени.
    max_state_age_ms: float = Field(default=200.0, gt=0)
    # Не выходить на железо, пока не доказано, что уставку можно писать часто.
    require_eeprom_check: bool = True


class BusCfg(BaseModel):
    model_config = _STRICT
    port: str
    baudrate: int = 115200
    parity: Literal["N", "E", "O"] = "E"
    bytesize: Literal[7, 8] = 8
    stopbits: Literal[1, 2] = 1
    timeout_ms: float = Field(default=50.0, gt=0)
    retries: int = Field(default=2, ge=0)
    inter_frame_us: float | None = None

    @property
    def char_time_us(self) -> float:
        """Время одного символа: старт + данные + чётность + стоп."""
        bits = 1 + self.bytesize + (0 if self.parity == "N" else 1) + self.stopbits
        return 1e6 * bits / self.baudrate

    @property
    def frame_gap_us(self) -> float:
        """Межкадровая пауза. Modbus требует 3.5 символа, но не менее 1750 мкс
        для скоростей выше 19200 (см. спецификацию Modbus over Serial Line)."""
        if self.inter_frame_us is not None:
            return self.inter_frame_us
        return 1750.0 if self.baudrate > 19200 else 3.5 * self.char_time_us


# --------------------------------------------------------------------------- #
#  Машина целиком
# --------------------------------------------------------------------------- #
class MachineConfig(BaseModel):
    model_config = _STRICT

    name: str
    version: int = 1
    geometry: GeometryCfg
    platform: PlatformCfg
    winch_defaults: dict[str, Any] = Field(default_factory=dict)
    winches: list[WinchCfg]
    tension: TensionCfg
    workspace: WorkspaceCfg
    motion: MotionCfg
    control: ControlCfg
    admittance: AdmittanceCfg
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    buses: dict[str, BusCfg]

    # ------------------------------------------------------------------ #
    @model_validator(mode="before")
    @classmethod
    def _merge_winch_defaults(cls, data: Any) -> Any:
        """Подмешивает winch_defaults в каждую лебёдку до валидации.

        В YAML лебёдка обычно задана одной строкой `{anchor: A1}` — всё
        остальное берётся из общего блока winch_defaults и может быть
        переопределено поштучно.
        """
        if not isinstance(data, dict):
            return data
        defaults = data.get("winch_defaults") or {}
        winches = data.get("winches")
        if isinstance(winches, list):
            data = dict(data)
            data["winches"] = [
                {**defaults, **w} if isinstance(w, dict) else w for w in winches
            ]
        return data

    @model_validator(mode="after")
    def _cross_checks(self) -> "MachineConfig":
        anchors = self.geometry.anchors
        ids = [a.id for a in anchors]
        if len(set(ids)) != len(ids):
            raise ValueError(f"повторяющиеся id якорей: {ids}")

        if len(self.winches) != len(anchors):
            raise ValueError(f"лебёдок {len(self.winches)}, а якорей {len(anchors)} — должно совпадать")

        known = set(ids)
        for w in self.winches:
            if w.anchor not in known:
                raise ValueError(f"лебёдка ссылается на неизвестный якорь {w.anchor!r}, есть {sorted(known)}")

        if len(self.platform.attachments) != len(anchors):
            raise ValueError(
                f"точек крепления {len(self.platform.attachments)}, а тросов {len(anchors)}"
            )

        seen: set[tuple[str, int]] = set()
        for a in anchors:
            if a.bus not in self.buses:
                raise ValueError(f"якорь {a.id}: шина {a.bus!r} не описана в buses, есть {sorted(self.buses)}")
            key = (a.bus, a.slave)
            if key in seen:
                raise ValueError(f"адрес {a.slave} на шине {a.bus} занят дважды — связь работать не будет")
            seen.add(key)

        n = len(anchors)
        if n < self.geometry.dof + 1:
            raise ValueError(
                f"{n} тросов при dof={self.geometry.dof}: для управляемости нужно "
                f"минимум dof+1 = {self.geometry.dof + 1}"
            )
        return self

    # ------------------------------------------------------------------ #
    @property
    def n_cables(self) -> int:
        return len(self.geometry.anchors)

    def anchor_by_id(self, anchor_id: str) -> AnchorCfg:
        for a in self.geometry.anchors:
            if a.id == anchor_id:
                return a
        raise ConfigError(f"нет якоря {anchor_id!r}")

    def winch_for(self, anchor_id: str) -> WinchCfg:
        for w in self.winches:
            if w.anchor == anchor_id:
                return w
        raise ConfigError(f"нет лебёдки для якоря {anchor_id!r}")

    def ordered_winches(self) -> list[WinchCfg]:
        """Лебёдки в том же порядке, что и якоря — порядок осей во всей системе."""
        return [self.winch_for(a.id) for a in self.geometry.anchors]

    @property
    def is_calibrated(self) -> bool:
        return all(w.is_calibrated for w in self.winches)

    def uncalibrated(self) -> list[str]:
        return [w.anchor for w in self.winches if not w.is_calibrated]


# --------------------------------------------------------------------------- #
#  Профиль привода
# --------------------------------------------------------------------------- #
#  Профиль — это ДАННЫЕ, а не код. Поддержать другой привод = положить рядом
#  ещё один YAML и указать его в запуске; менять Python при этом не нужно.
# --------------------------------------------------------------------------- #

#: сколько 16-битных регистров занимает величина каждого типа
REG_WORDS: dict[str, int] = {"u16": 1, "i16": 1, "u32": 2, "i32": 2}


class ParamSpec(BaseModel):
    """Параметр привода (группа P-xxx)."""

    model_config = ConfigDict(extra="allow")
    p: int | None = None
    type: Literal["u16", "i16", "u32", "i32"] = "u16"
    confirmed: bool = False
    address: int | None = None
    enum: dict[str, int] | None = None
    default: Any = None
    unit: str | None = None
    desc: str | None = None
    scale: float = 1.0  # сырое значение регистра * scale = инженерные единицы

    @field_validator("enum", mode="before")
    @classmethod
    def _stringify_keys(cls, v: Any) -> Any:
        # В YAML ключи вроде 115200 или 8N1 читаются то числом, то строкой —
        # приводим к строкам, чтобы поиск был однозначным.
        if isinstance(v, dict):
            return {str(k): int(val) for k, val in v.items()}
        return v

    @property
    def words(self) -> int:
        return REG_WORDS[self.type]


class MonitorSpec(BaseModel):
    """Мониторная величина (группа d-xxx)."""

    model_config = ConfigDict(extra="allow")
    id: str
    type: Literal["u16", "i16", "u32", "i32"] = "i16"
    address: int | None = None
    order: int | None = None
    critical: bool = False
    unit: str | None = None
    scale: float = 1.0  # сырое значение регистра * scale = инженерные единицы

    @property
    def words(self) -> int:
        return REG_WORDS[self.type]


class AddressingCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    param_base: int | None = None
    param_ram_base: int | None = None
    monitor_base: int | None = None
    monitor_function: int | None = None
    word_order: Literal["lo_hi", "hi_lo"] = "lo_hi"


class FunctionCodes(BaseModel):
    model_config = ConfigDict(extra="allow")
    read_holding: int = 3
    read_input: int = 4
    write_single: int = 6
    write_multiple: int = 16


class ProfileLimits(BaseModel):
    model_config = ConfigDict(extra="allow")
    max_slave_id: int = 247
    max_regs_per_read: int = 16


class InitStep(BaseModel):
    model_config = ConfigDict(extra="allow")
    param: str
    value: Any
    note: str | None = None


class DriveProfile(BaseModel):
    """Карта регистров и повадки конкретной модели привода."""

    model_config = ConfigDict(extra="allow")

    name: str
    vendor: str | None = None
    protocol: str = "modbus_rtu"
    addressing: AddressingCfg = Field(default_factory=AddressingCfg)
    function_codes: FunctionCodes = Field(default_factory=FunctionCodes)
    limits: ProfileLimits = Field(default_factory=ProfileLimits)
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    monitors: dict[str, MonitorSpec] = Field(default_factory=dict)
    init_sequence: list[InitStep] = Field(default_factory=list)
    hot_register: str | None = None
    eeprom_safe: bool | None = None
    alarms: dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_refs(self) -> "DriveProfile":
        if self.hot_register and self.hot_register not in self.params:
            raise ValueError(f"hot_register {self.hot_register!r} не описан в params")
        for step in self.init_sequence:
            if step.param not in self.params:
                raise ValueError(f"init_sequence ссылается на неизвестный параметр {step.param!r}")
        return self

    # ------------------------------------------------------------------ #
    #  Разрешение адресов
    # ------------------------------------------------------------------ #
    @property
    def is_discovered(self) -> bool:
        """Известны ли адреса настолько, чтобы выходить на реальное железо."""
        return self.addressing.param_base is not None and self.addressing.monitor_base is not None

    def _need_probe(self, what: str) -> ConfigError:
        return ConfigError(
            f"{what}: адрес неизвестен. В мануале карты регистров нет — её нужно снять "
            f"с привода утилитой `python tools/reg_probe.py`, после чего адреса "
            f"пропишутся в профиле {self.name!r}. До этого работает только симулятор."
        )

    def param(self, name: str) -> ParamSpec:
        try:
            return self.params[name]
        except KeyError:
            raise ConfigError(f"параметр {name!r} не описан в профиле {self.name!r}") from None

    def monitor(self, name: str) -> MonitorSpec:
        try:
            return self.monitors[name]
        except KeyError:
            raise ConfigError(f"монитор {name!r} не описан в профиле {self.name!r}") from None

    def param_address(self, name: str, *, ram: bool = False) -> int:
        """Адрес Modbus для параметра.

        Приоритет: явный address в профиле -> база + номер параметра.
        При ram=True берётся param_ram_base (диапазон без записи в EEPROM),
        если он найден; иначе честно откатываемся на обычный.
        """
        spec = self.param(name)
        if spec.address is not None and not ram:
            return spec.address
        if spec.p is None:
            raise self._need_probe(f"параметр {name!r} (номер P-xxx не известен)")
        base = self.addressing.param_ram_base if ram else self.addressing.param_base
        if base is None and ram:
            base = self.addressing.param_base
        if base is None:
            raise self._need_probe(f"параметр {name!r} (P-{spec.p:03d})")
        return base + spec.p

    def monitor_address(self, name: str) -> int:
        spec = self.monitor(name)
        if spec.address is not None:
            return spec.address
        if spec.order is None or self.addressing.monitor_base is None:
            raise self._need_probe(f"монитор {name!r} ({spec.id})")
        return self.addressing.monitor_base + spec.order

    def monitor_function_code(self) -> int:
        fn = self.addressing.monitor_function
        return fn if fn is not None else self.function_codes.read_holding

    def encode_value(self, param_name: str, value: Any) -> int:
        """Значение из конфига (число или строка перечисления) -> число для регистра."""
        spec = self.param(param_name)
        if spec.enum is not None:
            key = str(value)
            if key in spec.enum:
                return int(spec.enum[key])
            if isinstance(value, str):
                raise ConfigError(
                    f"{param_name}={value!r} — допустимо только {sorted(spec.enum)}"
                )
        elif isinstance(value, str):
            raise ConfigError(f"параметр {param_name!r} не перечисляемый, а задан строкой {value!r}")
        return int(value)

    def decode_value(self, param_name: str, raw: int) -> Any:
        """Обратное к encode_value: число -> имя из перечисления, если оно есть."""
        spec = self.param(param_name)
        if spec.enum:
            for k, v in spec.enum.items():
                if v == raw:
                    return k
        return raw

    def alarm_text(self, code: int) -> str:
        if code == 0:
            return "нет аварии"
        return self.alarms.get(code, f"неизвестный код аварии {code}")


# --------------------------------------------------------------------------- #
#  Загрузка и сохранение
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_MACHINE = CONFIG_DIR / "machine.yaml"
DEFAULT_PROFILE = CONFIG_DIR / "drive_t3d.yaml"


def _ruamel():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # не переносить длинные строки
    return y


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"нет файла конфигурации {path}")
    with io.open(path, encoding="utf-8") as fh:
        return _ruamel().load(fh)


def load_machine(path: str | Path = DEFAULT_MACHINE) -> MachineConfig:
    path = Path(path)
    try:
        return MachineConfig.model_validate(_read_yaml(path))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_profile(path: str | Path = DEFAULT_PROFILE) -> DriveProfile:
    path = Path(path)
    try:
        return DriveProfile.model_validate(_read_yaml(path))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _descend(node: Any, key: str) -> Any:
    """Шаг по пути: числовой ключ — индекс списка, иначе ключ словаря."""
    if isinstance(node, list):
        try:
            return node[int(key)]
        except (ValueError, IndexError) as exc:
            raise ConfigError(f"неверный индекс списка {key!r}: {exc}") from None
    try:
        return node[key]
    except (KeyError, TypeError):
        raise ConfigError(f"нет ключа {key!r}") from None


def patch_yaml(
    path: str | Path,
    updates: dict[str, Any],
    *,
    backup: bool = True,
    create: bool = False,
) -> None:
    """Точечно правит значения в YAML, сохраняя комментарии и порядок.

    Ключи — путь через точку, индексы списков числами:
        patch_yaml(p, {"tension.target_n": 35.0, "winches.0.count_empty": 8123456})

    Именно так панель сохраняет настройки: пояснения в файле остаются на месте,
    а не затираются машинным дампом.

    По умолчанию несуществующий ключ — ошибка: почти всегда это опечатка.
    create=True разрешает добавить последний ключ пути; это нужно калибровке,
    потому что лебёдка в файле записана одной строкой `{anchor: A1}`, а свои
    калибровочные поля получает только после первой калибровки.
    """
    path = Path(path)
    data = _read_yaml(path)

    for dotted, value in updates.items():
        parts = dotted.split(".")
        node = data
        for key in parts[:-1]:
            node = _descend(node, key)
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            if last not in node and not create:
                raise ConfigError(f"{path}: нет параметра {dotted!r} — опечатка?")
            node[last] = value

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8") as fh:
        _ruamel().dump(data, fh)
    tmp.replace(path)


def save_calibration(winch_values: dict[int, dict[str, Any]], path: str | Path = DEFAULT_MACHINE) -> None:
    """Записывает результаты калибровки в machine.yaml.

    winch_values: {индекс лебёдки: {"count_empty": ..., "length_at_empty_mm": ..., "ea_n": ...}}
    """
    allowed = {"count_empty", "length_at_empty_mm", "ea_n"}
    updates: dict[str, Any] = {}
    for idx, fields in winch_values.items():
        unknown = set(fields) - allowed
        if unknown:
            raise ConfigError(
                f"калибровка пытается записать не калибровочные поля {sorted(unknown)}; "
                f"разрешены только {sorted(allowed)}"
            )
        for key, value in fields.items():
            updates[f"winches.{idx}.{key}"] = value
    patch_yaml(path, updates, create=True)
