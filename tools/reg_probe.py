"""Снятие карты регистров привода.

    python tools/reg_probe.py --sim                     # репетиция без железа
    python tools/reg_probe.py --port COM3 --slave 1
    python tools/reg_probe.py --port COM3 --slave 1 --apply

В мануале на T3D есть номера параметров, но нет их адресов Modbus — таблица
была на диске, которого нет. Утилита восстанавливает карту опытом:

  1. Находит базу параметров по подписи. Мы точно знаем значения трёх
     параметров связи — раз обмен идёт, то P-181, P-182 и P-183 равны нашим
     настройкам. Три известных числа подряд однозначно указывают смещение.
  2. Проверяет находку по паспорту двигателя: номинальные обороты, ток и
     число пар полюсов должны совпасть с шильдиком.
  3. Находит счётчик позиции и порядок слов по повороту вала рукой:
     один оборот 23-битного энкодера — это ровно 8 388 608 импульсов.
  4. Проверяет, не ложится ли уставка скорости в EEPROM. Если ложится,
     цикл управления на 50 Гц сожжёт привод примерно за полчаса.

Найденное записывается в config/drive_t3d.yaml (только с ключом --apply).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdpr.config import BusCfg, DEFAULT_PROFILE, load_profile, patch_yaml  # noqa: E402
from drives import scanner  # noqa: E402
from drives.base import TransportError  # noqa: E402
from drives.modbus_rtu import ModbusRtuTransport, decode  # noqa: E402
from drives.sim import SimTransport  # noqa: E402

CPR_BY_BITS = {17: 131_072, 23: 8_388_608}


def rule(title: str = "") -> None:
    print("\n" + "─" * 72)
    if title:
        print(title)
        print("─" * 72)


def ask(prompt: str, auto: bool) -> None:
    if auto:
        print(f"  [авто] {prompt}")
        return
    input(f"  {prompt} — нажмите Enter...")


class Prober:
    def __init__(self, transport, slave: int, profile, *, baudrate: int,
                 parity: str, stopbits: int, auto: bool = False) -> None:
        self.t = transport
        self.slave = slave
        self.profile = profile
        self.baudrate, self.parity, self.stopbits = baudrate, parity, stopbits
        self.auto = auto
        self.found: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    def step_contact(self) -> bool:
        rule("1. Связь")
        for addr in (0, 1, 181):
            try:
                self.t.read_registers(self.slave, addr, 1)
                print(f"  привод {self.slave} отвечает (регистр {addr} читается)")
                return True
            except TransportError:
                continue
            except Exception as exc:  # ModbusException — тоже признак жизни
                print(f"  привод {self.slave} отвечает отказом на регистр {addr}: {exc}")
                return True
        print(f"  привод {self.slave} молчит. Проверьте адрес (P-181), скорость (P-182),")
        print("  полярность A/B и подано ли питание.")
        return False

    # ------------------------------------------------------------------ #
    def step_param_base(self) -> bool:
        rule("2. База параметров")
        sig = scanner.signature_for(
            self.profile, self.slave, self.baudrate, self.parity, self.stopbits
        )
        print(f"  ищу подпись: P-181={sig['slave_id']}, P-182={sig['baudrate']}, "
              f"P-183={sig['serial_format']}, P-184={sig['encoder_bits']}")

        def progress(addr, end, n):
            print(f"\r  снимаю адреса: 0x{addr:04X} / 0x{end:04X}, найдено регистров {n}   ",
                  end="", flush=True)

        candidates = scanner.probe_param_base(
            self.t, self.slave, self.profile, sig, progress=progress
        )
        print("\r" + " " * 70 + "\r", end="")

        if not candidates:
            print("  подпись не найдена.")
            print("  Возможные причины: другой порядок нумерации параметров, доступ")
            print("  к параметрам через отдельную функцию, либо разрядность энкодера")
            print("  не 23 (тогда запустите с --encoder-bits 17).")
            return False

        for c in candidates[:3]:
            print(f"  кандидат: база 0x{c.base:04X}, совпало {c.score} из 4 — {', '.join(c.matched)}")
        best = candidates[0]
        self.found["param_base"] = best.base

        if len(candidates) > 1 and candidates[1].score == best.score:
            mirror = candidates[1].base
            print(f"  найдено два одинаково подходящих диапазона: 0x{best.base:04X} и 0x{mirror:04X}.")
            print("  Обычно это значит, что параметры продублированы, и один диапазон")
            print("  не пишет в EEPROM. Проверю оба на шаге 4.")
            self.found["param_ram_base_candidate"] = mirror
        return True

    # ------------------------------------------------------------------ #
    def step_verify_by_nameplate(self) -> None:
        rule("3. Сверка с паспортом двигателя")
        base = self.found["param_base"]
        checks = [
            ("motor_rated_speed", "номинальные обороты", "об/мин"),
            ("motor_rated_current", "номинальный ток", "x0.1 А"),
            ("motor_pole_pairs", "пар полюсов", ""),
            ("encoder_bits", "разрядность энкодера", "бит"),
            ("control_mode", "режим управления", ""),
            ("speed_source", "источник команды скорости", ""),
        ]
        for name, label, unit in checks:
            spec = self.profile.params.get(name)
            if spec is None or spec.p is None:
                continue
            try:
                raw = self.t.read_registers(self.slave, base + spec.p, 1)[0]
            except Exception as exc:
                print(f"  {label:32s} не прочиталось ({exc})")
                continue
            shown = self.profile.decode_value(name, raw)
            print(f"  {label:32s} {shown} {unit}".rstrip())
        print("\n  Сверьте с шильдиком двигателя (80AST-A1C04025Z1: 4.6 А, 3000 об/мин).")
        print("  Если числа осмысленные — база найдена верно.")

    # ------------------------------------------------------------------ #
    def _dump(self, lo: int, hi: int) -> dict[int, int]:
        return scanner.dump_range(self.t, self.slave, lo, hi)

    def step_monitors(self, cpr: int, sweep: tuple[int, int]) -> bool:
        rule("4. Счётчик позиции и порядок слов")
        print(f"  Один оборот вала = {cpr} импульсов. Найду регистр, который")
        print("  на столько и изменится.\n")
        print("  ВАЖНО: привод должен быть ЗАПРЕЩЁН (снят SON), иначе он будет")
        print("  сопротивляться вращению.\n")

        runs = []
        for turns in (1.0, 2.5):
            ask(f"проверните барабан РОВНО на {turns} оборота в сторону намотки", self.auto)
            before = self._dump(*sweep)
            if self.auto:
                self._sim_rotate(cpr * turns)
            after = self._dump(*sweep)
            hits = scanner.analyze_rotation(before, after, counts_per_rev=cpr, turns=turns)
            print(f"    подходящих регистров: {len(hits)}")
            runs.append(hits)

        cands = scanner.confirm_across_rotations(runs)
        if not cands:
            print("  Ничего не подошло. Возможно, вал провернули слишком неточно —")
            print("  повторите, стараясь выдержать число оборотов, либо запустите")
            print("  с --tolerance 0.5.")
            return False

        if len(cands) > 1:
            print(f"  осталось кандидатов: {len(cands)}, разделяю малым доворотом")
            ask("проверните вал совсем чуть-чуть, меньше чем на 1/100 оборота", self.auto)
            before = self._dump(*sweep)
            if self.auto:
                self._sim_rotate(cpr // 400)
            after = self._dump(*sweep)
            refined = scanner.refine_by_small_rotation(before, after, cands)
            if refined:
                cands = refined

        for c in cands[:3]:
            print(f"  {c}")
        best = cands[0]
        self.found["position_address"] = best.address
        self.found["word_order"] = best.word_order
        self.found["monitor_base_hint"] = best.address

        value = decode(
            self.t.read_registers(self.slave, best.address, 2), "i32", best.word_order
        )
        print(f"\n  Сейчас счётчик показывает {value}.")
        print("  Сверьте с показанием d-PoS на панели самого привода —")
        print("  если совпало, адрес позиции найден точно.")
        return len(cands) == 1

    def _sim_rotate(self, counts: float) -> None:
        ax = getattr(self.t, "axes", {}).get(self.slave)
        if ax is not None:
            ax.position_counts += counts

    # ------------------------------------------------------------------ #
    def step_eeprom(self, writes: int) -> None:
        rule("5. Проверка записи уставки скорости")
        base = self.found["param_base"]
        spec = self.profile.params[self.profile.hot_register]
        addr = base + spec.p
        print(f"  Уставка скорости P-{spec.p:03d} лежит по адресу {addr} (0x{addr:04X}).")
        print(f"  Пишу туда {writes} раз значения 0 и 1 (привод запрещён, вал не тронется)")
        print("  и сравниваю задержку записи с задержкой чтения.\n")

        report = scanner.probe_eeprom(self.t, self.slave, addr, writes=writes)
        print(report.summary())
        self.found["eeprom_safe"] = report.safe

        mirror = self.found.get("param_ram_base_candidate")
        if mirror is not None and report.safe is not True:
            print(f"\n  Проверяю запасной диапазон 0x{mirror:04X}:")
            alt = scanner.probe_eeprom(self.t, self.slave, mirror + spec.p, writes=writes)
            print("  " + alt.verdict)
            if alt.safe:
                self.found["param_ram_base"] = mirror
                self.found["eeprom_safe"] = True
                print(f"  → уставку будем писать в 0x{mirror:04X}, параметры хранить в основном.")

        if self.found.get("eeprom_safe") is not True:
            print("\n  !!! Цикл управления запускать НЕЛЬЗЯ, пока это не решено.")
            print("  Ресурс EEPROM порядка 100 тысяч записей: на 50 Гц это ~33 минуты.")

    # ------------------------------------------------------------------ #
    def step_report(self, apply: bool, profile_path: Path) -> None:
        rule("Итог")
        if not self.found:
            print("  Ничего определить не удалось.")
            return
        for key, value in self.found.items():
            shown = f"0x{value:04X} ({value})" if isinstance(value, int) and value > 255 else value
            print(f"  {key:26s} {shown}")

        updates: dict[str, object] = {}
        if "param_base" in self.found:
            updates["addressing.param_base"] = int(self.found["param_base"])
        if "param_ram_base" in self.found:
            updates["addressing.param_ram_base"] = int(self.found["param_ram_base"])
        if "word_order" in self.found:
            updates["addressing.word_order"] = str(self.found["word_order"])
        if "position_address" in self.found:
            updates["monitors.actual_position.address"] = int(self.found["position_address"])
        if self.found.get("eeprom_safe") is not None:
            updates["eeprom_safe"] = bool(self.found["eeprom_safe"])

        if not updates:
            return
        print("\n  Будет записано в профиль:")
        for k, v in updates.items():
            print(f"    {k} = {v}")

        if apply:
            patch_yaml(profile_path, updates, create=True)
            print(f"\n  Записано в {profile_path}")
            print("  Остальные мониторы (момент, авария, обороты) ищутся тем же")
            print("  способом — запустите с --monitors после того, как подтвердите позицию.")
        else:
            print("\n  Ничего не записано. Добавьте --apply, чтобы сохранить в профиль.")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Снятие карты регистров привода T3D")
    ap.add_argument("--port")
    ap.add_argument("--slave", type=int, default=1)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--parity", default="E", choices=["N", "E", "O"])
    ap.add_argument("--stopbits", type=int, default=1, choices=[1, 2])
    ap.add_argument("--encoder-bits", type=int, default=23, choices=[17, 23])
    ap.add_argument("--sim", action="store_true", help="репетиция на симуляторе")
    ap.add_argument("--apply", action="store_true", help="записать найденное в профиль")
    ap.add_argument("--writes", type=int, default=60, help="сколько записей в тесте EEPROM")
    ap.add_argument("--sweep", default="0x1000-0x1100",
                    help="где искать мониторы, например 0x1000-0x1100")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="  %(message)s")

    profile = load_profile()
    lo, hi = (int(x, 0) for x in args.sweep.split("-"))

    if args.sim:
        # Репетиция должна изображать привод, карта которого ещё не снята —
        # даже если в профиле она уже подтверждена мануалом. Иначе
        # SimTransport возьмёт профиль как есть (реальные адреса, FC04 для
        # мониторов) вместо синтетической раскладки, которую и должна найти
        # разведка.
        blank = profile.model_copy(deep=True)
        blank.addressing.param_base = None
        blank.addressing.param_ram_base = None
        blank.addressing.monitor_base = None
        blank.addressing.monitor_function = None
        blank.eeprom_safe = None
        transport = SimTransport("sim", blank, slaves=[args.slave], latency_ms=0.5,
                                 baudrate=args.baud, parity=args.parity,
                                 stopbits=args.stopbits, encoder_bits=args.encoder_bits)
        transport.open()
        profile = transport.profile
        lo, hi = 0x1000, 0x1040
        print("РЕПЕТИЦИЯ НА СИМУЛЯТОРЕ. Повороты вала имитируются автоматически.")
    else:
        if not args.port:
            ap.error("укажите --port, либо --sim для репетиции")
        cfg = BusCfg(port=args.port, baudrate=args.baud, parity=args.parity,
                     stopbits=args.stopbits, timeout_ms=100, retries=1)
        transport = ModbusRtuTransport("probe", cfg)
        transport.open()

    prober = Prober(transport, args.slave, profile, baudrate=args.baud,
                    parity=args.parity, stopbits=args.stopbits, auto=args.sim)
    try:
        if not prober.step_contact():
            return 1
        if not prober.step_param_base():
            return 1
        prober.step_verify_by_nameplate()
        prober.step_monitors(CPR_BY_BITS[args.encoder_bits], (lo, hi))
        prober.step_eeprom(args.writes)
        prober.step_report(args.apply, DEFAULT_PROFILE)
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
