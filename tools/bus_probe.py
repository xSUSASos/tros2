"""Замер реального времени обмена — отсюда берётся частота цикла.

    python tools/bus_probe.py --sim               репетиция без железа
    python tools/bus_probe.py --port COM3
    python tools/bus_probe.py --port COM3 --seconds 20

Частоту цикла нельзя назначить, её можно только измерить. Один цикл
управления — это по два чтения на ось плюс запись уставки; на четырёх осях
двенадцать транзакций. Сколько они занимают, определяет всё остальное:
точность остановки равна скорости, умноженной на период цикла.

Чаще всего виноват не Modbus. У переходников на чипах FTDI по умолчанию
стоит latency timer 16 мс: чип копит байты, прежде чем отдать их хосту, и
один этот параметр растягивает цикл в разы. Утилита это распознаёт по тому,
что измеренное время сильно больше расчётного времени на проводе.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdpr.config import (  # noqa: E402
    DEFAULT_MACHINE,
    DEFAULT_PROFILE,
    ConfigError,
    load_machine,
    load_profile,
)
from drives.base import TransportError  # noqa: E402
from drives.t3d import build_drive_group  # noqa: E402


def wire_time_ms(bus, registers: int) -> float:
    """Сколько транзакция занимает на проводе, без учёта задержек хоста.

    Запрос — восемь байт, ответ — пять плюс по два на регистр, плюс
    обязательная тишина между кадрами. Для скоростей выше 19200 спецификация
    Modbus фиксирует её в 1.75 мс, и на быстрой шине именно эта пауза, а не
    сами байты, определяет время цикла: тринадцать транзакций — это уже
    двадцать три миллисекунды одного молчания.
    """
    request, response = 8, 5 + 2 * registers
    return (request + response) * bus.char_time_us / 1000.0 + bus.frame_gap_us / 1000.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Замер времени обмена на шине")
    ap.add_argument("--sim", action="store_true", help="репетиция на симуляторе")
    ap.add_argument("--port", help="последовательный порт, например COM3 или /dev/ttyUSB0")
    ap.add_argument("--seconds", type=float, default=10.0, help="сколько мерить")
    ap.add_argument("--machine", default=str(DEFAULT_MACHINE))
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="  %(message)s")

    machine = load_machine(args.machine)
    profile = load_profile(args.profile)
    if args.port:
        for bus in machine.buses.values():
            bus.port = args.port

    try:
        drives = build_drive_group(machine, profile, simulated=args.sim)
        drives.open()
    except (ConfigError, TransportError) as exc:
        print(f"\nНе удалось открыть шину:\n  {exc}\n")
        return 1

    bus = next(iter(machine.buses.values()))
    per_cycle = drives.n_axes * (len(drives.axes[0].spans) + 1) + 1
    print()
    print(f"Шина {bus.port}, {bus.baudrate} {bus.parity}{bus.stopbits}, осей {drives.n_axes}")
    print(f"Транзакций за цикл: {per_cycle} "
          f"({len(drives.axes[0].spans)} чтения и запись на ось, плюс одна на скорость)")
    print(f"Расчётное время на проводе: {wire_time_ms(bus, 7) * per_cycle:.1f} мс за цикл")
    print(f"Меряю {args.seconds:.0f} с...")

    samples: list[float] = []
    errors = 0
    deadline = time.perf_counter() + args.seconds
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        try:
            drives.read_states()
            drives.set_speeds([0.0] * drives.n_axes)
        except TransportError:
            errors += 1
            continue
        samples.append((time.perf_counter() - started) * 1000.0)

    drives.close()
    if not samples:
        print("\nНи один цикл не прошёл. Проверьте адреса приводов и полярность A/B.\n")
        return 1

    samples.sort()
    median = statistics.median(samples)
    worst = samples[int(0.99 * (len(samples) - 1))]
    wire = wire_time_ms(bus, 7) * per_cycle

    print()
    print(f"  циклов снято      {len(samples)}, сбоев {errors}")
    print(f"  медиана           {median:.1f} мс")
    print(f"  худшие 1 %        {worst:.1f} мс")
    print(f"  достижимая частота {1000.0 / worst:.1f} Гц "
          f"(по худшему, а не по среднему — цикл должен успевать всегда)")
    print()

    safe_hz = 1000.0 / (worst * 1.3)
    print(f"  Ставьте control.loop_hz не больше {safe_hz:.0f} — с запасом в треть на "
          f"колебания планировщика ОС.")
    print(f"  Сейчас в конфиге {machine.control.loop_hz:.0f} Гц.")
    if machine.control.loop_hz > safe_hz:
        print("  Это БОЛЬШЕ измеренного: цикл будет систематически не успевать, а")
        print("  контур, который не успевает, ведёт себя непредсказуемо.")

    feed = machine.motion.jog_feed_mms
    print()
    print(f"  Точность остановки при этой частоте и подаче {feed:.0f} мм/с: "
          f"около {feed * worst / 1000.0:.1f} мм.")
    print("  Это не настройка, а арифметика: пока хост читает энкодер и решает,")
    print("  что пора, мотор продолжает крутиться. Хотите точнее — снижайте")
    print("  скорость доводки, а не допуск.")

    if median > 2.5 * wire and not args.sim:
        print()
        print("  ВНИМАНИЕ: измеренное время в разы больше расчётного времени на")
        print(f"  проводе ({wire:.1f} мс). Первый подозреваемый — latency timer")
        print("  переходника USB-RS485: у чипов FTDI он по умолчанию 16 мс.")
        print("  Windows: свойства COM-порта, Advanced, Latency Timer -> 1")
        print("  Linux:   /sys/bus/usb-serial/devices/ttyUSB0/latency_timer -> 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
