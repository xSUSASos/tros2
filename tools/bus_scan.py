"""Поиск приводов на шине RS-485.

    python tools/bus_scan.py --list-ports
    python tools/bus_scan.py --port COM3
    python tools/bus_scan.py --port COM3 --slaves 1-8 --quick
    python tools/bus_scan.py --sim                 # показать работу без железа

Перебирает скорость, формат кадра и адрес, пока приводы не отзовутся.
Заводские настройки T3D — 19200 8E1 адрес 1 — проверяются первыми.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdpr.config import load_machine, load_profile  # noqa: E402
from drives import scanner  # noqa: E402
from drives.sim import SimTransport  # noqa: E402


def parse_slaves(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def list_ports() -> None:
    from serial.tools import list_ports as lp

    ports = list(lp.comports())
    if not ports:
        print("Последовательных портов не найдено.")
        print("Проверьте, что переходник USB-RS485 воткнут и драйвер встал.")
        return
    print("Доступные порты:")
    for p in ports:
        print(f"  {p.device:12s} {p.description}")
        if p.manufacturer:
            print(f"  {'':12s} производитель: {p.manufacturer}")


def run_sim_demo(args: argparse.Namespace) -> int:
    print("Демонстрация на симуляторе (железо не требуется).\n")
    profile = load_profile()
    sim = SimTransport("sim", profile, slaves=[1, 2, 3, 4], latency_ms=1.0)
    sim.open()
    found = []
    for slave in parse_slaves(args.slaves):
        if scanner._probe_slave(sim, slave, [0, 1, 181]):
            found.append(scanner.FoundDrive(slave, 115200, "E", 1, "чтение регистра 0"))
    report = scanner.BusScanReport(port="СИМУЛЯТОР", found=found, tried_combinations=len(parse_slaves(args.slaves)))
    print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Поиск приводов на шине RS-485")
    ap.add_argument("--port", help="последовательный порт, например COM3 или /dev/ttyUSB0")
    ap.add_argument("--list-ports", action="store_true", help="показать доступные порты и выйти")
    ap.add_argument("--sim", action="store_true", help="прогон на симуляторе без железа")
    ap.add_argument("--slaves", default="1-8", help="какие адреса проверять, например 1-32 или 1,2,5")
    ap.add_argument("--quick", action="store_true",
                    help="только заводские и целевые настройки (19200 8E1 и 115200 8E1)")
    ap.add_argument("--timeout", type=float, default=60.0, help="таймаут ответа, мс")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="остановиться, найдя столько приводов")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="  %(message)s",
    )

    if args.list_ports:
        list_ports()
        return 0
    if args.sim:
        return run_sim_demo(args)
    if not args.port:
        machine = load_machine()
        ports = {b.port for b in machine.buses.values()}
        ap.error(f"укажите --port (в конфиге сейчас: {', '.join(sorted(ports))}) "
                 f"или --list-ports, или --sim")

    slaves = parse_slaves(args.slaves)
    bauds = [19200, 115200] if args.quick else None
    formats = [("E", 1), ("N", 1)] if args.quick else None

    total = len(slaves) * len(bauds or scanner.BAUDRATES) * len(formats or scanner.FORMATS)
    print(f"Порт {args.port}: проверяю до {total} сочетаний "
          f"(адреса {min(slaves)}..{max(slaves)}), это может занять время.\n")

    state = {"n": 0}

    def progress(baud, parity, stop, slave, report):
        state["n"] += 1
        if state["n"] % 8 == 0 or slave == slaves[0]:
            print(f"\r  {baud} {parity}{stop}, адрес {slave:2d}   найдено: {len(report.found)}   ",
                  end="", flush=True)

    report = scanner.scan_bus(
        args.port, slaves=slaves, baudrates=bauds, formats=formats,
        timeout_ms=args.timeout, stop_after=args.stop_after, progress=progress,
    )
    print("\r" + " " * 70 + "\r", end="")
    print(report.summary())

    if report.found:
        print("\nДальше: снять карту регистров —")
        s = report.settings
        extra = f" --baud {s[0]} --parity {s[1]}" if s else ""
        print(f"  python tools/reg_probe.py --port {args.port} "
              f"--slave {sorted(d.slave for d in report.found)[0]}{extra}")
    return 0 if report.found else 1


if __name__ == "__main__":
    raise SystemExit(main())
