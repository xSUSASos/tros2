"""Запуск системы и веб-панели.

    python run.py --sim                 модель, железо не нужно
    python run.py                       железо по настройкам из config/machine.yaml
    python run.py --host 0.0.0.0        пустить в сеть (например, на Raspberry Pi)

Панель открывается по адресу, который скрипт напечатает при старте.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdpr.config import DEFAULT_MACHINE, DEFAULT_PROFILE, ConfigError  # noqa: E402
from cdpr.runtime import build_runtime  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Тросовая система: сервер и панель")
    ap.add_argument("--sim", action="store_true", help="работа на модели, без железа")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--machine", default=str(DEFAULT_MACHINE))
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument("--no-enable", action="store_true",
                    help="не разрешать приводы при старте (по умолчанию и так не разрешаются)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        runtime = build_runtime(args.machine, args.profile, simulated=args.sim)
    except ConfigError as exc:
        print("\nНе удалось собрать машину:\n")
        print(f"  {exc}\n")
        if not args.sim:
            print("Для работы без железа запустите: python run.py --sim\n")
        return 1

    import socket

    import uvicorn

    from api.server import create_app

    # Порт проверяем ДО запуска: иначе сообщение о занятом порте тонет в
    # выводе, а сервер молча не поднимается, и это выглядит как поломка.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((args.host, args.port))
    except OSError:
        print()
        print(f"  Порт {args.port} уже занят — скорее всего сервер уже запущен")
        print(f"  в другом окне. Закройте его либо возьмите другой порт:")
        print()
        print(f"      python run.py{' --sim' if args.sim else ''} --port {args.port + 1}")
        print()
        return 1
    finally:
        probe.close()

    app = create_app(runtime)

    print()
    print("=" * 70)
    print(f"  {runtime.machine.name}")
    print(f"  {'МОДЕЛЬ (железо не подключено)' if args.sim else 'РАБОТА С ЖЕЛЕЗОМ'}")
    print(f"  тросов: {runtime.machine.n_cables}, "
          f"{'откалибрована' if runtime.machine.is_calibrated else 'НЕ ОТКАЛИБРОВАНА'}")
    print(f"  разрешение приводов: {runtime.drives.enabler.describe()}")
    if not runtime.drives.enabler.is_automatic:
        print("  ВНИМАНИЕ: софт не может снять разрешение сам — кнопка аварийного")
        print("            стопа в панели только обнулит скорости. Настоящий стоп —")
        print("            размыкание цепи SON, держите его под рукой.")
    print()
    print(f"  Панель:  http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/")
    print(f"  API:     http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/api/docs")
    print("=" * 70)
    print()

    try:
        runtime.start()
    except Exception as exc:  # noqa: BLE001
        print(f"\nНе удалось запустить: {exc}\n")
        return 1

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
