"""Сборка машины: конфиг -> приводы -> контур управления.

Единственное место, где решается, работаем мы с железом или с моделью.
Всё остальное написано против одного интерфейса и разницы не замечает.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cdpr.config import (
    DEFAULT_MACHINE,
    DEFAULT_PROFILE,
    DriveProfile,
    MachineConfig,
    load_machine,
    load_profile,
)
from cdpr.controller import Controller
from cdpr.simulation import DEFAULT_EA_N, PlatformSimulator
from drives.t3d import T3DDriveGroup, build_drive_group

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    machine: MachineConfig
    profile: DriveProfile
    drives: T3DDriveGroup
    controller: Controller
    platform: PlatformSimulator | None = None
    machine_path: Path = DEFAULT_MACHINE
    profile_path: Path = DEFAULT_PROFILE
    clock: Any = None

    def step(self, dt: float = 0.02):
        """Один детерминированный шаг: время модели двигается ровно на dt.

        Работает только с виртуальными часами (build_runtime(virtual_clock=True)).
        На железе время двигать нельзя, там оно идёт само.
        """
        if self.clock is None:
            raise RuntimeError(
                "шаг вручную возможен только с виртуальными часами: "
                "соберите машину как build_runtime(simulated=True, virtual_clock=True)"
            )
        self.clock.advance(dt)
        return self.controller.cycle(dt)

    def start(self) -> None:
        self.drives.open()
        self.drives.initialize()
        self.controller.start()

    def shutdown(self) -> None:
        self.controller.stop()
        self.drives.close()

    def __enter__(self) -> "Runtime":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def _fill_simulated_calibration(machine: MachineConfig, *, drum_capacity_mm: float = 14_000.0) -> None:
    """Проставляет калибровку для модели.

    На железе эти числа даёт калибровка по реперам; модели же нужно с чего-то
    начать, поэтому берём «барабан пуст при нулевом отсчёте, троса намотано
    столько-то». Физику это не искажает: калибровочные параметры на то и
    параметры, что модель работает при любых их значениях.
    """
    for winch in machine.winches:
        if winch.count_empty is None:
            winch.count_empty = 0
        if winch.length_at_empty_mm is None:
            winch.length_at_empty_mm = drum_capacity_mm
        if winch.ea_n is None:
            winch.ea_n = DEFAULT_EA_N


def build_runtime(
    machine_path: str | Path = DEFAULT_MACHINE,
    profile_path: str | Path = DEFAULT_PROFILE,
    *,
    simulated: bool = False,
    virtual_clock: bool = False,
    start_pose_mm: Any = None,
    sim_options: dict[str, Any] | None = None,
) -> Runtime:
    machine = load_machine(machine_path)
    profile = load_profile(profile_path)
    platform: PlatformSimulator | None = None
    clock = None

    if simulated:
        _fill_simulated_calibration(machine)
        if virtual_clock:
            from drives.sim import VirtualClock

            clock = VirtualClock()
            sim_options = dict(sim_options or {})
            sim_options["clock"] = clock

    drives = build_drive_group(machine, profile, simulated=simulated, sim_options=sim_options)

    if simulated:
        platform = PlatformSimulator(machine)
        for name, transport in drives.transports.items():
            slaves = [a.slave for a in machine.geometry.anchors if a.bus == name]
            platform.attach(transport, slaves)

        centre = platform.kinematics.anchors.mean(axis=0)
        pose = np.asarray(start_pose_mm, dtype=float) if start_pose_mm is not None else np.array(
            [centre[0], centre[1], machine.workspace.z_min_mm
             + 0.4 * (machine.workspace.z_max_mm - machine.workspace.z_min_mm)]
        )
        counts = platform.place_at(pose)
        for i, anchor in enumerate(machine.geometry.anchors):
            axis = drives.transports[anchor.bus].axes[anchor.slave]
            axis.position_counts = float(counts[i])
        log.info("модель: платформа помещена в %s мм", np.round(pose).astype(int).tolist())

    controller = Controller(machine, drives.profile, drives)
    return Runtime(machine, drives.profile, drives, controller, platform,
                   Path(machine_path), Path(profile_path), clock)
