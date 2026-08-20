"""Хоминг: подтягивание коробки в угол и запись отсчётов.

Здесь курица и яйцо: чтобы знать, где коробка, нужна привязка, а чтобы
привязаться, надо привести коробку в известную точку. Разрывается это тем,
что режим работает НАПРЯМУЮ скоростями тросов и положением не пользуется
вовсе.

Трос выбранного угла наматывается с постоянной малой скоростью. Остальные три
работают в режиме следования: каждый стравливает ровно столько, чтобы держать
своё натяжение в низком коридоре. Никакой кинематики — четыре независимых
поведения, и ни одному из них не нужно знать, где платформа.

Следящие тросы намеренно несимметричны: стравливать они могут быстро, а
подбирать — только медленно. Иначе при шуме измерения момента трое могли бы
начать тянуть против того, который ведёт, и коробка встала бы посреди пути,
изображая упор.

Признак прибытия сложнее, чем кажется, и вот почему. Просто «натяжение выше
порога» не годится: подходя к углу, коробка уходит почти под модуль, и ведущий
трос принимает и её вес, и тягу остальных трёх — натяжение растёт само, без
всякого упора, до тех же величин. Скорость роста тоже не спасает: у самого
угла геометрия становится жёсткой, и натяжение растёт быстро в обоих случаях.

Однозначный признак ровно один: привод УПЁРСЯ В СВОЙ ПРЕДЕЛ МОМЕНТА и вал
встал. Больше он тянуть не может, значит дальше коробка не идёт. Поэтому на
время хоминга приводам ставится пониженный предел момента: он и защищает
трос, который здесь намеренно тянут до отказа, и служит признаком упора.
Обычный программный порог перетяга на это время поднимается — иначе он
сработал бы раньше, чем привод дойдёт до своего предела.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from cdpr.calibration import CornerRecord
from cdpr.modes.base import Mode, ModeOutput
from cdpr.state import ModeName

log = logging.getLogger(__name__)


class CornerHoming(Mode):
    """Объезжает углы и на каждом записывает отсчёты энкодеров."""

    name = ModeName.HOMING

    #: сколько секунд после трогания порог упора не проверяется
    SETTLE_BEFORE_ARRIVAL_S = 3.0
    #: доля от предела момента, начиная с которой считаем, что трос упёрся
    ARRIVAL_FRACTION = 0.8

    def __init__(self, corners: list[int] | None = None, *, feed_mms: float | None = None) -> None:
        self.corners = corners
        self.feed_mms = feed_mms
        self.index = 0
        self.phase = "подтягивание"
        self.records: list[CornerRecord] = []
        self.message = ""
        self._lock = threading.RLock()
        self._elapsed = 0.0
        self._over_s = 0.0
        self._settled_s = 0.0
        self._still_s = 0.0
        self._last_count: float | None = None
        self._limit_n = 0.0
        self._abort = False
        # Подтверждение упора: несколько циклов, а не доли секунды на глаз.
        # За это время мотор дотягивает трос, и чем оно длиннее, тем выше
        # подскочит натяжение — на капроне это чувствительно.
        self._confirm_s = 0.15
        self._arrival_n = 0.0
        self._follow_n = 0.0
        self._settle_s = 0.7
        self._timeout_s = 180.0
        self._mm_per_count = np.ones(1)

    @property
    def requires_homing(self) -> bool:
        return False

    @property
    def tolerates_slack(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    def enter(self, ctx) -> None:  # noqa: ANN001
        cfg = ctx.machine.homing
        if self.corners is None:
            self.corners = list(cfg.corners)
        if self.feed_mms is None:
            self.feed_mms = cfg.feed_mms
        self._settle_s = cfg.settle_s
        self._timeout_s = cfg.timeout_s
        self._follow_n = cfg.follow_tension_n
        self._mm_per_count = np.array([w.nominal_mm_per_count for w in ctx.winches])

        # Предел момента задан в процентах — переводим в ньютоны, чтобы
        # сравнивать с тем же, чем меряется натяжение везде остальное время.
        winch = ctx.winches[0]
        self._limit_n = abs(winch.torque_percent_to_force(cfg.torque_limit_percent))
        self._arrival_n = self.ARRIVAL_FRACTION * self._limit_n
        self._apply_torque_limit(ctx, cfg.torque_limit_percent)

        self.index = 0
        self.phase = "подтягивание"
        self.records = []
        self._reset_corner()
        self._abort = False
        log.info(
            "хоминг: углы %s, подача %.0f мм/с, упор при %.1f Н, следящие держат %.1f Н",
            self.corners, self.feed_mms, self._arrival_n, self._follow_n,
        )

    @property
    def tension_ceiling_n(self) -> float | None:
        # Потолок на время хоминга — сам предел момента привода с небольшим
        # запасом на шум измерения. Ниже ставить нельзя: упор не поймается.
        return 1.25 * self._limit_n if self._limit_n else None

    def exit(self, ctx) -> None:  # noqa: ANN001
        self._apply_torque_limit(ctx, ctx.machine.safety.drive_torque_limit_percent)

    @staticmethod
    def _apply_torque_limit(ctx, percent: float) -> None:  # noqa: ANN001
        """Меняет предел момента в самих приводах.

        Если параметр в профиле неизвестен, хоминг всё равно работает, но упор
        ловится хуже: тянуть трос будет нечему помешать, кроме программного
        порога. Об этом надо знать, поэтому пишем в журнал."""
        setter = getattr(ctx.drives, "set_torque_limit", None)
        if setter is None or not setter(percent):
            log.warning(
                "предел момента в приводах не выставлен — упор будет ловиться "
                "только программным порогом, а это менее надёжно"
            )

    def _reset_corner(self) -> None:
        self._elapsed = 0.0
        self._over_s = 0.0
        self._settled_s = 0.0
        self._still_s = 0.0
        self._last_count = None

    # ------------------------------------------------------------------ #
    @property
    def progress(self) -> float:
        return len(self.records) / max(1, len(self.corners or [1]))

    @property
    def current_corner(self) -> int | None:
        if self.corners is None or self.index >= len(self.corners):
            return None
        return self.corners[self.index]

    def abort(self) -> None:
        with self._lock:
            self._abort = True

    # ------------------------------------------------------------------ #
    def update(self, ctx, dt: float) -> ModeOutput:  # noqa: ANN001
        n = ctx.drives.n_axes
        zero = np.zeros(n)
        with self._lock:
            if self._abort:
                return ModeOutput(cable_velocity_mms=zero, done=True, message="хоминг прерван")

            corner = self.current_corner
            if corner is None:
                return ModeOutput(
                    cable_velocity_mms=zero, done=True,
                    message=f"хоминг закончен, углов пройдено {len(self.records)}",
                )

            tensions = ctx.state.tensions_n
            if tensions is None or len(tensions) != n:
                return ModeOutput(cable_velocity_mms=zero, message="нет данных о натяжении")

            counts = np.array([a.state.position_counts for a in ctx.drives.axes], dtype=float)
            self._elapsed += dt

            if self.phase == "успокоение":
                self._settled_s += dt
                if self._settled_s < self._settle_s:
                    return ModeOutput(cable_velocity_mms=zero, message="коробка успокаивается")
                self.records.append(CornerRecord(
                    corner=corner, counts=counts.copy(), tensions_n=tensions.copy(),
                    label=f"угол {corner}",
                ))
                log.info("хоминг: угол %d записан, натяжения %s Н",
                         corner, np.round(tensions, 1).tolist())
                self.index += 1
                self.phase = "подтягивание"
                self._reset_corner()
                return ModeOutput(
                    cable_velocity_mms=zero,
                    message=f"угол {corner} записан ({len(self.records)} из {len(self.corners)})",
                )

            # ── подтягивание ────────────────────────────────────────────
            if self._arrived(corner, counts, tensions, dt):
                self.phase = "успокоение"
                self._settled_s = 0.0
                return ModeOutput(cable_velocity_mms=zero,
                                  message=f"упёрлись в модуль {corner}, натяжение "
                                          f"{tensions[corner]:.1f} Н")

            if self._elapsed > self._timeout_s:
                return ModeOutput(
                    cable_velocity_mms=zero, done=True,
                    message=(
                        f"за {self._timeout_s:.0f} с коробка так и не упёрлась в модуль "
                        f"{corner}. Натяжения сейчас {np.round(tensions, 1).tolist()} Н при "
                        f"пороге {self._arrival_n:.1f} Н. Проверьте, наматывается ли трос "
                        f"{corner} в нужную сторону (winch.direction) и не мешает ли "
                        f"что-то коробке"
                    ),
                )

            return ModeOutput(
                cable_velocity_mms=self._velocities(corner, tensions),
                message=(f"угол {corner}: подтягиваю, натяжения "
                         f"{np.round(tensions, 1).tolist()} Н"),
            )

    # ------------------------------------------------------------------ #
    #: мм/с на ньютон ошибки натяжения у следящих тросов
    FOLLOW_GAIN = 12.0
    #: мёртвая зона по натяжению у следящих, Н — иначе они дёргаются на шуме
    FOLLOW_DEADBAND_N = 1.0
    #: предел скорости следящих, в долях подачи ведущего
    FOLLOW_SPEED_FACTOR = 2.0

    def _velocities(self, corner: int, tensions: np.ndarray) -> np.ndarray:
        """Ведущий трос выбирается, остальные держат своё натяжение.

        Симметрия по скорости здесь обязательна, и это неочевидно. Кажется
        разумным разрешить следящим быстро стравливать, но выбирать медленно —
        чтобы они не тянули против ведущего. На деле при переезде от угла к
        углу два троса из трёх обязаны УКОРОТИТЬСЯ на пару метров, и медленный
        предел их не пускает: они накапливают метры слабины, леска сходит с
        барабана, а записанные отсчёты не имеют отношения к геометрии.

        Усиление тоже не косметика: следящие обязаны успевать за ведущим.
        При слабом усилении коробку зажимает между тросами, натяжение ведущего
        растёт без всякого упора, и хоминг записывает угол посреди рамы.
        """
        feed = float(self.feed_mms)
        limit = self.FOLLOW_SPEED_FACTOR * feed
        velocity = np.zeros(len(tensions))
        velocity[corner] = feed
        for i in range(len(tensions)):
            if i == corner:
                continue
            error = self._follow_n - float(tensions[i])   # >0 — трос слаб, подобрать
            if abs(error) < self.FOLLOW_DEADBAND_N:
                continue
            velocity[i] = float(np.clip(error * self.FOLLOW_GAIN, -limit, limit))
        return velocity

    def _arrived(self, corner: int, counts: np.ndarray, tensions: np.ndarray,
                 dt: float) -> bool:
        """Упор — натяжение у предела момента И вставший вал одновременно.

        Порознь оба признака врут: натяжение растёт и на свободном подходе к
        углу, а вал замирает при любой заминке связи. Вместе они означают ровно
        одно — привод больше не может тянуть, то есть коробка стоит у корпуса.

        Первые секунды не в счёт: при трогании натяжение скачет, пока следящие
        тросы не разошлись, и на этом всплеске легко записать угол посреди
        рамы. Ошибка тихая и дорогая — вся система координат уезжает.
        """
        moved = 0.0
        if self._last_count is not None:
            moved = abs(float(counts[corner]) - self._last_count) * float(self._mm_per_count[corner])
        self._last_count = float(counts[corner])
        # Порог движения — четверть того, что вал прошёл бы за цикл на подаче.
        self._still_s = self._still_s + dt if moved < 0.25 * self.feed_mms * dt else 0.0

        if self._elapsed < self.SETTLE_BEFORE_ARRIVAL_S:
            self._over_s = 0.0
            return False

        if float(tensions[corner]) >= self._arrival_n:
            self._over_s += dt
        else:
            self._over_s = 0.0
        return self._over_s >= self._confirm_s and self._still_s >= self._confirm_s

    def describe(self) -> str:
        total = len(self.corners) if self.corners else 0
        return f"хоминг: угол {self.index + 1} из {total} ({self.phase})"
