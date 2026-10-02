"""Прогрев страниц: план расписания и ограниченный воркер (ring: application).

Что здесь и почему
------------------
Расписание прогрева жило внутри планировщика: словарь интервалов `_SCHEDULE` рядом с кодом,
который строит списки тикеров, и Redis-маркер «уже публиковали» через `GET` + `SET`.
Отсюда два дефекта:

1. **Маркер не атомарен.** ``GET`` (мимо) → публикация → ``SET``: между проверкой и записью
   вклинивается другая реплика, и слот публикуется **каждой** репликой. Для прогрева это
   худший случай: все воркеры одновременно тянут одни и те же цепочки у провайдера.
   Здесь вместо маркера — :class:`gex.adapters.cache.lease.RedisLease` (``SET NX PX`` до
   публикации), поэтому слот выполняется **ровно один раз** на кластер.
2. **Расписание — необъявленный список.** Что и с какой частотой прогревается, знал только
   цикл планировщика. Здесь это :class:`PrewarmSlot` — объявление: имя, интервал, что
   публиковать, описание. Из плана выводятся и расписание, и админский ответ, и проверка
   полноты («план ↔ обработчик»): слот без цели — ошибка, а не тихий пропуск.

Ограниченность воркера
----------------------
За тик выполняется не больше ``max_slots_per_tick`` слотов. На старте все слоты «просрочены»
одновременно (последнего запуска ещё нет), и без границы процесс выпустил бы весь прогрев
одной пачкой — то есть устроил бы провайдеру тот самый всплеск, против которого прогрев
и задуман.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional, Protocol

from gex.domain.schedule import MSK, slot_occurrence

from gex.application.jobs import FetchTask

logger = logging.getLogger(__name__)

#: Сколько слотов запускать за один тик (защита от одновременного старта всех).
DEFAULT_MAX_SLOTS_PER_TICK = 4

#: Срок аренды фиксированного слота: он живёт до следующего вхождения, то есть сутки+.
FIXED_SLOT_LEASE_TTL_S = 2 * 24 * 3600


class PublishesTasks(Protocol):
    """Публикатор задач (реализуется :class:`gex.application.queue.TaskPublisher`)."""

    def publish_many(self, tasks: Iterable[FetchTask]) -> int: ...


class AcquiresLeases(Protocol):
    """Аренды (реализуется :class:`gex.adapters.cache.lease.RedisLease`)."""

    def acquire(self, name: str, ttl_s: int) -> Optional[str]: ...
    def release(self, name: str, token: Optional[str]) -> bool: ...
    def is_held(self, name: str) -> bool: ...


@dataclass(frozen=True)
class PrewarmSlot:
    """Один пункт расписания прогрева.

    ``tasks`` — функция, а не список: цели зависят от вселенных, которые могут меняться
    (ad-hoc тикеры пользователя), и фиксировать их на момент сборки плана нельзя.
    """

    name: str
    interval_s: int
    tasks: Callable[[], list[FetchTask]]
    description: str = ""
    #: Фиксированные слоты МСК: публикуются только внутри окна вхождения, один раз на вхождение.
    fixed_msk: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.interval_s <= 0 and not self.fixed_msk:
            raise ValueError(f"слот {self.name}: интервал должен быть > 0")
        if not self.name:
            raise ValueError("у слота должно быть имя")


@dataclass(frozen=True)
class PrewarmPlan:
    """Объявленное расписание прогрева."""

    slots: tuple[PrewarmSlot, ...] = field(default_factory=tuple)

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots)

    def get(self, name: str) -> PrewarmSlot:
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise KeyError(f"слот {name!r} не объявлен в плане прогрева")

    def intervals(self) -> dict[str, int]:
        """Имя слота → интервал в секундах (для админки и диагностики)."""
        return {s.name: s.interval_s for s in self.slots}

    def validate(self) -> list[str]:
        """Проверка полноты «план ↔ обработчик».

        Возвращает список проблем (пусто — всё в порядке). Слот без целей или с пустым
        описанием — это объявление, которое ничего не делает: такое обязано быть видно
        на сборке, а не обнаруживаться по отсутствию данных в Redis.
        """
        problems: list[str] = []
        seen: set[str] = set()
        for slot in self.slots:
            if slot.name in seen:
                problems.append(f"слот {slot.name!r} объявлен дважды")
            seen.add(slot.name)
            tasks = slot.tasks
            if not callable(tasks):
                problems.append(f"слот {slot.name!r}: цели не вызываемы")
            if not slot.description.strip():
                problems.append(f"слот {slot.name!r}: нет описания — что именно прогревается?")
        return problems


class PrewarmWorker:
    """Ограниченный воркер прогрева: слот выполняется ровно один раз на кластер."""

    def __init__(
        self,
        plan: PrewarmPlan,
        publisher: PublishesTasks,
        lease: Optional[AcquiresLeases] = None,
        *,
        max_slots_per_tick: int = DEFAULT_MAX_SLOTS_PER_TICK,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._plan = plan
        self._publisher = publisher
        self._lease = lease
        self._max_slots_per_tick = max(1, int(max_slots_per_tick))
        self._clock = clock
        self._wall_clock = wall_clock
        self._last_run: dict[str, float] = {}
        self.fired: dict[str, int] = {}
        self.skipped_not_due = 0
        self.skipped_lease_held = 0
        self.failed: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  Один тик
    # ------------------------------------------------------------------ #
    def run_once(self, *, now: Optional[float] = None) -> list[str]:
        """Обработать слоты, которым пора. Возвращает имена выполненных.

        Порядок важен: сначала аренда, потом публикация. Наоборот (как было) —
        и слот выполнится столько раз, сколько реплик живёт одновременно.
        """
        now = self._clock() if now is None else now
        fired: list[str] = []
        for slot in self._plan.slots:
            if len(fired) >= self._max_slots_per_tick:
                break
            if not self._is_due(slot, now):
                self.skipped_not_due += 1
                continue
            if self._run_slot(slot):
                fired.append(slot.name)
        return fired

    def run(self, stop_event: threading.Event, *, interval_s: float = 30.0) -> None:
        """Крутить тики до ``stop_event`` (цикл планировщика)."""
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 — цикл обязан выживать
                logger.error("Ошибка цикла прогрева: %s", exc)
            stop_event.wait(interval_s)

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _is_due(self, slot: PrewarmSlot, now: float) -> bool:
        """Пора ли слоту (по интервалу процесса; кросс-процессную гонку снимает аренда)."""
        if slot.fixed_msk:
            return True  # окно вхождения проверяет _occurrence (по МСК-часам)
        last = self._last_run.get(slot.name)
        return last is None or (now - last) >= slot.interval_s

    def _run_slot(self, slot: PrewarmSlot) -> bool:
        name = self._lease_name(slot)
        if name is None:
            return False
        token = self._acquire(name, slot)
        if self._lease is not None and token is None:
            # Аренда занята: слот уже выполняет другой процесс. Это норма (не ошибка).
            self.skipped_lease_held += 1
            self._last_run[slot.name] = self._clock()
            return False

        try:
            tasks = slot.tasks()
        except Exception as exc:  # noqa: BLE001 — цель упала, освобождаем аренду
            self._failed(slot, exc)
            if self._lease is not None:
                self._lease.release(name, token)
            return False
        if not tasks:
            logger.debug("Слот %s: цели пусты — публиковать нечего", slot.name)
            if self._lease is not None:
                self._lease.release(name, token)
            self._last_run[slot.name] = self._clock()
            return False

        try:
            published = self._publisher.publish_many(tasks)
        except Exception as exc:  # noqa: BLE001
            self._failed(slot, exc)
            if self._lease is not None:
                self._lease.release(name, token)
            return False

        self._last_run[slot.name] = self._clock()
        self.fired[slot.name] = self.fired.get(slot.name, 0) + 1
        logger.info(
            "Прогрев %s: опубликовано %d/%d задач (интервал %ds)",
            slot.name, published, len(tasks), slot.interval_s,
        )
        if published == 0:
            # Ни одна задача не ушла: очередь недоступна. Аренду освобождаем, иначе
            # слот «сгорит» до конца интервала, ничего не сделав.
            if self._lease is not None:
                self._lease.release(name, token)
            return False
        return True

    def _failed(self, slot: PrewarmSlot, exc: BaseException) -> None:
        self.failed[slot.name] = self.failed.get(slot.name, 0) + 1
        logger.error("Прогрев %s не выполнен: %s", slot.name, exc)

    def _lease_name(self, slot: PrewarmSlot) -> Optional[str]:
        """Имя аренды: для фиксированного слота — с вхождением, иначе — по имени слота."""
        if not slot.fixed_msk:
            return f"prewarm:{slot.name}"
        occurrence = self._occurrence(slot)
        if occurrence is None:
            return None  # вне окна вхождения
        return f"prewarm:fixed:{slot.name}:{occurrence}"

    def _acquire(self, name: str, slot: PrewarmSlot) -> Optional[str]:
        if self._lease is None:
            return "nolease"  # аренды не настроены: работаем как единственный процесс
        if slot.fixed_msk:
            ttl = FIXED_SLOT_LEASE_TTL_S
        else:
            # Аренда живёт ровно интервал: когда он истечёт, слот снова доступен — и ни
            # секундой раньше (иначе реплики размножат прогрев).
            ttl = slot.interval_s
        return self._lease.acquire(name, ttl)

    def _occurrence(self, slot: PrewarmSlot) -> Optional[str]:
        """Ключ вхождения ``YYYY-MM-DD HH:MM``, если сейчас окно фиксированного слота.

        Окно считает домен (``gex.domain.schedule``): та же функция обслуживает планировщик,
        поэтому трактовка «23:00 МСК» здесь и там не может разойтись.
        """
        now_msk = datetime.fromtimestamp(self._wall_clock(), tz=MSK)
        return slot_occurrence(slot.fixed_msk, now_msk)

    def describe(self) -> dict:
        return {
            "slots": list(self._plan.names()),
            "fired": dict(self.fired),
            "failed": dict(self.failed),
            "skipped_not_due": self.skipped_not_due,
            "skipped_lease_held": self.skipped_lease_held,
        }

__all__ = [
    "DEFAULT_MAX_SLOTS_PER_TICK",
    "FIXED_SLOT_LEASE_TTL_S",
    "AcquiresLeases",
    "PrewarmPlan",
    "PrewarmSlot",
    "PrewarmWorker",
    "PublishesTasks",
]
