"""Динамический пул потребителей очереди: число воркеров — по нагрузке (ring: application).

Зачем
-----
Потребители очереди задач (``TaskConsumer``) были штучными: один на профиль, фиксированное
число. Планировщик же публикует задачи «пачками» — 18 свечей раз в 5 минут, 44 цепочки раз
в 15: одна пачка разгребается последовательно, и хвост ждёт. В простое воркер остаётся
запущенным просто так.

Здесь — пул: от ``min_workers`` до ``max_workers`` потребителей, число которых регулируется
по глубине очереди профиля (backlog = сумма ``XLEN`` по стримам):

* backlog выше ``high_watermark`` → добавляем воркера (с cooldown — защита от осцилляций);
* backlog пуст и все воркеры простаивали (по дельте ``stats.processed``) → убираем воркера
  (не ниже ``min_workers``), не прерывая занятого.

Каждый воркер — отдельный ``TaskConsumer`` со своим consumer-именем: pending-наборы не
смешиваются, а недоделанную работу умершего воркера страхует ``claim_stale`` соседей.
Группа потребителей общая, поэтому произвольное число потоков и процессов **делит** работу,
а не дублирует её: at-least-once + дедупликация по ``idempotency_key`` остаются сквозными.

Границы
-------
``autoscale_once`` отделён от ``run`` — по тому же принципу, что ``handle_once`` у
``TaskConsumer``: одно решение проверяется тестом без потоков и таймеров. Максимум воркеров
считается **на процесс**; при запуске нескольких процессов на один профиль (docker scale)
суммарная ёмкость складывается — ограничителем остаётся общий rate limiter провайдеров.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from gex.application.jobs import CONSUMER_PROFILES, FetchTask
from gex.application.worker import TaskConsumer
from gex.ports.job_queue import JobQueuePort

logger = logging.getLogger(__name__)

#: Порог backlog, после которого пул добавляет воркера.
DEFAULT_HIGH_WATERMARK = 8
#: Период контроля состава пула.
DEFAULT_CHECK_INTERVAL_S = 5.0
#: Минимальная пауза между изменениями состава (защита от осцилляций).
DEFAULT_COOLDOWN_S = 12.0


@dataclass(frozen=True)
class PoolConfig:
    """Диапазон и пороги пула.

    ``autoscale=False`` фиксирует состав на ``min_workers`` (аварийный/отладочный режим).
    """

    min_workers: int = 1
    max_workers: int = 4
    autoscale: bool = True
    high_watermark: int = DEFAULT_HIGH_WATERMARK
    check_interval_s: float = DEFAULT_CHECK_INTERVAL_S
    cooldown_s: float = DEFAULT_COOLDOWN_S

    def __post_init__(self) -> None:
        if self.min_workers < 1:
            raise ValueError(f"min_workers={self.min_workers}: пул без воркеров бессмыслен")
        if self.max_workers < self.min_workers:
            raise ValueError(
                f"max_workers={self.max_workers} < min_workers={self.min_workers}"
            )


@dataclass
class _Worker:
    """Воркер пула: потребитель + его очередь (нужна для чтения backlog)."""

    name: str
    consumer: TaskConsumer
    port: JobQueuePort


def parse_profiles(value: str) -> tuple[str, ...]:
    """Разобрать ``QUEUE_CONSUMERS`` («all» | csv профилей) во включённые профили.

    Неизвестные имена отбрасываются с предупреждением; пустой или полностью неизвестный
    список трактуется как ``all`` — лучше потреблять всё, чем из-за опечатки копить задачи
    в очереди без потребителей.
    """
    raw = [p.strip().lower() for p in str(value or "all").split(",") if p.strip()]
    if not raw or "all" in raw:
        return tuple(CONSUMER_PROFILES)
    known = tuple(dict.fromkeys(p for p in raw if p in CONSUMER_PROFILES))
    unknown = [p for p in raw if p not in CONSUMER_PROFILES]
    if unknown:
        logger.warning(
            "QUEUE_CONSUMERS: неизвестные профили %s — пропускаю (доступно: %s, all)",
            ", ".join(unknown), ", ".join(CONSUMER_PROFILES),
        )
    return known or tuple(CONSUMER_PROFILES)


class ConsumerPool:
    """Пул потребителей одного профиля с автоскейлом по backlog.

    Parameters
    ----------
    profile :
        Имя профиля из :data:`gex.application.jobs.CONSUMER_PROFILES` (для имён и логов).
    port_factory :
        Фабрика очереди по имени воркера: каждому потребителю — свой consumer-статус,
        но общая группа (см. ``StreamJobQueue``).
    handler :
        Обработчик задачи (тот же, что у одиночного ``TaskConsumer``).
    config :
        Диапазон и пороги (по умолчанию — :class:`PoolConfig`).
    clock :
        Монотонные часы; инжектируются тестами (``tick``/``autoscale_once``).
    """

    def __init__(
        self,
        profile: str,
        port_factory: Callable[[str], JobQueuePort],
        handler: Callable[[FetchTask], None],
        config: Optional[PoolConfig] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if profile not in CONSUMER_PROFILES:
            raise ValueError(f"профиль {profile!r} не объявлен в CONSUMER_PROFILES")
        self._profile = profile
        self._port_factory = port_factory
        self._handler = handler
        self._config = config or PoolConfig()
        self._clock = clock
        self._workers: list[_Worker] = []
        self._last_processed: dict[str, int] = {}
        self._last_change: Optional[float] = None
        self._next_index = 0
        self.scale_ups = 0
        self.scale_downs = 0
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  Состояние
    # ------------------------------------------------------------------ #
    @property
    def profile(self) -> str:
        return self._profile

    @property
    def size(self) -> int:
        """Текущее число воркеров."""
        return len(self._workers)

    def describe(self) -> dict:
        return {
            "profile": self._profile,
            "size": self.size,
            "min": self._config.min_workers,
            "max": self._config.max_workers,
            "autoscale": self._config.autoscale,
            "scale_ups": self.scale_ups,
            "scale_downs": self.scale_downs,
            "workers": [w.name for w in self._workers],
        }

    # ------------------------------------------------------------------ #
    #  Жизненный цикл
    # ------------------------------------------------------------------ #
    def start(self, *, autopilot: bool = True) -> "ConsumerPool":
        """Поднять ``min_workers`` воркеров и (по желанию) контрольный цикл.

        ``autopilot=False`` — только воркеры: состав затем меняется вручную через
        :meth:`autoscale_once` (так тесты и отладочные сценарии проверяют решения
        детерминированно, без гонки с фоновым потоком).
        """
        self._stop_event = threading.Event()
        for _ in range(self._config.min_workers):
            self._spawn(reason="старт")
        if autopilot:
            self._thread = threading.Thread(
                target=self.run,
                args=(self._stop_event,),
                daemon=True,
                name=f"consumer-pool-{self._profile}",
            )
            self._thread.start()
        logger.info(
            "Пул «%s» запущен: %d воркер(ов), диапазон %d..%d%s%s",
            self._profile, self.size,
            self._config.min_workers, self._config.max_workers,
            "" if self._config.autoscale else " (автоскейл выключен)",
            "" if autopilot else " (без контрольного цикла)",
        )
        return self

    def stop(self, *, timeout: float = 7.0) -> None:
        """Остановить контрольный цикл и всех воркеров (graceful)."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            # join — до него autoscale_once может менять состав; после — только мы.
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Пул «%s»: контрольный цикл не остановился за %.1f с", self._profile, timeout)
        for worker in list(self._workers):
            worker.consumer.stop(timeout=3.0)
        self._workers.clear()
        logger.info(
            "Пул «%s» остановлен (%d воркер(ов); up=%d, down=%d)",
            self._profile, self.scale_ups + self._config.min_workers, self.scale_ups, self.scale_downs,
        )

    def run(self, stop_event: threading.Event) -> None:
        """Контрольный цикл: решение по составу раз в ``check_interval_s``."""
        while not stop_event.is_set():
            try:
                self.autoscale_once()
            except Exception as exc:  # noqa: BLE001 — цикл обязан выживать
                logger.error("Пул «%s»: ошибка автопарка — %s", self._profile, exc)
            stop_event.wait(self._config.check_interval_s)

    # ------------------------------------------------------------------ #
    #  Одно решение (тестируется без потоков)
    # ------------------------------------------------------------------ #
    def autoscale_once(self, *, now: Optional[float] = None) -> Optional[str]:
        """Одно решение по составу пула.

        Возвращает ``\"up\"``/``\"down\"`` при изменении состава, иначе ``None``.
        Счётчики ``processed`` снимаются в конце прохода — дельта за прошедший интервал
        показывает, кто простаивал.
        """
        now = self._clock() if now is None else now
        decision: Optional[str] = None

        if self._config.autoscale:
            backlog = self._backlog()
            if (
                backlog > self._config.high_watermark
                and self.size < self._config.max_workers
                and self._cooldown_passed(now)
            ):
                self._spawn(reason=f"backlog {backlog} > {self._config.high_watermark}")
                self.scale_ups += 1
                self._last_change = now
                decision = "up"
            elif (
                backlog <= 0
                and self.size > self._config.min_workers
                and self._cooldown_passed(now)
            ):
                idle = self._pick_idle()
                if idle is not None:
                    self._despawn(idle, reason="очередь пуста, воркер простаивал")
                    self.scale_downs += 1
                    self._last_change = now
                    decision = "down"

        # Снимок счётчиков — после решения: следующая дельта считается от него.
        self._last_processed = {w.name: w.consumer.stats.processed for w in self._workers}
        return decision

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _cooldown_passed(self, now: float) -> bool:
        return self._last_change is None or (now - self._last_change) >= self._config.cooldown_s

    def _backlog(self) -> int:
        """Суммарная глубина очередей профиля (со стрима любого воркера — kinds общие)."""
        if not self._workers:
            return 0
        depth = self._workers[0].port.depth()
        return int(sum(depth.values()))

    def _pick_idle(self) -> Optional[_Worker]:
        """Воркер-кандидат на остановку: минимальная дельта ``processed`` за интервал.

        Занятого воркера не трогаем: если все что-то обработали — пул не сокращается
        (лучше подождать следующий тик, чем прервать работу). При равных дельтах
        останавливаем последнего добавленного — первые воркеры живут долго.
        """
        if not self._workers:
            return None
        deltas = [
            (w, w.consumer.stats.processed - self._last_processed.get(w.name, w.consumer.stats.processed))
            for w in self._workers
        ]
        idle = [(w, d) for w, d in deltas if d <= 0]
        if not idle:
            return None
        return min(idle, key=lambda pair: (pair[1], -self._workers.index(pair[0])))[0]

    def _spawn(self, *, reason: str) -> None:
        self._next_index += 1
        name = f"{self._profile}-{self._next_index}"
        port = self._port_factory(name)
        consumer = TaskConsumer(port, handler=self._handler)
        consumer.start(name=f"task-consumer-{name}")
        self._workers.append(_Worker(name=name, consumer=consumer, port=port))
        self._last_processed[name] = 0
        logger.info("Пул «%s»: +воркер %s (%d/%d) — %s", self._profile, name, self.size, self._config.max_workers, reason)

    def _despawn(self, worker: _Worker, *, reason: str) -> None:
        self._workers.remove(worker)
        self._last_processed.pop(worker.name, None)
        worker.consumer.stop(timeout=3.0)
        logger.info("Пул «%s»: −воркер %s (%d/%d) — %s", self._profile, worker.name, self.size, self._config.max_workers, reason)


__all__ = [
    "DEFAULT_CHECK_INTERVAL_S",
    "DEFAULT_COOLDOWN_S",
    "DEFAULT_HIGH_WATERMARK",
    "ConsumerPool",
    "PoolConfig",
    "parse_profiles",
]
