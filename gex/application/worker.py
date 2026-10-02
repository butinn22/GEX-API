"""Разбор очереди задач: цикл потребителя (ring: application).

Что здесь и почему
------------------
Раньше цикл потребителя был частью ``task_queue.py`` и был сшит с транспортом: ``BRPOP``,
переподключение Redis, чтение ``settings`` внутри цикла. Здесь остаётся **логика разбора**
(взять сообщение → проверить, не дубликат ли → вызвать обработчик → подтвердить или
отправить в DLQ → периодически перехватить зависшее), а транспорт приходит портом.

Что даёт переход на at-least-once
---------------------------------
Прежний цикл забирал задачу из списка (``BRPOP``) — сообщение исчезало из Redis **до**
обработки. Падение процесса посреди фетча означало безвозвратную потерю задачи, а ошибка
обработчика — только строчку в логе. Теперь: подтверждение после успеха, DLQ для отказов,
``claim_stale`` для задач, оставшихся у умершего воркера, и дедупликация по
``idempotency_key`` (иначе повторная выдача выполняла бы работу дважды).

``handle_once`` отделён от ``run`` намеренно: цикл в потоке проверять неудобно, а один
проход — обычная функция, и именно её покрывают тесты.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from gex.application.jobs import FetchTask, job_to_task
from gex.ports.job_queue import Delivery, JobQueuePort

logger = logging.getLogger(__name__)

#: Как часто перехватывать сообщения, оставшиеся у умерших потребителей.
DEFAULT_CLAIM_INTERVAL_S = 60.0

#: Пауза после холостого прохода очереди, секунды.
#: Зачем нужна: чтение может вернуться мгновенно без сообщений — не только когда очередь
#: пуста, но и когда транспорт отдаёт пустой результат вместо ошибки (адаптер глотает
#: исключение и возвращает ``[]``). Без паузы такой проход превращается в горячий цикл:
#: наблюдалось ~2 300 итераций в секунду и 59 МБ лога за 12 минут.
DEFAULT_IDLE_PAUSE_S = 1.0


@dataclass
class ConsumerStats:
    """Счётчики потребителя: без них деградация очереди не видна."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0
    dead_lettered: int = 0
    claimed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "dead_lettered": self.dead_lettered,
            "claimed": self.claimed,
            "errors": self.errors[-5:],
        }


class TaskConsumer:
    """Потребитель задач поверх :class:`JobQueuePort`.

    Parameters
    ----------
    port :
        Очередь (Redis Streams в проде).
    handler :
        Обработчик задачи. Исключение из него означает «задача не выполнена»: сообщение
        уходит в DLQ, а не теряется и не блокирует очередь.
    """

    def __init__(
        self,
        port: JobQueuePort,
        handler: Callable[[FetchTask], None],
        *,
        count: int = 10,
        block_ms: Optional[int] = 1000,
        claim_interval_s: float = DEFAULT_CLAIM_INTERVAL_S,
        min_idle_ms: int = 60_000,
        idle_pause_s: float = DEFAULT_IDLE_PAUSE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._port = port
        self._handler = handler
        self._count = count
        self._block_ms = block_ms
        self._claim_interval_s = claim_interval_s
        self._min_idle_ms = min_idle_ms
        self._idle_pause_s = idle_pause_s
        self._clock = clock
        self._last_claim = 0.0
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self.stats = ConsumerStats()

    # ------------------------------------------------------------------ #
    #  Один проход
    # ------------------------------------------------------------------ #
    def handle_once(self, *, count: Optional[int] = None, block_ms: Optional[int] = None) -> int:
        """Разобрать пачку сообщений. Возвращает число успешно обработанных.

        Зависшие сообщения перехватываются перед обычным чтением: задача, выданная
        упавшему воркеру, важнее новой — иначе очередь «залипает» в неподтверждённых.
        """
        self._maybe_claim()
        deliveries = self._port.read(
            count=count or self._count,
            block_ms=block_ms if block_ms is not None else self._block_ms,
        )
        handled = 0
        for delivery in deliveries:
            if self._process(delivery):
                handled += 1
        return handled

    def _maybe_claim(self) -> None:
        now = self._clock()
        if now - self._last_claim < self._claim_interval_s:
            return
        self._last_claim = now
        try:
            claimed = self._port.claim_stale(min_idle_ms=self._min_idle_ms, count=self._count)
        except Exception as exc:  # noqa: BLE001 — перехват не должен ронять цикл
            logger.warning("Перехват зависших задач не удался: %s", exc)
            return
        for delivery in claimed:
            self.stats.claimed += 1
            self._process(delivery)

    def _process(self, delivery: Delivery) -> bool:
        """Дедупликация → обработка → подтверждение или DLQ."""
        job = delivery.job
        try:
            if not self._port.should_process(job):
                # Повторная доставка уже выполненной задачи: подтверждаем и идём дальше.
                self.stats.skipped += 1
                self._port.ack(delivery)
                return False
        except Exception as exc:  # noqa: BLE001 — дедупликация недоступна
            # Redis не ответил на маркер: считаем задачу новой. Обработать дважды лучше,
            # чем не обработать вообще — обработчик идемпотентен по контракту.
            logger.warning("Дедупликация недоступна (%s) — обрабатываю как новую", exc)

        task = job_to_task(job)
        try:
            self._handler(task)
        except Exception as exc:  # noqa: BLE001 — отказ обработчика
            self.stats.failed += 1
            self.stats.errors.append(f"{task.task_type}/{task.ticker}: {exc}")
            logger.error(
                "Задача не выполнена: %s/%s/%s — %s",
                task.task_type, task.provider, task.ticker, exc,
            )
            try:
                if self._port.dead_letter(delivery, str(exc)):
                    self.stats.dead_lettered += 1
            except Exception as dlq_exc:  # noqa: BLE001
                logger.error("DLQ недоступна: %s — сообщение останется выданным", dlq_exc)
            return False

        self.stats.processed += 1
        self._port.ack(delivery)
        return True

    # ------------------------------------------------------------------ #
    #  Цикл
    # ------------------------------------------------------------------ #
    def run(self, stop_event: threading.Event) -> None:
        """Крутить разбор до ``stop_event``. Ошибки транспорта не убивают цикл."""
        logger.info("TaskConsumer запущен (порт: %s)", type(self._port).__name__)
        while not stop_event.is_set():
            try:
                handled = self.handle_once()
            except Exception as exc:  # noqa: BLE001 — цикл обязан выживать
                logger.warning("Ошибка цикла очереди: %s — пауза 5 с", exc)
                stop_event.wait(5)
                continue
            if not handled:
                # Холостой проход: пауза обязательна. `stop_event.wait` (а не `sleep`) — чтобы
                # остановка оставалась мгновенной.
                stop_event.wait(self._idle_pause_s)
        logger.info("TaskConsumer остановлен: %s", self.stats.as_dict())

    def start(self, *, name: str = "task-consumer") -> threading.Thread:
        """Запустить цикл в демон-потоке. Остановка — :meth:`stop`."""
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self.run, args=(self._stop_event,), daemon=True, name=name
        )
        self._thread.start()
        return self._thread

    def stop(self, *, timeout: float = 5.0) -> None:
        """Остановить цикл и дождаться потока."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


__all__ = ["DEFAULT_CLAIM_INTERVAL_S", "DEFAULT_IDLE_PAUSE_S", "ConsumerStats", "TaskConsumer"]
