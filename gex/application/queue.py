"""Публикация задач в очередь: фасад «задача → сообщение» (ring: application).

Зачем фасад
-----------
Планировщик и админка работают с :class:`~gex.application.jobs.FetchTask` — им не нужно
знать ни про ``Job``, ни про выбор потока, ни про приоритеты транспорта. Преобразование
живёт здесь, поэтому модель задачи (:mod:`gex.application.jobs`) остаётся описанием работы,
а транспорт — деталью за портом.

Совместимость имён
------------------
``publish_many``/``queue_length``/``clear_queues`` сохранены: на них завязаны планировщик,
админка, метрики и ручка фетчера. Это не «наследие ради наследия» — замена транспорта
не должна тянуть за собой переписывание вызывающих, у которых нет другой причины меняться.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

from gex.application.jobs import FetchTask, task_to_job
from gex.ports.job_queue import JobQueuePort

logger = logging.getLogger(__name__)

#: Как часто повторять предупреждение «очередь недоступна», секунды.
#: Планировщик публикует десятки задач за проход: без ограничителя каждая дала бы
#: отдельную строку с одним и тем же текстом. Так было, пока Redis просто лежал —
#: а после его удаления это стало единственным, что видно в логе каждые 5 минут.
_UNAVAILABLE_LOG_INTERVAL_S = 300.0
_last_unavailable_log = 0.0
_unavailable_suppressed = 0


def _warn_unavailable(queue: str = "", *, clock=time.monotonic) -> None:
    """Сообщить об отключённой очереди не чаще раза в интервал; остальное — в счётчик."""
    global _last_unavailable_log, _unavailable_suppressed  # noqa: PLW0603

    now = clock()
    if now - _last_unavailable_log < _UNAVAILABLE_LOG_INTERVAL_S:
        _unavailable_suppressed += 1
        return
    suppressed = _unavailable_suppressed
    _unavailable_suppressed = 0
    _last_unavailable_log = now
    suffix = f" (подавлено повторов: {suppressed})" if suppressed else ""
    logger.warning(
        "Очередь задач недоступна (нет Redis) — задачи не ставятся%s%s",
        f" [очередь: {queue}]" if queue else "", suffix,
    )


class TaskPublisher:
    """Публикация :class:`FetchTask` через :class:`JobQueuePort`."""

    def __init__(self, port: JobQueuePort, *, queue: Optional[str] = None) -> None:
        self._port = port
        self._default_queue = queue
        self.published = 0

    @property
    def port(self) -> JobQueuePort:
        """Очередь-порт: нужна композиционному корню, чтобы собрать потребителя."""
        return self._port

    def publish(self, task: FetchTask, *, queue: Optional[str] = None) -> bool:
        """Поставить задачу. ``False`` — очередь недоступна (не исключение).

        Планировщик публикует десятки задач за проход: отказ одной не должен прерывать
        остальные и ронять поток планировщика.
        """
        target = queue or self._default_queue or task.queue
        try:
            message_id = self._port.publish(task_to_job(task), queue=target)
        except Exception as exc:  # noqa: BLE001 — отказ публикации не критичен для цикла
            logger.warning("Публикация задачи %s/%s не удалась: %s", task.task_type, task.ticker, exc)
            return False
        if message_id is None:
            if self._port_available():
                logger.warning(
                    "Задача %s/%s не поставлена (очередь %s)", task.task_type, task.ticker, target,
                )
            else:
                # Очереди нет вообще (Redis удалён/лежит) — это одно сообщение на
                # интервал, а не по строке на каждую задачу планировщика.
                _warn_unavailable(target)
            return False
        self.published += 1
        return True

    def _port_available(self) -> bool:
        """Работает ли очередь (если порт умеет отвечать — иначе считаем, что работает)."""
        available = getattr(self._port, "available", None)
        if available is None:
            return True
        try:
            return bool(available())
        except Exception:  # noqa: BLE001 — метрика доступности не должна ломать публикацию
            return True

    def publish_many(self, tasks: Iterable[FetchTask]) -> int:
        """Опубликовать список задач. Возвращает число успешных."""
        return sum(1 for task in tasks if self.publish(task))

    # ------------------------------------------------------------------ #
    #  Наблюдаемость и обслуживание (используют админка и метрики)
    # ------------------------------------------------------------------ #
    def queue_length(self, queue: str = "") -> int:
        """Сколько сообщений ждёт разбора (по всем очередям, если имя не задано)."""
        try:
            return sum(self._port.depth(queue or None).values())
        except Exception as exc:  # noqa: BLE001 — метрика не должна ронять запрос
            logger.debug("Глубина очереди недоступна: %s", exc)
            return 0

    def pending(self, queue: str = "") -> int:
        """Сколько сообщений выдано и не подтверждено (признак зависших задач)."""
        pending_fn = getattr(self._port, "pending", None)
        if pending_fn is None:
            return 0
        try:
            return sum(pending_fn(queue or None).values())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pending недоступен: %s", exc)
            return 0

    def dead_letters(self, queue: str = "") -> int:
        """Сколько задач в DLQ."""
        try:
            return int(self._port.dead_letters(queue or None))
        except Exception as exc:  # noqa: BLE001
            logger.debug("DLQ недоступна: %s", exc)
            return 0

    def clear_queues(self) -> None:
        """Очистить очереди (ручка админки и тесты)."""
        try:
            clear = getattr(self._port, "clear", None)
            if clear is not None:
                clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Очистка очередей не удалась: %s", exc)

    def describe(self) -> dict:
        """Сводка состояния очереди (админка/метрики)."""
        return {
            "length": self.queue_length(),
            "pending": self.pending(),
            "dead_letters": self.dead_letters(),
            "published": self.published,
        }


__all__ = ["TaskPublisher"]
