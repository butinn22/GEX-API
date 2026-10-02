"""Порт очереди фоновых задач.

Один транспорт вместо двух: списки ``gex:queue:*`` (``task_queue.py``) выведены в итер. 28,
реализация — Redis Streams (``gex/adapters/queue/redis_streams.py``); RabbitMQ — при
срабатывании триггеров из ROADMAP v2 §1.3.

Гарантии контракта:
  * доставка **at-least-once** — сообщение подтверждается **после** обработки, а не при
    выдаче, поэтому падение воркера не теряет задачу; обработчик обязан быть идемпотентным
    по ``job.idempotency_key`` (дедупликация — ``should_process``);
  * есть DLQ для задач, которые не удалось обработать;
  * приоритет влияет на порядок разбора (``interactive`` важнее ``background``).

Почему в порту есть методы выдачи
--------------------------------
Первая версия порта описывала только ``publish``/``depth``/``dead_letters``. Этого
недостаточно, чтобы выполнить собственное обещание: «at-least-once» невозможно выразить
без ``ack`` (подтвердить после успеха) и ``claim_stale`` (перехватить то, что осталось
у умершего потребителя). Порт, который не может выразить свой контракт, — это подсказка
реализации, а не контракт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = ["Priority", "JobResult", "Job", "Delivery", "JobQueuePort"]


class Priority(str, Enum):
    """Приоритет задачи: интерактивные (пользователь ждёт) важнее фоновых (прогрев)."""

    INTERACTIVE = "interactive"
    BACKGROUND = "background"


@dataclass(frozen=True)
class Job:
    """Задача для фонового исполнителя.

    ``idempotency_key`` — обязателен: без него повторная доставка (at-least-once) приведёт к
    дублированию работы, а для слотов вида «IMOEX breadth 23:00» — к двойному пересчёту.
    Дедупликация — на стороне потребителя через ``SET NX EX`` маркер.
    """

    task_type: str
    idempotency_key: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.BACKGROUND
    provider: str | None = None


@dataclass(frozen=True)
class JobResult:
    """Результат выполнения задачи (для метрик и для DLQ-записи)."""

    ok: bool
    error: str | None = None
    skipped: bool = False  # задача не выполнялась (например, покрыта дедупликацией)


@dataclass(frozen=True)
class Delivery:
    """Выданное потребителю сообщение.

    ``message_id`` и ``stream`` нужны, чтобы подтвердить **именно** это сообщение: ``ack``
    принимает доставку, а не пару строк, которые легко перепутать местами.
    """

    stream: str
    message_id: str
    job: Job
    redelivered: bool = False


@runtime_checkable
class JobQueuePort(Protocol):
    """Публикация и разбор фоновых задач.

    Порт описывает обе стороны контракта: без ``read``/``ack``/``dead_letter`` его
    собственное обещание (at-least-once + DLQ) выразить нельзя — подтверждать сообщение
    можно только после успешной обработки, а «не подтверждённое» должно быть видно.
    """

    def publish(self, job: Job, *, queue: str | None = None) -> str | None:
        """Поставить задачу; вернуть идентификатор сообщения (``None`` при отказе, например queue full)."""
        ...

    def read(self, *, count: int = 10, block_ms: int | None = None) -> list[Delivery]:
        """Прочитать новые сообщения в порядке приоритета (интерактивные — первыми).

        Возвращает пустой список, если сообщений нет: это нормальный результат цикла,
        а не ошибка.
        """
        ...

    def ack(self, delivery: Delivery) -> bool:
        """Подтвердить обработку. Без подтверждения сообщение остаётся выданным и вернётся."""
        ...

    def dead_letter(self, delivery: Delivery, error: str) -> bool:
        """Отправить необработанную задачу в DLQ и подтвердить исходное сообщение."""
        ...

    def claim_stale(self, *, min_idle_ms: int = 60_000, count: int = 10) -> list[Delivery]:
        """Забрать сообщения, выданные потребителю, который умер, не подтвердив их."""
        ...

    def should_process(self, job: Job, *, ttl: int | None = None) -> bool:
        """Первый ли это раз, когда мы видим ``idempotency_key`` (дедупликация at-least-once)."""
        ...

    def depth(self, queue: str | None = None) -> dict[str, int]:
        """Глубина очередей (для наблюдаемости и админки)."""
        ...

    def dead_letters(self, queue: str | None = None) -> int:
        """Сколько задач ушло в DLQ."""
        ...

    def clear(self, queue: str | None = None) -> None:
        """Очистить очереди (ручка админки и тесты)."""
        ...
