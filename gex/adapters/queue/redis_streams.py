"""Redis Streams — единственная очередь задач (ring: adapters).

Порт: :mod:`gex.ports.job_queue`. Что даёт Streams против прежних списков
-----------------------------------------------------------------------
``task_queue.py`` использовал ``LPUSH``/``BRPOP``: сообщение **исчезало из Redis до**
запуска обработчика. Падение воркера посреди фетча означало безвозвратную потерю задачи,
а необработанный сбой оставался только в логе. Контракт порта требует другого:
at-least-once, DLQ и дедупликацию по ``idempotency_key``. Streams с группой потребителей
даёт это штатно: сообщение подтверждается (``XACK``) уже **после** успешной обработки,
зависшее забирается повторно (``XAUTOCLAIM``), а провал уходит в отдельный поток DLQ.

Приоритет
---------
Контракт: «приоритет влияет на порядок разбора (``interactive`` важнее ``background``)».
Поэтому у вида задачи **два** потока, и ``read()`` сначала читает интерактивные и только
если они пусты — фоновые. Читать оба одним ``XREADGROUP`` было бы проще, но тогда порядок
определял бы Redis, а не контракт.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from gex.ports.job_queue import Delivery, Job, JobResult, Priority
from typing import Protocol

logger = logging.getLogger(__name__)

#: Префикс потока задач: ``gex:q:{kind}`` (interactive) и ``gex:q:{kind}:bg`` (background).
STREAM_PREFIX = "gex:q:"

#: Префикс потока «мёртвых» задач.
DLQ_PREFIX = "gex:q:dlq:"

#: Префикс маркера дедупликации (``SET NX EX`` по ``idempotency_key``).
DEDUP_PREFIX = "gex:q:dedup:"

#: Суффикс фонового потока (приоритет ``background``).
BACKGROUND_SUFFIX = ":bg"

#: Имя группы потребителей по умолчанию: одно на все потоки, чтобы воркеры делили работу.
DEFAULT_GROUP = "gex-workers"

#: Ограничение длины потока: очередь не должна расти бесконечно при остановленном
#: потребителе.
DEFAULT_MAXLEN = 10_000

#: Срок жизни маркера дедупликации: столько задача считается «уже выполнялась».
DEFAULT_DEDUP_TTL = 300

#: Сколько ждать «зависшее» сообщение, прежде чем забрать его у мёртвого потребителя.
DEFAULT_CLAIM_IDLE_MS = 60_000


class StreamsRedis(Protocol):
    """Минимум клиента Redis, нужный очереди.

    Протокол вместо нетипизированного параметра: он и документирует контракт, и позволяет
    статически поймать «адаптер зовёт метод, которого у клиента нет» — этот класс
    дефекта встречался трижды (``nx``/``px``, ``script_load``). Полный список методов
    зависит от наследника (``StreamsClientMixin``), поэтому здесь только то, что
    использует очередь.
    """

    def xadd(self, name: str, fields: dict, *, maxlen: Optional[int] = None) -> Optional[str]: ...
    def xgroup_create(self, name: str, group: str, *, mkstream: bool = True) -> bool: ...
    def xreadgroup(self, group: str, consumer: str, streams: dict, *, count: int = 10,
                   block_ms: Optional[int] = None) -> list: ...
    def xack(self, name: str, group: str, *message_ids: str) -> int: ...
    def xautoclaim(self, name: str, group: str, consumer: str, *, min_idle_ms: int,
                   count: int = 10) -> list: ...
    def xlen(self, name: str) -> int: ...
    def xpending(self, name: str, group: str) -> dict: ...
    def delete(self, key: str) -> bool: ...
    def set(self, key: str, value: object, ex: Optional[int] = None, nx: bool = False) -> Optional[bool]: ...


def stream_name(kind: str, priority: Priority = Priority.INTERACTIVE) -> str:
    """Имя потока для вида задачи и приоритета.

    >>> stream_name("ohlcv", Priority.INTERACTIVE)
    'gex:q:ohlcv'
    >>> stream_name("ohlcv", Priority.BACKGROUND)
    'gex:q:ohlcv:bg'
    """
    base = f"{STREAM_PREFIX}{kind}"
    return f"{base}{BACKGROUND_SUFFIX}" if priority == Priority.BACKGROUND else base


def dlq_name(kind: str) -> str:
    """Имя потока DLQ для вида задачи."""
    return f"{DLQ_PREFIX}{kind}"


def base_kind(stream: str) -> str:
    """Вид задачи из имени потока (обратная операция к :func:`stream_name`).

    >>> base_kind("gex:q:ohlcv:bg")
    'ohlcv'
    """
    kind = stream[len(STREAM_PREFIX):] if stream.startswith(STREAM_PREFIX) else stream
    if kind.endswith(BACKGROUND_SUFFIX):
        kind = kind[: -len(BACKGROUND_SUFFIX)]
    return kind


def dedup_key(idempotency_key: str) -> str:
    return f"{DEDUP_PREFIX}{idempotency_key}"


@dataclass
class QueueStats:
    """Наблюдаемость очереди: длина потока и число неподтверждённых сообщений."""

    depth: dict[str, int] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.depth.values())

    @property
    def total_pending(self) -> int:
        return sum(self.pending.values())


class StreamJobQueue:
    """Очередь задач на Redis Streams (реализация :class:`JobQueuePort`).

    Все потоки читаются одной группой: несколько воркеров делят работу между собой,
    а не дублируют её.
    """

    def __init__(
        self,
        redis: Optional[StreamsRedis],
        *,
        kinds: Iterable[str],
        group: str = DEFAULT_GROUP,
        consumer: str = "worker",
        maxlen: int = DEFAULT_MAXLEN,
        dedup_ttl: int = DEFAULT_DEDUP_TTL,
        clock=time.monotonic,
    ) -> None:
        self._redis = redis
        self._kinds = tuple(dict.fromkeys(kinds))  # уникальные, порядок сохраняем
        if not self._kinds:
            raise ValueError("нужен хотя бы один вид задачи: иначе очередь нечего читать")
        self._group = group
        self._consumer = consumer
        self._maxlen = maxlen
        self._dedup_ttl = dedup_ttl
        self._clock = clock
        self.published = 0
        self.acked = 0
        self.duplicates_skipped = 0
        self.dead_lettered = 0

    # ------------------------------------------------------------------ #
    #  Служебное
    # ------------------------------------------------------------------ #
    def _ready(self) -> bool:
        """Есть ли рабочий Redis.

        ``None`` означает «очередь не сконфигурирована» — это не исключение: приложение
        обязано подниматься и без Redis, а очередь просто сообщает, что недоступна.
        """
        return self._redis is not None

    def available(self) -> bool:
        """Очередь реально работает: есть клиент **и** он подключён.

        Отличается от :meth:`_ready`: то отвечает «клиент сконфигурирован», а не
        «соединение живо». Когда Redis удалён или лежит, разница видна в логах:
        ``_ready() == True`` (клиент есть), но каждая публикация молча возвращает
        ``None``, и вызывающий не может отличить «задача не поставлена» от
        «очереди нет вообще».
        """
        if self._redis is None:
            return False
        connected = getattr(self._redis, "connected", None)
        if connected is None:
            # Клиент без признака подключения (тестовый/другой транспорт) — считаем рабочим.
            return True
        return bool(connected)

    @property
    def kinds(self) -> tuple[str, ...]:
        return self._kinds

    def all_streams(self) -> tuple[str, ...]:
        """Все потоки очереди (оба приоритета), в порядке разбора: interactive → bg."""
        return tuple(
            stream_name(kind, priority)
            for priority in (Priority.INTERACTIVE, Priority.BACKGROUND)
            for kind in self._kinds
        )

    # ------------------------------------------------------------------ #
    #  Публикация
    # ------------------------------------------------------------------ #
    def publish(self, job: Job, *, queue: Optional[str] = None) -> Optional[str]:
        """Поставить задачу. Возвращает id сообщения или ``None`` при отказе.

        ``queue`` — вид задачи (``ohlcv``/``chain``/…). Провайдер и параметры уезжают
        в ``payload``; вид задачи остаётся в имени потока, чтобы потребитель не разбирал
        JSON ради маршрутизации.
        """
        if not self._ready():
            return None
        kind = queue or job.task_type
        if kind not in self._kinds:
            logger.warning("Очередь %s не объявлена (известные: %s)", kind, list(self._kinds))
            return None
        stream = stream_name(kind, job.priority)
        payload = {
            "task_type": job.task_type,
            "idempotency_key": job.idempotency_key,
            "provider": job.provider or "",
            "payload": json.dumps(job.payload, ensure_ascii=False),
        }
        message_id = self._redis.xadd(stream, payload, maxlen=self._maxlen)
        if message_id is None:
            return None
        self.published += 1
        return message_id

    # ------------------------------------------------------------------ #
    #  Разбор
    # ------------------------------------------------------------------ #
    def ensure_groups(self) -> int:
        """Создать группу потребителей на всех потоках (идемпотентно)."""
        if not self._ready():
            return 0
        created = 0
        for stream in self.all_streams():
            if self._redis.xgroup_create(stream, self._group, mkstream=True):
                created += 1
        for kind in self._kinds:
            self._redis.xgroup_create(dlq_name(kind), self._group, mkstream=True)
        return created

    def read(self, *, count: int = 10, block_ms: Optional[int] = None) -> list[Delivery]:
        """Прочитать новые сообщения: сначала interactive, потом background.

        Интерактивные (пользователь ждёт) разбираются первыми — это и есть реализация
        приоритета; читать оба потока одним запросом означало бы отдать порядок Redis.
        """
        for priority in (Priority.INTERACTIVE, Priority.BACKGROUND):
            streams = {stream_name(kind, priority): ">" for kind in self._kinds}
            deliveries = self._read_streams(streams, count=count, block_ms=block_ms)
            if deliveries:
                return deliveries
        return []

    def _read_streams(self, streams: dict, *, count: int, block_ms: Optional[int]) -> list[Delivery]:
        if not self._ready():
            return []
        response = self._redis.xreadgroup(
            self._group, self._consumer, streams, count=count, block_ms=block_ms
        )
        return [
            delivery
            for stream, messages in response
            for delivery in (
                self._to_delivery(stream, message_id, fields)
                for message_id, fields in messages
            )
            if delivery is not None
        ]

    def _to_delivery(self, stream: str, message_id: str, fields: dict) -> Optional[Delivery]:
        """Поля сообщения → :class:`Job`. Неразбираемое сообщение вернёт ``None``."""
        try:
            raw = fields.get("payload") or "{}"
            payload = json.loads(raw) if isinstance(raw, str) else {}
            priority = (
                Priority.BACKGROUND if stream.endswith(BACKGROUND_SUFFIX) else Priority.INTERACTIVE
            )
            job = Job(
                task_type=str(fields.get("task_type") or base_kind(stream)),
                idempotency_key=str(fields.get("idempotency_key") or ""),
                payload=payload if isinstance(payload, dict) else {},
                priority=priority,
                provider=str(fields.get("provider") or "") or None,
            )
        except (ValueError, TypeError) as exc:
            # Подтверждать такое сообщение нельзя (оно потеряется молча), но и читать
            # его бесконечно тоже нельзя: потребитель отправит его в DLQ.
            logger.warning("Сообщение %s в %s не разбирается: %s", message_id, stream, exc)
            return None
        return Delivery(stream=stream, message_id=message_id, job=job)

    def ack(self, delivery: Delivery) -> bool:
        """Подтвердить обработку (``XACK``). Без этого сообщение остаётся выданным."""
        if not self._ready():
            return False
        ok = bool(self._redis.xack(delivery.stream, self._group, delivery.message_id))
        if ok:
            self.acked += 1
        return ok

    def dead_letter(self, delivery: Delivery, error: str) -> bool:
        """Отправить задачу в DLQ и подтвердить исходное сообщение.

        Порядок важен: сначала запись в DLQ, потом ack. Обратный порядок потерял бы
        задачу, если процесс упадёт между двумя вызовами.
        """
        if not self._ready():
            return False
        stream = dlq_name(base_kind(delivery.stream))
        fields = {
            "task_type": delivery.job.task_type,
            "idempotency_key": delivery.job.idempotency_key,
            "provider": delivery.job.provider or "",
            "payload": json.dumps(delivery.job.payload, ensure_ascii=False),
            "error": str(error)[:500],
            "source_stream": delivery.stream,
            "source_id": delivery.message_id,
        }
        if self._redis.xadd(stream, fields, maxlen=1000) is None:
            logger.error("DLQ недоступна: задача %s остаётся выданной", delivery.message_id)
            return False
        self.dead_lettered += 1
        self.ack(delivery)
        return True

    def claim_stale(
        self, *, min_idle_ms: int = DEFAULT_CLAIM_IDLE_MS, count: int = 10
    ) -> list[Delivery]:
        """Забрать сообщения, у которых потребитель умер, не подтвердив их.

        Вторая половина at-least-once: без повторного захвата задача, выданная упавшему
        воркеру, осталась бы выданной навсегда.
        """
        if not self._ready():
            return []
        claimed: list[Delivery] = []
        for stream in self.all_streams():
            for message_id, fields in self._redis.xautoclaim(
                stream, self._group, self._consumer, min_idle_ms=min_idle_ms, count=count
            ):
                delivery = self._to_delivery(stream, message_id, fields)
                if delivery is not None:
                    claimed.append(
                        Delivery(
                            stream=delivery.stream,
                            message_id=delivery.message_id,
                            job=delivery.job,
                            redelivered=True,
                        )
                    )
        if claimed:
            logger.info("Очередь: перехвачено зависших сообщений: %d", len(claimed))
        return claimed

    # ------------------------------------------------------------------ #
    #  Дедупликация (требование контракта)
    # ------------------------------------------------------------------ #
    def should_process(self, job: Job, *, ttl: Optional[int] = None) -> bool:
        """Первый ли это раз, когда мы видим ``idempotency_key``.

        at-least-once означает возможную повторную доставку, поэтому обработчик обязан
        быть идемпотентным; маркер ``SET NX EX`` делает это без изменений в обработчике.
        """
        if not self._ready():
            return True  # очередь недоступна: не мешаем обработке
        if not job.idempotency_key:
            return True  # без ключа дедуплицировать нечего — обрабатываем
        first = self._redis.set(dedup_key(job.idempotency_key), "1", ex=ttl or self._dedup_ttl, nx=True)
        if not first:
            self.duplicates_skipped += 1
            logger.debug("Дубликат задачи %s — пропускаю", job.idempotency_key)
        return bool(first)

    def forget(self, job: Job) -> bool:
        """Снять маркер дедупликации (нужна повторная попытка после провала)."""
        if not self._ready() or not job.idempotency_key:
            return False
        return bool(self._redis.delete(dedup_key(job.idempotency_key)))

    # ------------------------------------------------------------------ #
    #  Наблюдаемость (JobQueuePort)
    # ------------------------------------------------------------------ #
    def _streams_for(self, queue: Optional[str]) -> tuple[str, ...]:
        if queue:
            return tuple(
                stream_name(queue, p) for p in (Priority.INTERACTIVE, Priority.BACKGROUND)
            )
        return self.all_streams()

    def depth(self, queue: Optional[str] = None) -> dict[str, int]:
        """Глубина потоков: имя → число сообщений."""
        if not self._ready():
            return {}
        return {stream: self._redis.xlen(stream) for stream in self._streams_for(queue)}

    def pending(self, queue: Optional[str] = None) -> dict[str, int]:
        """Выданные и не подтверждённые сообщения (индикатор зависших задач)."""
        if not self._ready():
            return {}
        out: dict[str, int] = {}
        for stream in self._streams_for(queue):
            summary = self._redis.xpending(stream, self._group)
            out[stream] = int(summary.get("pending", 0)) if isinstance(summary, dict) else 0
        return out

    def dead_letters(self, queue: Optional[str] = None) -> int:
        """Сколько задач в DLQ."""
        if not self._ready():
            return 0
        kinds = (queue,) if queue else self._kinds
        return sum(self._redis.xlen(dlq_name(kind)) for kind in kinds)

    def stats(self) -> QueueStats:
        return QueueStats(depth=self.depth(), pending=self.pending())

    def clear(self, queue: Optional[str] = None) -> None:
        """Очистить потоки (тесты и ручное вмешательство)."""
        if not self._ready():
            return
        for stream in self._streams_for(queue):
            self._redis.delete(stream)
        for kind in ((queue,) if queue else self._kinds):
            self._redis.delete(dlq_name(kind))

    def describe(self) -> dict:
        """Сводка для логов и админки."""
        stats = self.stats()
        return {
            "available": self.available(),
            "kinds": list(self._kinds),
            "group": self._group,
            "consumer": self._consumer,
            "depth": stats.depth,
            "pending": stats.pending,
            "dead_letters": self.dead_letters(),
            "published": self.published,
            "acked": self.acked,
            "duplicates_skipped": self.duplicates_skipped,
            "dead_lettered": self.dead_lettered,
        }


def make_result(skipped: bool = False, error: Optional[str] = None) -> JobResult:
    """Результат обработки (для метрик и DLQ-записи)."""
    return JobResult(ok=error is None, error=error, skipped=skipped)


__all__ = [
    "BACKGROUND_SUFFIX",
    "DEDUP_PREFIX",
    "DEFAULT_CLAIM_IDLE_MS",
    "DEFAULT_DEDUP_TTL",
    "DEFAULT_GROUP",
    "DEFAULT_MAXLEN",
    "DLQ_PREFIX",
    "STREAM_PREFIX",
    "QueueStats",
    "StreamsRedis",
    "StreamJobQueue",
    "base_kind",
    "dedup_key",
    "dlq_name",
    "make_result",
    "stream_name",
]
