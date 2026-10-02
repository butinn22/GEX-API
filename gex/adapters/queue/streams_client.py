"""Streams-операции клиента Redis — очередь задач (ring: adapters).

Зачем отдельным модулем
-----------------------
``RedisClient`` перешёл порог «god-класса» (>300 строк), когда в него добавили очереди:
Streams — отдельная группа операций (потоки, группы потребителей, подтверждения), и держать
её в общем файле значило растить класс, который и без того самый большой в проекте.
Миксин сохраняет привычный вызов ``client.xadd(...)`` (вызывающие не меняются), но держит
эти девять методов отдельно.

Типы полей
----------
Redis принимает в полях потока только строки и числа, поэтому объект сериализуется в JSON
на этой границе: ``_encode_field`` — перед отправкой, ``_decode_field`` — при чтении
(Redis отдаёт bytes).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

#: Те же имена исключений, что использует :class:`gex.redis_client.RedisClient`.
#: Миксин может импортироваться раньше клиента (клиент импортирует его), поэтому
#: подстраховка нужна и здесь: без неё миксин не импортировался бы в окружении без redis.
try:  # pragma: no cover — зависит от наличия пакета redis
    from redis import RedisError, ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
except ImportError:  # pragma: no cover
    RedisError = Exception  # type: ignore[misc,assignment]
    RedisConnectionError = Exception  # type: ignore[misc,assignment]
    RedisTimeoutError = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

__all__ = ["StreamsClientMixin", "_decode_field", "_encode_field"]


def _encode_field(value: Any) -> Any:
    """Поле stream: строки/числа как есть, остальное — JSON."""
    if isinstance(value, (str, bytes, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _decode_field(value: Any) -> str:
    """Поле stream в строку (Redis отдаёт bytes)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else str(value)


#: Как часто повторять предупреждение о постоянной ошибке контейнера, секунды.
#: Одинаковые строки в цикле давали 59 МБ лога за 12 минут; сигнал при этом не усиливался.
_ERROR_LOG_INTERVAL_S = 30.0
_error_log_state: dict[str, float] = {}
_error_log_suppressed: dict[str, int] = {}


#: ``XAUTOCLAIM`` появился только в Redis 6.2. На более старых серверах команда
#: неизвестна, и каждый проход цикла потребления давал бы одно и то же
#: предупреждение (``unknown command 'XAUTOCLAIM'``) — поэтому возможность
#: запоминается навсегда, а перехват идёт по старому пути: ``XPENDING`` + ``XCLAIM``
#: (есть с Redis 5.0). Флаг на уровне модуля: он описывает сервер, а не клиент.
_XAUTOCLAIM_UNSUPPORTED = False


def _is_unknown_command(exc: Exception) -> bool:
    """Сервер не знает команду (старая версия Redis, а не сбой соединения)."""
    return "unknown command" in str(exc).lower()


def reset_capability_cache() -> None:
    """Сбросить remembered-возможности сервера (для тестов и переподключения)."""
    global _XAUTOCLAIM_UNSUPPORTED  # noqa: PLW0603
    _XAUTOCLAIM_UNSUPPORTED = False


def _log_transport_error(key: str, exc: Exception, *, clock=None) -> None:
    """Записать сбой транспорта не чаще интервала; остальное — в счётчик подавленных.

    Сигнал сохраняется (первый сбой виден сразу, далее — раз в интервал с числом
    подавленных), а флуд исчезает. Молчаливое подавление было бы хуже: сбой очереди
    перестал бы быть заметным вообще.
    """
    import time as _time

    now = (clock or _time.monotonic)()
    last = _error_log_state.get(key)
    if last is not None and now - last < _ERROR_LOG_INTERVAL_S:
        _error_log_suppressed[key] = _error_log_suppressed.get(key, 0) + 1
        return
    suppressed = _error_log_suppressed.pop(key, 0)
    _error_log_state[key] = now
    if suppressed:
        logger.warning("Redis %s error: %s (подавлено повторов: %d)", key, exc, suppressed)
    else:
        logger.warning("Redis %s error: %s", key, exc)


class StreamsClientMixin:
    """Потоки Redis для очереди задач: at-least-once, подтверждения, DLQ.

    Требует от наследника полей ``_conn`` и ``_connected`` и исключений
    ``RedisError``/``RedisConnectionError``/``RedisTimeoutError`` в пространстве имён:
    миксин рассчитан только на :class:`gex.redis_client.RedisClient`.
    """

    _conn: Any
    _connected: bool

    # ------------------------------------------------------------------ #
    #  Streams: очередь задач (at-least-once + DLQ)
    # ------------------------------------------------------------------ #
    # Раньше очередь задач ходила в Redis через приватное поле `_conn`
    # (`redis._conn.lpush(...)`): так обходились и обработка ошибок клиента, и учёт
    # соединений, а очередь не могла получить ни подтверждений, ни DLQ.
    def xadd(self, name: str, fields: dict, *, maxlen: Optional[int] = None) -> Optional[str]:
        """Добавить сообщение в stream. Возвращает id сообщения или ``None``."""
        if not self._connected or self._conn is None:
            return None
        try:
            payload = {k: _encode_field(v) for k, v in fields.items()}
            message_id = self._conn.xadd(name, payload, maxlen=maxlen, approximate=True)
            return message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis XADD error for '%s': %s", name, exc)
            return None

    def xgroup_create(self, name: str, group: str, *, mkstream: bool = True) -> bool:
        """Создать группу потребителей. ``True``, если группа есть (``BUSYGROUP`` — тоже ``True``)."""
        if not self._connected or self._conn is None:
            return False
        try:
            self._conn.xgroup_create(name, group, id="0", mkstream=mkstream)
            return True
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            # BUSYGROUP — группа уже существует: повторный старт воркера не ошибка
            if "BUSYGROUP" in str(exc).upper():
                return True
            logger.warning("Redis XGROUP CREATE error for '%s': %s", name, exc)
            return False

    def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict,
        *,
        count: int = 10,
        block_ms: Optional[int] = None,
    ) -> list:
        """Прочитать новые сообщения группы.

        Возвращает ``[(stream, [(message_id, fields), ...]), ...]``; пустой список —
        «новых сообщений нет» (нормальный результат цикла, а не ошибка).
        """
        if not self._connected or self._conn is None:
            return []
        try:
            response = self._conn.xreadgroup(
                group, consumer, streams, count=count, block=block_ms
            )
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            # Постоянная ошибка (например, Redis без Streams) приходит сюда на каждом проходе
            # цикла, поэтому запись — через ограничитель частоты.
            _log_transport_error("XREADGROUP", exc)
            return []
        out: list = []
        for stream, messages in response or []:
            decoded = [
                (
                    message_id.decode() if isinstance(message_id, bytes) else str(message_id),
                    {_decode_field(k): _decode_field(v) for k, v in fields.items()},
                )
                for message_id, fields in messages
            ]
            key = stream.decode() if isinstance(stream, bytes) else str(stream)
            out.append((key, decoded))
        return out

    def xack(self, name: str, group: str, *message_ids: str) -> int:
        """Подтвердить обработку (at-least-once: без ack сообщение останется выданным)."""
        if not self._connected or self._conn is None:
            return 0
        try:
            return int(self._conn.xack(name, group, *message_ids))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis XACK error for '%s': %s", name, exc)
            return 0

    def xautoclaim(
        self,
        name: str,
        group: str,
        consumer: str,
        *,
        min_idle_ms: int,
        count: int = 10,
    ) -> list:
        """Забрать «зависшие» сообщения (потребитель умер, ack не пришёл).

        Это и есть механизм, отличающий at-least-once от «потеряли задачу вместе
        с упавшим воркером».

        На Redis < 6.2 команды ``XAUTOCLAIM`` нет: ловим «unknown command» один
        раз, запоминаем и дальше ходим через :meth:`_xautoclaim_legacy`
        (``XPENDING`` + ``XCLAIM``). Без этого перехват либо спамил бы лог на
        каждом проходе, либо молча перестал бы работать.
        """
        global _XAUTOCLAIM_UNSUPPORTED  # noqa: PLW0603

        if not self._connected or self._conn is None:
            return []
        if _XAUTOCLAIM_UNSUPPORTED:
            return self._xautoclaim_legacy(
                name, group, consumer, min_idle_ms=min_idle_ms, count=count
            )
        try:
            _cursor, messages, _deleted = self._conn.xautoclaim(
                name, group, consumer, min_idle_time=min_idle_ms, count=count
            )
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            if _is_unknown_command(exc):
                _XAUTOCLAIM_UNSUPPORTED = True
                logger.warning(
                    "Redis не поддерживает XAUTOCLAIM (нужен Redis >= 6.2): %s. "
                    "Перехват зависших сообщений идёт через XPENDING+XCLAIM.",
                    exc,
                )
                return self._xautoclaim_legacy(
                    name, group, consumer, min_idle_ms=min_idle_ms, count=count
                )
            # Постоянная ошибка приходит на каждом проходе цикла — через ограничитель.
            _log_transport_error("XAUTOCLAIM", exc)
            return []
        return [
            (
                message_id.decode() if isinstance(message_id, bytes) else str(message_id),
                {_decode_field(k): _decode_field(v) for k, v in fields.items()},
            )
            for message_id, fields in (messages or [])
        ]

    def _xautoclaim_legacy(
        self,
        name: str,
        group: str,
        consumer: str,
        *,
        min_idle_ms: int,
        count: int = 10,
    ) -> list:
        """Перехват для Redis < 6.2: ``XPENDING`` (кто висит) + ``XCLAIM`` (забрать).

        ``XAUTOCLAIM`` — это ровно эти две команды в одном вызове, поэтому
        поведение совпадает; разница в том, что ``XPENDING`` возвращает сводку по
        всем выданным, и отфильтровать по времени простоя приходится на клиенте
        (аргумент ``IDLE`` у ``XPENDING`` тоже появился в 6.2).
        """
        try:
            entries = self._conn.xpending_range(name, group, min="-", max="+", count=count * 5)
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            _log_transport_error("XPENDING", exc)
            return []

        stale_ids: list[str] = []
        for entry in entries or []:
            idle = int(entry.get("time_since_delivered") or 0)
            if idle < min_idle_ms:
                continue
            message_id = entry.get("message_id")
            stale_ids.append(
                message_id.decode() if isinstance(message_id, bytes) else str(message_id)
            )
            if len(stale_ids) >= count:
                break
        if not stale_ids:
            return []

        try:
            claimed = self._conn.xclaim(name, group, consumer, min_idle_ms, stale_ids)
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            _log_transport_error("XCLAIM", exc)
            return []
        return [
            (
                message_id.decode() if isinstance(message_id, bytes) else str(message_id),
                {_decode_field(k): _decode_field(v) for k, v in fields.items()},
            )
            for message_id, fields in (claimed or [])
        ]

    def xlen(self, name: str) -> int:
        """Число сообщений в stream (глубина очереди)."""
        if not self._connected or self._conn is None:
            return 0
        try:
            return int(self._conn.xlen(name))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis XLEN error for '%s': %s", name, exc)
            return 0

    def xpending(self, name: str, group: str) -> dict:
        """Сводка выданных и не подтверждённых сообщений (для наблюдаемости)."""
        if not self._connected or self._conn is None:
            return {}
        try:
            summary = self._conn.xpending(name, group)
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis XPENDING error for '%s': %s", name, exc)
            return {}
        if isinstance(summary, dict):
            return summary
        try:
            return {"pending": int(summary[0]), "min_id": summary[1], "max_id": summary[2]}
        except (IndexError, TypeError):
            return {"pending": 0}

    def xtrim(self, name: str, maxlen: int) -> int:
        """Обрезать stream до ``maxlen`` (защита от неограниченного роста)."""
        if not self._connected or self._conn is None:
            return 0
        try:
            return int(self._conn.xtrim(name, maxlen=maxlen, approximate=True))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis XTRIM error for '%s': %s", name, exc)
            return 0

    def xdel(self, name: str, *message_ids: str) -> int:
        """Удалить сообщения из stream."""
        if not self._connected or self._conn is None or not message_ids:
            return 0
        try:
            return int(self._conn.xdel(name, *message_ids))
        except (RedisError, RedisConnectionError, RedisTimeoutError) as exc:
            logger.debug("Redis XDEL error for '%s': %s", name, exc)
            return 0
