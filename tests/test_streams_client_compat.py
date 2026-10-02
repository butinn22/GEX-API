"""Совместимость очереди со старыми версиями Redis (ring: adapters/queue).

``XAUTOCLAIM`` появился в Redis 6.2. На более старом сервере команда неизвестна,
и перехват зависших сообщений ломался: лог забивался одинаковыми
``unknown command 'XAUTOCLAIM'``, а задачи умершего воркера оставались выданными
навсегда. Тесты ниже фиксируют переход на ``XPENDING`` + ``XCLAIM``.
"""
from __future__ import annotations

import pytest

from gex.adapters.queue import streams_client
from gex.adapters.queue.streams_client import StreamsClientMixin, reset_capability_cache

try:  # pragma: no cover — зависит от наличия пакета redis
    from redis.exceptions import ResponseError as _ResponseError
except ImportError:  # pragma: no cover
    _ResponseError = Exception


@pytest.fixture(autouse=True)
def _reset_caps():
    """Флаг «сервер не умеет XAUTOCLAIM» — модульный: сбрасываем между тестами."""
    reset_capability_cache()
    yield
    reset_capability_cache()


class _OldRedisConn:
    """Соединение с Redis < 6.2: ``XAUTOCLAIM`` нет, ``XPENDING``/``XCLAIM`` есть."""

    def __init__(self, entries=None):
        self.xautoclaim_calls = 0
        self.claimed_ids: list[str] = []
        self.entries = entries if entries is not None else [
            {"message_id": b"1-0", "consumer": b"dead-worker",
             "time_since_delivered": 120_000, "times_delivered": 1},
            {"message_id": b"2-0", "consumer": b"dead-worker",
             "time_since_delivered": 500, "times_delivered": 1},
        ]

    def xautoclaim(self, *args, **kwargs):
        self.xautoclaim_calls += 1
        raise _ResponseError("unknown command 'XAUTOCLAIM'")

    def xpending_range(self, name, group, min="-", max="+", count=None, **kwargs):
        return self.entries

    def xclaim(self, name, group, consumer, min_idle_time, message_ids, **kwargs):
        self.claimed_ids = list(message_ids)
        self.min_idle_time = min_idle_time
        return [(mid, {b"task_type": b"ohlcv", b"payload": b"{}"}) for mid in message_ids]


class _Client(StreamsClientMixin):
    def __init__(self, conn):
        self._conn = conn
        self._connected = True


def test_xautoclaim_falls_back_to_xpending_xclaim():
    """Команда неизвестна → перехват всё равно работает, через XPENDING+XCLAIM."""
    conn = _OldRedisConn()
    client = _Client(conn)

    claimed = client.xautoclaim("gex:q:ohlcv", "gex-workers", "me", min_idle_ms=60_000)

    assert [mid for mid, _fields in claimed] == ["1-0"]
    assert conn.claimed_ids == ["1-0"]


def test_legacy_path_skips_fresh_messages():
    """Сообщение с простоем меньше порога не считается зависшим."""
    conn = _OldRedisConn(entries=[
        {"message_id": b"1-0", "consumer": b"w", "time_since_delivered": 10, "times_delivered": 1},
    ])
    client = _Client(conn)

    assert client.xautoclaim("gex:q:ohlcv", "g", "me", min_idle_ms=60_000) == []
    assert conn.claimed_ids == []


def test_unsupported_command_is_remembered():
    """Один раз узнали, что команды нет — больше не дёргаем сервер и не спамим лог."""
    conn = _OldRedisConn()
    client = _Client(conn)

    client.xautoclaim("gex:q:ohlcv", "g", "me", min_idle_ms=1)
    client.xautoclaim("gex:q:ohlcv", "g", "me", min_idle_ms=1)
    client.xautoclaim("gex:q:ohlcv", "g", "me", min_idle_ms=1)

    assert conn.xautoclaim_calls == 1  # запомнили с первого раза
    assert streams_client._XAUTOCLAIM_UNSUPPORTED is True


def test_modern_redis_keeps_using_xautoclaim():
    """Если команда поддерживается, старый путь не подменяет её."""
    class _ModernConn:
        def __init__(self):
            self.calls = 0

        def xautoclaim(self, name, group, consumer, min_idle_time=0, count=10):
            self.calls += 1
            return (b"0-0", [(b"7-0", {b"task_type": b"ohlcv"})], [])

    conn = _ModernConn()
    client = _Client(conn)

    claimed = client.xautoclaim("gex:q:ohlcv", "g", "me", min_idle_ms=60_000)

    assert [mid for mid, _f in claimed] == ["7-0"]
    assert conn.calls == 1
    assert streams_client._XAUTOCLAIM_UNSUPPORTED is False


def test_disconnected_client_does_not_call_redis():
    """Redis удалён/недоступен — никаких обращений и никаких ошибок."""
    conn = _OldRedisConn()
    client = _Client(conn)
    client._connected = False

    assert client.xautoclaim("gex:q:ohlcv", "g", "me", min_idle_ms=60_000) == []
    assert conn.xautoclaim_calls == 0
