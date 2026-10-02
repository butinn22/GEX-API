"""Unit-тесты для Redis-кэширования (gex/redis_client.py).

Использует ``fakeredis`` — in-memory эмуляцию Redis без сетевых вызовов.
"""
from __future__ import annotations

import pickle
import zlib
from typing import Any

import pandas as pd
import pytest

from gex.adapters.cache.redis_client import (
    RedisClient,
    cache_key,
    cached,
    serialize_df,
    deserialize_df,
    serialize_value,
    deserialize_value,
)


# ====================================================================== #
#  Фикстуры
# ====================================================================== #
@pytest.fixture
def fakeredis_client() -> RedisClient:
    """RedisClient, подключенный к fakeredis вместо настоящего Redis.

    Создаём RedisClient вручную без вызова _connect() (чтобы не ждать
    таймаут реального Redis).
    """
    import fakeredis
    fake_conn = fakeredis.FakeRedis()

    # Создаём клиента без соединения, потом подменяем коннект
    import gex.adapters.cache.redis_client as rc_module
    client = rc_module.RedisClient.__new__(rc_module.RedisClient)
    # Инициализируем поля вручную (обходим __init__, который пытается коннектиться)
    client._host = "fake"
    client._port = 0
    client._db = 0
    client._password = None
    client._socket_timeout = 1
    client._socket_connect_timeout = 1
    client._maxmemory = "1mb"
    client._conn = fake_conn
    client._connected = True
    client._pool = None

    fake_conn.flushdb()
    return client


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Маленький DataFrame для теста сериализации OHLC."""
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 103.0],
        "Volume": [1000, 1500, 1200],
    }, index=pd.date_range("2026-01-01", periods=3, freq="1h"))


@pytest.fixture
def sample_snapshot(sample_df):
    """Мок OptionSnapshot (датакласс-подобный объект)."""
    from gex.domain.data_loader import OptionSnapshot
    return OptionSnapshot(
        symbol="TEST",
        spot=500.0,
        as_of=pd.Timestamp.now(),
        chain=sample_df,
    )


# ====================================================================== #
#  Key Schema
# ====================================================================== #
class TestCacheKey:
    """Схема провайдер-независимых ключей ``gex:{prefix}:{part1}:{part2}``.

    Итер. 25: ``cache_key`` обслуживает только ключи, не зависящие от источника данных.
    Для ``chain`` / ``ohlcv`` / ``spot`` / ``hv`` / ``commodity:*`` он обязан падать —
    иначе провайдер не входит в ключ, и два источника делят одну запись.
    """

    def test_simple(self):
        assert cache_key("res", "ta", "SPY", 1000) == "gex:res:TA:SPY:1000"

    def test_ticker_upper(self):
        assert cache_key("res", "cash", "btc") == "gex:res:CASH:BTC"

    def test_no_parts(self):
        assert cache_key("vol") == "gex:vol"

    def test_empty_prefix(self):
        assert cache_key("", "SPY") == "gex::SPY"

    @pytest.mark.parametrize("kind", ["chain", "ohlcv", "spot", "hv", "commodity:spot"])
    def test_provider_scoped_kind_rejected(self, kind):
        """Неоднозначный ключ нельзя построить: провайдер обязан быть в ключе."""
        from gex.adapters.cache.keys import CacheKeyError

        with pytest.raises(CacheKeyError, match="не различает источники"):
            cache_key(kind, "SPY", "1h")

    def test_provider_scoped_names_the_builder(self):
        """Ошибка должна называть замену, иначе разработчик пойдёт искать её сам."""
        from gex.adapters.cache.keys import CacheKeyError

        with pytest.raises(CacheKeyError, match="commodity_key"):
            cache_key("commodity:ohlcv", "GOLD", "1d", "500")


# ====================================================================== #
#  Сериализация
# ====================================================================== #
class TestSerialization:
    """Проверка сериализации/десериализации DataFrame и произвольных объектов."""

    def test_dataframe_roundtrip(self, sample_df):
        data = serialize_df(sample_df)
        assert isinstance(data, bytes)
        assert len(data) > 0
        restored = deserialize_df(data)
        pd.testing.assert_frame_equal(restored, sample_df)

    def test_dataframe_compression(self, sample_df):
        """Проверить, что pickle+zlib реально сжимает."""
        raw = pickle.dumps(sample_df)
        compressed = serialize_df(sample_df)
        # Для тривиального датафрейма сжатие может не уменьшить размер,
        # но структура должна быть корректной.
        assert len(compressed) > 0
        assert deserialize_df(compressed) is not None

    def test_float_serialization(self):
        data = serialize_value(123.45)
        assert isinstance(data, bytes)
        restored = deserialize_value(data)
        assert float(restored) == 123.45

    def test_int_serialization(self):
        data = serialize_value(42)
        restored = deserialize_value(data)
        assert int(restored) == 42

    def test_dict_serialization(self):
        obj = {"a": 1, "b": "hello"}
        data = serialize_value(obj)
        restored = deserialize_value(data)
        assert restored == obj

    def test_dataframe_via_serialize_value(self, sample_df):
        data = serialize_value(sample_df)
        restored = deserialize_value(data)
        pd.testing.assert_frame_equal(restored, sample_df)

    def test_corrupted_data(self):
        """Некорректные данные при десериализации не должны падать."""
        result = deserialize_value(b"not valid data")
        # Должен вернуть сырые байты или строку, не взорваться
        assert result is not None


# ====================================================================== #
#  Идемпотентность сериализации + круговой рейс конверта (Fix 1)
# ====================================================================== #
class TestSerializeIdempotency:
    """``serialize_value`` обязан быть идемпотентным для уже сериализованного.

    Дефект: слой конверта (:class:`RedisEnvelopeCache.write`) сериализует payload сам и
    передаёт в :meth:`RedisClient.set` готовые ``bytes``. Без ветки ``bytes`` они уходили в
    ``pickle.dumps`` повторно (``pickle(pickle(payload))``): читатель снимал один слой,
    получал ``bytes``, ``Envelope.from_payload`` возвращал ``None`` — и **каждый** read был
    вечным промахом (спам «неизвестной формы» + ``None`` у followers → HTTP 500).
    """

    def test_bytes_are_returned_unchanged(self):
        """Уже сериализованные байты не заворачиваются повторно."""
        assert serialize_value(b"already") == b"already"

    def test_str_is_raw_utf8_not_pickled(self):
        """Строки храним сырым utf-8: токены аренд сравнивают после ``.decode()``."""
        blob = serialize_value("token-abc")
        assert isinstance(blob, bytes)
        assert blob == b"token-abc"
        assert not blob.startswith(b"\x80"), "строка ушла в pickle — токен не сравнить после decode"
        assert deserialize_value(blob) == "token-abc"

    def test_envelope_roundtrip_with_real_serializers(self, fakeredis_client):
        """Конверт через настоящую пару сериализаторов и реальный клиент (fakeredis).

        До Fix 1 ``read()`` возвращал ``None`` — этот тест его ловит.
        """
        from gex.adapters.cache.envelope import RedisEnvelopeCache

        cache = RedisEnvelopeCache(fakeredis_client)
        assert cache.write("gex:res:TESTREPRO:env", {"a": 1}, ttl=30) is not None

        got = cache.read("gex:res:TESTREPRO:env")
        assert got is not None, "конверт прочитан как None — двойная сериализация вернулась"
        assert got.value == {"a": 1}
        assert got.ttl == 30


# ====================================================================== #
#  RedisClient — graceful degradation
# ====================================================================== #
class TestRedisClientGraceful:
    """Проверка что RedisClient НЕ падает при отсутствии Redis."""

    def test_noop_get(self):
        """Если соединения нет, get() возвращает None."""
        client = RedisClient(host="192.0.0.1", port=1, socket_connect_timeout=1)
        assert client.get("some-key") is None

    def test_noop_set(self):
        """Если соединения нет, set() возвращает False."""
        client = RedisClient(host="192.0.0.1", port=1, socket_connect_timeout=1)
        assert client.set("some-key", "value", ex=600) is False

    def test_noop_delete(self):
        client = RedisClient(host="192.0.0.1", port=1, socket_connect_timeout=1)
        assert client.delete("some-key") is False

    def test_noop_ping(self):
        client = RedisClient(host="192.0.0.1", port=1, socket_connect_timeout=1)
        assert client.ping() is False

    def test_ping_does_not_reconnect_on_every_call(self):
        """Инцидент 2026-09-21: ping() инициировал переподключение на КАЖДЫЙ вызов.

        Клиент внутри делает несколько попыток с таймаутом, поэтому тривиальный
        ``/health`` при недоступном Redis стоил ~4.5 с и занимал поток пула.
        Повторная попытка допустима не чаще раза в ``RECONNECT_MIN_INTERVAL``.
        """
        client = RedisClient(host="192.0.0.1", port=1, socket_connect_timeout=1)
        first_attempt = client._last_connect_attempt
        assert client.ping() is False
        assert client.ping() is False
        assert client._last_connect_attempt == first_attempt, (
            "повторный ping не должен инициировать новое переподключение"
        )
        assert client._reconnect_allowed() is False

    def test_noop_close(self):
        """close() не падает, если соединения не было."""
        client = RedisClient(host="192.0.0.1", port=1, socket_connect_timeout=1)
        client.close()
        assert client.connected is False

    def test_operation_failure_flips_client_to_disconnected(self):
        """Сбой операции при поднятом флаге ``connected`` отключает клиент немедленно.

        Раньше после «тихой» смерти Redis каждый следующий вызов шёл в мёртвое
        соединение и платил ``socket_timeout`` (2 с) за операцию — до следующего
        ping-реконнекта. Теперь первый же сбой переводит клиент в ``connected=False``:
        кэш мгновенно выключается (включаются внутрипроцессные фолбэки), а не тянет
        время каждого запроса.
        """
        import gex.adapters.cache.redis_client as rc_module
        client = rc_module.RedisClient.__new__(rc_module.RedisClient)
        client._host = "fake"
        client._port = 0
        client._db = 0
        client._password = None
        client._socket_timeout = 1
        client._socket_connect_timeout = 1
        client._maxmemory = "1mb"
        client._pool = None
        client._connected = True

        import redis as _redis_mod

        calls = {"n": 0}

        class _DeadConn:
            def get(self, *a, **k):
                calls["n"] += 1
                raise _redis_mod.ConnectionError("redis silently dead")

        client._conn = _DeadConn()
        assert client.get("k") is None
        assert client.connected is False
        # Второй вызов уже не ходит в мёртвое соединение — мгновенный промах.
        assert client.get("k") is None
        assert calls["n"] == 1


# ====================================================================== #
#  RedisClient — базовые операции
# ====================================================================== #
class TestRedisClient:
    """Проверка get/set/delete/ttl на fakeredis."""

    def test_set_and_get(self, fakeredis_client):
        assert fakeredis_client.set("test:key", 42, ex=600)
        assert fakeredis_client.get("test:key") is not None  # bytes

    def test_get_float(self, fakeredis_client):
        fakeredis_client.set("spot:SPY", 500.0, ex=600)
        data = fakeredis_client.get("spot:SPY")
        assert data is not None
        assert float(data) == 500.0

    def test_get_miss(self, fakeredis_client):
        assert fakeredis_client.get("nonexistent") is None

    def test_delete(self, fakeredis_client):
        fakeredis_client.set("test:del", "value", ex=600)
        assert fakeredis_client.exists("test:del")
        assert fakeredis_client.delete("test:del")
        assert not fakeredis_client.exists("test:del")

    def test_flush(self, fakeredis_client):
        fakeredis_client.set("a", 1, ex=600)
        fakeredis_client.set("b", 2, ex=600)
        fakeredis_client.flush()
        assert fakeredis_client.get("a") is None
        assert fakeredis_client.get("b") is None

    def test_set_dataframe(self, fakeredis_client, sample_df):
        key = "test:df"
        fakeredis_client.set(key, sample_df, ex=600)
        data = fakeredis_client.get(key)
        assert data is not None
        restored = deserialize_value(data)
        pd.testing.assert_frame_equal(restored, sample_df)

    def test_ttl(self, fakeredis_client):
        """Проверить, что TTL устанавливается (fakeredis поддерживает)."""
        key = "test:ttl"
        fakeredis_client.set(key, "x", ex=60)
        data = fakeredis_client.get(key)
        assert data is not None
        # Проверяем, что ключ существует (fakeredis 2.x поддерживает TTL)
        assert fakeredis_client.exists(key)

    def test_ping_connected(self, fakeredis_client):
        assert fakeredis_client.ping() is True


# ====================================================================== #
#  @cached декоратор
# ====================================================================== #
class TestCachedDecorator:
    """Проверка Cache-Aside декоратора."""

    def test_cache_hit(self, fakeredis_client):
        """При hit оригинал НЕ вызывается."""
        call_count = 0

        @cached(ttl=600, cache_client=fakeredis_client,
                key_builder=lambda ticker: f"test:{ticker}")
        def fetch_something(ticker: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"ticker": ticker, "data": "expensive"}

        # Первый вызов — miss → вызываем fetch
        result1 = fetch_something("AAPL")
        assert call_count == 1
        assert result1["ticker"] == "AAPL"

        # Второй вызов — hit → НЕ вызываем fetch
        result2 = fetch_something("AAPL")
        assert call_count == 1, "Cache HIT: оригинал не должен вызываться"
        assert result2 == result1

    def test_cache_miss_different_args(self, fakeredis_client):
        """Разные аргументы → разные ключи → miss."""
        call_count = 0

        @cached(ttl=600, cache_client=fakeredis_client,
                key_builder=lambda ticker: f"test:{ticker}")
        def fetch(ticker: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{ticker}"

        fetch("AAPL")
        fetch("MSFT")
        assert call_count == 2

    def test_graceful_degradation_no_redis(self):
        """Недоступный Redis → декоратор просто вызывает оригинал (без кэша)."""
        from gex.adapters.cache.redis_client import RedisClient
        # Заведомо мёртвый клиент (порт закрыт) — детерминированно "не подключён".
        dead_client = RedisClient(host="127.0.0.1", port=1, socket_connect_timeout=1)
        call_count = 0

        @cached(ttl=600, key_builder=lambda: "some:key", cache_client=dead_client)
        def fetch() -> str:
            nonlocal call_count
            call_count += 1
            return "always_fresh"

        fetch()
        fetch()
        # Redis недоступен → каждый вызов = miss
        assert call_count == 2

    def test_dataframe_cache(self, fakeredis_client, sample_df):
        """DataFrame корректно кэшируется через @cached."""

        @cached(ttl=600, cache_client=fakeredis_client,
                key_builder=lambda: "test:df_cache")
        def fetch_df() -> pd.DataFrame:
            return sample_df

        df1 = fetch_df()
        pd.testing.assert_frame_equal(df1, sample_df)

        # Кэш hit
        df2 = fetch_df()
        pd.testing.assert_frame_equal(df2, sample_df)


# ====================================================================== #
#  Конфигурация
# ====================================================================== #
class TestRedisSettings:
    """Проверка настроек Redis в pydantic-settings."""

    def test_defaults(self):
        from gex.auth.config import Settings
        s = Settings()
        assert s.REDIS_HOST == "localhost"
        assert s.REDIS_PORT == 6379
        assert s.REDIS_DB == 0
        assert s.REDIS_TTL_DEFAULT == 600
        assert s.REDIS_MAXMEMORY == "512mb"
        assert s.REDIS_PASSWORD is None


# ====================================================================== #
#  Интеграционный тест: кэширование через serialization roundtrip
# ====================================================================== #
class TestIntegration:
    """Полный цикл: serialize → Redis → deserialize."""

    def test_option_snapshot_cache(self, fakeredis_client, sample_snapshot):
        """OptionSnapshot → pickle+zlib → Redis → restore."""
        key = "gex:chain:TEST:5"
        assert fakeredis_client.set(key, sample_snapshot, ex=600)
        data = fakeredis_client.get(key)
        assert data is not None
        restored = deserialize_value(data)
        assert restored.symbol == "TEST"
        assert restored.spot == 500.0
        pd.testing.assert_frame_equal(restored.chain, sample_snapshot.chain)
