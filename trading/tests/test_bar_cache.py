"""Redis bar cache: 500 bars per instrument, FIFO eviction, memory fallback."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading.adapters.cache.bar_cache import (
    MAX_BARS_PER_INSTRUMENT,
    MemoryBarCache,
    RedisBarCache,
    bar_from_json,
    bar_to_json,
    cache_key,
    loop_bar_cache,
    reset_loop_bar_caches,
)
from trading.domain import Bar

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_probe_state():
    """The Redis probe cooldown is process-global — isolate it per test."""
    reset_loop_bar_caches()
    yield
    reset_loop_bar_caches()



def make_bars(n: int, start: int = 0) -> list[Bar]:
    out = []
    for i in range(start, start + n):
        price = 100.0 + i
        out.append(Bar(
            timestamp=T0 + timedelta(days=i),
            open=price, high=price + 1, low=price - 1, close=price, volume=10.0,
        ))
    return out


class FakePipeline:
    def __init__(self, store: dict):
        self._store = store
        self._ops = []

    def delete(self, key):
        self._ops.append(("delete", key))

    def lpush(self, key, value):
        self._ops.append(("lpush", key, value))

    def ltrim(self, key, start, end):
        self._ops.append(("ltrim", key, start, end))

    async def execute(self):
        for op in self._ops:
            if op[0] == "delete":
                self._store.pop(op[1], None)
            elif op[0] == "lpush":
                self._store.setdefault(op[1], []).insert(0, op[2])
            elif op[0] == "ltrim":
                self._store[op[1]] = self._store[op[1]][op[2]:op[3] + 1]
        self._ops.clear()


class FakeRedis:
    """Minimal async-redis stand-in (list ops only)."""

    def __init__(self):
        self.lists: dict[str, list[str]] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.lists)

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        if end == -1:
            return items[start:]
        return items[start:end + 1]


class TestSerialization:
    def test_bar_json_roundtrip(self):
        bar = make_bars(1)[0]
        assert bar_from_json(bar_to_json(bar)) == bar

    def test_cache_key_format(self):
        assert cache_key("bybit", "btc-usdt", "1h") == "bars:bybit:BTC-USDT:1h"


class CacheContract:
    """Shared behavior for Redis and memory implementations."""

    async def make_cache(self):
        raise NotImplementedError

    async def test_put_get_roundtrip_chronological(self):
        cache = await self.make_cache()
        bars = make_bars(50)
        await cache.put_bars("k", bars)
        got = await cache.get_bars("k", 50)
        assert got == bars  # chronological order restored

    async def test_get_newest_subset(self):
        cache = await self.make_cache()
        bars = make_bars(100)
        await cache.put_bars("k", bars)
        got = await cache.get_bars("k", 10)
        assert got == bars[-10:]

    async def test_miss_when_too_few_cached(self):
        cache = await self.make_cache()
        await cache.put_bars("k", make_bars(30))
        assert await cache.get_bars("k", 50) is None

    async def test_miss_when_limit_exceeds_capacity(self):
        cache = await self.make_cache()
        await cache.put_bars("k", make_bars(500))
        assert await cache.get_bars("k", 501) is None

    async def test_fifo_eviction_keeps_newest_500(self):
        cache = await self.make_cache()
        await cache.put_bars("k", make_bars(600))
        got = await cache.get_bars("k", 500)
        expected = make_bars(600)[-500:]
        assert got == expected
        assert got[0].timestamp == expected[0].timestamp  # oldest 100 evicted

    async def test_merge_dedupes_overlapping_windows(self):
        cache = await self.make_cache()
        await cache.put_bars("k", make_bars(100, start=0))
        await cache.put_bars("k", make_bars(100, start=50))  # overlaps 50 bars
        got = await cache.get_bars("k", 150)
        assert got == make_bars(150, start=0)

    async def test_unknown_key_misses(self):
        cache = await self.make_cache()
        assert await cache.get_bars("nope", 10) is None


class TestRedisBarCache(CacheContract):
    async def make_cache(self):
        return RedisBarCache(FakeRedis())

    async def test_max_capacity_is_500(self):
        assert MAX_BARS_PER_INSTRUMENT == 500

    async def test_stored_newest_first_in_redis(self):
        redis = FakeRedis()
        cache = RedisBarCache(redis)
        bars = make_bars(5)
        await cache.put_bars("k", bars)
        stored = redis.lists["k"]
        assert bar_from_json(stored[0]).timestamp == bars[-1].timestamp  # head = newest


class TestMemoryBarCache(CacheContract):
    async def make_cache(self):
        return MemoryBarCache()


class TestFactory:
    async def test_falls_back_to_memory_when_redis_down(self):
        cache = await loop_bar_cache(redis_url="redis://127.0.0.1:1/0")
        assert isinstance(cache, MemoryBarCache)
        reset_loop_bar_caches()

    async def test_dead_redis_is_probed_once_per_cooldown(self, monkeypatch):
        """A hanging Redis must not cost a TCP timeout on every event loop."""
        import trading.adapters.cache.bar_cache as bc

        calls = []

        async def fake_probe(url):
            calls.append(url)
            return None  # unreachable

        monkeypatch.setattr(bc, "_probe_redis", fake_probe)
        reset_loop_bar_caches()
        first = await loop_bar_cache(redis_url="redis://dead:1/0")
        bc._LOOP_CACHES.clear()  # a fresh event loop, but the cooldown stands
        second = await loop_bar_cache(redis_url="redis://dead:1/0")
        assert isinstance(first, MemoryBarCache) and isinstance(second, MemoryBarCache)
        assert calls == ["redis://dead:1/0"]  # probed once, then cooled down

    async def test_probe_cache_is_per_loop(self, monkeypatch):
        import trading.adapters.cache.bar_cache as bc

        async def fake_probe(url):
            return None

        monkeypatch.setattr(bc, "_probe_redis", fake_probe)
        reset_loop_bar_caches()
        a = await loop_bar_cache(redis_url="redis://dead:1/0")
        b = await loop_bar_cache(redis_url="redis://dead:1/0")
        assert a is b  # same loop → same instance

    async def test_reset_clears_probe_cooldown(self, monkeypatch):
        import trading.adapters.cache.bar_cache as bc

        calls = []

        async def fake_probe(url):
            calls.append(url)
            return None

        monkeypatch.setattr(bc, "_probe_redis", fake_probe)
        await loop_bar_cache(redis_url="redis://dead:1/0")
        reset_loop_bar_caches()  # must also forget the negative probe state
        await loop_bar_cache(redis_url="redis://dead:1/0")
        assert len(calls) == 2
