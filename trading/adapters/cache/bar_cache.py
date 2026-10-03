"""Redis-backed OHLCV bar cache: up to 500 bars per instrument, FIFO eviction.

Layout: one Redis **LIST** per ``bars:{source}:{symbol}:{timeframe}`` key,
newest bar at the head. Writes merge by timestamp (historical bars are
immutable, so re-fetching an overlapping window never creates duplicates) and
``LTRIM 0 499`` evicts the oldest bars first — a FIFO, which is safe precisely
because old bars never change.

When Redis is unreachable the factory degrades to :class:`MemoryBarCache`
(same contract, per-process) so backtests keep working offline.
"""
from __future__ import annotations

import asyncio
import json
import time
import weakref
from datetime import datetime
from typing import Any, Protocol

from trading.domain import Bar

__all__ = [
    "MAX_BARS_PER_INSTRUMENT",
    "BarCache",
    "RedisBarCache",
    "MemoryBarCache",
    "bar_to_json",
    "bar_from_json",
    "cache_key",
    "loop_bar_cache",
    "reset_loop_bar_caches",
]

#: Hard cap per instrument — the FIFO evicts the oldest bars beyond this.
MAX_BARS_PER_INSTRUMENT = 500


def bar_to_json(bar: Bar) -> str:
    return json.dumps({
        "ts": bar.timestamp.isoformat(),
        "o": bar.open, "h": bar.high, "l": bar.low, "c": bar.close, "v": bar.volume,
    }, separators=(",", ":"))


def bar_from_json(raw: str | bytes) -> Bar:
    d = json.loads(raw)
    return Bar(
        timestamp=datetime.fromisoformat(d["ts"]),
        open=d["o"], high=d["h"], low=d["l"], close=d["c"], volume=d["v"],
    )


def cache_key(source: str, symbol: str, timeframe: str) -> str:
    return f"bars:{source}:{symbol.strip().upper()}:{timeframe}"


def _merge(existing: list[Bar], new: list[Bar], max_bars: int) -> list[Bar]:
    """Union by timestamp (new wins), chronological, capped to the newest bars."""
    by_ts = {b.timestamp: b for b in existing}
    by_ts.update({b.timestamp: b for b in new})
    return sorted(by_ts.values(), key=lambda b: b.timestamp)[-max_bars:]


class BarCache(Protocol):
    async def get_bars(self, key: str, limit: int) -> list[Bar] | None:
        """Newest ``limit`` bars (chronological) or None on a miss."""

    async def put_bars(self, key: str, bars: list[Bar]) -> None:
        """Merge bars into the cache (FIFO-evicting beyond the cap)."""


class RedisBarCache:
    def __init__(self, redis: Any, *, max_bars: int = MAX_BARS_PER_INSTRUMENT) -> None:
        self._redis = redis
        self.max_bars = max_bars

    async def get_bars(self, key: str, limit: int) -> list[Bar] | None:
        if limit < 1 or limit > self.max_bars:
            return None  # oversized windows bypass the FIFO cache
        raw = await self._redis.lrange(key, 0, limit - 1)  # head = newest
        if len(raw) < limit:
            return None
        return [bar_from_json(r) for r in reversed(raw)]

    async def put_bars(self, key: str, bars: list[Bar]) -> None:
        if not bars:
            return
        raw = await self._redis.lrange(key, 0, -1)
        merged = _merge([bar_from_json(r) for r in raw], list(bars), self.max_bars)
        pipe = self._redis.pipeline()
        pipe.delete(key)
        for bar in merged:  # LPUSH oldest→newest leaves the newest at the head
            pipe.lpush(key, bar_to_json(bar))
        pipe.ltrim(key, 0, self.max_bars - 1)
        await pipe.execute()


class MemoryBarCache:
    """Same contract, plain dict — offline/test fallback."""

    def __init__(self, *, max_bars: int = MAX_BARS_PER_INSTRUMENT) -> None:
        self.max_bars = max_bars
        self._lists: dict[str, list[Bar]] = {}

    async def get_bars(self, key: str, limit: int) -> list[Bar] | None:
        if limit < 1 or limit > self.max_bars:
            return None
        bars = self._lists.get(key)
        if bars is None or len(bars) < limit:
            return None
        return bars[-limit:]

    async def put_bars(self, key: str, bars: list[Bar]) -> None:
        if bars:
            self._lists[key] = _merge(self._lists.get(key, []), list(bars), self.max_bars)

    def clear(self) -> None:
        self._lists.clear()


#: One cache per event loop (redis.asyncio connections are loop-bound; Celery
#: runs every task in a fresh loop — same reasoning as ``loop_registry``).
_LOOP_CACHES: "weakref.WeakKeyDictionary[Any, BarCache]" = weakref.WeakKeyDictionary()

#: A dead Redis must not cost a TCP timeout *per event loop*: when a probe
#: fails, wait this long before probing again (see ``_next_probe``). Without
#: the cooldown, a Windows host whose 6379 is a dead port proxy makes every
#: test/task loop pay the full connect timeout.
_PROBE_COOLDOWN = 30.0
_PROBE_TIMEOUT = 0.5
_next_probe = float("-inf")  # -inf = "never probed, probe now"


async def _probe_redis(redis_url: str) -> BarCache | None:
    """Return a RedisBarCache if ``redis_url`` answers PING, else None."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, socket_connect_timeout=_PROBE_TIMEOUT)
        await client.ping()
        return RedisBarCache(client)
    except Exception:
        return None


async def loop_bar_cache(redis_url: str | None = None) -> BarCache:
    """The bar cache for the running loop; Redis if reachable, memory otherwise."""
    global _next_probe
    loop = asyncio.get_running_loop()
    cached = _LOOP_CACHES.get(loop)
    if cached is not None:
        return cached

    cache: BarCache | None = None
    if time.monotonic() >= _next_probe:
        if redis_url is None:
            from trading.config import settings
            redis_url = settings.redis_url
        cache = await _probe_redis(redis_url)
        if cache is None:
            _next_probe = time.monotonic() + _PROBE_COOLDOWN
    if cache is None:
        cache = MemoryBarCache()
    _LOOP_CACHES[loop] = cache
    return cache


def reset_loop_bar_caches() -> None:
    """Forget all loop caches and the probe cooldown (tests)."""
    global _next_probe
    _LOOP_CACHES.clear()
    _next_probe = float("-inf")
