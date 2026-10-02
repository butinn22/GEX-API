"""Tests for the TTL cache."""
from __future__ import annotations

import asyncio

from trading.adapters.cache import TtlCache


async def test_ttl_cache_set_get_delete():
    c = TtlCache(default_ttl=60.0)
    await c.set("k", "v")
    assert await c.get("k") == "v"
    await c.delete("k")
    assert await c.get("k") is None


async def test_ttl_cache_expiry():
    c = TtlCache(default_ttl=0.01)
    await c.set("k", "v")
    assert await c.get("k") == "v"
    await asyncio.sleep(0.03)
    assert await c.get("k") is None


async def test_ttl_cache_miss_returns_none():
    assert await TtlCache().get("missing") is None
