"""Strategy Lab "Run optimization" hang — regression tests.

The user-visible bug: pressing *Run optimization* left the panel stuck on
"sweeping N combinations…" forever, even with a tiny grid. The sweep itself is
fast; what can stall a request indefinitely sits *before* it — the bar cache's
Redis probe (a broker that accepts connections but never replies hung the
probe's PING await forever) — and the browser fetch had no ceiling either.
These tests pin the server-side bounds; the frontend got a matching
``timeoutMs`` ceiling in ``json()``.
"""
from __future__ import annotations

import asyncio

import httpx

from trading.adapters.cache import bar_cache as bc
from trading.adapters.persistence import database as db
from trading.main import app

# ── endpoint regression: a small-grid optimize must always answer ──────


async def test_optimize_small_grid_answers_within_budget():
    """POST /backtest/optimize with a 4-combo grid returns before a 60 s ceiling.

    Mirrors the exact payload the Strategy Lab UI sends (``pineOptimize``):
    ``trend_confluence_pine`` + SYNTH + two 2-value grid axes.
    """
    await db.init_db()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/v1/auth/token",
                         json={"username": "admin", "password": "admin"})
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        body = {
            "strategy": "trend_confluence_pine",
            "symbol": "SYNTH",
            "params": {},
            "grid": {"zone_atr": [0.5, 0.8], "min_confluence": [2, 3]},
            "objective": "profit_win",
            "timeframe": "1d",
            "limit": 1000,
            "save_preset": False,
            "run_token": "regress-opt-1",
        }
        # A hang here is the bug; wait_for turns it into a loud failure.
        r = await asyncio.wait_for(
            c.post("/api/v1/backtest/optimize", json=body, headers=headers),
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "SYNTH"
        assert data["strategy"] == "trend_confluence_pine"
        assert data["n_candidates"] == 4
        assert "best" in data and "leaderboard" in data


# ── the stalled-broker probe must degrade, not hang ────────────────────


class _StalledRedis:
    """Stand-in for a half-dead broker: connects fine, never answers PING."""

    def __init__(self) -> None:
        self.closed = False

    async def ping(self) -> None:
        await asyncio.sleep(30)  # far beyond the probe's ceilings

    async def aclose(self) -> None:
        self.closed = True


async def test_probe_of_stalled_redis_falls_back_to_memory(monkeypatch):
    """A broker that accepts but never replies degrades to the memory cache.

    Regression for the Strategy Lab hang: ``_probe_redis`` used to await PING
    with no read timeout, so every uncached ``load_bars`` (and therefore every
    optimize/backtest request) stalled forever.
    """
    stalled = _StalledRedis()

    def _fake_from_url(url: str, **_: object) -> _StalledRedis:
        return stalled

    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", _fake_from_url)
    monkeypatch.setattr(bc, "_next_probe", float("-inf"))
    bc.reset_loop_bar_caches()
    try:
        cache = await asyncio.wait_for(
            bc.loop_bar_cache(redis_url="redis://localhost:6379/0"), timeout=5
        )
    finally:
        bc.reset_loop_bar_caches()
    assert isinstance(cache, bc.MemoryBarCache)
    assert stalled.closed  # the abandoned client must not leak


async def test_probe_cooldown_still_applies_after_stall(monkeypatch):
    """The stall path also arms the 30 s cooldown (one probe per cooldown)."""
    probes: list[str] = []

    class _Recording(_StalledRedis):
        pass

    def _fake_from_url(url: str, **_: object) -> _Recording:
        probes.append(url)
        return _Recording()

    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", _fake_from_url)
    monkeypatch.setattr(bc, "_next_probe", float("-inf"))
    bc.reset_loop_bar_caches()
    try:
        await bc.loop_bar_cache(redis_url="redis://stalled/0")
        await bc.loop_bar_cache(redis_url="redis://stalled/0")
    finally:
        bc.reset_loop_bar_caches()
    assert len(probes) == 1  # second call served from the per-loop cache


# ── an L2 cache that raises must not break the fetch ───────────────────


class _ExplodingCache:
    """L2 cache whose reads/writes raise (e.g. socket timeout mid-session)."""

    async def get_bars(self, key: str, limit: int):
        raise TimeoutError("redis stalled")

    async def put_bars(self, key: str, bars) -> None:
        raise TimeoutError("redis stalled")


async def test_load_bars_survives_l2_cache_failure(monkeypatch):
    """L2 read/write errors degrade to fetching; the request still succeeds."""
    import trading.adapters.cache as cache_pkg
    from trading.application.backtest.portfolio import TickerSpec, load_bars

    async def _exploding(redis_url: str | None = None):
        return _ExplodingCache()

    monkeypatch.setattr(cache_pkg, "loop_bar_cache", _exploding)
    bars = await asyncio.wait_for(
        load_bars(TickerSpec(symbol="SYNTH", source="synthetic",
                             timeframe="1d", limit=400)),
        timeout=30,
    )
    assert len(bars) >= 100
