"""Data-loading performance guards.

These lock in the three fixes that made a portfolio backtest fast:

1. **Bounded yahoo window** — we ask for the bars we need, not the ticker's whole
   history (AAPL's full daily history is ~11,500 bars / ~1.3 MB, of which we used
   the last 1,500).
2. **Loop-scoped fetcher registry** — one client set per event loop instead of a
   fresh set (and TLS handshake) per ticker.
3. **TTL bar cache** — re-running a backtest must not re-download.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from trading.adapters.fetchers import aclose_loop_registry, default_registry, loop_registry
from trading.adapters.fetchers.yfinance import YFinanceFetcher, _lookback_seconds
from trading.application.backtest.portfolio import (
    TickerSpec,
    bar_cache_stats,
    clear_bar_cache,
    load_bars,
)

# ── 1. bounded yahoo window ────────────────────────────────────────────


def _capture(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "chart": {"result": [{
                "timestamp": [1700000000, 1700086400],
                "indicators": {"quote": [{
                    "open": [100.0, 101.0], "high": [102.0, 103.0],
                    "low": [99.0, 100.0], "close": [101.0, 102.0],
                    "volume": [1000, 1100],
                }]},
            }]},
        })
    return handler


async def test_yahoo_daily_request_is_bounded_not_full_history():
    """The period1=0 bug: full history downloaded, then sliced away."""
    requests: list[httpx.Request] = []
    fetcher = YFinanceFetcher(transport=httpx.MockTransport(_capture(requests)))
    await fetcher.get_ohlcv("AAPL", "1d", limit=800)

    params = dict(requests[0].url.params)
    period1, period2 = int(params["period1"]), int(params["period2"])
    assert period1 > 0, "must not request epoch 0 (all history)"
    # ~800 trading days => bounded window, comfortably under 10 years.
    span_days = (period2 - period1) / 86400
    assert 1000 < span_days < 4000, span_days


async def test_yahoo_window_scales_with_limit():
    small, large = _lookback_seconds("1d", 100), _lookback_seconds("1d", 2000)
    assert small < large
    # 100 daily bars must not pull a decade of data.
    assert small / 86400 < 400


async def test_yahoo_intraday_respects_server_history_cap():
    """1m data only exists for 7 days at Yahoo; never ask for more.

    The cap is applied when building the request (not by ``_lookback_seconds``),
    so assert on what actually goes over the wire.
    """
    requests: list[httpx.Request] = []
    fetcher = YFinanceFetcher(transport=httpx.MockTransport(_capture(requests)))
    await fetcher.get_ohlcv("AAPL", "1m", limit=100_000)

    params = dict(requests[0].url.params)
    span_days = (int(params["period2"]) - int(params["period1"])) / 86400
    assert span_days <= 7.0, span_days


async def test_yahoo_explicit_start_wins_over_window():
    requests: list[httpx.Request] = []
    fetcher = YFinanceFetcher(transport=httpx.MockTransport(_capture(requests)))
    from datetime import datetime, timezone

    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await fetcher.get_ohlcv("AAPL", "1d", start=start, limit=10)
    assert int(dict(requests[0].url.params)["period1"]) == int(start.timestamp())


# ── 2. loop-scoped registry ────────────────────────────────────────────


async def test_loop_registry_is_stable_within_a_loop():
    assert loop_registry() is loop_registry()


def test_loop_registry_is_per_loop():
    """A registry is bound to its loop (httpx clients are loop-affine)."""
    seen: list[int] = []

    async def grab() -> None:
        seen.append(id(loop_registry()))

    asyncio.run(grab())  # loop A
    asyncio.run(grab())  # loop B
    assert len(seen) == 2 and seen[0] != seen[1]


async def test_aclose_loop_registry_drops_the_registry():
    first = loop_registry()
    await aclose_loop_registry()
    assert loop_registry() is not first


def test_default_registry_has_the_three_real_sources():
    from trading.domain import Exchange

    reg = default_registry()
    assert reg.get(Exchange.MOEX) is not None
    assert reg.get(Exchange.YFINANCE) is not None
    assert reg.get(Exchange.BYBIT) is not None


# ── 3. TTL bar cache ───────────────────────────────────────────────────


async def test_second_load_is_served_from_cache():
    clear_bar_cache()
    spec = TickerSpec(symbol="CACHEME", source="synthetic", limit=200)
    first = await load_bars(spec)
    stats_after_first = bar_cache_stats()
    second = await load_bars(spec)

    assert stats_after_first["misses"] == 1 and stats_after_first["hits"] == 0
    assert bar_cache_stats()["hits"] == 1
    assert [b.close for b in first] == [b.close for b in second]
    # Same objects: Bar is frozen, so sharing is safe and avoids a copy.
    assert second[0] is first[0]


async def test_refresh_bypasses_the_cache():
    clear_bar_cache()
    spec = TickerSpec(symbol="REFRESHME", source="synthetic", limit=120)
    await load_bars(spec)
    await load_bars(spec, refresh=True)
    stats = bar_cache_stats()
    assert stats["hits"] == 0 and stats["misses"] == 2


async def test_different_limit_is_a_different_cache_entry():
    clear_bar_cache()
    await load_bars(TickerSpec(symbol="DIFF", source="synthetic", limit=100))
    await load_bars(TickerSpec(symbol="DIFF", source="synthetic", limit=150))
    stats = bar_cache_stats()
    assert stats["misses"] == 2 and stats["hits"] == 0 and stats["size"] == 2


async def test_cache_can_be_disabled_by_ttl(monkeypatch):
    """``TRADING_DATA_CACHE_TTL=0`` must turn the cache off completely."""
    from types import SimpleNamespace

    import trading.application.backtest.portfolio as P

    clear_bar_cache()
    # Settings is a frozen dataclass, so swap the module-level reference instead.
    monkeypatch.setattr(P, "settings", SimpleNamespace(data_cache_ttl=0))
    spec = TickerSpec(symbol="NOCACHE", source="synthetic", limit=100)
    await load_bars(spec)
    await load_bars(spec)
    assert bar_cache_stats()["hits"] == 0
    assert bar_cache_stats()["size"] == 0


async def test_bar_cache_stats_reset():
    await load_bars(TickerSpec(symbol="RESETME", source="synthetic", limit=80))
    assert bar_cache_stats()["misses"] == 1
    clear_bar_cache()
    assert bar_cache_stats() == {"hits": 0, "misses": 0, "size": 0}


# ── the engine itself is not the bottleneck ────────────────────────────


async def test_backtest_engine_is_fast_for_a_full_portfolio():
    """Guard against future per-bar regressions in the replay loop."""
    import time

    from trading.application.backtest.portfolio import (
        PortfolioBacktestConfig,
        run_portfolio_backtest,
    )

    specs = [
        TickerSpec(symbol=f"S{i}", source="synthetic", limit=1500, strategy="sma_crossover",
                   params={"fast": 10, "slow": 30}, weight=0.5)
        for i in range(4)
    ]
    await run_portfolio_backtest(specs, PortfolioBacktestConfig())  # warm the cache
    start = time.perf_counter()
    await run_portfolio_backtest(specs, PortfolioBacktestConfig())
    elapsed = time.perf_counter() - start
    # 6000 bars through 4 strategies. Generous bound so slow CI doesn't flake.
    assert elapsed < 2.0, f"engine replay took {elapsed:.2f}s"
