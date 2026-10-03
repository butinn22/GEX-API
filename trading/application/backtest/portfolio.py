"""Multi-ticker portfolio backtest.

Runs an independent backtest per ticker — each with its **own** strategy and
individual settings — then combines the per-ticker equity curves into a single
capital-weighted portfolio curve on a shared time grid.

Why per-ticker then combine
---------------------------
Each ticker trades its own book: signal → next-bar fill → fees/slippage, exactly
as the single-symbol engine does. We do not try to make one giant strategy emit
cross-asset signals (that is a different, coupled design). The portfolio layer is
deliberately an *aggregation* layer:

1. load bars for every ticker concurrently (``asyncio.gather``);
2. backtest each ticker in isolation with its allocated capital;
3. align the equity curves on the union of their timestamps, carrying each
   curve forward (capital that has not started trading is parked as cash);
4. sum the aligned curves → portfolio equity;
5. compute portfolio metrics + per-ticker attribution + return correlation.

A ticker that fails to load (bad symbol, exchange down) is isolated into
``errors`` and the rest of the portfolio still runs.
"""
from __future__ import annotations

import asyncio
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from trading.application.instruments import resolve_symbol
from trading.application.strategy_factory import build_strategy
from trading.adapters.cache import TtlCache
from trading.config import settings
from trading.domain import Bar, DataFetchError, Exchange, StrategyError
from trading.ports import Strategy

from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import BacktestMetrics, compute_metrics, returns_from_equity

if TYPE_CHECKING:  # pragma: no cover
    from trading.application.cancellation import CancelToken

__all__ = [
    "TickerSpec",
    "PortfolioBacktestConfig",
    "TickerBacktest",
    "CorrelationMatrix",
    "PortfolioBacktestResult",
    "run_portfolio_backtest",
    "resolve_exchanges",
    "load_bars",
    "clear_bar_cache",
    "bar_cache_stats",
]

#: OHLCV cache keyed by the full request. Process-wide (not loop-bound) because
#: the entries are inert frozen ``Bar`` data — this also lets a Celery worker that
#: runs each task in a fresh event loop still reuse data across tasks.
_bar_cache = TtlCache(default_ttl=settings.data_cache_ttl)
_bar_cache_hits = 0
_bar_cache_misses = 0


def clear_bar_cache() -> None:
    """Drop all cached bars (tests, or after a manual data refresh)."""
    global _bar_cache_hits, _bar_cache_misses
    _bar_cache.clear()
    _bar_cache_hits = 0
    _bar_cache_misses = 0


def bar_cache_stats() -> dict[str, int]:
    """Cache counters — surfaced for tests and ops debugging."""
    return {"hits": _bar_cache_hits, "misses": _bar_cache_misses, "size": len(_bar_cache)}


_SOURCES: dict[str, list[Exchange]] = {
    "moex": [Exchange.MOEX],
    "yfinance": [Exchange.YFINANCE],
    "bybit": [Exchange.BYBIT],
    "webull": [Exchange.WEBULL],
    "auto": [Exchange.MOEX, Exchange.YFINANCE, Exchange.BYBIT],
}


# ── configuration ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TickerSpec:
    """One portfolio leg: a symbol + the strategy and settings it trades with."""

    symbol: str
    strategy: str = "sma_crossover"
    params: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    capital: float | None = None  # explicit allocation; overrides ``weight``
    source: str = "auto"  # auto | moex | yfinance | bybit | webull | synthetic
    timeframe: str = "1d"
    limit: int = 5000
    enabled: bool = True


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    position_fraction: float = 0.95
    periods_per_year: int = 252


# ── results ────────────────────────────────────────────────────────────


@dataclass
class TickerBacktest:
    symbol: str
    strategy: str
    source: str
    timeframe: str
    weight: float
    capital: float
    result: BacktestResult

    @property
    def metrics(self) -> BacktestMetrics:
        return self.result.metrics

    @property
    def equity_curve(self) -> np.ndarray:
        return self.result.equity_curve

    @property
    def total_return(self) -> float:
        return self.result.metrics.total_return


@dataclass
class CorrelationMatrix:
    symbols: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]


@dataclass
class PortfolioBacktestResult:
    equity_curve: np.ndarray
    times: tuple[datetime, ...]
    metrics: BacktestMetrics
    initial_cash: float
    tickers: tuple[TickerBacktest, ...]
    errors: tuple[dict[str, str], ...] = ()
    correlation: CorrelationMatrix | None = None
    #: Per-ticker equity curves forward-filled onto ``times`` (same length/order
    #: as ``tickers``), so charts and attribution line up exactly.
    aligned_equity: tuple[np.ndarray, ...] = ()

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def per_ticker_return(self) -> dict[str, float]:
        return {t.symbol: t.total_return for t in self.tickers}


# ── data resolution ────────────────────────────────────────────────────


def resolve_exchanges(symbol: str, source: str) -> tuple[list[Exchange] | None, str]:
    """Map ``(symbol, source)`` → ``(exchanges, fetch_symbol)``.

    ``exchanges`` is ``None`` for the synthetic source. ``auto`` resolves the
    ticker to the exchange that actually lists it (MOEX/Bybit/yFinance) so we
    never ask the wrong exchange for a symbol.
    """
    info = resolve_symbol(symbol)
    if source in ("", "auto"):
        return [Exchange(info["exchange"])], info["fetch_symbol"]
    if source == "synthetic":
        return None, symbol
    exchanges = _SOURCES.get(source)
    if exchanges is None:
        raise DataFetchError(f"unknown source '{source}'")
    if Exchange(info["exchange"]) in exchanges:
        return list(exchanges), info["fetch_symbol"]
    return list(exchanges), symbol


async def load_bars(spec: TickerSpec, *, refresh: bool = False) -> list[Bar]:
    """Load OHLCV for one spec (synthetic or a real fetcher with fallback).

    Results are cached for ``settings.data_cache_ttl`` seconds, so re-running a
    backtest after tweaking a parameter is instant instead of re-downloading every
    ticker. Pass ``refresh=True`` to force a fetch (e.g. a "reload data" button).

    ``SYNTH`` is the built-in synthetic instrument: it always resolves to the
    deterministic synthetic feed so demos and tests never touch the network.
    """
    global _bar_cache_hits, _bar_cache_misses

    cache_key = "|".join(
        (spec.symbol.upper(), spec.source, spec.timeframe, str(spec.limit))
    )
    if settings.data_cache_ttl > 0 and not refresh:
        cached = await _bar_cache.get(cache_key)
        if cached is not None:
            _bar_cache_hits += 1
            return cached
    _bar_cache_misses += 1

    # L2: Redis FIFO bar cache (≤ 500 bars/instrument), shared across processes
    # and event loops; the in-process TTL cache above is only L1. Oversized
    # windows bypass L2 (it is capped by design).
    from trading.adapters.cache import (
        MAX_BARS_PER_INSTRUMENT,
        cache_key as bar_cache_key,
        loop_bar_cache,
    )

    l2 = None
    l2_key = bar_cache_key(spec.source or "auto", spec.symbol, spec.timeframe)
    if spec.limit <= MAX_BARS_PER_INSTRUMENT:
        l2 = await loop_bar_cache()
        if not refresh:
            cached_l2 = await l2.get_bars(l2_key, spec.limit)
            if cached_l2 is not None:
                return cached_l2

    if spec.source == "synthetic" or (
        spec.source in ("", "auto") and spec.symbol.upper() == "SYNTH"
    ):
        from trading.adapters.fetchers.synthetic import SyntheticFetcher

        seed = zlib.crc32(spec.symbol.encode()) if spec.symbol.upper() != "SYNTH" else 0
        bars = await SyntheticFetcher(seed=seed).get_ohlcv(
            spec.symbol, spec.timeframe, limit=spec.limit
        )
    else:
        from trading.adapters.fetchers import loop_registry

        exchanges, fetch_symbol = resolve_exchanges(spec.symbol, spec.source)
        assert exchanges is not None  # synthetic handled above
        # ``loop_registry()`` reuses one client set per event loop instead of
        # rebuilding fetchers (and their TLS handshakes) for every ticker.
        bars = await loop_registry().get_ohlcv(
            exchanges, fetch_symbol, spec.timeframe, limit=spec.limit
        )

    if settings.data_cache_ttl > 0:
        await _bar_cache.set(cache_key, bars, ttl=settings.data_cache_ttl)
    if l2 is not None:
        await l2.put_bars(l2_key, bars)  # write-through (FIFO-evicts oldest)
    return bars


# ── allocation ─────────────────────────────────────────────────────────


def _allocate(specs: Sequence[TickerSpec], total: float) -> list[float]:
    """Split ``total`` across specs: explicit ``capital`` first, remainder by weight."""
    capitals: list[float | None] = [s.capital if s.capital is not None else None for s in specs]
    explicit = sum(c for c in capitals if c is not None)
    remaining = max(total - explicit, 0.0)
    weights = [max(s.weight, 0.0) for s in specs]
    wsum = sum(w for w, c in zip(weights, capitals) if c is None) or 1.0
    return [
        float(c) if c is not None else remaining * w / wsum
        for w, c in zip(weights, capitals)
    ]


# ── correlation ────────────────────────────────────────────────────────


def _correlation(
    tickers: Sequence[TickerBacktest], grids: Sequence[np.ndarray]
) -> CorrelationMatrix | None:
    series: list[np.ndarray] = []
    symbols: list[str] = []
    for t, grid in zip(tickers, grids):
        r = returns_from_equity(grid)
        if r.size >= 2 and not np.allclose(r, r[0]):  # skip zero-variance curves
            series.append(r)
            symbols.append(t.symbol)
    if len(symbols) < 2:
        return None
    n = min(len(s) for s in series)
    mat = np.vstack([s[:n] for s in series])
    corr = np.corrcoef(mat)
    corr = np.nan_to_num(corr, nan=0.0)
    return CorrelationMatrix(
        symbols=tuple(symbols),
        matrix=tuple(tuple(float(x) for x in row) for row in corr),
    )


# ── engine ─────────────────────────────────────────────────────────────


def _align(
    curves: Sequence[np.ndarray], times: Sequence[Sequence[datetime]]
) -> tuple[list[datetime], list[np.ndarray]]:
    """Forward-fill each curve onto the sorted union of all timestamps."""
    union = sorted({ts for tss in times for ts in tss})
    idx = {ts: i for i, ts in enumerate(union)}
    aligned: list[np.ndarray] = []
    for curve, tss in zip(curves, times):
        out = np.empty(len(union), dtype=float)
        pos = 0
        last = float(curve[0])  # before the first bar the allocation is cash
        for j, ts in enumerate(union):
            while pos < len(tss) and tss[pos] <= ts:
                last = float(curve[pos])
                pos += 1
            out[j] = last
        aligned.append(out)
    return union, aligned


async def run_portfolio_backtest(
    specs: Sequence[TickerSpec],
    config: PortfolioBacktestConfig | None = None,
    *,
    bars_by_symbol: Mapping[str, Sequence[Bar]] | None = None,
    cancel: "CancelToken | None" = None,
    refresh: bool = False,
) -> PortfolioBacktestResult:
    """Backtest every enabled spec and aggregate into one portfolio curve.

    ``bars_by_symbol`` injects pre-loaded data (tests / offline); otherwise each
    ticker is fetched concurrently through the real fetcher registry.

    ``cancel`` makes the run interruptible: it is checked between tickers and
    the event loop is yielded after each leg so a concurrent cancel request can
    actually be served.

    ``refresh`` bypasses the OHLCV cache so every ticker is re-downloaded.
    """
    cfg = config or PortfolioBacktestConfig()
    if cancel is not None:
        cancel.check()
    active = [s for s in specs if s.enabled]
    if not active:
        raise ValueError("portfolio backtest requires at least one enabled ticker")

    capitals = _allocate(active, cfg.initial_cash)
    injected = dict(bars_by_symbol or {})

    async def _load(spec: TickerSpec) -> list[Bar]:
        if spec.symbol in injected:
            return list(injected[spec.symbol])
        return await load_bars(spec, refresh=refresh)

    loaded = await asyncio.gather(*(_load(s) for s in active), return_exceptions=True)
    if cancel is not None:
        cancel.check()

    tickers: list[TickerBacktest] = []
    errors: list[dict[str, str]] = []
    curves: list[np.ndarray] = []
    time_grids: list[tuple[datetime, ...]] = []

    for spec, capital, bars in zip(active, capitals, loaded):
        if cancel is not None:
            cancel.check()
        if isinstance(bars, BaseException):
            errors.append({"symbol": spec.symbol, "error": str(bars)})
            continue
        try:
            return_exc = None
            bars = sorted(bars, key=lambda b: b.timestamp)
            if len(bars) < 2:
                raise DataFetchError(f"need at least 2 bars, got {len(bars)}")
            strategy: Strategy = build_strategy(spec.strategy, spec.symbol, spec.params)
            ticker_cfg = BacktestConfig(
                initial_cash=max(capital, 1.0),
                fee_rate=cfg.fee_rate,
                slippage=cfg.slippage,
                position_fraction=cfg.position_fraction,
                periods_per_year=cfg.periods_per_year,
            )
            result = await run_backtest(strategy, bars, ticker_cfg)
        except (DataFetchError, StrategyError, ValueError, KeyError) as exc:
            errors.append({"symbol": spec.symbol, "error": str(exc)})
            continue

        weight = capital / cfg.initial_cash if cfg.initial_cash else 0.0
        ticker = TickerBacktest(
            symbol=spec.symbol, strategy=spec.strategy, source=spec.source,
            timeframe=spec.timeframe, weight=weight, capital=capital, result=result,
        )
        tickers.append(ticker)
        curves.append(result.equity_curve)
        time_grids.append(tuple(b.timestamp for b in bars))
        # Hand the loop back so a pending POST /backtest/cancel is served.
        await asyncio.sleep(0)

    if cancel is not None:
        cancel.check()
    if not tickers:
        raise DataFetchError(
            "no ticker produced a result: "
            + "; ".join(f"{e['symbol']}: {e['error']}" for e in errors)
        )

    union, aligned = _align(curves, time_grids)
    if cancel is not None:
        cancel.check()
    # Capital belonging to failed legs stays as cash so the curve still starts
    # at ``initial_cash`` and the drawdown/return math stays honest.
    cash_floor = max(cfg.initial_cash - sum(t.capital for t in tickers), 0.0)
    portfolio_equity = np.sum(np.vstack(aligned), axis=0) + cash_floor

    trade_pnls = [t.realized_pnl for tb in tickers for t in tb.result.trades]
    metrics = compute_metrics(
        portfolio_equity, trade_pnls, periods_per_year=cfg.periods_per_year
    )

    return PortfolioBacktestResult(
        equity_curve=portfolio_equity,
        times=tuple(union),
        metrics=metrics,
        initial_cash=cfg.initial_cash,
        tickers=tuple(tickers),
        errors=tuple(errors),
        correlation=_correlation(tickers, aligned),
        aligned_equity=tuple(aligned),
    )
