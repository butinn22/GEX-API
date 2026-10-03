"""Tests for the trend_confluence strategy, trade analysis and optimizer."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from trading.application.backtest.engine import BacktestConfig, Trade, run_backtest
from trading.application.backtest.optimize import optimize_strategy
from trading.application.backtest.trade_analysis import (
    analyze_trades,
    recommend_adjustments,
)
from trading.application.strategies.trend_confluence import (
    TrendConfluenceStrategy,
    resolve_walls,
)
from trading.application.strategy_factory import build_strategy
from trading.domain import Bar, PositionSide, Side


def _trending_bars(n: int = 600, *, start: float = 100.0, drift: float = 0.0015) -> list[Bar]:
    """Deterministic noisy uptrend with pullbacks (no RNG — reproducible)."""
    bars: list[Bar] = []
    price = start
    t0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        # Sine wobble creates the pullbacks the strategy is meant to buy.
        import math

        ret = drift + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        high = max(prev, price) * 1.004
        low = min(prev, price) * 0.996
        bars.append(Bar(timestamp=t0 + timedelta(days=i), open=prev,
                        high=high, low=low, close=price, volume=1000.0))
    return bars


# ── walls parsing ──────────────────────────────────────────────────────


def test_resolve_walls_from_levels() -> None:
    w = resolve_walls({"enabled": True, "call_wall": 700, "put_wall": 600, "gamma_flip": 650,
                       "walls": [{"strike": 620, "oi": 50000, "kind": "put"}],
                       "min_wall_oi": 10000})
    assert w.enabled and w.call_wall == 700 and w.put_wall == 600 and w.gamma_flip == 650
    assert len(w.significant_walls()) == 1
    assert w.significant_walls()[0].strike == 620


def test_resolve_walls_from_chain() -> None:
    # Cumulative net GEX: [-5, -7, -4, +4] → zero-gamma crossing between 110
    # and 120 (interpolated to 115).
    w = resolve_walls({"strikes": [90, 100, 110, 120], "gex_net": [-5, -2, 3, 8]})
    assert w.gamma_flip is not None and 110 < w.gamma_flip < 120
    assert w.call_wall == 120  # largest positive net GEX
    assert w.put_wall == 90  # largest negative net GEX


def test_resolve_walls_empty() -> None:
    assert not resolve_walls(None).enabled
    assert not resolve_walls({}).enabled


# ── strategy behaviour ─────────────────────────────────────────────────


def test_factory_builds_trend_confluence() -> None:
    strat = build_strategy("trend_confluence", "SYNTH", {"zone_atr": 0.4, "min_confluence": 2})
    assert isinstance(strat, TrendConfluenceStrategy)
    assert strat._p.zone_atr == 0.4


def test_no_signals_during_warmup() -> None:
    bars = _trending_bars(400)
    strat = TrendConfluenceStrategy("SYNTH", params={"trendline_refresh": 10})
    sigs = asyncio.run(strat.generate_signals(bars))
    assert sigs, "expected at least one signal on a trending series"
    first = min(s.timestamp for s in sigs)
    assert first >= bars[strat.min_bars - 1].timestamp


def test_backtest_never_drives_equity_negative() -> None:
    """Regression: exits must reuse the entry strength.

    A full-strength exit after a half-strength entry overshoots the position
    and silently flips it — with an uptrend that turned 'exits' into runaway
    shorts and equity went negative.
    """
    bars = _trending_bars(600)
    strat = build_strategy("trend_confluence", "SYNTH", {"trendline_refresh": 10})
    res = asyncio.run(run_backtest(strat, bars, BacktestConfig()))
    assert res.equity_curve.min() > 0
    assert len(res.trades) > 0
    # Long-only in a clean uptrend: shorts should be rare and never ruinous.
    short_pnl = sum(t.realized_pnl for t in res.trades if t.side is PositionSide.SHORT)
    assert short_pnl > -res.equity_curve[-1] * 0.5


def test_prepare_streaming_equivalence() -> None:
    """prepare()+on_bar must emit the same signals as generate_signals()."""
    bars = _trending_bars(400)
    a = TrendConfluenceStrategy("SYNTH", params={"trendline_refresh": 10})
    batch = [(s.side, s.timestamp, s.reason) for s in asyncio.run(a.generate_signals(bars))]

    b = TrendConfluenceStrategy("SYNTH", params={"trendline_refresh": 10})
    asyncio.run(b.prepare(bars))
    streamed: list = []
    for bar in bars:
        streamed.extend((s.side, s.timestamp, s.reason) for s in asyncio.run(b.on_bar(bar)))
    assert batch == streamed


def test_call_wall_blocks_long_entries() -> None:
    bars = _trending_bars(400)
    # CALL wall sitting just above the start of the series caps every long.
    strat = TrendConfluenceStrategy("SYNTH", params={
        "trendline_refresh": 10,
        "options": {"enabled": True, "call_wall": 101.0},
    })
    sigs = asyncio.run(strat.generate_signals(bars))
    entries = [s for s in sigs if s.side is Side.BUY and s.reason.startswith("add_long")]
    assert not entries, "no long entries should fire right under a large CALL wall"


# ── trade analysis ─────────────────────────────────────────────────────


def _trade(pnl: float, side: PositionSide = PositionSide.LONG) -> Trade:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return Trade(symbol="X", side=side, entry_price=100.0, exit_price=100.0 + pnl,
                 quantity=1.0, realized_pnl=pnl, entry_time=ts, exit_time=ts)


def test_analyze_trades_math() -> None:
    trades = [_trade(100), _trade(200), _trade(-50), _trade(-100), _trade(-50)]
    a = analyze_trades(trades)
    assert a.n_trades == 5 and a.wins == 2 and a.losses == 3
    assert a.win_rate == pytest.approx(0.4)
    assert a.gross_profit == 300 and a.gross_loss == -200
    assert a.profit_factor == pytest.approx(1.5)
    assert a.avg_win == pytest.approx(150)
    assert a.avg_loss == pytest.approx(-200 / 3)
    assert a.payoff_ratio == pytest.approx(150 / (200 / 3))
    assert a.expectancy == pytest.approx(20)
    assert a.max_loss_streak == 3
    assert a.max_win_streak == 2
    assert a.histogram["counts"] and sum(a.histogram["counts"]) == 5


def test_analyze_trades_by_side() -> None:
    trades = [_trade(100, PositionSide.LONG), _trade(-80, PositionSide.SHORT)]
    a = analyze_trades(trades)
    assert a.long.n == 1 and a.long.wins == 1 and a.long.pnl == 100
    assert a.short.n == 1 and a.short.wins == 0 and a.short.pnl == -80


def test_analyze_trades_empty() -> None:
    a = analyze_trades([])
    assert a.n_trades == 0 and a.win_rate == 0.0
    assert a.as_dict()["profit_factor"] is None


def test_recommendations_fire_on_weak_shorts() -> None:
    trades = [_trade(150, PositionSide.LONG) for _ in range(6)]
    trades += [_trade(-90, PositionSide.SHORT) for _ in range(5)]
    recs = recommend_adjustments(analyze_trades(trades), {"allow_short": True})
    params = {r["parameter"] for r in recs}
    assert "allow_short" in params


def test_recommendations_low_sample() -> None:
    recs = recommend_adjustments(analyze_trades([_trade(10)]), {})
    assert recs and recs[0]["parameter"] == "min_confluence"


# ── optimizer ──────────────────────────────────────────────────────────


def test_optimize_returns_ranked_candidates() -> None:
    bars = _trending_bars(400)
    res = optimize_strategy(
        "trend_confluence", "SYNTH", bars,
        grid={"zone_atr": [0.3, 0.5], "atr_trail_mult": [2.0, 3.0]},
        cfg=BacktestConfig(),
    )
    assert res.n_candidates == 4
    assert res.best_params["zone_atr"] in (0.3, 0.5)
    scores = [c.score for c in res.leaderboard]
    assert scores == sorted(scores, reverse=True)
    assert res.trade_analysis is not None
    payload = res.as_dict()
    assert payload["leaderboard"] and "baseline" in payload and "best" in payload
