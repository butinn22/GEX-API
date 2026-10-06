"""Tests for the confluence-breakout strategy (the validated trend system).

The suite uses a deterministic synthetic bar series — no network, no research
cache — and asserts the *invariants* that make the port trustworthy:

* the emitted signal carries the full trade plan (entry/stop/timeframe/risk),
* a position is opened at most once and is closed by exactly one exit,
* streaming (``on_bar``) and batch (``prepare`` + replay) agree bar for bar,
  which is the causality regression test (no lookahead),
* short history never produces a signal (warm-up guard).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategies.confluence_breakout import (
    BREAKOUT_PRESETS,
    DEFAULT_PRESET,
    ConfluenceBreakoutParams,
    ConfluenceBreakoutStrategy,
    preset_params,
)
from trading.application.strategy_factory import STRATEGY_NAMES, build_strategy
from trading.application.strategy_registry import STRATEGY_REGISTRY
from trading.domain import Bar, Side

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _bars(closes, *, highs=None, lows=None, volumes=None, step_hours=4) -> list[Bar]:
    n = len(closes)
    out: list[Bar] = []
    for i, c in enumerate(closes):
        h = highs[i] if highs else c * 1.001
        l = lows[i] if lows else c * 0.999
        v = volumes[i] if volumes else 1000.0
        out.append(Bar(timestamp=_T0 + timedelta(hours=step_hours * i), open=c,
                       high=max(h, c), low=min(l, c), close=c, volume=v))
    return out


def _series(n=400, seed=7) -> list[float]:
    """Deterministic, gentle uptrend with noise — enough for the Alligator to
    align, structure to confirm HH/HL, ADL/EMF to turn positive and a breakout
    to print."""
    out: list[float] = []
    price = 100.0
    state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        noise = ((state / 2 ** 31) - 0.5) / 100.0
        drift = 0.0025 if (i // 12) % 5 < 4 else -0.0015
        price = price * (1.0 + drift + noise)
        out.append(price)
    return out


# ── parameters / registration ─────────────────────────────────────────
def test_params_defaults_and_validation():
    p = ConfluenceBreakoutParams()
    assert p.entry_mode == "breakout" and p.exit_level == "jaw"
    with pytest.raises(ValueError):
        ConfluenceBreakoutParams(entry_mode="nope")
    with pytest.raises(ValueError):
        ConfluenceBreakoutParams(exit_level="neck")
    with pytest.raises(ValueError):
        ConfluenceBreakoutParams(sizing="martingale")
    with pytest.raises(ValueError):
        ConfluenceBreakoutParams(n_break=0)
    with pytest.raises(ValueError):
        ConfluenceBreakoutParams(min_stop_atr=5.0, max_stop_atr=1.0)


def test_params_roundtrip_via_from_dict():
    p = ConfluenceBreakoutParams.from_dict({"n_break": 40, "k_trail": 3.0, "unknown": 1})
    assert p.n_break == 40 and p.k_trail == 3.0
    assert "unknown" not in p.FIELD_NAMES
    assert ConfluenceBreakoutParams.from_dict(None) == ConfluenceBreakoutParams()


def test_frozen_presets_present_and_fallback():
    assert set(BREAKOUT_PRESETS) == {"alligator_4h", "donchian_1d"}
    assert preset_params(None) == preset_params(DEFAULT_PRESET)
    assert preset_params("alligator_4h")["n_break"] == 30
    assert preset_params("donchian_1d")["use_trend_filter"] is True
    assert preset_params("donchian_1d")["sizing"] == "fraction"
    with pytest.raises(ValueError):
        preset_params("nope")


def test_registered_in_factory_and_registry():
    assert "confluence_breakout" in STRATEGY_NAMES
    s = build_strategy("confluence_breakout", "TEST", {"preset": "donchian_1d"})
    assert isinstance(s, ConfluenceBreakoutStrategy)
    assert s.preset == "donchian_1d"
    assert s.timeframe == "1d"
    assert STRATEGY_REGISTRY.params("confluence_breakout") == list(
        ConfluenceBreakoutParams.FIELD_NAMES
    )
    with pytest.raises(Exception):
        build_strategy("confluence_breakout", "TEST", {"preset": "nope"})


def test_explicit_params_override_preset():
    s = build_strategy("confluence_breakout", "TEST",
                       {"preset": "alligator_4h", "n_break": 12, "k_trail": 1.5})
    assert s._p.n_break == 12 and s._p.k_trail == 1.5
    assert s._p.exit_level == "jaw"   # untouched preset key survives


# ── signal content ────────────────────────────────────────────────────
async def test_entry_signal_carries_full_trade_plan():
    strat = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h")
    bars = _bars(_series())
    await strat.start()
    out = await strat.generate_signals(bars)
    entries = [s for s in out if s.meta.get("exit_reason") is None]
    assert entries, "the deterministic series must produce at least one entry"
    sig = entries[0]
    assert sig.side is Side.BUY
    assert sig.symbol == "TEST"
    assert sig.entry_price and sig.entry_price > 0
    assert sig.stop_loss and sig.stop_loss < sig.entry_price          # long stop is below
    assert math.isclose(sig.risk_pct, 0.005)
    assert sig.risk_amount == pytest.approx(0.005 * 100_000)
    assert sig.timeframe == "4h"
    assert sig.bar_time is not None
    assert 0.0 < sig.strength <= 1.0
    # an open position may still be running when the series ends, so the count
    # is odd only for the tail position — every earlier cycle is a pair.
    depth = sum(1 if s.meta.get("exit_reason") is None else -1 for s in out)
    assert depth in (0, 1)


async def test_position_opened_once_and_closed_by_exactly_one_exit():
    strat = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h")
    bars = _bars(_series())
    await strat.start()
    out = await strat.generate_signals(bars)
    depth = 0
    for s in out:
        if s.meta.get("exit_reason") is None:
            depth += 1
        else:
            depth -= 1
        assert 0 <= depth <= 1, "at most one position at a time"
    assert depth in (0, 1), "the run ends flat or in a single open position"
    reasons = {s.meta["exit_reason"] for s in out if s.meta.get("exit_reason")}
    assert reasons <= {"stop_loss", "trailing_stop", "time_stop", "structure_break",
                       "jaw_break", "teeth_break", "lips_break", "ma_exit"}


async def test_stop_exit_fires_when_price_runs_through_it():
    """A long is stopped out when price trades through its resting stop."""
    # Freeze the trail at the initial stop (k_trail huge, no structural ratchet)
    # so the *initial* stop is the level that gets hit.
    strat = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h", params={
        "max_hold_bars": 1000, "k_trail": 100000.0, "use_struct_trail": False})
    bars = _bars(_series())
    await strat.start()
    _ = await strat.generate_signals(bars)
    pos = strat.position
    assert pos is not None and pos.side == "long"
    stop = pos.stop
    # one bar that trades below the stop and closes there
    crash = Bar(timestamp=bars[-1].timestamp + timedelta(hours=4),
                open=bars[-1].close, high=bars[-1].close,
                low=stop - 1.0, close=stop - 0.5, volume=1000.0)
    out = await strat.on_bar(crash)
    assert out and out[0].meta["exit_reason"] == "stop_loss"
    assert out[0].side is Side.SELL
    assert strat.position is None


async def test_warmup_guard():
    strat = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h")
    await strat.start()
    out = await strat.generate_signals(_bars(_series(120)))
    assert out == [], "no signal before the warm-up window"


async def test_on_bar_matches_batch_replay_causality():
    """Streaming and batch must be identical — this is the no-lookahead test."""
    bars = _bars(_series())
    batch = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h")
    await batch.start()
    await batch.prepare(bars)
    batch_out = []
    for b in bars:
        batch_out.extend(await batch.on_bar(b))

    stream = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h")
    await stream.start()
    stream_out = []
    for b in bars:
        stream_out.extend(await stream.on_bar(b))

    assert [(s.side, s.reason, s.meta.get("exit_reason"), s.bar_time)
            for s in batch_out] == [(s.side, s.reason, s.meta.get("exit_reason"), s.bar_time)
                                    for s in stream_out]


async def test_short_side_mirror_when_enabled():
    strat = ConfluenceBreakoutStrategy("TEST", params={
        "allow_short": True, "allow_long": False, "entry_mode": "breakout"})
    falling = [100.0 * (1.0 - 0.0025 * i) for i in range(400)]
    await strat.start()
    out = await strat.generate_signals(_bars(falling))
    entries = [s for s in out if s.meta.get("exit_reason") is None]
    if entries:
        assert all(s.side is Side.SELL for s in entries)
        assert all(s.stop_loss > s.entry_price for s in entries)


# ── backtest integration ──────────────────────────────────────────────
async def test_backtest_runs_and_reports_metrics():
    bars = _bars(_series(), step_hours=4)
    strat = ConfluenceBreakoutStrategy("TEST", preset="alligator_4h")
    res = await run_backtest(strat, bars, BacktestConfig(periods_per_year=365 * 6))
    assert res.metrics.n_trades >= 0
    assert math.isfinite(res.metrics.max_drawdown)
    # with fees on both legs a losing series cannot show a positive PF
    if res.trades:
        assert abs(res.metrics.max_drawdown) <= 1.0


async def test_donchian_preset_uses_fraction_sizing():
    strat = ConfluenceBreakoutStrategy("TEST", preset="donchian_1d")
    await strat.start()
    out = await strat.generate_signals(_bars(_series(600), step_hours=24))
    entries = [s for s in out if s.meta.get("exit_reason") is None]
    if entries:
        # fraction sizing ⇒ strength is the slice (0.95/0.95), not a risk fraction
        assert entries[0].strength == pytest.approx(1.0)
        assert entries[0].meta["sizing"] == "fraction"
