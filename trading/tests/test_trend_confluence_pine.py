"""Tests for the Pine-confluence strategy (percent TP / trailing / guarded adds).

Covers the user-facing contract:
* TP and trailing are SEPARATE on/off switches with percent distances;
* a TP hit closes only ``tp_close_pct`` of the position, keeps the strategy
  in-position, respects the cooldown, and rescales exit strengths;
* adds fire only into an open position of the matching side (position-state
  guard) — never as an opener, never onto an exit bar;
* no exit of any kind is emitted while flat.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategy_factory import build_strategy
from trading.application.strategies.trend_confluence_pine import (
    PINE_PARAM_NAMES,
    PineConfluenceParams,
    PineConfluenceStrategy,
)
from trading.domain import Bar, Price, StrategyError

TS0 = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _trending_bars(n: int = 600, *, start: float = 100.0, drift: float = 0.0015) -> list[Bar]:
    bars: list[Bar] = []
    price = start
    for i in range(n):
        ret = drift + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        bars.append(Bar(timestamp=TS0 + timedelta(days=i), open=prev,
                        high=max(prev, price) * 1.004, low=min(prev, price) * 0.996,
                        close=price, volume=1000.0))
    return bars


def _key(sig) -> tuple:
    return (sig.timestamp, sig.side, sig.reason, round(float(sig.price), 10))


# ── params ─────────────────────────────────────────────────────────────


def test_params_defaults_and_validation() -> None:
    p = PineConfluenceParams.from_dict(None)
    assert p.tp_enabled and p.tp_percent == 2.0 and p.tp_close_pct == 50.0
    assert p.tp_cooldown_bars == 3
    assert p.trailing_enabled and p.trailing_percent == 1.0
    assert p.allow_adds is False and p.add_cooldown_bars == 10
    # unified knobs still flow through
    p2 = PineConfluenceParams.from_dict({"tp_percent": 3.5, "zone_atr": 0.7,
                                         "emf_mode": "require"})
    assert p2.tp_percent == 3.5 and p2.zone_atr == 0.7 and p2.emf_mode == "require"
    # unknown keys are filtered, bad values rejected
    p3 = PineConfluenceParams.from_dict({"nonsense": 1})
    assert p3.tp_percent == 2.0
    with pytest.raises(ValueError):
        PineConfluenceParams.from_dict({"tp_percent": 0})
    with pytest.raises(ValueError):
        PineConfluenceParams.from_dict({"tp_close_pct": 101})
    with pytest.raises(ValueError):
        PineConfluenceParams.from_dict({"trailing_percent": -1})
    assert "tp_enabled" in PINE_PARAM_NAMES and "trailing_percent" in PINE_PARAM_NAMES


def test_factory_builds_and_validates() -> None:
    s = build_strategy("trend_confluence_pine", "SYNTH",
                       {"tp_percent": 3.5, "allow_adds": True})
    assert isinstance(s, PineConfluenceStrategy)
    assert s._pp.tp_percent == 3.5 and s._pp.allow_adds
    with pytest.raises(StrategyError):
        build_strategy("trend_confluence_pine", "SYNTH", {"tp_percent": 0})


def test_overlapping_overlays_default_off_but_overridable() -> None:
    s = PineConfluenceStrategy("SYNTH")
    assert s._up.use_risk_exits is False
    assert s._p.use_trailing is False
    assert s._p.tp_r == 0.0
    s2 = PineConfluenceStrategy("SYNTH", params={"use_risk_exits": True, "tp_r": 1.5})
    assert s2._up.use_risk_exits is True and s2._p.tp_r == 1.5


# ── position-state guard ───────────────────────────────────────────────


async def _prepared(params: dict) -> tuple[PineConfluenceStrategy, list[Bar]]:
    bars = _trending_bars(400)
    s = PineConfluenceStrategy("SYNTH", params=params)
    await s.prepare(bars)
    return s, bars


def test_no_exit_while_flat() -> None:
    s, bars = asyncio.run(_prepared({}))
    i = s.min_bars + 5
    sig = s._exit_check(i, float(bars[i].close), 2.0,
                        Price(float(bars[i].close)), bars[i].timestamp, regime=1)
    assert sig is None


def test_no_add_while_flat_even_if_emf_flag_true() -> None:
    s, bars = asyncio.run(_prepared({"allow_adds": True}))
    # Force the EMF add column on everywhere: a flat strategy must still
    # never emit an add (no opening adds).
    n = len(bars)
    s._emf_features = pd.DataFrame({
        "long_add_signal": [True] * n, "short_add_signal": [True] * n,
    })
    for i in range(s.min_bars, min(s.min_bars + 20, n)):
        out = s._signal_at(i, bars[i].timestamp)
        assert all(sig.reason not in ("add_long", "add_short") for sig in out)


# ── percent take profit ────────────────────────────────────────────────


def test_tp_partial_close_keeps_position_and_rescales() -> None:
    s, bars = asyncio.run(_prepared({"tp_enabled": True, "tp_percent": 2.0,
                                     "tp_close_pct": 50, "tp_cooldown_bars": 3,
                                     "trailing_enabled": False, "use_emf_exits": False}))
    s._open("long", 100.0, 0.4, atr=2.0)
    s._best = 100.0
    px = Price(103.0)
    ts = bars[300].timestamp
    sig = s._exit_check(300, 103.0, 2.0, px, ts, regime=1)
    assert sig is not None and sig.reason == "take_profit_pct"
    assert sig.strength == pytest.approx(0.2)        # 0.4 × 50%
    assert s._side == "long"                          # partial: still in
    assert s._entry_strength == pytest.approx(0.2)    # remaining size
    # cooldown: same level, next bar → blocked
    assert s._exit_check(301, 103.0, 2.0, px, ts, regime=1) is None
    # after the cooldown another slice fires at the same anchored level
    sig2 = s._exit_check(303, 103.0, 2.0, px, ts, regime=1)
    assert sig2 is not None and sig2.reason == "take_profit_pct"
    assert sig2.strength == pytest.approx(0.1)        # 0.2 × 50%
    assert s._entry_strength == pytest.approx(0.1)


def test_tp_disabled_emits_no_tp() -> None:
    s, bars = asyncio.run(_prepared({"tp_enabled": False, "trailing_enabled": False,
                                     "stop_atr": 0.0, "use_trailing": False}))
    s._open("long", 100.0, 0.4, atr=2.0)
    s._best = 110.0
    ts = bars[300].timestamp
    sig = s._exit_check(300, 115.0, 2.0, Price(115.0), ts, regime=1)
    assert sig is None or sig.reason != "take_profit_pct"


def test_short_tp_anchored_below_entry() -> None:
    s, bars = asyncio.run(_prepared({"trailing_enabled": False, "use_emf_exits": False,
                                     "invalidation_atr": 1e9, "stop_atr": 0.0}))
    s._open("short", 100.0, 0.4, atr=2.0)
    s._best = 100.0
    ts = bars[300].timestamp
    assert s._exit_check(300, 98.5, 2.0, Price(98.5), ts, regime=-1) is None
    sig = s._exit_check(301, 98.0, 2.0, Price(98.0), ts, regime=-1)
    assert sig is not None and sig.reason == "take_profit_pct"
    assert sig.side.value == "buy"


# ── percent trailing stop ──────────────────────────────────────────────


def test_trailing_ratchet_exits_full_position() -> None:
    s, bars = asyncio.run(_prepared({"trailing_enabled": True,
                                     "trailing_percent": 1.0, "tp_enabled": False,
                                     "use_emf_exits": False}))
    s._open("long", 100.0, 0.4, atr=2.0)
    s._best = 110.0                      # trail = 108.9
    ts = bars[300].timestamp
    # above the trail → nothing
    assert s._exit_check(300, 109.0, 2.0, Price(109.0), ts, regime=1) is None
    # below the trail → full exit at the rescaled strength, position closed
    sig = s._exit_check(301, 108.8, 2.0, Price(108.8), ts, regime=1)
    assert sig is not None and sig.reason == "trailing_stop_pct"
    assert sig.strength == pytest.approx(0.4)
    assert s._side == "flat"


def test_trailing_disabled_never_fires() -> None:
    s, bars = asyncio.run(_prepared({"trailing_enabled": False, "tp_enabled": False,
                                     "stop_atr": 0.0}))
    s._open("long", 100.0, 0.4, atr=2.0)
    s._best = 130.0
    ts = bars[300].timestamp
    sig = s._exit_check(300, 101.0, 2.0, Price(101.0), ts, regime=1)
    assert sig is None or sig.reason != "trailing_stop_pct"


def test_trailing_and_tp_are_independent() -> None:
    # trailing on + tp off must still protect; tp on + trailing off must still bank
    a, bars = asyncio.run(_prepared({"trailing_enabled": True, "tp_enabled": False}))
    a._open("long", 100.0, 0.4, atr=2.0); a._best = 105.0
    ts = bars[300].timestamp
    assert a._exit_check(300, 103.9, 2.0, Price(103.9), ts, regime=1).reason == "trailing_stop_pct"
    b, _ = asyncio.run(_prepared({"trailing_enabled": False, "tp_enabled": True}))
    b._open("long", 100.0, 0.4, atr=2.0); b._best = 105.0
    sig = b._exit_check(300, 103.9, 2.0, Price(103.9), ts, regime=1)
    assert sig.reason == "take_profit_pct" and b._side == "long"


# ── adds (pyramiding) ──────────────────────────────────────────────────


def test_add_into_open_position_and_bookkeeping() -> None:
    s, bars = asyncio.run(_prepared({"allow_adds": True, "add_size_mult": 0.25,
                                     "add_cooldown_bars": 5}))
    s._open("long", 100.0, 0.4, atr=2.0)
    n = len(bars)
    s._emf_features = pd.DataFrame({"long_add_signal": [True] * n,
                                    "short_add_signal": [False] * n})
    i = s.min_bars + 2
    sig = s._try_add(i, bars[i].timestamp)
    assert sig is not None and sig.reason == "add_long" and sig.side.value == "buy"
    assert sig.strength == pytest.approx(0.1)           # 0.4 × 0.25
    assert s._entry_strength == pytest.approx(0.5)      # full exits match 1.25×
    # cooldown
    assert s._try_add(i + 1, bars[i + 1].timestamp) is None
    # wrong-side flag → no add
    s._emf_features = pd.DataFrame({"long_add_signal": [False] * n,
                                    "short_add_signal": [True] * n})
    assert s._try_add(i + 6, bars[i + 6].timestamp) is None
    # position guard: adds never reopen after a close
    s._close_position()
    s._emf_features = pd.DataFrame({"long_add_signal": [True] * n,
                                    "short_add_signal": [True] * n})
    assert s._try_add(i + 12, bars[i + 12].timestamp) is None


# ── integration ────────────────────────────────────────────────────────


async def _replay(s: PineConfluenceStrategy, bars: list[Bar]) -> list:
    out = []
    for b in bars:
        out.extend(await s.on_bar(b))
    return out


def test_batch_streaming_equivalence() -> None:
    bars = _trending_bars(500)
    params = {"allow_adds": True, "tp_percent": 1.5, "trailing_percent": 0.8}
    a = PineConfluenceStrategy("SYNTH", params=params)
    batch = asyncio.run(a.generate_signals(bars))
    b = PineConfluenceStrategy("SYNTH", params=params)
    streamed = asyncio.run(_replay(b, bars))
    assert [_key(s) for s in batch] == [_key(s) for s in streamed]


def test_backtest_runs_with_adds_tp_trailing() -> None:
    bars = _trending_bars(700)
    strat = build_strategy("trend_confluence_pine", "SYNTH", {
        "allow_adds": True, "tp_percent": 2.0, "trailing_percent": 1.0,
        "trendline_refresh": 10,
    })
    res = asyncio.run(run_backtest(strat, bars, BacktestConfig()))
    assert res.equity_curve.min() > 0
    assert len(res.trades) > 0
