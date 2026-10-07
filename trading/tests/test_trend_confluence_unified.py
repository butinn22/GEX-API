"""Tests for the unified strategy (Trend Confluence × EMF+ADL × Momentum)."""
from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta

import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategies.trend_confluence_unified import (
    UNIFIED_STRATEGY_NAME,
    UNIFIED_STRATEGY_VERSION,
    UnifiedTrendParams,
    UnifiedTrendStrategy,
)
from trading.application.strategy_factory import build_strategy
from trading.domain import Bar, Side


def _trending_bars(n: int = 600, *, start: float = 100.0, drift: float = 0.0015) -> list[Bar]:
    """Deterministic noisy uptrend with pullbacks (same shape as TC tests)."""
    bars: list[Bar] = []
    price = start
    t0 = datetime(2022, 1, 1, tzinfo=UTC)
    for i in range(n):
        ret = drift + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        high = max(prev, price) * 1.004
        low = min(prev, price) * 0.996
        bars.append(Bar(timestamp=t0 + timedelta(days=i), open=prev,
                        high=high, low=low, close=price, volume=1000.0))
    return bars


# ── schema ────────────────────────────────────────────────────────────


def test_params_inherit_tc_fields_and_validate_modes() -> None:
    p = UnifiedTrendParams.from_dict({
        "zone_atr": 0.4, "emf_mode": "bonus", "momentum_period": 15,
        "emf": {"atr_tp_mult": 3.0}, "options": {"enabled": True},
    })
    assert p.zone_atr == 0.4
    assert p.emf_mode == "bonus"
    assert p.momentum_period == 15
    assert p.emf == {"atr_tp_mult": 3.0}
    # unknown keys are filtered (same convention as TC params)
    assert "options" not in UnifiedTrendParams.__dataclass_fields__

    with pytest.raises(ValueError):
        UnifiedTrendParams(emf_mode="nonsense")
    with pytest.raises(ValueError):
        UnifiedTrendParams(momentum_period=0)


def test_default_params_are_a_real_merge() -> None:
    p = UnifiedTrendParams()
    assert p.use_emf and p.emf_mode == "bonus"
    assert p.use_emf_exits and p.use_risk_exits
    assert p.use_momentum and p.momentum_mode == "bonus"
    assert p.momentum_period == 10  # the standalone momentum default


def test_factory_builds_unified_strategy() -> None:
    strat = build_strategy(UNIFIED_STRATEGY_NAME, "SYNTH", {
        "zone_atr": 0.4,
        "emf_mode": "bonus",
        "emf": {"atr_tp_mult": 2.5},
    })
    assert isinstance(strat, UnifiedTrendStrategy)
    assert strat.name == UNIFIED_STRATEGY_NAME
    assert strat.version == UNIFIED_STRATEGY_VERSION
    assert strat._p.zone_atr == 0.4
    assert strat._up.emf_mode == "bonus"
    assert strat._emf_settings.atr_tp_mult == 2.5


def test_factory_rejects_bad_params() -> None:
    from trading.domain import StrategyError

    with pytest.raises(StrategyError):
        build_strategy(UNIFIED_STRATEGY_NAME, "SYNTH", {"emf_mode": "bogus"})


# ── behaviour ──────────────────────────────────────────────────────────


def test_unified_produces_signals_on_trend() -> None:
    bars = _trending_bars(500)
    strat = UnifiedTrendStrategy("SYNTH", params={"trendline_refresh": 10})
    sigs = asyncio.run(strat.generate_signals(bars))
    assert sigs, "expected signals on a trending series"
    first = min(s.timestamp for s in sigs)
    assert first >= bars[strat.min_bars - 1].timestamp


def test_emf_require_mode_is_stricter_than_off() -> None:
    bars = _trending_bars(500)
    strict = UnifiedTrendStrategy("SYNTH", params={
        "trendline_refresh": 10, "emf_mode": "require",
    })
    loose = UnifiedTrendStrategy("SYNTH", params={
        "trendline_refresh": 10, "emf_mode": "off", "use_emf_exits": False,
        "use_risk_exits": False,
    })
    strict_sigs = asyncio.run(strict.generate_signals(bars))
    loose_sigs = asyncio.run(loose.generate_signals(bars))
    strict_entries = [s for s in strict_sigs if s.reason.startswith("add_")]
    loose_entries = [s for s in loose_sigs if s.reason.startswith("add_")]
    # The intersection can only reduce (or keep) the entry count.
    assert len(strict_entries) <= len(loose_entries)
    # "off" + no EMF exits must reproduce pure trend-confluence behaviour.
    pure = build_strategy("trend_confluence", "SYNTH", {"trendline_refresh": 10})
    pure_sigs = asyncio.run(pure.generate_signals(bars))
    pure_entries = [s for s in pure_sigs if s.reason.startswith("add_")]
    assert [(s.side, s.timestamp) for s in loose_entries] == [
        (s.side, s.timestamp) for s in pure_entries
    ]


def test_momentum_gate_blocks_entries_against_roc() -> None:
    bars = _trending_bars(500)
    gated = UnifiedTrendStrategy("SYNTH", params={
        "trendline_refresh": 10, "momentum_mode": "gate",
    })
    ungated = UnifiedTrendStrategy("SYNTH", params={
        "trendline_refresh": 10, "momentum_mode": "off",
    })
    gated_sigs = asyncio.run(gated.generate_signals(bars))
    ungated_sigs = asyncio.run(ungated.generate_signals(bars))
    g = [s for s in gated_sigs if s.reason.startswith("add_")]
    u = [s for s in ungated_sigs if s.reason.startswith("add_")]
    assert len(g) <= len(u)
    # every gated long must sit on a bar with positive ROC
    closes = [b.close for b in bars]
    period = gated._up.momentum_period
    for s in g:
        i = next(j for j, b in enumerate(bars) if b.timestamp == s.timestamp)
        roc = closes[i] - closes[i - period]
        if s.side is Side.BUY:
            assert roc > 0


def test_bonus_modes_cap_strength_at_one() -> None:
    strat = UnifiedTrendStrategy("SYNTH", params={
        "emf_mode": "bonus", "momentum_mode": "bonus",
        "emf_bonus": 0.9, "momentum_bonus": 0.9,
    })
    bars = _trending_bars(400)
    sigs = asyncio.run(strat.generate_signals(bars))
    assert all(0.0 <= s.strength <= 1.0 for s in sigs)


def test_prepare_streaming_equivalence() -> None:
    bars = _trending_bars(400)
    a = UnifiedTrendStrategy("SYNTH", params={"trendline_refresh": 10})
    batch = [(s.side, s.timestamp, s.reason) for s in asyncio.run(a.generate_signals(bars))]

    b = UnifiedTrendStrategy("SYNTH", params={"trendline_refresh": 10})
    asyncio.run(b.prepare(bars))
    streamed: list = []
    for bar in bars:
        streamed.extend(
            (s.side, s.timestamp, s.reason) for s in asyncio.run(b.on_bar(bar))
        )
    assert batch == streamed


def test_backtest_runs_and_equity_stays_positive() -> None:
    bars = _trending_bars(600)
    strat = build_strategy(UNIFIED_STRATEGY_NAME, "SYNTH", {"trendline_refresh": 10})
    res = asyncio.run(run_backtest(strat, bars, BacktestConfig()))
    assert res.equity_curve.min() > 0
    assert len(res.trades) > 0


def test_registry_lists_unified() -> None:
    from trading.application.strategy_registry import STRATEGY_REGISTRY

    assert UNIFIED_STRATEGY_NAME in STRATEGY_REGISTRY.names()
    params = STRATEGY_REGISTRY.params(UNIFIED_STRATEGY_NAME)
    for knob in ("emf_mode", "momentum_period", "use_risk_exits", "emf", "options"):
        assert knob in params
    assert "zone_atr" in params  # TC core fields are part of the schema
