"""Tests for the strategy registry and runner."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from trading.application.strategies import BuyAndHold, MeanReversion, SmaCrossover
from trading.application.strategies.confluence_breakout import (
    ConfluenceBreakoutParams,
    ConfluenceBreakoutStrategy,
)
from trading.application.strategy_registry import STRATEGY_REGISTRY, StrategyRunner
from trading.domain import Bar


def _bar(i: int, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_registry_names_and_params():
    names = set(STRATEGY_REGISTRY.names())
    assert names == {"sma_crossover", "buy_and_hold", "mean_reversion", "momentum",
                     "sma_crossover_ls", "gex_emf", "trend_confluence",
                     "trend_confluence_unified", "trend_confluence_pine",
                     "confluence_breakout"}
    assert STRATEGY_REGISTRY.params("sma_crossover") == ["fast", "slow"]
    assert STRATEGY_REGISTRY.params("buy_and_hold") == []
    assert STRATEGY_REGISTRY.params("sma_crossover_ls") == ["long_fast", "long_slow", "short_fast", "short_slow"]
    assert "gex_emf" in STRATEGY_REGISTRY.names()
    assert "trend_confluence_unified" in STRATEGY_REGISTRY.names()
    assert "confluence_breakout" in STRATEGY_REGISTRY.names()
    assert STRATEGY_REGISTRY.params("confluence_breakout") == list(
        ConfluenceBreakoutParams.FIELD_NAMES
    )


def test_registry_builds_confluence_breakout_with_preset():
    s = STRATEGY_REGISTRY.build("confluence_breakout", "X", preset="donchian_1d")
    assert isinstance(s, ConfluenceBreakoutStrategy)
    assert s.preset == "donchian_1d" and s.timeframe == "1d"


def test_registry_build():
    s = STRATEGY_REGISTRY.build("sma_crossover", "X", fast=5, slow=20)
    assert isinstance(s, SmaCrossover) and s.fast == 5 and s.slow == 20
    m = STRATEGY_REGISTRY.build("mean_reversion", "X", period=10)
    assert isinstance(m, MeanReversion)
    with pytest.raises(KeyError):
        STRATEGY_REGISTRY.build("nonexistent", "X")


async def test_runner_generates_signals():
    bars = [_bar(i, 100.0 + i) for i in range(10)]
    runner = StrategyRunner(BuyAndHold("X"), publish=True)
    signals = await runner.run_bars(bars)
    assert len(signals) == 1  # buy-and-hold emits exactly one BUY
