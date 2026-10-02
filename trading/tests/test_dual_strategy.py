"""Tests for the dual long/short SMA crossover strategy."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategies.dual_sma_crossover import DualSmaCrossover
from trading.domain import Bar, Side


def _bar(i: int, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def _bars(closes) -> list[Bar]:
    return [_bar(i, c) for i, c in enumerate(closes)]


def test_long_only_when_short_disabled():
    # down → up → down: long leg fires BUY then SELL; short leg disabled.
    closes = list(range(30, 10, -1)) + list(range(10, 40)) + list(range(40, 10, -1))
    s = DualSmaCrossover("X", long_enabled=True, long_fast=3, long_slow=5,
                         short_enabled=False, short_fast=3, short_slow=5)
    signals = asyncio.run(s.generate_signals(_bars(closes)))
    assert [sig.side for sig in signals] == [Side.BUY, Side.SELL]


def test_short_only_when_long_disabled():
    # up → down → up: fast SMA crosses below (short entry SELL) then above (short exit BUY).
    closes = list(range(10, 40)) + list(range(40, 10, -1)) + list(range(10, 40))
    s = DualSmaCrossover("X", long_enabled=False, short_enabled=True,
                         short_fast=3, short_slow=5)
    signals = asyncio.run(s.generate_signals(_bars(closes)))
    assert signals[0].side is Side.SELL
    assert Side.BUY in [sig.side for sig in signals]


def test_independent_windows_change_entry_bar():
    # A tighter long window crosses earlier than a wide one → first BUY earlier.
    closes = [10.0] * 8 + list(range(10, 30)) + [29.0] * 5
    tight = DualSmaCrossover("X", long_fast=2, long_slow=4, short_enabled=False)
    wide = DualSmaCrossover("X", long_fast=3, long_slow=8, short_enabled=False)
    tight_sigs = asyncio.run(tight.generate_signals(_bars(closes)))
    wide_sigs = asyncio.run(wide.generate_signals(_bars(closes)))
    assert len(tight_sigs) >= 1 and len(wide_sigs) >= 1
    assert tight_sigs[0].timestamp <= wide_sigs[0].timestamp


def test_backtest_with_dual_strategy_produces_trades():
    closes = list(range(100, 80, -1)) + list(range(80, 130)) + list(range(130, 90, -1))
    bars = _bars(closes)
    s = DualSmaCrossover("X", long_fast=3, long_slow=8, short_fast=3, short_slow=8)
    result = asyncio.run(run_backtest(s, bars, BacktestConfig(initial_cash=10000)))
    assert result.metrics.n_trades >= 1
    assert result.metrics.n_trades == len(result.trades)


def test_rejects_bad_windows():
    with pytest.raises(ValueError):
        DualSmaCrossover("X", long_fast=10, long_slow=5)
    with pytest.raises(ValueError):
        DualSmaCrossover("X", short_fast=10, short_slow=5)
