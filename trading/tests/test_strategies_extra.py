"""Tests for mean-reversion and momentum strategies."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from trading.application.strategies.mean_reversion import MeanReversion
from trading.application.strategies.momentum import Momentum
from trading.domain import Bar, Side


def _bar(i: int, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_momentum_signals_up_then_down():
    closes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 10, 8, 6, 4]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    signals = asyncio.run(Momentum("X", period=3).generate_signals(bars))
    assert [s.side for s in signals] == [Side.BUY, Side.SELL]


def test_mean_reversion_emits_on_dip():
    # flat → sharp dip → recovery: a mean-reversion dip produces a BUY then a SELL.
    closes = [100.0] * 25 + [80.0] * 4 + [100.0] * 10
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    signals = asyncio.run(MeanReversion("X", period=20).generate_signals(bars))
    sides = [s.side for s in signals]
    assert Side.BUY in sides and Side.SELL in sides
    assert sides.index(Side.BUY) < sides.index(Side.SELL)


def test_mean_reversion_incremental_matches_batch():
    closes = [100.0] * 25 + [80.0] * 4 + [100.0] * 10
    bars = [_bar(i, c) for i, c in enumerate(closes)]

    batch = asyncio.run(MeanReversion("X", period=20).generate_signals(bars))
    inc = MeanReversion("X", period=20)
    inc_signals = []
    for b in bars:
        inc_signals.extend(asyncio.run(inc.on_bar(b)))

    assert [s.side for s in batch] == [s.side for s in inc_signals]
