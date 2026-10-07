"""Tests for the example strategies."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.application.strategies.sma_crossover import SmaCrossover
from trading.domain import Bar, Side


def _bar(i: int, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_buy_and_hold_single_entry():
    s = BuyAndHold("AAPL")
    sigs = asyncio.run(s.on_bar(_bar(0, 100.0)))
    assert len(sigs) == 1 and sigs[0].side is Side.BUY
    assert asyncio.run(s.on_bar(_bar(1, 101.0))) == []  # held


def test_sma_crossover_signals():
    s = SmaCrossover("X", fast=2, slow=3)
    closes = [10.0, 10.0, 10.0, 11.0, 9.0, 9.0]
    signals = asyncio.run(s.generate_signals([_bar(i, c) for i, c in enumerate(closes)]))
    assert [sig.side for sig in signals] == [Side.BUY, Side.SELL]


def test_sma_crossover_incremental_matches_batch():
    closes = [10.0, 10.0, 10.0, 11.0, 9.0, 9.0, 12.0, 13.0]
    bars = [_bar(i, c) for i, c in enumerate(closes)]

    batch = SmaCrossover("X", fast=2, slow=3)
    batch_signals = asyncio.run(batch.generate_signals(bars))

    inc = SmaCrossover("X", fast=2, slow=3)
    inc_signals = []
    for b in bars:
        inc_signals.extend(asyncio.run(inc.on_bar(b)))

    assert [s.side for s in batch_signals] == [s.side for s in inc_signals]


def test_sma_crossover_rejects_bad_windows():
    import pytest
    with pytest.raises(ValueError):
        SmaCrossover("X", fast=10, slow=5)
