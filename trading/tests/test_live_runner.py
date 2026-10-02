"""Tests for the live strategy manager."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from trading.application.live_runner import LiveStrategyManager
from trading.application.signal_hub import signal_hub
from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.domain import Bar


def _bar(i: int) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    return Bar(timestamp=ts, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.0 + i, volume=1.0)


async def _finite(bars):
    for b in bars:
        yield b


async def test_manager_runs_and_publishes():
    q = signal_hub.subscribe()
    mgr = LiveStrategyManager()
    bars = [_bar(i) for i in range(5)]
    mgr.start("x", BuyAndHold("X"), _finite(bars))
    for _ in range(100):
        if not mgr.is_running("x"):
            break
        await asyncio.sleep(0.01)
    assert not mgr.is_running("x")  # finite feed → task completed
    msg = q.get_nowait()
    assert msg["type"] == "signal"
    signal_hub.unsubscribe(q)


async def test_manager_stop_cancels():
    mgr = LiveStrategyManager()

    async def infinite():
        while True:
            yield _bar(0)
            await asyncio.sleep(0.01)

    mgr.start("y", BuyAndHold("Y"), infinite())
    await asyncio.sleep(0.05)
    assert mgr.is_running("y")
    assert "y" in mgr.running()
    assert mgr.stop("y") is True
    assert not mgr.is_running("y")
    assert mgr.stop("y") is False  # already stopped
