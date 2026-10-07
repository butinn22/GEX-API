"""Tests for the live strategy engine."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading.application.audit import AuditLog
from trading.application.live_engine import LiveEngine
from trading.application.signal_hub import signal_hub
from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.domain import Bar


def _bar(i: int, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


async def _gen(bars):
    for b in bars:
        yield b


async def test_live_engine_emits_signal_and_audits():
    q = signal_hub.subscribe()
    audit = AuditLog()
    bars = [_bar(i, 100.0 + i) for i in range(5)]
    engine = LiveEngine(BuyAndHold("X"), audit=audit, publish=True)
    count = await engine.run(_gen(bars))
    assert count == 1  # buy-and-hold emits one BUY
    assert len(audit.entries("signal")) == 1
    msg = q.get_nowait()
    assert msg["type"] == "signal" and msg["side"] == "buy"
    signal_hub.unsubscribe(q)


async def test_live_engine_respects_publish_false():
    q = signal_hub.subscribe()
    engine = LiveEngine(BuyAndHold("X"), publish=False)
    count = await engine.run(_gen([_bar(0, 100.0), _bar(1, 101.0)]))
    assert count == 1
    assert q.empty()  # not published
    signal_hub.unsubscribe(q)
