"""Tests for the pub/sub signal hub."""
from __future__ import annotations

from trading.application.signal_hub import SignalHub


async def test_hub_fanout_and_unsubscribe():
    hub = SignalHub()
    q1, q2 = hub.subscribe(), hub.subscribe()
    assert hub.subscriber_count == 2

    hub.publish({"type": "signal", "side": "buy"})
    assert (await q1.get())["side"] == "buy"
    assert (await q2.get())["side"] == "buy"

    hub.unsubscribe(q1)
    assert hub.subscriber_count == 1
    hub.publish({"type": "signal", "side": "sell"})
    assert (await q2.get())["side"] == "sell"  # q2 still subscribed
    assert q1.empty()  # q1 unsubscribed, receives nothing

    hub.unsubscribe(q2)
    assert hub.subscriber_count == 0


async def test_hub_multiple_messages_in_order():
    hub = SignalHub()
    q = hub.subscribe()
    for i in range(3):
        hub.publish({"i": i})
    assert [ (await q.get())["i"] for _ in range(3) ] == [0, 1, 2]


def test_hub_publish_is_sync_and_nonblocking():
    # publish() must be callable from sync code (strategies/engine call it directly)
    hub = SignalHub()
    hub.publish({"x": 1})  # no subscribers — must not raise
