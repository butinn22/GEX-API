"""Tests for the domain event bus."""
from __future__ import annotations

import asyncio

from trading.domain.events import DomainEvent, DomainEventBus


async def test_bus_dispatches_to_subscribers():
    bus = DomainEventBus()
    received = []

    async def handler(event: DomainEvent) -> None:
        received.append(event.payload)

    bus.subscribe("signal", handler)
    await bus.publish(DomainEvent("signal", {"side": "buy"}))
    assert received == [{"side": "buy"}]
    assert bus.subscriber_count("signal") == 1

    bus.unsubscribe("signal", handler)
    await bus.publish(DomainEvent("signal", {"side": "sell"}))
    assert len(received) == 1  # unsubscribed, no further delivery
    assert bus.subscriber_count("signal") == 0


def test_bus_supports_sync_handlers():
    bus = DomainEventBus()
    got = []
    bus.subscribe("tick", lambda e: got.append(1))
    asyncio.run(bus.publish(DomainEvent("tick")))
    assert got == [1]
