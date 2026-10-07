"""Domain event bus (pub/sub) for strategy/order/position events.

Pure stdlib. Handlers may be sync or async; the bus awaits awaitable results.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = ["DomainEvent", "DomainEventBus"]

Handler = Callable[["DomainEvent"], Awaitable[None] | None]


@dataclass(frozen=True)
class DomainEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class DomainEventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}

    def subscribe(self, event_name: str, handler: Handler) -> None:
        self._handlers.setdefault(event_name, []).append(handler)

    def unsubscribe(self, event_name: str, handler: Handler) -> None:
        handlers = self._handlers.get(event_name, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: DomainEvent) -> None:
        for handler in list(self._handlers.get(event.name, [])):
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result

    def subscriber_count(self, event_name: str) -> int:
        return len(self._handlers.get(event_name, []))
