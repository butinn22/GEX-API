"""In-process pub/sub hubs for live streaming (signals, orders, positions).

Each hub fans a published message out to every subscriber queue. Consumers are
the WebSocket endpoints; publishers are strategies / the execution engine.
Single-process, single-event-loop — sufficient for the trading context, which
runs one engine per process (Redis pub/sub would back a multi-worker scale-out).
"""
from __future__ import annotations

import asyncio
from typing import Any

__all__ = ["SignalHub", "signal_hub", "order_hub", "position_hub"]


class SignalHub:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, message: dict[str, Any]) -> None:
        for q in list(self._subs):
            q.put_nowait(message)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


signal_hub = SignalHub()
order_hub = SignalHub()
position_hub = SignalHub()
