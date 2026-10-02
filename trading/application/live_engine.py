"""Live strategy engine: drive a strategy over a realtime bar stream.

Consumes any async iterable of ``Bar``, calls ``on_bar``, and fans signals out to
the signal hub, the domain event bus, and the audit log. A ``polling_bar_stream``
helper wraps a fetcher as a polled stream for live feeds.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime

from trading.application.audit import AuditLog
from trading.application.signal_hub import signal_hub
from trading.domain import Bar
from trading.domain.events import DomainEvent, DomainEventBus
from trading.ports import Strategy

__all__ = ["LiveEngine", "polling_bar_stream"]


class LiveEngine:
    def __init__(
        self,
        strategy: Strategy,
        *,
        event_bus: DomainEventBus | None = None,
        audit: AuditLog | None = None,
        publish: bool = True,
    ) -> None:
        self.strategy = strategy
        self.event_bus = event_bus
        self.audit = audit
        self.publish = publish

    async def run(self, bars: AsyncIterable[Bar]) -> int:
        """Process bars until the stream ends; return the number of signals."""
        await self.strategy.start()
        count = 0
        try:
            async for bar in bars:
                for sig in await self.strategy.on_bar(bar):
                    count += 1
                    if self.publish:
                        signal_hub.publish({
                            "type": "signal", "strategy": sig.strategy, "symbol": sig.symbol,
                            "side": sig.side.value, "reason": sig.reason,
                            "timestamp": sig.timestamp.isoformat(),
                        })
                    if self.event_bus is not None:
                        await self.event_bus.publish(
                            DomainEvent("signal", {"strategy": sig.strategy, "symbol": sig.symbol,
                                                   "side": sig.side.value})
                        )
                    if self.audit is not None:
                        self.audit.record("signal", actor=sig.strategy, symbol=sig.symbol,
                                          side=sig.side.value, reason=sig.reason)
        finally:
            await self.strategy.shutdown()
        return count


async def polling_bar_stream(
    fetcher,
    symbol: str,
    timeframe: str,
    *,
    poll_seconds: float = 60.0,
    limit: int = 500,
) -> AsyncIterator[Bar]:
    """Poll a fetcher forever, yielding only bars not seen before (dedup by timestamp)."""
    seen: set[datetime] = set()
    while True:
        for bar in await fetcher.get_ohlcv(symbol, timeframe, limit=limit):
            if bar.timestamp not in seen:
                seen.add(bar.timestamp)
                yield bar
        await asyncio.sleep(poll_seconds)
