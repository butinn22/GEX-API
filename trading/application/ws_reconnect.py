"""WebSocket reconnect manager — exponential backoff on (re)connect failures."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

__all__ = ["WSReconnectManager"]

T = TypeVar("T")


class WSReconnectManager:
    def __init__(self, *, base_delay: float = 1.0, max_delay: float = 30.0,
                 factor: float = 2.0, max_attempts: int | None = None) -> None:
        if base_delay < 0 or max_delay < 0 or factor < 1:
            raise ValueError("invalid backoff parameters")
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.factor = factor
        self.max_attempts = max_attempts

    def delay(self, attempt: int) -> float:
        """Exponential backoff: base * factor**attempt, capped at max_delay."""
        return min(self.base_delay * (self.factor ** max(attempt, 0)), self.max_delay)

    async def run(self, connect: Callable[[], Awaitable[T]],
                  *, on_connected: Callable[[T], Awaitable[None]] | None = None) -> T:
        """Call ``connect`` until it succeeds (backing off on failure)."""
        attempt = 0
        while self.max_attempts is None or attempt < self.max_attempts:
            try:
                session = await connect()
            except Exception:
                await asyncio.sleep(self.delay(attempt))
                attempt += 1
                continue
            if on_connected is not None:
                await on_connected(session)
            return session
        raise ConnectionError("websocket reconnect attempts exhausted")
