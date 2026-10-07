"""Live strategy manager — start/stop background strategy tasks.

Each strategy runs as an asyncio task consuming an async bar feed through the
``LiveEngine`` (signals → signal hub / event bus / audit). Use ``polling_bar_stream``
for a 24/7 feed, or a finite generator for tests/demos.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable

from trading.application.audit import AuditLog
from trading.application.live_engine import LiveEngine
from trading.domain import Bar
from trading.ports import Strategy

__all__ = ["LiveStrategyManager", "live_manager"]


class LiveStrategyManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, name: str, strategy: Strategy, feed: AsyncIterable[Bar]) -> str:
        if name in self._tasks and not self._tasks[name].done():
            return name
        task = asyncio.create_task(self._run(name, strategy, feed))
        self._tasks[name] = task
        return name

    async def _run(self, name: str, strategy: Strategy, feed: AsyncIterable[Bar]) -> None:
        engine = LiveEngine(strategy, audit=AuditLog())
        try:
            await engine.run(feed)
        finally:
            self._tasks.pop(name, None)

    def stop(self, name: str) -> bool:
        task = self._tasks.get(name)
        if task is None:
            return False
        task.cancel()
        self._tasks.pop(name, None)
        return True

    def is_running(self, name: str) -> bool:
        task = self._tasks.get(name)
        return task is not None and not task.done()

    def running(self) -> list[str]:
        return [n for n in self._tasks if self.is_running(n)]


live_manager = LiveStrategyManager()
