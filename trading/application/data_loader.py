"""Parallel historical data loader (fetches many symbols via asyncio.gather)."""
from __future__ import annotations

import asyncio
from datetime import datetime

from trading.adapters.fetchers.registry import FetcherRegistry
from trading.domain import Bar, Exchange

__all__ = ["HistoricalDataLoader"]


class HistoricalDataLoader:
    def __init__(self, registry: FetcherRegistry, exchanges: list[Exchange]) -> None:
        self.registry = registry
        self.exchanges = exchanges

    async def load(
        self,
        symbols: list[str],
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> dict[str, list[Bar]]:
        """Fetch all symbols in parallel; a symbol that fails is omitted."""
        results = await asyncio.gather(
            *[
                self.registry.get_ohlcv(self.exchanges, s, timeframe, start=start, end=end, limit=limit)
                for s in symbols
            ],
            return_exceptions=True,
        )
        out: dict[str, list[Bar]] = {}
        for symbol, res in zip(symbols, results):
            if isinstance(res, Exception):
                continue
            out[symbol] = res
        return out
