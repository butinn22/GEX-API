"""Tests for the parallel historical data loader."""
from __future__ import annotations

from datetime import UTC, datetime

from trading.adapters.fetchers.registry import FetcherRegistry
from trading.application.data_loader import HistoricalDataLoader
from trading.domain import Bar, Exchange
from trading.ports import BaseFetcher


class FakeFetcher(BaseFetcher):
    exchange = Exchange.YFINANCE

    async def get_instruments(self):
        return []

    async def get_ohlcv(self, symbol, timeframe, *, start=None, end=None, limit=500):
        return [
            Bar(datetime(2024, 1, 1 + i, tzinfo=UTC), 1, 2, 0.5, 1.5, 10)
            for i in range(limit)
        ]

    async def get_orderbook(self, symbol, *, depth=20):
        raise NotImplementedError

    async def get_trades(self, symbol, *, limit=100):
        raise NotImplementedError


async def test_loader_parallel_fetch():
    reg = FetcherRegistry()
    reg.register(FakeFetcher())
    loader = HistoricalDataLoader(reg, [Exchange.YFINANCE])
    data = await loader.load(["A", "B", "C"], "1d", limit=10)
    assert set(data) == {"A", "B", "C"}
    assert all(len(bars) == 10 for bars in data.values())
