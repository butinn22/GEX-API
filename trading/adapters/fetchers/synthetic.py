"""Synthetic fetcher: deterministic GBM candles (dev/demo/testing).

Real fetchers (MOEX/Bybit/yFinance/Webull) live in ``gex/adapters`` and get
wrapped behind ``BaseFetcher`` in a later slice; this one lets the platform run
end-to-end without network access.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trading.domain import Bar, Exchange, Instrument, OrderBook, Tick
from trading.ports import BaseFetcher

__all__ = ["SyntheticFetcher"]


class SyntheticFetcher(BaseFetcher):
    exchange = Exchange.YFINANCE

    def __init__(self, *, seed: int = 0, mu: float = 0.0002, sigma: float = 0.02) -> None:
        self._rng = np.random.default_rng(seed)
        self.mu = mu
        self.sigma = sigma

    async def get_instruments(self) -> list[Instrument]:
        return [Instrument("SYNTH", self.exchange, base_asset="SYNTH", quote_asset="USD")]

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        n = max(limit, 1)
        start_dt = start or datetime(2023, 1, 1, tzinfo=UTC)
        closes = 100.0 * np.exp(np.cumsum(self._rng.normal(self.mu, self.sigma, n)))
        bars: list[Bar] = []
        for i in range(n):
            close = float(closes[i])
            prev = float(closes[i - 1]) if i > 0 else close
            open_ = prev
            high = max(open_, close) * (1 + abs(self._rng.normal(0, self.sigma / 2)))
            low = min(open_, close) * (1 - abs(self._rng.normal(0, self.sigma / 2)))
            ts = start_dt + timedelta(days=i)
            bars.append(Bar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=1000.0))
        return bars

    async def get_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        return OrderBook()  # synthetic source has no live book

    async def get_trades(self, symbol: str, *, limit: int = 100) -> list[Tick]:
        return []
