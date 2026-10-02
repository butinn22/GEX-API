"""Unified async fetcher port — one interface for every market-data source.

Each concrete fetcher (ISS/MOEX, Bybit, yFinance, Webull, or a future venue)
implements this ABC and registers itself in a ``FetcherRegistry`` keyed by
``exchange``. Consumers depend on this port, never on a concrete fetcher, so
fallback (ISS → yfinance) and rate-limiting stay orthogonal concerns.

Rate limiting is enforced *outside* the fetcher: the application wraps a fetcher
with a decorator/adaptor that calls the Redis-backed ``RateLimitPort`` before each
network call. That keeps each fetcher dumb (just normalise + return) and makes
the throttle policy configurable per exchange.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

from trading.domain import Bar, Exchange, Instrument, OrderBook, Tick

__all__ = ["BaseFetcher"]


class BaseFetcher(ABC):
    """Async market-data interface: instruments, OHLCV, order book, trades."""

    exchange: ClassVar[Exchange]

    @abstractmethod
    async def get_instruments(self) -> list[Instrument]:
        """All tradable instruments for this venue (optionally filtered by caller)."""
        ...

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        """Candles for ``symbol`` on ``timeframe`` (e.g. "1m", "5m", "1h", "1d").

        Returns bars in ascending time order. ``start``/``end`` are UTC and
        inclusive; ``limit`` caps the count when the range is open-ended.
        """
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        """Current order book snapshot (bids desc, asks asc)."""
        ...

    @abstractmethod
    async def get_trades(self, symbol: str, *, limit: int = 100) -> list[Tick]:
        """Most recent trades for ``symbol`` (newest last)."""
        ...
