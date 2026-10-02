"""Fetcher registry + cross-source fallback."""
from __future__ import annotations

import asyncio
import weakref
from datetime import datetime
from typing import Any

from trading.domain import Bar, DataFetchError, Exchange, OrderBook, Tick
from trading.ports import BaseFetcher

__all__ = ["FetcherRegistry", "default_registry", "loop_registry", "aclose_loop_registry"]


class FetcherRegistry:
    """Holds fetchers keyed by ``Exchange`` and provides fallback queries."""

    def __init__(self) -> None:
        self._fetchers: dict[Exchange, BaseFetcher] = {}

    def register(self, fetcher: BaseFetcher) -> None:
        self._fetchers[fetcher.exchange] = fetcher

    def get(self, exchange: Exchange) -> BaseFetcher | None:
        return self._fetchers.get(exchange)

    def exchanges(self) -> list[Exchange]:
        return list(self._fetchers)

    async def aclose(self) -> None:
        """Close every fetcher's HTTP client (releases its connection pool)."""
        for fetcher in self._fetchers.values():
            close = getattr(fetcher, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:  # a failed close must not mask the real error
                pass

    async def get_ohlcv(
        self,
        exchanges: list[Exchange],
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        """Try each source in order; return the first success, else raise."""
        errors: list[str] = []
        for ex in exchanges:
            fetcher = self._fetchers.get(ex)
            if fetcher is None:
                continue
            try:
                bars = await fetcher.get_ohlcv(symbol, timeframe, start=start, end=end, limit=limit)
                if bars:
                    return bars
                errors.append(f"{ex.value}: empty")
            except DataFetchError as exc:
                errors.append(f"{ex.value}: {exc}")
        raise DataFetchError(f"all sources failed for {symbol}: {'; '.join(errors)}")

    async def get_orderbook(self, exchanges: list[Exchange], symbol: str, *, depth: int = 20) -> OrderBook:
        errors: list[str] = []
        for ex in exchanges:
            fetcher = self._fetchers.get(ex)
            if fetcher is None:
                continue
            try:
                return await fetcher.get_orderbook(symbol, depth=depth)
            except DataFetchError as exc:
                errors.append(f"{ex.value}: {exc}")
        raise DataFetchError(f"no orderbook source for {symbol}: {'; '.join(errors)}")

    async def get_trades(self, exchanges: list[Exchange], symbol: str, *, limit: int = 100) -> list[Tick]:
        errors: list[str] = []
        for ex in exchanges:
            fetcher = self._fetchers.get(ex)
            if fetcher is None:
                continue
            try:
                return await fetcher.get_trades(symbol, limit=limit)
            except DataFetchError as exc:
                errors.append(f"{ex.value}: {exc}")
        raise DataFetchError(f"no trade source for {symbol}: {'; '.join(errors)}")


def default_registry() -> FetcherRegistry:
    """Registry with the real public fetchers (MOEX → yFinance → Bybit fallback)."""
    from .bybit import BybitFetcher
    from .moex import MoexIssFetcher
    from .yfinance import YFinanceFetcher

    reg = FetcherRegistry()
    reg.register(MoexIssFetcher())
    reg.register(YFinanceFetcher())
    reg.register(BybitFetcher())
    return reg


#: One registry per event loop. Keyed by a *weak* reference to the loop object so
#: the entry dies with the loop instead of pinning it. httpx clients are bound to
#: the loop that first uses them, so sharing a single registry across loops would
#: raise ("Event loop is closed") — and Celery runs every task in its own loop.
_LOOP_REGISTRIES: "weakref.WeakKeyDictionary[Any, FetcherRegistry]" = weakref.WeakKeyDictionary()


def loop_registry() -> FetcherRegistry:
    """The registry for the running event loop, created on first use.

    Reusing it means one TLS handshake per exchange per run instead of one per
    ticker — previously every ticker built a brand-new set of fetchers (and their
    httpx clients), then never closed them.
    """
    loop = asyncio.get_running_loop()
    registry = _LOOP_REGISTRIES.get(loop)
    if registry is None:
        registry = default_registry()
        _LOOP_REGISTRIES[loop] = registry
    return registry


async def aclose_loop_registry() -> None:
    """Close and forget the running loop's registry (app shutdown / task exit)."""
    registry = _LOOP_REGISTRIES.pop(asyncio.get_running_loop(), None)
    if registry is not None:
        await registry.aclose()
