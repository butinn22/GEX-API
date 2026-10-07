"""Bybit fetcher (V5 public REST, async httpx).

Public market endpoints need no auth. Covers instruments, OHLCV, order book,
and recent trades.
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx

from trading.domain import (
    Bar,
    BookLevel,
    Exchange,
    Instrument,
    OrderBook,
    Tick,
)
from trading.ports import BaseFetcher

from .http_util import get_json

__all__ = ["BybitFetcher"]

_BASE = "https://api.bybit.com"

_INTERVAL = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W", "1mo": "M"}


class BybitFetcher(BaseFetcher):
    exchange = Exchange.BYBIT

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=_BASE, transport=transport)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_instruments(self) -> list[Instrument]:
        data = await get_json(
            self._client, "/v5/market/instruments-info", params={"category": "spot"}
        )
        out = []
        for item in data.get("result", {}).get("list", []):
            out.append(
                Instrument(
                    symbol=item["symbol"],
                    exchange=self.exchange,
                    base_asset=item.get("baseCoin", ""),
                    quote_asset=item.get("quoteCoin", ""),
                    tick_size=float(item.get("priceFilter", {}).get("tickSize") or 0.0),
                    lot_size=float(item.get("lotSizeFilter", {}).get("basePrecision") or 0.0),
                )
            )
        return out

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        interval = _INTERVAL.get(timeframe, "D")
        params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start is not None:
            params["start"] = int(start.timestamp() * 1000)
        if end is not None:
            params["end"] = int(end.timestamp() * 1000)
        data = await get_json(self._client, "/v5/market/kline", params=params)
        rows = data.get("result", {}).get("list", []) or []
        bars: list[Bar] = []
        for row in reversed(rows):  # API returns newest-first
            t_ms = int(row[0])
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(t_ms / 1000, tz=UTC),
                    open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5]),
                )
            )
        return bars

    async def get_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        data = await get_json(
            self._client, "/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": min(depth, 50)},
        )
        bids = tuple(BookLevel(float(l[0]), float(l[1])) for l in data["result"]["b"])
        asks = tuple(BookLevel(float(l[0]), float(l[1])) for l in data["result"]["a"])
        return OrderBook(bids=bids, asks=asks)

    async def get_trades(self, symbol: str, *, limit: int = 100) -> list[Tick]:
        data = await get_json(
            self._client, "/v5/market/recent-trade",
            params={"category": "spot", "symbol": symbol, "limit": min(limit, 1000)},
        )
        out = []
        for t in data.get("result", {}).get("list", []):
            out.append(
                Tick(
                    timestamp=datetime.fromtimestamp(int(t["time"]) / 1000, tz=UTC),
                    price=float(t["price"]),
                    volume=float(t["size"]),
                    side="buy" if t.get("side") == "Buy" else "sell",
                )
            )
        return out
