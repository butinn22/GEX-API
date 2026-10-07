"""Webull fetcher (best-effort, public REST).

Webull's public API requires device headers and resolves symbols to numeric
``tickerId`` via a search endpoint; this fetcher implements that flow but is
marked best-effort because the endpoints may reject unauthenticated requests.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx

from trading.domain import Bar, DataFetchError, Exchange, Instrument, OrderBook, Tick
from trading.ports import BaseFetcher

__all__ = ["WebullFetcher"]

_BASE = "https://quotes-gw.webullfintech.com"
_TYPE = {"1m": "m1", "5m": "m5", "15m": "m15", "1h": "m60", "1d": "d", "1w": "w"}


class WebullFetcher(BaseFetcher):
    exchange = Exchange.WEBULL

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        resolve_ticker_id: Callable[[str], int] | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE,
            transport=transport,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        self._resolve = resolve_ticker_id

    async def close(self) -> None:
        await self._client.aclose()

    async def get_instruments(self) -> list[Instrument]:
        raise DataFetchError("webull has no public symbol catalogue (use search)")

    async def _ticker_id(self, symbol: str) -> int:
        if self._resolve is not None:
            return self._resolve(symbol)
        r = await self._client.get(
            "/api/search/pc/tickers", params={"keyword": symbol, "pageIndex": 1, "pageSize": 5}
        )
        if r.status_code != 200:
            raise DataFetchError(f"webull search -> HTTP {r.status_code}")
        data = r.json()
        tickers = (data.get("data") or {}).get("tickers") or []
        if not tickers:
            raise DataFetchError(f"webull: no ticker for {symbol}")
        return int(tickers[0]["tickerId"])

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        ticker_id = await self._ticker_id(symbol)
        body = {"tickerId": ticker_id, "type": _TYPE.get(timeframe, "d"), "count": min(limit, 1000)}
        r = await self._client.post("/api/wlas/charts/query", json=body)
        if r.status_code != 200:
            raise DataFetchError(f"webull chart -> HTTP {r.status_code}")
        data = r.json()
        rows = ((data.get("data") or {}).get("stock") or {}).get("data") or []
        bars: list[Bar] = []
        for row in rows:
            # row: [timestamp, open, high, low, close, volume, ...]
            if len(row) < 6 or row[0] is None:
                continue
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(int(row[0]), tz=UTC),
                    open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5] or 0.0),
                )
            )
        if not bars:
            raise DataFetchError(f"webull: no candles for {symbol}")
        return bars

    async def get_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        raise DataFetchError("webull order book requires authenticated stream")

    async def get_trades(self, symbol: str, *, limit: int = 100) -> list[Tick]:
        raise DataFetchError("webull trade tape requires authenticated stream")
