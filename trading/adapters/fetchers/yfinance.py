"""yFinance fetcher (Yahoo Finance public chart API, async httpx).

Covers OHLCV only: Yahoo has no public order book / trade-tape endpoint, so
``get_orderbook``/``get_trades`` raise ``DataFetchError`` (the registry falls
through to another source for those).
"""
from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx

from trading.domain import Bar, DataFetchError, Exchange, Instrument, OrderBook, Tick
from trading.ports import BaseFetcher

from .http_util import get_json

__all__ = ["YFinanceFetcher"]

_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

#: ``timeframe -> (yahoo interval, hard server-side history cap in seconds)``.
#: The cap is Yahoo's own limit for the interval (e.g. 1-minute data only goes
#: back 7 days); asking for more silently returns what it has.
_INTERVAL_RANGE = {
    "1m": ("1m", 7 * 86400),
    "5m": ("5m", 60 * 86400),
    "15m": ("15m", 60 * 86400),
    "1h": ("60m", 730 * 86400),
    "1d": ("1d", 0),
    "1wk": ("1wk", 0),
    "1mo": ("1mo", 0),
}

#: Rough calendar-day span of one bar, per timeframe.
_BAR_DAYS = {
    "1m": 1 / 390,  # ~390 trading minutes per US session
    "5m": 5 / 390,
    "15m": 15 / 390,
    "1h": 1 / 6.5,
    "1d": 1.45,  # a trading day spans ~1.45 calendar days (weekends/holidays)
    "1wk": 7.0,
    "1mo": 31.0,
}

#: Extra headroom so holidays/suspensions can't leave us short of ``limit`` bars.
_WINDOW_SAFETY = 1.6

#: Never ask for less than this, and don't add pointless headroom near the cap.
_MIN_WINDOW_DAYS = 5.0


def _lookback_seconds(timeframe: str, limit: int) -> int:
    """Calendar seconds needed to obtain ``limit`` bars of ``timeframe``.

    Sizing the request to the data we actually return is the difference between
    a ~90 KB response and a ~1.3 MB one: Yahoo otherwise hands back the ticker's
    entire daily history (AAPL alone is ~11,500 bars) which we then slice to
    ``[-limit:]`` and throw away.
    """
    per_bar = _BAR_DAYS.get(timeframe, _BAR_DAYS["1d"])
    days = max(limit, 1) * per_bar * _WINDOW_SAFETY
    return int(max(days, _MIN_WINDOW_DAYS) * 86400)



class YFinanceFetcher(BaseFetcher):
    exchange = Exchange.YFINANCE

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE,
            transport=transport,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_instruments(self) -> list[Instrument]:
        # Yahoo has no symbol catalogue endpoint; return an empty list.
        return []

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Bar]:
        interval, server_cap = _INTERVAL_RANGE.get(timeframe, ("1d", 0))
        now = int(time.time())
        params = {"interval": interval, "includePrePost": "false"}
        # Size the window to the requested bar count instead of pulling all
        # history: period1=0 returns every bar since inception (~1.3 MB for AAPL)
        # only for us to slice off the last `limit`.
        wanted = _lookback_seconds(timeframe, limit)
        if server_cap:  # intraday history has a hard server-side limit
            wanted = min(wanted, server_cap)
        params["period1"] = int(start.timestamp()) if start is not None else now - wanted
        params["period2"] = int(end.timestamp()) if end is not None else now
        data = await get_json(self._client, f"/{symbol}", params=params)
        try:
            result = data["chart"]["result"][0]
        except (KeyError, IndexError) as exc:
            raise DataFetchError(f"yahoo: no data for {symbol}") from exc
        ts = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        bars: list[Bar] = []
        for i, t in enumerate(ts):
            o, h, l, c = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i]
            if None in (o, h, l, c):
                continue
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(t, tz=UTC),
                    open=float(o), high=float(h), low=float(l), close=float(c),
                    volume=float(quote["volume"][i] or 0.0),
                )
            )
        return bars[-limit:]

    async def get_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        raise DataFetchError("yfinance has no order book")

    async def get_trades(self, symbol: str, *, limit: int = 100) -> list[Tick]:
        raise DataFetchError("yfinance has no trade tape")
