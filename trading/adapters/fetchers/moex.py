"""MOEX ISS fetcher (public REST, async httpx).

Covers OHLCV, order book, trades, and the securities catalogue for the main
shares board (TQBR). Timestamps are exchange-local (Europe/Moscow, UTC+3, no
DST since 2014) and are normalised to UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from trading.domain import (
    Bar,
    BookLevel,
    DataFetchError,
    Exchange,
    Instrument,
    OrderBook,
    Tick,
)
from trading.ports import BaseFetcher

from .http_util import get_json

__all__ = ["MoexIssFetcher"]

_BASE = "https://iss.moex.com"
_MOSCOW = timezone(timedelta(hours=3))

_INTERVAL = {"1m": 1, "10m": 10, "1h": 60, "1d": 24, "1w": 7, "1mo": 31}

#: Calendar-day lookback per timeframe, sized for ~500 candles (MOEX caps at 500).
_WINDOW_DAYS = {"1m": 3, "10m": 10, "1h": 60, "1d": 700, "1w": 2500, "1mo": 15000}


def _rows(block: dict, keys: list[str]) -> list[dict]:
    cols = block.get("columns", [])
    idx = {c: i for i, c in enumerate(cols)}
    out = []
    for row in block.get("data", []):
        out.append({k: row[idx[k]] if idx.get(k) is not None and idx[k] < len(row) else None for k in keys})
    return out


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=_MOSCOW).astimezone(timezone.utc)


class MoexIssFetcher(BaseFetcher):
    exchange = Exchange.MOEX

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=_BASE, transport=transport)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_instruments(self) -> list[Instrument]:
        data = await get_json(self._client, "/iss/engines/stock/markets/shares/securities.json")
        out = []
        for r in _rows(data.get("securities", {}), ["SECID", "LOTSIZE", "MINSTEP"]):
            out.append(
                Instrument(
                    symbol=r["SECID"],
                    exchange=self.exchange,
                    lot_size=float(r["LOTSIZE"] or 1.0),
                    tick_size=float(r["MINSTEP"] or 0.0),
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
        interval = _INTERVAL.get(timeframe, 24)
        till = end or datetime.now(timezone.utc)
        # Fetch the LATEST window: MOEX returns the oldest history unless a recent
        # `from` is given.
        from_dt = start or (till - timedelta(days=_WINDOW_DAYS.get(timeframe, 700)))
        params: dict = {
            "interval": interval, "iss.meta": "off", "limit": min(limit, 500),
            "from": from_dt.strftime("%Y-%m-%d"), "till": till.strftime("%Y-%m-%d"),
        }
        data = await get_json(
            self._client,
            f"/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json",
            params=params,
        )
        bars: list[Bar] = []
        for r in _rows(data.get("candles", {}), ["open", "close", "high", "low", "volume", "begin"]):
            ts = _parse_time(r["begin"])
            if ts is None or None in (r["open"], r["close"], r["high"], r["low"]):
                continue
            bars.append(
                Bar(
                    timestamp=ts,
                    open=float(r["open"]), high=float(r["high"]),
                    low=float(r["low"]), close=float(r["close"]),
                    volume=float(r["volume"] or 0.0),
                )
            )
        if not bars:
            raise DataFetchError(f"moex: no candles for {symbol}")
        return bars

    async def get_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        data = await get_json(
            self._client,
            f"/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/orderbook.json",
            params={"iss.meta": "off"},
        )
        bids, asks = [], []
        for r in _rows(data.get("orderbook", {}), ["BUYSELL", "PRICE", "QUANTITY"]):
            level = BookLevel(float(r["PRICE"] or 0.0), float(r["QUANTITY"] or 0.0))
            (bids if r["BUYSELL"] == "B" else asks).append(level)
        bids.sort(key=lambda l: l.price, reverse=True)
        asks.sort(key=lambda l: l.price)
        return OrderBook(bids=tuple(bids[:depth]), asks=tuple(asks[:depth]))

    async def get_trades(self, symbol: str, *, limit: int = 100) -> list[Tick]:
        data = await get_json(
            self._client,
            f"/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/trades.json",
            params={"iss.meta": "off", "limit": min(limit, 500)},
        )
        out: list[Tick] = []
        for r in _rows(data.get("trades", {}), ["TRADETIME", "PRICE", "QUANTITY", "BUYSELL"]):
            ts = _parse_time(r["TRADETIME"])
            if ts is None or r["PRICE"] is None:
                continue
            out.append(
                Tick(
                    timestamp=ts,
                    price=float(r["PRICE"]),
                    volume=float(r["QUANTITY"] or 0.0),
                    side="buy" if r["BUYSELL"] == "B" else "sell",
                )
            )
        return out
