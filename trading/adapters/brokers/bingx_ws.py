"""BingX WebSocket streams (market data + user data).

Message payloads are parsed into domain types; the raw JSON shapes follow the
Binance-compatible schema BingX mirrors (``e`` event, ``s`` symbol, ``k`` kline,
``b``/``a`` depth). Subscribe message envelopes are built by helper functions and
should be confirmed against the current BingX WS docs before live use — the
parsers are the stable, tested part.

``connect`` is injectable so the stream is testable without a live socket.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from trading.application.ws_reconnect import WSReconnectManager
from trading.domain import Bar, BookLevel, OrderBook, Tick

__all__ = [
    "parse_trade", "parse_kline", "parse_depth",
    "build_subscribe", "BingxMarketStream",
]

_MARKET_URL = "wss://open-api-ws.bingx.com/market"


def _ts_ms(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def parse_trade(msg: dict) -> Tick | None:
    """Map a ``trade`` event to a :class:`Tick` (None if not a trade)."""
    if msg.get("e") != "trade":
        return None
    return Tick(
        timestamp=_ts_ms(msg["T"]),
        price=float(msg["p"]),
        volume=float(msg["q"]),
        side="buy" if msg.get("m") is True else "sell",
    )


def parse_kline(msg: dict) -> Bar | None:
    """Map a ``kline`` event to a :class:`Bar` (None if not a kline)."""
    if msg.get("e") != "kline" or "k" not in msg:
        return None
    k = msg["k"]
    return Bar(
        timestamp=_ts_ms(k["t"]),
        open=float(k["o"]), high=float(k["h"]), low=float(k["l"]),
        close=float(k["c"]), volume=float(k.get("v", 0.0)),
    )


def parse_depth(msg: dict) -> OrderBook | None:
    """Map a ``depth`` event to an :class:`OrderBook` (None if not depth)."""
    if msg.get("e") != "depth":
        return None
    bids = tuple(BookLevel(float(l[0]), float(l[1])) for l in msg.get("b", []))
    asks = tuple(BookLevel(float(l[0]), float(l[1])) for l in msg.get("a", []))
    return OrderBook(bids=bids, asks=asks, timestamp=_ts_ms(msg.get("E", 0)))


def build_subscribe(sub_id: str, data_type: str, *, symbol: str | None = None) -> str:
    """Build a subscribe message (BingX market-WS envelope)."""
    payload: dict[str, Any] = {"id": sub_id, "dataType": data_type}
    if symbol is not None:
        payload["data"] = {"pair": symbol}
    return json.dumps(payload)


class BingxMarketStream:
    """Connect to the public market stream and yield parsed domain objects."""

    def __init__(
        self,
        symbol: str,
        channels: list[str],
        *,
        url: str = _MARKET_URL,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        self.symbol = symbol
        self.channels = channels
        self.url = url
        self._connect = connect

    async def _open(self):
        if self._connect is not None:
            return await self._connect(self.url)
        import websockets
        return await websockets.connect(self.url)

    async def __aiter__(self) -> AsyncIterator[Tick | Bar | OrderBook]:
        async with await self._open() as ws:
            for i, channel in enumerate(self.channels):
                await ws.send(build_subscribe(f"sub-{i}", f"{self.symbol}@{channel}", symbol=self.symbol))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                for parsed in (parse_trade(msg), parse_kline(msg), parse_depth(msg)):
                    if parsed is not None:
                        yield parsed
                        break
