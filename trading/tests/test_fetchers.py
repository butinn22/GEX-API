"""Tests for the real data fetchers (offline, mocked HTTP transport)."""
from __future__ import annotations

import httpx
import pytest

from trading.adapters.fetchers import (
    BybitFetcher,
    FetcherRegistry,
    MoexIssFetcher,
    WebullFetcher,
    YFinanceFetcher,
    validate_bars,
)
from trading.domain import Bar, DataFetchError, Exchange
from trading.ports import BaseFetcher


def _yahoo_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={
        "chart": {"result": [{
            "timestamp": [1700000000, 1700086400],
            "indicators": {"quote": [{
                "open": [100.0, 101.0], "high": [102.0, 103.0],
                "low": [99.0, 100.0], "close": [101.0, 102.0],
                "volume": [1000, 1100],
            }]},
        }]},
    })


async def test_yfinance_ohlcv():
    f = YFinanceFetcher(transport=httpx.MockTransport(_yahoo_handler))
    bars = await f.get_ohlcv("AAPL", "1d")
    assert len(bars) == 2
    assert bars[0].open == 100.0 and bars[1].close == 102.0
    with pytest.raises(DataFetchError):
        await f.get_orderbook("AAPL")


async def test_bybit_ohlcv_reversed_to_ascending():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"list": [
            ["1700086400000", "101", "103", "100", "102", "1100", "0"],  # newest
            ["1700000000000", "100", "102", "99", "101", "1000", "0"],   # oldest
        ]}})

    f = BybitFetcher(transport=httpx.MockTransport(handler))
    bars = await f.get_ohlcv("BTCUSDT", "1d")
    assert bars[0].close == 101.0  # ascending order
    assert bars[1].close == 102.0


async def test_bybit_orderbook_and_trades():
    def handler(request: httpx.Request) -> httpx.Response:
        if "orderbook" in str(request.url):
            return httpx.Response(200, json={"result": {
                "b": [["100.5", "2"], ["100.0", "1"]],
                "a": [["101.0", "1"], ["101.5", "2"]],
            }})
        return httpx.Response(200, json={"result": {"list": [
            {"time": "1700000000000", "price": "100.5", "size": "1.5", "side": "Buy"},
        ]}})

    f = BybitFetcher(transport=httpx.MockTransport(handler))
    book = await f.get_orderbook("BTCUSDT")
    assert book.best_bid == 100.5 and book.best_ask == 101.0
    trades = await f.get_trades("BTCUSDT")
    assert trades[0].side == "buy" and trades[0].volume == 1.5


async def test_moex_ohlcv_columns_based():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candles": {
            "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
            "data": [[100.0, 101.0, 102.0, 99.0, 1000000, 1000, "2024-01-01 10:00:00", "2024-01-01 10:59:59"]],
        }})

    f = MoexIssFetcher(transport=httpx.MockTransport(handler))
    bars = await f.get_ohlcv("SBER", "1d")
    assert len(bars) == 1
    assert bars[0].close == 101.0
    assert bars[0].timestamp.tzinfo is not None  # normalised to UTC


async def test_moex_orderbook_sides_sorted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"orderbook": {
            "columns": ["BOARDID", "SECID", "BUYSELL", "PRICE", "QUANTITY", "SEQNUM", "UPDATETIME", "DECIMALS"],
            "data": [
                ["TQBR", "SBER", "B", "100.5", 10, 1, "2024-01-01 10:00:00", 1],
                ["TQBR", "SBER", "S", "101.0", 8, 2, "2024-01-01 10:00:00", 1],
                ["TQBR", "SBER", "B", "99.5", 20, 3, "2024-01-01 10:00:00", 1],
            ],
        }})

    f = MoexIssFetcher(transport=httpx.MockTransport(handler))
    book = await f.get_orderbook("SBER")
    assert book.best_bid == 100.5  # bids sorted desc
    assert book.best_ask == 101.0


async def test_webull_with_resolver():
    f = WebullFetcher(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
            "data": {"stock": {"data": [
                [1700000000, 100.0, 102.0, 99.0, 101.0, 1000],
                [1700086400, 101.0, 103.0, 100.0, 102.0, 1100],
            ]}},
        })),
        resolve_ticker_id=lambda s: 913256135,
    )
    bars = await f.get_ohlcv("AAPL", "1d")
    assert len(bars) == 2 and bars[0].close == 101.0


class _FakeFetcher(BaseFetcher):
    exchange = Exchange.YFINANCE

    def __init__(self, *, fail=False):
        self._fail = fail

    async def get_ohlcv(self, symbol, timeframe, *, start=None, end=None, limit=500):
        if self._fail:
            raise DataFetchError("boom")
        return [Bar(bars_timestamp(), 1, 2, 0.5, 1.5, 10)]

    async def get_orderbook(self, symbol, *, depth=20):
        raise DataFetchError("no")

    async def get_trades(self, symbol, *, limit=100):
        raise DataFetchError("no")

    async def get_instruments(self):
        return []


def bars_timestamp():
    from datetime import datetime, timezone
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


async def test_registry_fallback():
    reg = FetcherRegistry()
    reg.register(_FakeFetcher(fail=True))
    good = _FakeFetcher(fail=False)
    reg.register(good)
    bars = await reg.get_ohlcv([Exchange.YFINANCE], "X", "1d")
    assert len(bars) == 1

    # both fail → DataFetchError
    reg2 = FetcherRegistry()
    reg2.register(_FakeFetcher(fail=True))
    with pytest.raises(DataFetchError):
        await reg2.get_ohlcv([Exchange.YFINANCE], "X", "1d")


def test_validate_bars_dedup_and_sort():
    from datetime import datetime, timedelta, timezone
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [
        Bar(ts + timedelta(days=1), 1, 2, 0.5, 1.5, 10),
        Bar(ts, 1, 2, 0.5, 1.5, 10),
        Bar(ts, 1, 2, 0.5, 1.4, 10),  # duplicate timestamp
    ]
    out = validate_bars(bars)
    assert [b.timestamp for b in out] == [ts, ts + timedelta(days=1)]


def test_validate_bars_rejects_bad():
    from datetime import datetime, timezone
    with pytest.raises(DataFetchError):
        validate_bars([])
    # Bar itself guards high/low; validate_bars additionally guards negative values
    with pytest.raises(DataFetchError):
        validate_bars([Bar(datetime(2024, 1, 1, tzinfo=timezone.utc), -1, -1, -1, -1, 10)])  # negative close
    with pytest.raises(DataFetchError):
        validate_bars([Bar(datetime(2024, 1, 1, tzinfo=timezone.utc), 1, 2, 0.5, 1.5, -5)])  # negative volume
