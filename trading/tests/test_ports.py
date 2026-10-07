"""Tests for the trading ports (async interfaces)."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from trading.domain import (
    Bar,
    Exchange,
    Instrument,
    Order,
    OrderBook,
    OrderIntent,
    OrderType,
    Portfolio,
    Quantity,
    Side,
    Tick,
)
from trading.ports import BaseFetcher, BrokerAdapter, Strategy

# ── Abstractness ──────────────────────────────────────────────────────


def test_base_fetcher_is_abstract():
    with pytest.raises(TypeError):
        BaseFetcher()  # type: ignore[abstract]


def test_broker_adapter_is_abstract():
    with pytest.raises(TypeError):
        BrokerAdapter()  # type: ignore[abstract]


def test_strategy_is_abstract():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_incomplete_fetcher_still_abstract():
    class Partial(BaseFetcher):
        exchange = Exchange.YFINANCE

        async def get_instruments(self):
            return []

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


# ── Concrete implementations satisfy the port ─────────────────────────


class FakeFetcher(BaseFetcher):
    exchange = Exchange.YFINANCE

    async def get_instruments(self):
        return [Instrument("AAPL", self.exchange, base_asset="AAPL", quote_asset="USD")]

    async def get_ohlcv(self, symbol, timeframe, *, start=None, end=None, limit=500):
        return [Bar(datetime(2024, 1, 1), 1, 2, 0.5, 1.5, 10)]

    async def get_orderbook(self, symbol, *, depth=20):
        return OrderBook()

    async def get_trades(self, symbol, *, limit=100):
        return [Tick(datetime(2024, 1, 1), 1.0, 1.0, "buy")]


class FakeBroker(BrokerAdapter):
    exchange = Exchange.BINGX

    async def get_accounts(self):
        return []

    async def get_portfolio(self):
        return Portfolio(cash=0.0)

    async def get_positions(self):
        return []

    async def place_order(self, intent):
        return intent.to_order("o1")

    async def cancel_order(self, order_id):
        raise NotImplementedError

    async def get_order_status(self, order_id):
        raise NotImplementedError


class FakeStrategy(Strategy):
    name = "fake"

    async def on_bar(self, bar):
        return []

    async def on_tick(self, tick):
        return []

    async def generate_signals(self, bars):
        return []


def test_fake_fetcher_works():
    f = FakeFetcher()
    assert f.exchange is Exchange.YFINANCE
    bars = asyncio.run(f.get_ohlcv("AAPL", "1d", limit=10))
    assert len(bars) == 1 and bars[0].close == 1.5
    book = asyncio.run(f.get_orderbook("AAPL"))
    assert book.mid_price is None


def test_fake_broker_place_order():
    b = FakeBroker()
    intent = OrderIntent("AAPL", Side.BUY, Quantity(1), OrderType.MARKET)
    o = asyncio.run(b.place_order(intent))
    assert isinstance(o, Order)
    assert o.symbol == "AAPL"


def test_fake_strategy_runs():
    s = FakeStrategy()
    assert s.name == "fake"
    out = asyncio.run(s.generate_signals([]))
    assert out == []
