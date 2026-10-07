"""Tests for the order execution engine (routing, retry, kill switch)."""
from __future__ import annotations

import pytest

from trading.application.execution import BrokerRouter, ExecutionEngine, KillSwitch
from trading.domain import (
    BrokerError,
    Exchange,
    Order,
    OrderIntent,
    OrderRejectedError,
    OrderType,
    Portfolio,
    Quantity,
    Side,
)
from trading.ports import BrokerAdapter


class FakeBroker(BrokerAdapter):
    exchange = Exchange.BINGX

    def __init__(self, *, fail_count: int = 0, reject: bool = False):
        self.calls = 0
        self.fail_count = fail_count
        self.reject = reject

    async def place_order(self, intent: OrderIntent) -> Order:
        self.calls += 1
        if self.reject:
            raise OrderRejectedError("rejected by exchange")
        if self.calls <= self.fail_count:
            raise BrokerError("transient network error")
        return Order(id="o1", symbol=intent.symbol, side=intent.side,
                     quantity=intent.quantity.value, order_type=intent.order_type)

    async def get_accounts(self):
        return []

    async def get_portfolio(self):
        return Portfolio(cash=1000.0)

    async def get_positions(self):
        return []

    async def cancel_order(self, order_id: str) -> Order:
        return Order(id=order_id, symbol="X", side=Side.BUY, quantity=0.0,
                     order_type=OrderType.MARKET)

    async def get_order_status(self, order_id: str) -> Order:
        raise NotImplementedError


def _intent() -> OrderIntent:
    return OrderIntent("BTC-USDT", Side.BUY, Quantity(0.1), OrderType.MARKET)


async def test_router_routes_and_missing():
    router = BrokerRouter()
    b = FakeBroker()
    router.register(b)
    assert router.get(Exchange.BINGX) is b
    with pytest.raises(BrokerError):
        router.get(Exchange.TBANK)


async def test_place_order_success():
    router = BrokerRouter()
    router.register(FakeBroker())
    engine = ExecutionEngine(router)
    order = await engine.place_order(Exchange.BINGX, _intent())
    assert order.id == "o1"


async def test_retry_transient_then_success():
    router = BrokerRouter()
    b = FakeBroker(fail_count=2)
    router.register(b)
    engine = ExecutionEngine(router, max_retries=4, base_delay=0.0)
    await engine.place_order(Exchange.BINGX, _intent())
    assert b.calls == 3  # 2 transient failures then success


async def test_no_retry_on_rejection():
    router = BrokerRouter()
    b = FakeBroker(reject=True)
    router.register(b)
    engine = ExecutionEngine(router, base_delay=0.0)
    with pytest.raises(OrderRejectedError):
        await engine.place_order(Exchange.BINGX, _intent())
    assert b.calls == 1  # permanent rejection → no retry


async def test_raises_after_max_retries():
    router = BrokerRouter()
    b = FakeBroker(fail_count=99)
    router.register(b)
    engine = ExecutionEngine(router, max_retries=3, base_delay=0.0)
    with pytest.raises(BrokerError):
        await engine.place_order(Exchange.BINGX, _intent())
    assert b.calls == 3


def test_kill_switch_threshold():
    ks = KillSwitch(max_drawdown=0.2)
    assert ks.update(100.0) is False
    assert ks.update(95.0) is False  # 5% dd
    assert ks.update(79.0) is True  # 21% dd > 20%
    assert ks.update(50.0) is True
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=1.5)
