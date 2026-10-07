"""Tests for the broker health monitor and WS reconnect manager."""
from __future__ import annotations

import pytest

from trading.application.broker_health import BrokerHealthMonitor
from trading.application.ws_reconnect import WSReconnectManager
from trading.domain import (
    Exchange,
    Order,
    OrderIntent,
    Portfolio,
)
from trading.ports import BrokerAdapter


class _Broker(BrokerAdapter):
    exchange = Exchange.BINGX

    async def get_accounts(self):
        return []

    async def get_portfolio(self):
        return Portfolio(cash=0.0)

    async def get_positions(self):
        return []

    async def place_order(self, intent: OrderIntent) -> Order:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> Order:
        raise NotImplementedError

    async def get_order_status(self, order_id: str) -> Order:
        raise NotImplementedError


class _DownBroker(_Broker):
    exchange = Exchange.TBANK

    async def get_accounts(self):
        raise RuntimeError("down")


async def test_health_monitor_sets_status():
    mon = BrokerHealthMonitor()
    assert await mon.check(Exchange.BINGX, _Broker()) is True
    assert await mon.check(Exchange.TBANK, _DownBroker()) is False


def test_backoff_schedule_caps():
    m = WSReconnectManager(base_delay=1.0, max_delay=10.0, factor=2.0)
    assert m.delay(0) == 1.0
    assert m.delay(1) == 2.0
    assert m.delay(2) == 4.0
    assert m.delay(20) == 10.0  # capped


async def test_reconnect_until_success():
    attempts = {"n": 0}

    async def connect():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("fail")
        return "session"

    m = WSReconnectManager(base_delay=0.0, max_attempts=5)
    session = await m.run(connect)
    assert session == "session"
    assert attempts["n"] == 3


async def test_reconnect_exhausts_attempts():
    async def connect():
        raise ConnectionError("always fails")

    m = WSReconnectManager(base_delay=0.0, max_attempts=2)
    with pytest.raises(ConnectionError):
        await m.run(connect)
