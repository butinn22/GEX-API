"""Tests for the TBANK adapter (dry-run path + SDK status mapping)."""
from __future__ import annotations

import asyncio

from trading.adapters.brokers.tbank import _STATUS, TbankBroker
from trading.domain import OrderIntent, OrderStatus, OrderType, Quantity, Side


def test_dry_run_without_token():
    broker = TbankBroker("")  # empty token → dry-run
    assert broker.dry_run is True
    accounts = asyncio.run(broker.get_accounts())
    assert accounts[0].currency == "RUB"
    assert asyncio.run(broker.get_positions()) == []
    assert asyncio.run(broker.get_portfolio()).currency == "RUB"


def test_dry_run_place_order_pending():
    broker = TbankBroker("")
    intent = OrderIntent("SBER", Side.BUY, Quantity(10), OrderType.MARKET)
    order = asyncio.run(broker.place_order(intent))
    assert order.status is OrderStatus.PENDING
    assert order.symbol == "SBER"


def test_status_mapping():
    assert _STATUS["FILL"] is OrderStatus.FILLED
    assert _STATUS["PARTIALLY_FILL"] is OrderStatus.PARTIAL
    assert _STATUS["REJECTED"] is OrderStatus.REJECTED
    assert _STATUS["NEW"] is OrderStatus.OPEN


def test_live_session_construction_with_token():
    # A token yields a live (non-dry-run) adapter, but the SDK session is built
    # lazily (its ctor registers against the API) — constructing the adapter must
    # NOT hit the network.
    broker = TbankBroker("fake-token", account_id="123", sandbox=True)
    assert broker.dry_run is False
    assert broker._session is None  # lazy — no network on construction
