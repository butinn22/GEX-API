"""Tests for BingX HMAC signing and the broker adapter (offline, mocked transport)."""
from __future__ import annotations

import hashlib
import hmac

import httpx
import pytest

from trading.adapters.brokers.bingx import BingxBroker, BingxClient, build_query, sign_hmac_sha256
from trading.domain import OrderIntent, OrderType, Quantity, Side


def test_sign_matches_hmac_reference():
    secret, message = "s3cret", "symbol=BTC-USDT&timestamp=1700000000000"
    expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    assert sign_hmac_sha256(secret, message) == expected
    assert len(sign_hmac_sha256(secret, message)) == 64  # hex sha256


def test_build_query_sorts_and_encodes():
    q = build_query({"z": 1, "a": "x y", "b": 2})
    assert q == "a=x%20y&b=2&z=1"  # sorted, space → %20


def test_signed_params_include_timestamp_and_signature():
    client = BingxClient("key", "secret")
    query, headers = client._signed_params({"symbol": "BTC-USDT"})
    assert headers["X-BX-APIKEY"] == "key"
    assert "timestamp=" in query
    assert "signature=" in query
    # signature must be the HMAC of the query WITHOUT the signature fragment
    base = query.split("&signature=")[0]
    assert query.endswith(sign_hmac_sha256("secret", base))


def test_broker_accounts_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-BX-APIKEY") == "key"
        assert "signature=" in str(request.url)
        return httpx.Response(200, json={
            "code": 0,
            "data": {"uid": "u1", "balance": {
                "balance": "1000", "availableMargin": "900", "usedMargin": "100"}},
        })

    client = BingxClient("key", "secret", transport=httpx.MockTransport(handler))
    broker = BingxBroker(client)
    import asyncio
    accounts = asyncio.run(broker.get_accounts())
    assert accounts[0].cash == pytest.approx(1000)
    assert accounts[0].buying_power == pytest.approx(900)
    assert accounts[0].currency == "USDT"


def test_broker_raises_on_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 100001, "msg": "bad request"})

    client = BingxClient("key", "secret", transport=httpx.MockTransport(handler))
    broker = BingxBroker(client)
    import asyncio
    from trading.domain import BrokerError
    with pytest.raises(BrokerError):
        asyncio.run(broker.get_accounts())


def test_place_order_maps_to_domain():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "data": {"order": {"orderId": "12345"}}})

    client = BingxClient("key", "secret", transport=httpx.MockTransport(handler))
    broker = BingxBroker(client)
    import asyncio
    intent = OrderIntent("BTC-USDT", Side.BUY, Quantity(0.1), OrderType.MARKET, strategy="s", reason="r")
    order = asyncio.run(broker.place_order(intent))
    assert order.id == "12345"
    assert order.symbol == "BTC-USDT"
