"""WebSocket endpoint auth + smoke tests (Starlette TestClient)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trading.main import app


def _token(client: TestClient) -> str:
    r = client.post("/api/v1/auth/token", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_ws_signals_hello_and_close():
    with TestClient(app) as client:
        token = _token(client)
        with client.websocket_connect(
            "/ws/signals", subprotocols=["gex.jwt", token]
        ) as ws:
            msg = ws.receive_json()
            assert msg["type"] == "hello"
            assert msg["stream"] == "signals"


def test_ws_orders_and_positions_hello():
    with TestClient(app) as client:
        token = _token(client)
        with client.websocket_connect("/ws/orders", subprotocols=["gex.jwt", token]) as ws:
            assert ws.receive_json()["stream"] == "orders"
        with client.websocket_connect(
            "/ws/positions", subprotocols=["gex.jwt", token]
        ) as ws:
            assert ws.receive_json()["stream"] == "positions"


def test_ws_token_via_query_fallback():
    with TestClient(app) as client:
        token = _token(client)
        with client.websocket_connect(f"/ws/signals?token={token}") as ws:
            assert ws.receive_json()["stream"] == "signals"


def test_ws_anonymous_is_rejected():
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/signals") as ws:
                ws.receive_json()


def test_ws_bad_token_is_rejected():
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/ws/signals", subprotocols=["gex.jwt", "not-a-jwt"]
            ) as ws:
                ws.receive_json()
