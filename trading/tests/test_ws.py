"""WebSocket endpoint smoke test (Starlette TestClient)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from trading.main import app


def test_ws_signals_hello_and_close():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signals") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "hello"
            assert msg["stream"] == "signals"


def test_ws_orders_and_positions_hello():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/orders") as ws:
            assert ws.receive_json()["stream"] == "orders"
        with client.websocket_connect("/ws/positions") as ws:
            assert ws.receive_json()["stream"] == "positions"
