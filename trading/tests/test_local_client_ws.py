"""Local signal client WebSocket protocol (/ws/client)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from trading.api.local_client_ws import registry
from trading.main import app


@pytest.fixture
def ws_client():
    with TestClient(app) as c:  # context manager runs lifespan (dispatcher start)
        yield c
    # tests share the module-level registry; leave no sessions behind
    for snap in registry.snapshot():
        registry.disconnect(snap["session_id"])


def open_ws(c):
    return c.websocket_connect("/ws/client")


class TestHandshake:
    def test_handshake_ack_carries_session_and_timing(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "bot-1", "version": "1"})
            ack = ws.receive_json()
            assert ack["type"] == "handshake_ack"
            assert ack["session_id"]
            assert ack["heartbeat_interval"] > 0
            assert ack["heartbeat_timeout"] > ack["heartbeat_interval"]
            assert ack["max_tickers"] == 20
            assert registry.get(ack["session_id"]).client_id == "bot-1"

    def test_messages_before_handshake_are_rejected(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "subscribe", "tickers": ["BTC-USDT"]})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "NO_HANDSHAKE"


class TestHeartbeat:
    def test_ping_gets_pong(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "b"})
            ws.receive_json()
            ws.send_json({"type": "ping", "ts": 123.0})
            pong = ws.receive_json()
            assert pong == {"type": "pong", "ts": 123.0}


class TestSubscribe:
    def test_subscribe_ack(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "b"})
            ws.receive_json()
            ws.send_json({"type": "subscribe", "tickers": ["btc-usdt", "SBER"]})
            ack = ws.receive_json()
            assert ack["type"] == "subscribe_ack"
            assert ack["tickers"] == ["BTC-USDT", "SBER"]
            assert ack["count"] == 2

    def test_over_limit_returns_error(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "b"})
            ws.receive_json()
            ws.send_json({"type": "subscribe", "tickers": [f"T{i}" for i in range(21)]})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "MAX_TICKERS"


class TestSignalIngest:
    def test_signal_ack_with_native_payloads(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "emf-bot"})
            ws.receive_json()
            ws.send_json({"type": "signal", "payload": {
                "symbol": "BTC-USDT", "side": "buy", "strength": 0.8,
                "quantity": 0.001, "price": 64200.5, "reason": "ema50-cross",
            }})
            ack = ws.receive_json()
            assert ack["type"] == "signal_ack"
            assert ack["signal_id"]
            assert ack["symbol"] == "BTC-USDT"
            bingx = ack["native_payloads"]["bingx"]
            assert bingx["symbol"] == "BTC-USDT"
            assert bingx["side"] == "BUY"
            assert bingx["positionSide"] == "LONG"
            assert bingx["type"] == "MARKET"
            tbank = ack["native_payloads"]["tbank"]
            assert tbank["direction"] == "ORDER_DIRECTION_BUY"
            assert tbank["order_type"] == "ORDER_TYPE_MARKET"

    def test_bad_signal_returns_error(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "b"})
            ws.receive_json()
            ws.send_json({"type": "signal", "payload": {"side": "buy"}})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "BAD_SIGNAL"

    def test_signal_before_handshake_rejected(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "signal", "payload": {
                "symbol": "X", "side": "buy", "quantity": 1}})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "NO_HANDSHAKE"

    def test_unknown_message_type(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "b"})
            ws.receive_json()
            ws.send_json({"type": "explode"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "BAD_MESSAGE"


class TestProducerAuth:
    def test_handshake_without_token_is_rejected(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "client_id": "anon"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "UNAUTHORIZED"

    def test_handshake_with_bad_token_is_rejected(self, ws_client):
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "wrong", "client_id": "anon"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "UNAUTHORIZED"


class TestHealthEndpoint:
    def test_lists_active_sessions(self, ws_client):
        r = ws_client.post("/api/v1/auth/token",
                           json={"username": "admin", "password": "admin"})
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        with open_ws(ws_client) as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token", "client_id": "health-bot"})
            ack = ws.receive_json()
            resp = ws_client.get("/api/v1/local-clients", headers=headers)
            assert resp.status_code == 200
            clients = resp.json()["clients"]
            mine = [c for c in clients if c["session_id"] == ack["session_id"]]
            assert mine and mine[0]["client_id"] == "health-bot"
            assert mine[0]["state"] == "active"
            assert mine[0]["stale"] is False

    @pytest.mark.real_auth
    def test_requires_auth(self, ws_client):
        assert ws_client.get("/api/v1/local-clients").status_code == 401
