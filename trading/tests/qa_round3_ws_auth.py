"""QA Round-3 — INDEPENDENT WebSocket auth verification (B6).

Proves unauthorised connects are rejected on both WS surfaces:

* ``/ws/{signals,orders,positions}`` — JWT via ``Sec-WebSocket-Protocol``
  (fallback ``?token=``); validated BEFORE accept → anon/tampered handshake
  fails.
* ``/ws/client`` — token inside the ``handshake`` frame; an invalid/missing
  token never registers the session.

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_ws_auth.py -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from trading.api.local_client_ws import registry
from trading.main import app

pytestmark = pytest.mark.real_auth


def _token(client: TestClient) -> str:
    r = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
    )
    assert r.status_code == 200
    return r.json()["access_token"]


def _rejected(client: TestClient, path: str, subprotocols=None) -> bool:
    """True when the server refuses to open a usable stream."""
    try:
        with client.websocket_connect(path, subprotocols=subprotocols) as ws:
            ws.receive_json()  # a usable stream would send hello
        return False
    except WebSocketDisconnect:
        return True


# ── browser streams ───────────────────────────────────────────────────
@pytest.mark.parametrize("path", ["/ws/signals", "/ws/orders", "/ws/positions"])
def test_anonymous_stream_rejected(path):
    with TestClient(app) as client:
        assert _rejected(client, path) is True, f"{path} accepted an anonymous handshake"


def test_bad_jwt_stream_rejected():
    with TestClient(app) as client:
        assert _rejected(client, "/ws/signals", subprotocols=["gex.jwt", "not-a-jwt"])


def test_valid_jwt_subprotocol_accepted():
    with TestClient(app) as client:
        token = _token(client)
        with client.websocket_connect(
            "/ws/signals", subprotocols=["gex.jwt", token]
        ) as ws:
            assert ws.receive_json()["type"] == "hello"


def test_valid_jwt_query_fallback_accepted():
    with TestClient(app) as client:
        token = _token(client)
        with client.websocket_connect(f"/ws/signals?token={token}") as ws:
            assert ws.receive_json()["type"] == "hello"


# ── machine producer /ws/client ───────────────────────────────────────
def _client_snapshots() -> set[str]:
    return {s["session_id"] for s in registry.snapshot()}


def test_ws_client_handshake_without_token_unauthorized():
    with TestClient(app) as client:
        before = _client_snapshots()
        with client.websocket_connect("/ws/client") as ws:
            ws.send_json({"type": "handshake", "client_id": "anon"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "UNAUTHORIZED"
        assert _client_snapshots() == before, "unauthorised handshake registered a session"


def test_ws_client_handshake_bad_token_unauthorized():
    with TestClient(app) as client:
        before = _client_snapshots()
        with client.websocket_connect("/ws/client") as ws:
            ws.send_json({"type": "handshake", "token": "wrong", "client_id": "anon"})
            assert ws.receive_json()["code"] == "UNAUTHORIZED"
        assert _client_snapshots() == before


def test_ws_client_valid_static_token_registers():
    with TestClient(app) as client:
        before = _client_snapshots()
        with client.websocket_connect("/ws/client") as ws:
            ws.send_json(
                {"type": "handshake", "token": "test-client-token", "client_id": "bot"}
            )
            ack = ws.receive_json()
            assert ack["type"] == "handshake_ack"
        for sid in _client_snapshots() - before:
            registry.disconnect(sid)


def test_ws_client_valid_jwt_token_registers():
    with TestClient(app) as client:
        token = _token(client)
        before = _client_snapshots()
        with client.websocket_connect("/ws/client") as ws:
            ws.send_json({"type": "handshake", "token": token, "client_id": "bot2"})
            assert ws.receive_json()["type"] == "handshake_ack"
        for sid in _client_snapshots() - before:
            registry.disconnect(sid)
