"""WebSocket per-IP connection-cap tests (Round 2, design §3.4).

``BaseHTTPMiddleware`` never sees the ASGI ``websocket`` scope, so the HTTP
rate-limit middleware cannot bound WebSocket connections; a dedicated per-IP
concurrent-connection cap is enforced at the WS accept path in both
``trading/api/websockets.py`` and ``trading/api/local_client_ws.py``.

(The HTTP login bucket / lockout / bounded-registry behaviour is pinned in
``test_rate_limit_round2.py``.)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trading.api.ws_limits import WSConnectionLimiter


class _FakeWS:
    """Stand-in exposing just the ``client``/``headers`` a limiter reads."""

    def __init__(self, host: str = "1.2.3.4", headers: dict | None = None) -> None:
        self.client = type("_C", (), {"host": host})()
        self.headers = headers or {}


def test_ws_connection_limiter_counts_per_ip():
    lim = WSConnectionLimiter(max_per_ip=2)
    a, b, c = _FakeWS(), _FakeWS(), _FakeWS()
    assert lim.acquire(a) is True
    assert lim.acquire(b) is True
    assert lim.acquire(c) is False  # at the cap
    lim.release(a)
    assert lim.acquire(c) is True
    assert lim.count(c) == 2
    lim.release(b)
    lim.release(c)
    assert lim.count(c) == 0


def test_ws_connection_limiter_is_per_client():
    lim = WSConnectionLimiter(max_per_ip=1)
    assert lim.acquire(_FakeWS(host="1.1.1.1")) is True
    assert lim.acquire(_FakeWS(host="1.1.1.1")) is False
    assert lim.acquire(_FakeWS(host="2.2.2.2")) is True  # unrelated client


def test_app_ws_endpoint_enforces_the_cap(monkeypatch):
    from trading.api import websockets as wsmod
    from trading.main import app

    monkeypatch.setattr(wsmod.ws_limiter, "_max", 1)
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
        )
        token = r.json()["access_token"]
        with client.websocket_connect(
            "/ws/signals", subprotocols=["gex.jwt", token]
        ) as ws:
            assert ws.receive_json()["stream"] == "signals"
            # Second concurrent connection from the same client is refused.
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/ws/signals", subprotocols=["gex.jwt", token]
                ) as ws2:
                    ws2.receive_json()
