"""Rate-limit middleware + WebSocket connection-cap tests (Round 2, design §3).

Covers the A4 findings end-to-end against the **current** middleware API:

* a dedicated **login bucket** (tighter than the global one) → 429 + Retry-After;
* a **failed-login lockout** that blocks even a later correct password;
* a **successful login clears** the failure counter, and the lock **expires**;
* ``X-Forwarded-For`` honoured **only** when ``trust_proxy`` is on;
* the bucket registry is **LRU-bounded** (no unbounded growth);
* the per-IP **WebSocket connection cap** (the HTTP limiter never sees WS).

The throttle/lockout tests pass explicit small limits so they are deterministic
and independent of the process-wide ``TRADING_LOGIN_RATE_LIMIT`` the app uses.
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from trading.api.middleware import LOGIN_PATH, RateLimitMiddleware, _LRUBuckets
from trading.api.ws_limits import WSConnectionLimiter

_GOOD = {"username": "admin", "password": "good"}
_BAD = {"username": "admin", "password": "nope"}


def _login_app() -> FastAPI:
    """Minimal app: the login path (200 on _GOOD, else 401) + a plain GET."""
    app = FastAPI()

    @app.post(LOGIN_PATH)
    async def login(request: Request) -> JSONResponse:
        body = await request.json()
        if body == _GOOD:
            return JSONResponse({"ok": True})
        return JSONResponse({"detail": "bad creds"}, status_code=401)

    @app.get("/api/v1/thing")
    async def thing() -> dict:
        return {"ok": True}

    return app


def _client(**mw) -> TestClient:
    app = _login_app()
    app.add_middleware(RateLimitMiddleware, per_seconds=60.0, **mw)
    return TestClient(app)


# ── global bucket ─────────────────────────────────────────────────────
def test_global_bucket_429_and_retry_after():
    c = _client(requests=2)
    assert [c.get("/api/v1/thing").status_code for _ in range(2)] == [200, 200]
    r = c.get("/api/v1/thing")
    assert r.status_code == 429
    assert r.json()["detail"] == "rate limit exceeded"
    assert int(r.headers["retry-after"]) >= 1


# ── login bucket ──────────────────────────────────────────────────────
def test_login_bucket_throttles_and_advertises_retry_after():
    c = _client(requests=100_000, login_requests=2)
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 200
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 200
    r = c.post(LOGIN_PATH, json=_GOOD)
    assert r.status_code == 429
    assert r.json()["detail"] == "rate limit exceeded"
    assert int(r.headers["retry-after"]) >= 1


def test_login_bucket_is_independent_of_global():
    c = _client(requests=100_000, login_requests=1)
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 200
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 429  # login bucket spent
    assert c.get("/api/v1/thing").status_code == 200  # global bucket untouched


# ── login lockout ─────────────────────────────────────────────────────
def test_lockout_after_failures_blocks_even_a_correct_password():
    c = _client(
        requests=100_000, login_requests=100_000, lockout_failures=3, lockout_seconds=900
    )
    for _ in range(3):
        assert c.post(LOGIN_PATH, json=_BAD).status_code == 401
    r = c.post(LOGIN_PATH, json=_GOOD)
    assert r.status_code == 429
    assert r.json()["detail"] == "too many failed login attempts"
    assert int(r.headers["retry-after"]) >= 1


def test_successful_login_resets_the_failure_counter():
    c = _client(
        requests=100_000, login_requests=100_000, lockout_failures=3, lockout_seconds=900
    )
    c.post(LOGIN_PATH, json=_BAD)
    c.post(LOGIN_PATH, json=_BAD)
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 200  # resets to zero
    c.post(LOGIN_PATH, json=_BAD)
    c.post(LOGIN_PATH, json=_BAD)
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 200  # not locked yet


def test_lockout_expires_after_the_window():
    c = _client(
        requests=100_000, login_requests=100_000, lockout_failures=2, lockout_seconds=0.2
    )
    c.post(LOGIN_PATH, json=_BAD)
    c.post(LOGIN_PATH, json=_BAD)
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 429
    time.sleep(0.5)  # > lockout window
    assert c.post(LOGIN_PATH, json=_GOOD).status_code == 200


# ── trusted proxy ─────────────────────────────────────────────────────
def test_trust_proxy_off_ignores_x_forwarded_for():
    c = _client(requests=2, trust_proxy=False)
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
    # every request shares the single "testclient" bucket → the 3rd is limited
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "3.3.3.3"}).status_code == 429


def test_trust_proxy_on_keys_by_forwarded_for():
    c = _client(requests=2, trust_proxy=True)
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
    assert c.get("/api/v1/thing", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429


# ── bounded registry ──────────────────────────────────────────────────
def test_bucket_registry_is_lru_bounded():
    reg = _LRUBuckets(capacity=1.0, refill_rate=1.0, max_size=3)
    for i in range(50):
        reg.acquire(f"client-{i}")
    assert len(reg) == 3  # never grows past the cap


# ── WebSocket connection cap ──────────────────────────────────────────
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
            # A second concurrent connection from the same client is refused.
            with pytest.raises(Exception):
                with client.websocket_connect(
                    "/ws/signals", subprotocols=["gex.jwt", token]
                ) as ws2:
                    ws2.receive_json()
