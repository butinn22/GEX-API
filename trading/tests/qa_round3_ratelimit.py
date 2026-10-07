"""QA Round-3 — INDEPENDENT rate-limit + WS-cap verification (C12).

Covers: global burst 429+Retry-After, tighter login bucket, failed-login
lockout (+ success clears it), bounded bucket registry, X-Forwarded-For only
honoured with trust_proxy, and the per-IP concurrent WS connection cap.

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_ratelimit.py -q
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.websockets import WebSocketDisconnect

import trading.api.ws_limits as wl
from trading.api.middleware import RateLimitMiddleware, _LRUBuckets
from trading.main import app


async def _ok(_request):
    return JSONResponse({"ok": True})


def _app(**mw_kwargs) -> Starlette:
    a = Starlette(routes=[Route("/", _ok)])
    a.add_middleware(RateLimitMiddleware, **mw_kwargs)
    return a


# ── global burst ──────────────────────────────────────────────────────
def test_global_burst_429_with_retry_after():
    a = _app(requests=3, per_seconds=60.0, login_requests=1000)
    with TestClient(a) as c:
        assert [c.get("/").status_code for _ in range(3)] == [200, 200, 200]
        r = c.get("/")
        assert r.status_code == 429
        assert r.json()["detail"] == "rate limit exceeded"
        assert int(r.headers["retry-after"]) >= 1


# ── login bucket is tighter + independent ─────────────────────────────
def test_login_bucket_is_tighter_than_global():
    async def _login(_request):
        return JSONResponse({"ok": True})

    a = Starlette(routes=[Route("/api/v1/auth/token", _login, methods=["POST"])])
    a.add_middleware(RateLimitMiddleware, requests=1000, per_seconds=60.0,
                     login_requests=3, login_window=60.0)
    with TestClient(a) as c:
        assert [c.post("/api/v1/auth/token").status_code for _ in range(3)] == [200] * 3
        assert c.post("/api/v1/auth/token").status_code == 429


# ── failed-login lockout ──────────────────────────────────────────────
def _login_app(fail_on_header: bool = True) -> Starlette:
    async def _login(request):
        fail = fail_on_header and request.headers.get("x-fail")
        return JSONResponse({"ok": not fail}, status_code=401 if fail else 200)

    a = Starlette(routes=[Route("/api/v1/auth/token", _login, methods=["POST"])])
    a.add_middleware(RateLimitMiddleware, requests=1000, per_seconds=60.0,
                     login_requests=1000, login_window=60.0,
                     lockout_failures=3, lockout_seconds=60.0)
    return a


def test_lockout_after_consecutive_failures():
    with TestClient(_login_app()) as c:
        for _ in range(3):
            assert c.post("/api/v1/auth/token", headers={"x-fail": "1"}).status_code == 401
        r = c.post("/api/v1/auth/token", headers={"x-fail": "1"})
        assert r.status_code == 429
        assert "failed login" in r.json()["detail"]
        assert int(r.headers["retry-after"]) >= 1


def test_success_clears_failure_counter():
    with TestClient(_login_app()) as c:
        c.post("/api/v1/auth/token", headers={"x-fail": "1"})
        c.post("/api/v1/auth/token", headers={"x-fail": "1"})
        assert c.post("/api/v1/auth/token").status_code == 200  # success resets
        # two more failures must NOT lock (counter was cleared)
        for _ in range(2):
            assert c.post("/api/v1/auth/token", headers={"x-fail": "1"}).status_code == 401


# ── bounded registry ──────────────────────────────────────────────────
def test_bucket_registry_is_bounded():
    reg = _LRUBuckets(capacity=1_000, refill_rate=1_000, max_size=5)
    for i in range(500):
        reg.acquire(f"ip-{i}")
    assert len(reg) <= 5, f"registry grew to {len(reg)} (unbounded)"


# ── X-Forwarded-For trust ─────────────────────────────────────────────
def test_xff_ignored_without_trust_proxy():
    a = _app(requests=2, per_seconds=60.0, login_requests=1000, trust_proxy=False)
    with TestClient(a) as c:
        assert c.get("/", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert c.get("/", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
        # a fresh spoofed IP cannot dodge the socket-peer bucket
        assert c.get("/", headers={"X-Forwarded-For": "3.3.3.3"}).status_code == 429


def test_xff_honoured_with_trust_proxy():
    a = _app(requests=2, per_seconds=60.0, login_requests=1000, trust_proxy=True)
    with TestClient(a) as c:
        assert c.get("/", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
        assert c.get("/", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
        assert c.get("/", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 429
        # a different forwarded IP has its own bucket
        assert c.get("/", headers={"X-Forwarded-For": "10.0.0.2"}).status_code == 200


# ── WS connection cap (class + app wiring) ────────────────────────────
class _FakeWS:
    def __init__(self, host: str = "1.2.3.4", headers=None):
        self.client = type("C", (), {"host": host})()
        self.headers = headers or {}


def test_ws_limiter_class_caps_per_ip():
    lim = wl.WSConnectionLimiter(max_per_ip=2)
    a, b, c_ = _FakeWS(), _FakeWS(), _FakeWS()
    assert lim.acquire(a) is True
    assert lim.acquire(b) is True
    assert lim.acquire(c_) is False, "3rd concurrent connection was not capped"
    lim.release(a)
    assert lim.acquire(c_) is True


def _token(client: TestClient) -> str:
    r = client.post("/api/v1/auth/token", json={"username": "admin", "password": "admin"})
    return r.json()["access_token"]


def test_app_ws_cap_enforced_end_to_end():
    old = wl.ws_limiter._max
    wl.ws_limiter._max = 1
    try:
        with TestClient(app) as client:
            tok = _token(client)
            with client.websocket_connect(
                "/ws/signals", subprotocols=["gex.jwt", tok]
            ) as ws1:
                assert ws1.receive_json()["type"] == "hello"
                rejected = False
                try:
                    with client.websocket_connect(
                        "/ws/signals", subprotocols=["gex.jwt", tok]
                    ) as ws2:
                        ws2.receive_json()
                except WebSocketDisconnect:
                    rejected = True
                assert rejected, "2nd concurrent WS was not capped (cap=1)"
    finally:
        wl.ws_limiter._max = old
