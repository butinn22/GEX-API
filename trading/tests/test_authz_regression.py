"""Authorization-boundary regression tests (audit additions).

These guard behavioural gaps the audit found *uncovered*:

* a tampered/garbage bearer token must be rejected (only "no token" was tested);
* the rate-limit middleware must advertise ``Retry-After`` on a 429;
* authentication must be enforced *before* body validation (an anonymous
  malformed request is 401, never 422 — otherwise an attacker learns the
  validator's shape anonymously);
* every JWT-guarded router must reject anonymous access (a systematic sweep so
  a future route added without ``Depends(require_auth)`` fails loudly);
* deleting / patching a non-existent resource is a clean 404.

All tests are deterministic and offline (ASGITransport; no network).
"""
from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import ApiKeyRow
from trading.api.middleware import RateLimitMiddleware
from trading.main import app

# Every test here exercises the *real* bearer guard (never the conftest
# auto-auth override): the whole point is to prove anonymous access is rejected.
pytestmark = pytest.mark.real_auth


# ── fixtures ──────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def client():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(ApiKeyRow))
        await s.commit()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _auth(client) -> dict[str, str]:
    r = await client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
    )
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── 1. token validation ───────────────────────────────────────────────
@pytest.mark.parametrize(
    "bad_token",
    [
        "garbage",
        "not.a.jwt",
        # HS256 token signed with a *different* secret must not verify
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJhZG1pbiIsImlhdCI6MTAwMDAwMDAwMCwiZXhwIjo5OTk5OTk5OTk5fQ."
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ],
)
async def test_tampered_token_is_rejected(client, bad_token):
    r = await client.get("/api/v1/keys", headers={"Authorization": f"Bearer {bad_token}"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid or expired token"


async def test_missing_token_is_rejected(client):
    r = await client.get("/api/v1/keys")
    assert r.status_code == 401
    assert r.json()["detail"] == "Not authenticated"


# ── 2. auth precedes body validation ──────────────────────────────────
async def test_auth_enforced_before_body_validation(client):
    # Malformed body + no token → 401, never 422 (validator shape must not leak)
    r = await client.post("/api/v1/keys", json={})
    assert r.status_code == 401

    r = await client.post(
        "/api/v1/signal-keys", json={"exchange": "nope", "tickers": []}
    )
    assert r.status_code == 401


async def test_authenticated_malformed_body_is_422(client):
    headers = await _auth(client)
    r = await client.post(
        "/api/v1/keys",
        json={"exchange": "coinbase", "api_key": "x"},  # not a valid Literal
        headers=headers,
    )
    assert r.status_code == 422


# ── 3. systematic anonymous-access sweep of guarded routers ───────────
_GUARDED_ANON_REQUESTS = [
    ("GET", "/api/v1/keys"),
    ("POST", "/api/v1/keys"),
    ("DELETE", "/api/v1/keys/999999"),
    ("PATCH", "/api/v1/keys/999999/settings"),
    ("GET", "/api/v1/signal-keys"),
    ("POST", "/api/v1/signal-keys"),
    ("DELETE", "/api/v1/signal-keys/999999"),
    ("PATCH", "/api/v1/signal-keys/999999"),
    ("POST", "/api/v1/baskets/export"),
    ("POST", "/api/v1/baskets/deploy"),
    ("GET", "/api/v1/signals"),
    ("GET", "/api/v1/signals/positions"),
    ("GET", "/api/v1/signals/stats"),
    ("DELETE", "/api/v1/signals/positions"),
    ("POST", "/api/v1/signals/engine/start"),
    ("POST", "/api/v1/signals/engine/stop"),
    ("GET", "/api/v1/signals/export/signals.csv"),
    ("GET", "/api/v1/export/live-trades.csv"),
    ("POST", "/api/v1/orders"),
    # Round 2 — the five routers that were previously public.
    ("POST", "/api/v1/presets"),
    ("GET", "/api/v1/presets"),
    ("POST", "/api/v1/presets/1/promote"),
    ("DELETE", "/api/v1/presets/999999"),
    ("POST", "/api/v1/backtest"),
    ("POST", "/api/v1/backtest/optimize/global"),
    ("GET", "/api/v1/backtest/cancel"),
    ("GET", "/api/v1/strategies"),
    ("POST", "/api/v1/strategies/sma_crossover/start"),
    ("POST", "/api/v1/strategies/sma_crossover/stop"),
    ("GET", "/api/v1/portfolio"),
    ("GET", "/api/v1/positions"),
    ("GET", "/api/v1/orders"),
    ("GET", "/api/v1/data/ohlcv/BTC"),
    ("GET", "/api/v1/data/instruments"),
    ("GET", "/api/v1/data/sources"),
    # Round 2 — management actions added for the audit's create→purge gaps.
    ("DELETE", "/api/v1/orders/xyz"),
    ("DELETE", "/api/v1/orders"),
    ("DELETE", "/api/v1/backtest/results"),
    ("DELETE", "/api/v1/backtest/results/1"),
    ("DELETE", "/api/v1/signal-keys/999999/signals"),
    ("POST", "/api/v1/signal-keys/cache/purge"),
]


@pytest.mark.parametrize("method,path", _GUARDED_ANON_REQUESTS,
                         ids=[f"{m} {p}" for m, p in _GUARDED_ANON_REQUESTS])
async def test_guarded_routes_reject_anonymous(client, method, path):
    r = await client.request(method, path, json={})
    assert r.status_code in (401, 403), (
        f"{method} {path} is reachable without a token (got {r.status_code}); "
        "it must depend on require_auth"
    )


# ── 3b. the two intentional public exceptions stay public ─────────────
async def test_login_endpoint_is_public(client):
    r = await client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
    )
    assert r.status_code == 200  # never 401 for a valid login


async def test_backtest_cancel_is_public_capability(client):
    # sendBeacon cannot set an Authorization header → this route must not 401.
    r = await client.post("/api/v1/backtest/cancel/some-run-token")
    assert r.status_code != 401


# ── 4. clean 404s for non-existent resources ──────────────────────────
async def test_delete_missing_api_key_404(client):
    headers = await _auth(client)
    r = await client.delete("/api/v1/keys/999999", headers=headers)
    assert r.status_code == 404


async def test_delete_missing_signal_key_404(client):
    headers = await _auth(client)
    r = await client.delete("/api/v1/signal-keys/999999", headers=headers)
    assert r.status_code == 404


async def test_patch_missing_signal_key_404(client):
    headers = await _auth(client)
    r = await client.patch("/api/v1/signal-keys/999999", headers=headers)
    assert r.status_code == 404


# ── 5. rate-limit 429 advertises Retry-After ──────────────────────────
async def _hello(_request):
    return JSONResponse({"ok": True})


def test_rate_limit_429_has_retry_after_header():
    app_ = Starlette(routes=[Route("/", _hello)])
    app_.add_middleware(RateLimitMiddleware, requests=2, per_seconds=60.0)
    with TestClient(app_) as c:
        assert [c.get("/").status_code for _ in range(2)] == [200, 200]
        r = c.get("/")
        assert r.status_code == 429
        assert r.json()["detail"] == "rate limit exceeded"
        retry_after = r.headers.get("retry-after")
        assert retry_after is not None
        assert int(retry_after) >= 1  # clients need a concrete back-off
