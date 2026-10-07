"""Tests for API middleware (rate limit), CORS, and the Monte-Carlo endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from trading.api.middleware import RateLimitMiddleware
from trading.domain import TradingError
from trading.main import app


async def _hello(request):
    return JSONResponse({"ok": True})


def test_rate_limit_middleware():
    s = Starlette(routes=[Route("/", _hello)])
    s.add_middleware(RateLimitMiddleware, requests=3, per_seconds=60.0)
    client = TestClient(s)
    assert [client.get("/").status_code for _ in range(3)] == [200, 200, 200]
    assert client.get("/").status_code == 429


def test_cors_header_present():
    with TestClient(app) as client:
        r = client.get("/health", headers={"Origin": "http://example.com"})
        assert r.headers.get("access-control-allow-origin") == "*"


def test_trading_error_handler_registered():
    assert TradingError in app.exception_handlers


def test_monte_carlo_endpoint():
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        r = client.post("/api/v1/backtest/monte-carlo",
                        json={"symbol": "SYNTH", "n_paths": 200, "n_steps": 50},
                        headers=headers)
        assert r.status_code == 200
        d = r.json()
        assert d["n_paths"] == 200
        assert d["p5"] <= d["mean"] <= d["p95"]
