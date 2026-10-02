"""End-to-end API tests (httpx ASGI transport against the real app)."""
from __future__ import annotations

import httpx
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import ApiKeyRow
from trading.main import app


@pytest_asyncio.fixture
async def client():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(ApiKeyRow))  # clean slate
        await s.commit()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_login_and_key_crud(client):
    # wrong password → 401
    r = await client.post("/api/v1/auth/token", json={"username": "admin", "password": "nope"})
    assert r.status_code == 401
    # correct login
    r = await client.post("/api/v1/auth/token", json={"username": "admin", "password": "admin"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # unauthorized without token
    assert (await client.get("/api/v1/keys")).status_code == 401

    # add
    r = await client.post("/api/v1/keys", json={
        "exchange": "bingx", "label": "main", "api_key": "PUBKEY123456", "api_secret": "SEC",
    }, headers=headers)
    assert r.status_code == 201
    out = r.json()
    assert out["api_key_masked"] == "PUBK…3456"
    assert out["exchange"] == "bingx"

    # list
    r = await client.get("/api/v1/keys", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 1

    # delete
    r = await client.delete(f"/api/v1/keys/{out['id']}", headers=headers)
    assert r.status_code == 204
    assert len((await client.get("/api/v1/keys", headers=headers)).json()) == 0


async def test_backtest_synthetic(client):
    r = await client.post("/api/v1/backtest", json={
        "strategy": "sma_crossover", "symbol": "SYNTH", "fast": 5, "slow": 20, "source": "synthetic",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["strategy"] == "sma_crossover"
    assert len(data["equity_curve"]) > 0
    m = data["metrics"]
    for key in ("sharpe", "max_drawdown", "win_rate", "n_trades", "n_periods"):
        assert key in m
    assert data["n_trades"] >= 0


async def test_backtest_with_bars(client):
    bars = [
        {"timestamp": f"2024-01-{i+1:02d}T00:00:00Z", "open": 100 + i,
         "high": 103 + i, "low": 99 + i, "close": 101 + i, "volume": 1000}
        for i in range(30)
    ]
    r = await client.post("/api/v1/backtest", json={
        "strategy": "buy_and_hold", "symbol": "X", "bars": bars,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["equity_curve"]) == 30
    assert data["metrics"]["total_return"] > 0  # rising prices


async def test_strategies_and_portfolio(client):
    r = await client.get("/api/v1/strategies")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert "sma_crossover" in names and "buy_and_hold" in names

    r = await client.get("/api/v1/portfolio")
    assert r.status_code == 200
    assert r.json()["configured"] is False  # no live keys


async def test_orders_require_credentials(client):
    r = await client.post("/api/v1/auth/token", json={"username": "admin", "password": "admin"})
    token = r.json()["access_token"]
    # valid auth but no broker creds → 400
    r = await client.post("/api/v1/orders", json={
        "exchange": "bingx", "symbol": "BTC-USDT", "side": "buy", "quantity": 0.1,
    }, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400
    # no auth → 401
    r = await client.post("/api/v1/orders", json={
        "exchange": "bingx", "symbol": "BTC-USDT", "side": "buy", "quantity": 0.1,
    })
    assert r.status_code == 401


async def test_data_sources_and_synthetic(client):
    r = await client.get("/api/v1/data/sources")
    assert "moex" in r.json() and "synthetic" in r.json()
    r = await client.get("/api/v1/data/ohlcv/SYNTH", params={"source": "synthetic", "limit": 5})
    assert r.status_code == 200
    assert len(r.json()) == 5
