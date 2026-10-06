"""Tests for signal API keys: lifecycle, generation, dashboard, exports."""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import (
    KeySignalRow,
    KeyTradeRow,
    SignalKeyRow,
    StrategyPresetRow,
)
from trading.application.signal_keys import SignalKeyError, SignalKeyService
from trading.domain import Bar
from trading.main import app


def _bars(n: int = 400, *, seed: float = 100.0, drift: float = 0.0015) -> list[Bar]:
    bars: list[Bar] = []
    price = seed
    t0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ret = drift + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        bars.append(Bar(timestamp=t0 + timedelta(days=i), open=prev,
                        high=max(prev, price) * 1.004, low=min(prev, price) * 0.996,
                        close=price, volume=1000.0))
    return bars


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        for table in (KeySignalRow, KeyTradeRow, SignalKeyRow, StrategyPresetRow):
            await s.execute(delete(table))
        await s.commit()
        yield s


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _auth(client):
    r = asyncio.get_event_loop()
    # login synchronously via a nested loop is awkward — use the client in async tests


# ── service ────────────────────────────────────────────────────────────


async def test_create_requires_valid_exchange(session):
    svc = SignalKeyService(session)
    with pytest.raises(SignalKeyError):
        await svc.create(exchange="coinbase", tickers=["BTC"])
    with pytest.raises(SignalKeyError):
        await svc.create(exchange="bingx", tickers=[])


async def test_created_key_is_unique_and_references_config(session):
    svc = SignalKeyService(session)
    row1, _ = await svc.create(exchange="bingx", tickers=["BTC", "ETH"],
                              params_by_ticker={"BTC": {"zone_atr": 0.4}})
    row2, _ = await svc.create(exchange="tbank", tickers=["SBER"])
    assert row1.key != row2.key
    assert row1.key.startswith("sk_") and len(row1.key) >= 40

    config = json.loads(row1.config_json)
    assert config["strategy"] == "trend_confluence_unified"
    assert config["tickers"][0] == {"symbol": "BTC", "params": {"zone_atr": 0.4}, "preset_id": None}
    assert config["tickers"][1]["symbol"] == "ETH"
    assert "exchange" not in config  # exchange lives on the row, not in config


async def test_broker_support_warning_for_unroutable_ticker(session):
    svc = SignalKeyService(session)
    # NVDA is a US equity: no order route on bingx → warning, still accepted.
    _row, warnings = await svc.create(exchange="bingx", tickers=["NVDA"])
    assert warnings and "NVDA" in warnings[0]


async def test_generate_produces_signals_and_trades(session):
    svc = SignalKeyService(session)
    row, _ = await svc.create(exchange="bingx", tickers=["AAA", "BBB"])
    bars = {"AAA": _bars(400), "BBB": _bars(400, seed=50.0, drift=0.002)}
    report = await svc.generate(key=row, refresh=False, bars_by_symbol=bars)

    assert report["n_signals"] > 0
    assert report["n_trades"] > 0
    assert "sharpe" in report["metrics"] and "total_return" in report["metrics"]
    summary = svc.summary(row.id)
    assert summary and summary.equity and summary.metrics["final_equity"] > 0

    signals = await svc.signals(row.id)
    trades = await svc.trades(row.id)
    assert len(signals) == report["n_signals"]
    assert len(trades) == report["n_trades"]
    for s in signals:
        assert s.strategy == "trend_confluence_unified"
        assert s.source == "live" and s.symbol in {"AAA", "BBB"}
    for t in trades:
        assert t.direction in ("long", "short")
        assert t.exit_time >= t.entry_time
        assert t.source == "replay"

    # idempotent regeneration: rows are replaced, not duplicated
    report2 = await svc.generate(key=row, refresh=False, bars_by_symbol=bars)
    assert len(await svc.trades(row.id)) == report2["n_trades"] == len(trades)
    assert row.last_used_at is not None


async def test_generate_resolves_preset_params(session):
    from trading.application.presets import PresetService

    presets = PresetService(session)
    saved = await presets.save(symbol="CCC", params={"emf_mode": "require"})

    svc = SignalKeyService(session)
    row, _ = await svc.create(exchange="tbank", tickers=["ccc"])
    config = json.loads(row.config_json)
    tick = next(t for t in config["tickers"] if t["symbol"] == "CCC")
    # Strategy Hub follow-live: the key stores a reference, not a params copy —
    # generation resolves the ticker's active/live version from the store.
    assert tick["preset_id"] is None
    assert tick["params"] is None

    await svc.generate(key=row, refresh=False,
                      bars_by_symbol={"CCC": _bars(400)})
    trades = await svc.trades(row.id)
    assert all(t.preset_id == saved.id for t in trades)


async def test_revoked_key_cannot_generate(session):
    svc = SignalKeyService(session)
    row, _ = await svc.create(exchange="bingx", tickers=["BTC"])
    await svc.revoke(row.id)
    with pytest.raises(SignalKeyError):
        await svc.generate(key=row)
    with pytest.raises(SignalKeyError):
        await svc.set_active(row.id, True)  # a revoked key stays dead
    again = await svc.get(row.id)
    assert again.revoked_at is not None


# ── API + dashboard ────────────────────────────────────────────────────


async def _token(client) -> dict:
    r = await client.post("/api/v1/auth/token",
                          json={"username": "admin", "password": "admin"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_signal_key_api_workflow(client, session):
    headers = await _token(client)

    # unauthorized without a token
    assert (await client.get("/api/v1/signal-keys")).status_code == 401

    r = await client.post("/api/v1/signal-keys", json={
        "exchange": "bingx", "label": "test", "tickers": ["AAA", "BBB"],
    }, headers=headers)
    assert r.status_code == 201
    out = r.json()
    key = out["key"]
    assert key.startswith("sk_")
    assert out["config"]["strategy"] == "trend_confluence_unified"

    # generation via the API (synthetic data source)
    r = await client.post(f"/api/v1/signal-keys/{key}/generate", headers=headers)
    assert r.status_code == 200
    report = r.json()
    assert report["n_signals"] >= 0 and "metrics" in report

    # dashboard page
    r = await client.get(f"/API_KEY/{key}")
    assert r.status_code == 200
    assert key in r.text and "Live strategy dashboard" in r.text

    # dashboard data
    r = await client.get(f"/API_KEY/{key}/data")
    assert r.status_code == 200
    data = r.json()
    assert data["key"] == key
    assert "totals" in data and "metrics" in data and "last_updated" in data

    # charts (server-side SVG, same functions as the backtest report)
    r = await client.get(f"/API_KEY/{key}/charts")
    assert r.status_code == 200
    charts = r.json()
    assert "<svg" in charts.get("equity", "") or charts["equity"] == ""

    # exports
    r = await client.get(f"/API_KEY/{key}/trades.csv")
    assert r.status_code == 200
    assert r.text.splitlines()[0].startswith("trade_id,api_key,broker,ticker")
    r = await client.get(f"/API_KEY/{key}/trades.xlsx")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"  # xlsx is a zip

    # unknown / revoked keys
    assert (await client.get("/API_KEY/sk_missing")).status_code == 404
    r = await client.delete(f"/api/v1/signal-keys/{out['id']}", headers=headers)
    assert r.status_code == 204
    r = await client.get(f"/API_KEY/{key}")
    assert r.status_code == 410
    r = await client.post(f"/api/v1/signal-keys/{key}/generate", headers=headers)
    assert r.status_code in (400, 404)


async def test_dashboard_refresh_endpoint(client, session):
    headers = await _token(client)
    r = await client.post("/api/v1/signal-keys", json={
        "exchange": "tbank", "tickers": ["SYNTH"], "source": "synthetic",
        "limit": 300,
    }, headers=headers)
    assert r.status_code == 201
    key = r.json()["key"]

    r = await client.post(f"/API_KEY/{key}/refresh")
    assert r.status_code == 200
    report = r.json()
    assert report["key"] == key
