"""QA Round-3 — INDEPENDENT order-idempotency (C10) + engine validation (C11).

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_backend.py -q
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import OrderRow
from trading.api.routers import portfolio as portfolio_router
from trading.application.signal_engine import signal_engine
from trading.domain import Order, OrderStatus
from trading.main import app


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(sa_delete(OrderRow))
        await s.commit()
        yield s


@pytest.fixture(autouse=True)
def _clear_idem_cache():
    portfolio_router._ORDER_IDEMPOTENCY.clear()
    yield
    portfolio_router._ORDER_IDEMPOTENCY.clear()


async def _order_rows(session) -> int:
    return (await session.execute(select(func.count()).select_from(OrderRow))).scalar_one()


def _fake_broker(calls: dict):
    class _FakeBroker:
        async def place_order(self, intent):
            calls["n"] += 1
            return Order(
                id=f"idem-{uuid.uuid4().hex}",
                symbol=intent.symbol,
                side=intent.side,
                quantity=float(intent.quantity.value),
                order_type=intent.order_type,
                status=OrderStatus.OPEN,
            )

    async def _broker(_exchange, _session, _svc):
        return _FakeBroker()

    return _broker


BODY = {"exchange": "bingx", "symbol": "BTC-USDT", "side": "buy", "quantity": 0.1}


# ── C10: idempotency ──────────────────────────────────────────────────
async def test_same_idempotency_key_places_exactly_one(session, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(portfolio_router, "_broker", _fake_broker(calls))
    with TestClient(app) as client:
        r1 = client.post("/api/v1/orders", json=BODY, headers={"Idempotency-Key": "qa-k1"})
        r2 = client.post("/api/v1/orders", json=BODY, headers={"Idempotency-Key": "qa-k1"})
        r3 = client.post("/api/v1/orders", json=BODY, headers={"Idempotency-Key": "qa-k1"})
    assert r1.status_code == r2.status_code == r3.status_code == 202
    assert r1.json()["id"] == r2.json()["id"] == r3.json()["id"]
    assert calls["n"] == 1, "retry re-placed the order at the broker"
    assert await _order_rows(session) == 1, "retry persisted a duplicate order row"


async def test_distinct_idempotency_keys_place_two(session, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(portfolio_router, "_broker", _fake_broker(calls))
    with TestClient(app) as client:
        a = client.post("/api/v1/orders", json=BODY, headers={"Idempotency-Key": "qa-a"})
        b = client.post("/api/v1/orders", json=BODY, headers={"Idempotency-Key": "qa-b"})
    assert a.status_code == b.status_code == 202
    assert a.json()["id"] != b.json()["id"]
    assert calls["n"] == 2
    assert await _order_rows(session) == 2


async def test_no_key_preserves_old_behaviour(session, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(portfolio_router, "_broker", _fake_broker(calls))
    with TestClient(app) as client:
        client.post("/api/v1/orders", json=BODY)
        client.post("/api/v1/orders", json=BODY)
    assert calls["n"] == 2, "without a key each request must place an order"


# ── C11: engine start validation ──────────────────────────────────────
@pytest.mark.parametrize("poll", [0, -1, -3.7, "0"])
async def test_engine_start_rejects_nonpositive_poll_seconds(session, poll):
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/signals/engine/start",
            json={"symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": poll, "bars": 400},
        )
    assert r.status_code == 422, f"poll_seconds={poll!r} was accepted (busy-loop risk)"


async def test_engine_start_rejects_zero_bars(session):
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/signals/engine/start",
            json={"symbols": ["ETHUSDT"], "poll_seconds": 30, "bars": 0},
        )
    assert r.status_code == 422


async def test_engine_start_accepts_positive_poll_seconds(session, monkeypatch):
    seen = {}

    async def _fake_start(config):
        seen["poll_seconds"] = config.poll_seconds
        return {"running": True}

    monkeypatch.setattr(signal_engine, "start", _fake_start)
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/signals/engine/start",
            json={"symbols": ["ETHUSDT"], "poll_seconds": 30, "bars": 400},
        )
    assert r.status_code == 200
    assert seen["poll_seconds"] == 30.0


def test_openapi_documents_poll_seconds_bound():
    schema = app.openapi()
    props = schema["components"]["schemas"]["SignalEngineStartRequest"]["properties"]
    assert props["poll_seconds"]["exclusiveMinimum"] == 0
