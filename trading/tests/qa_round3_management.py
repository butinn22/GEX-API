"""QA Round-3 — INDEPENDENT management-action verification (C14, WS-7).

Every create→purge gap the audit flagged must exist AND behave:

* ``DELETE /backtest/results`` + ``/results/{id}`` (stored backtest results)
* ``DELETE /orders`` + ``/orders/{id}`` (order history)
* ``DELETE /signal-keys/{id}/signals`` (direct ``key_signals`` purge)
* ``POST /signal-keys/cache/purge`` (in-memory ``_summary_cache`` purge)

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_management.py -q
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete as sa_delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import (
    BacktestResultRow,
    KeySignalRow,
    OrderRow,
    SignalKeyRow,
)
from trading.application import signal_keys as sk_mod
from trading.main import app


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        for model in (SignalKeyRow, KeySignalRow, BacktestResultRow, OrderRow):
            await s.execute(sa_delete(model))
        await s.commit()
        yield s


# ── orders ────────────────────────────────────────────────────────────
async def test_order_delete_and_purge_exist_and_work(session):
    oid = f"qa-{uuid.uuid4().hex}"
    session.add(OrderRow(id=oid, exchange="bingx", symbol="BTC-USDT", side="buy",
                         quantity=1.0, order_type="market", status="open",
                         filled_quantity=0.0))
    await session.commit()
    with TestClient(app) as c:
        assert c.delete(f"/api/v1/orders/{oid}").status_code == 204
        assert c.delete(f"/api/v1/orders/{oid}").status_code == 404
        r = c.delete("/api/v1/orders")
    assert r.status_code == 200 and "deleted" in r.json()


# ── backtest results ──────────────────────────────────────────────────
async def test_backtest_result_delete_and_purge_exist_and_work(session):
    row = BacktestResultRow(strategy="sma_crossover", symbol="SYNTH",
                            metrics_json="{}", trades_json="[]")
    session.add(row)
    await session.commit()
    await session.refresh(row)
    rid = row.id
    with TestClient(app) as c:
        assert c.delete(f"/api/v1/backtest/results/{rid}").status_code == 204
        assert c.delete(f"/api/v1/backtest/results/{rid}").status_code == 404
        r = c.delete("/api/v1/backtest/results")
    assert r.status_code == 200 and "deleted" in r.json()


# ── key_signals direct delete ─────────────────────────────────────────
async def test_signal_key_signals_delete(session):
    key = uuid.uuid4().hex
    session.add(SignalKeyRow(key=key, exchange="bingx", label="x",
                             config_json="{}", active=True))
    await session.commit()
    from sqlalchemy import select

    row = (await session.execute(
        select(SignalKeyRow).where(SignalKeyRow.key == key)
    )).scalar_one()
    session.add(KeySignalRow(key_id=row.id, symbol="BTC", side="buy",
                             state="long_entry", strategy="s", price=1.0,
                             timestamp=datetime.now(UTC)))
    await session.commit()
    with TestClient(app) as c:
        r = c.delete(f"/api/v1/signal-keys/{row.id}/signals")
        assert r.status_code == 200 and r.json()["deleted"] == 1
        assert c.delete("/api/v1/signal-keys/999999/signals").status_code == 404


# ── summary cache purge ───────────────────────────────────────────────
def test_summary_cache_purge_clears_cache():
    sk_mod._summary_cache.clear()
    sk_mod._summary_cache[123] = object()  # type: ignore[assignment]
    sk_mod._summary_cache[456] = object()  # type: ignore[assignment]
    try:
        with TestClient(app) as c:
            r = c.post("/api/v1/signal-keys/cache/purge")
        assert r.status_code == 200
        assert r.json()["purged"] == 2
        assert sk_mod._summary_cache == {}, "cache not actually purged"
    finally:
        sk_mod._summary_cache.clear()
