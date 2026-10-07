"""Round-2 backend-correctness tests (ENG-04/05/07 + management actions).

* the ``/API_KEY/{key}`` page HTML-escapes attacker-controlled ticker/strategy
  values (stored-XSS sink) and renders an HTML error page (not raw JSON) for an
  unknown/revoked key;
* ``POST /orders`` honours an ``Idempotency-Key`` (a retry never double-places);
* ``POST /signals/engine/start`` rejects ``poll_seconds<=0`` at the edge;
* the previously-missing management actions exist: order delete/purge, backtest
  result delete/purge, direct ``key_signals`` delete, and a summary-cache purge.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import (
    BacktestResultRow,
    KeySignalRow,
    OrderRow,
    SignalKeyRow,
)
from trading.api.routers import portfolio as portfolio_router
from trading.domain import Order, OrderStatus
from trading.main import app


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        for model in (SignalKeyRow, KeySignalRow, BacktestResultRow, OrderRow):
            await s.execute(sa_delete(model))
        await s.commit()
        yield s


async def _get_row(session, key: str) -> SignalKeyRow:
    return (await session.execute(
        select(SignalKeyRow).where(SignalKeyRow.key == key)
    )).scalar_one()


async def _make_key(session, *, config: dict, revoked: bool = False) -> str:
    key = uuid.uuid4().hex
    session.add(SignalKeyRow(
        key=key, exchange="bingx", label="x",
        config_json=json.dumps(config), active=not revoked,
    ))
    await session.commit()
    if revoked:
        row = await _get_row(session, key)
        row.revoked_at = datetime.now(UTC)
        await session.commit()
    return key


# ── ENG-04: dashboard escaping + HTML error page ──────────────────────
async def test_dashboard_escapes_tickers_html(session):
    key = await _make_key(session, config={
        "tickers": [{"symbol": "<img src=x onerror=alert(1)>"}],
        "strategy": "<script>alert(2)</script>",
    })
    with TestClient(app) as client:
        r = client.get(f"/API_KEY/{key}")
    assert r.status_code == 200
    body = r.text
    assert "<img src=x" not in body            # the raw payload never reaches the DOM
    assert "<script>alert(2)</script>" not in body
    assert "&lt;img" in body                    # escaped instead


async def test_dashboard_unknown_key_is_html_not_json(session):
    with TestClient(app) as client:
        r = client.get("/API_KEY/deadbeefdeadbeef")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text.lower() and "unknown api key" in r.text.lower()


async def test_dashboard_revoked_key_is_html(session):
    key = await _make_key(session, config={"tickers": []}, revoked=True)
    with TestClient(app) as client:
        r = client.get(f"/API_KEY/{key}")
    assert r.status_code == 410
    assert r.headers["content-type"].startswith("text/html")
    assert "revoked" in r.text.lower()


# ── ENG-05: order idempotency ─────────────────────────────────────────
async def test_place_order_is_idempotent_with_header(session, monkeypatch):
    calls = {"n": 0}
    order_id = f"idem-{uuid.uuid4().hex}"

    class _FakeBroker:
        async def place_order(self, intent):
            calls["n"] += 1
            return Order(
                id=order_id, symbol=intent.symbol, side=intent.side,
                quantity=float(intent.quantity.value), order_type=intent.order_type,
                status=OrderStatus.OPEN,
            )

    async def _fake_broker(_exchange, _session, _svc):
        return _FakeBroker()

    monkeypatch.setattr(portfolio_router, "_broker", _fake_broker)
    body = {"exchange": "bingx", "symbol": "BTC-USDT", "side": "buy", "quantity": 0.1}
    with TestClient(app) as client:
        r1 = client.post("/api/v1/orders", json=body, headers={"Idempotency-Key": "k-1"})
        r2 = client.post("/api/v1/orders", json=body, headers={"Idempotency-Key": "k-1"})
    assert r1.status_code == 202 and r2.status_code == 202
    assert r1.json()["id"] == r2.json()["id"] == order_id
    assert calls["n"] == 1  # the retry never placed a second order


# ── ENG-07: engine start validates poll_seconds ───────────────────────
async def test_engine_start_rejects_nonpositive_poll_seconds(session):
    with TestClient(app) as client:
        r = client.post("/api/v1/signals/engine/start", json={
            "symbols": ["ETHUSDT"], "timeframe": "4h",
            "poll_seconds": 0, "bars": 400,
        })
    assert r.status_code == 422


# ── management actions ────────────────────────────────────────────────
async def test_order_delete_and_purge(session):
    oid = f"del-{uuid.uuid4().hex}"
    session.add(OrderRow(id=oid, exchange="bingx", symbol="BTC-USDT", side="buy",
                         quantity=1.0, order_type="market", status="open",
                         filled_quantity=0.0))
    await session.commit()
    with TestClient(app) as client:
        assert client.delete(f"/api/v1/orders/{oid}").status_code == 204
        assert client.delete(f"/api/v1/orders/{oid}").status_code == 404
        r = client.delete("/api/v1/orders")
    assert r.status_code == 200 and "deleted" in r.json()


async def test_backtest_result_delete_and_purge(session):
    row = BacktestResultRow(strategy="sma_crossover", symbol="SYNTH",
                            metrics_json="{}", trades_json="[]")
    session.add(row)
    await session.commit()
    await session.refresh(row)
    with TestClient(app) as client:
        listed = client.get("/api/v1/backtest/results?limit=5")
        assert listed.status_code == 200
        assert any(x["id"] == row.id and x["symbol"] == "SYNTH" for x in listed.json())
        assert client.delete(f"/api/v1/backtest/results/{row.id}").status_code == 204
        assert client.delete(f"/api/v1/backtest/results/{row.id}").status_code == 404
        r = client.delete("/api/v1/backtest/results")
    assert r.status_code == 200 and "deleted" in r.json()


async def test_signal_key_signals_delete_and_cache_purge(session):
    key = uuid.uuid4().hex
    session.add(SignalKeyRow(key=key, exchange="bingx", label="x",
                             config_json="{}", active=True))
    await session.commit()
    row = await _get_row(session, key)
    session.add(KeySignalRow(key_id=row.id, symbol="BTC", side="buy",
                             state="long_entry", strategy="s", price=1.0,
                             timestamp=datetime.now(UTC)))
    await session.commit()
    with TestClient(app) as client:
        r = client.delete(f"/api/v1/signal-keys/{row.id}/signals")
        assert r.status_code == 200 and r.json()["deleted"] == 1
        assert client.delete("/api/v1/signal-keys/999999/signals").status_code == 404
        purge = client.post("/api/v1/signal-keys/cache/purge")
    assert purge.status_code == 200 and "purged" in purge.json()


async def test_credentials_validate_missing_key_404():
    with TestClient(app) as client:
        assert client.post("/api/v1/keys/999999/validate").status_code == 404


async def test_order_cancel_missing_order_404():
    with TestClient(app) as client:
        assert client.post("/api/v1/orders/does-not-exist/cancel").status_code == 404


async def test_bulk_delete_orders(session):
    ids = [f"bd-{uuid.uuid4().hex}" for _ in range(2)]
    for oid in ids:
        session.add(OrderRow(id=oid, exchange="bingx", symbol="X", side="buy",
                             quantity=1.0, order_type="market", status="open",
                             filled_quantity=0.0))
    await session.commit()
    with TestClient(app) as client:
        r = client.post("/api/v1/orders/bulk-delete", json={"ids": [*ids, "missing-1"]})
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == 2 and body["missing"] == ["missing-1"]


async def test_bulk_delete_results(session):
    rows = [BacktestResultRow(strategy="s", symbol="SYNTH", metrics_json="{}", trades_json="[]")
            for _ in range(2)]
    for row in rows:
        session.add(row)
    await session.commit()
    for row in rows:
        await session.refresh(row)
    with TestClient(app) as client:
        r = client.post("/api/v1/backtest/results/bulk-delete",
                        json={"ids": [row.id for row in rows] + [99_999_999]})
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == 2 and 99_999_999 in body["missing"]


async def test_signal_keys_bulk_disable(session):
    keys = []
    for _ in range(2):
        k = uuid.uuid4().hex
        session.add(SignalKeyRow(key=k, exchange="bingx", label="x",
                                 config_json="{}", active=True))
        keys.append(k)
    await session.commit()
    rows = [await _get_row(session, k) for k in keys]
    with TestClient(app) as client:
        r = client.post("/api/v1/signal-keys/bulk",
                        json={"ids": [rows[0].id, rows[1].id, 99_999_999], "action": "disable"})
    assert r.status_code == 200
    body = r.json()
    assert body["updated"] == 2 and 99_999_999 in body["missing"]


def test_signal_derived_tables_have_non_destructive_fks():
    """R5-7: derived signal rows carry ON DELETE SET NULL FKs (non-destructive)."""
    from trading.adapters.persistence.models import (
        KeySignalRow,
        KeyTradeRow,
        SignalPositionRow,
    )

    for table, col in ((KeySignalRow.__table__, "key_id"),
                       (SignalPositionRow.__table__, "key_id"),
                       (KeyTradeRow.__table__, "key_id")):
        fks = list(table.c[col].foreign_keys)
        assert fks, f"{table.name}.{col} is missing the FK constraint"
        assert fks[0].column.table.name == "signal_keys"
        assert fks[0].ondelete == "SET NULL"
    # SET NULL is only valid if the column is nullable.
    assert KeyTradeRow.__table__.c.key_id.nullable is True
