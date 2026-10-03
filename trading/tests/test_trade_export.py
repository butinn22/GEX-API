"""CSV/XLSX trade export + live order-row classification."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.bulk import TaskResultStore
from trading.adapters.persistence.models import ApiKeyRow, BacktestResultRow, OrderRow
from trading.application.backtest.trade_log import TradeState, event_from_fill
from trading.application.reporting.trade_export import (
    events_from_order_rows,
    events_to_csv,
    events_to_xlsx,
)
from trading.domain import Fill, Position, PositionSide, Side

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_events():
    pos0 = Position("BTC-USDT")
    f1 = Fill(order_id="1", symbol="BTC-USDT", side=Side.BUY, price=100.0, quantity=2.0, timestamp=T0)
    e1 = event_from_fill(pos0, f1, strategy="emf", reason="entry")
    pos1 = pos0.apply_fill(f1)
    f2 = Fill(order_id="2", symbol="BTC-USDT", side=Side.BUY, price=101.0, quantity=1.0, timestamp=T0 + timedelta(days=1))
    e2 = event_from_fill(pos1, f2, strategy="emf", reason="add")
    pos2 = pos1.apply_fill(f2)
    f3 = Fill(order_id="3", symbol="BTC-USDT", side=Side.SELL, price=110.0, quantity=3.0, timestamp=T0 + timedelta(days=2))
    e3 = event_from_fill(pos2, f3, strategy="emf", reason="exit")
    return [e1, e2, e3]


class TestCsv:
    def test_header_and_states(self):
        out = events_to_csv(make_events())
        rows = list(csv.reader(io.StringIO(out)))
        assert rows[0] == ["timestamp", "symbol", "state", "direction", "side",
                           "price", "quantity", "realized_pnl", "pct_return",
                           "strategy", "reason"]
        assert [r[2] for r in rows[1:]] == ["long_entry", "long_add", "long_exit"]
        assert [r[3] for r in rows[1:]] == ["long", "long", "long"]

    def test_pnl_and_pct_on_exit(self):
        out = events_to_csv(make_events())
        rows = list(csv.DictReader(io.StringIO(out)))
        exit_row = rows[2]
        # avg entry = (100*2 + 101*1)/3 = 100.333…; exit 110 × 3 units
        assert float(exit_row["realized_pnl"]) == pytest.approx(29.0)
        assert float(exit_row["pct_return"]) == pytest.approx(110 / (301 / 3) - 1)

    def test_empty_events_still_has_header(self):
        out = events_to_csv([])
        assert out.strip().count(",") == 10  # header only, 11 columns


class TestXlsx:
    def test_valid_zip_with_sheet(self):
        blob = events_to_xlsx(make_events())
        assert blob[:2] == b"PK"
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = set(z.namelist())
            assert "xl/worksheets/sheet1.xml" in names
            sheet = z.read("xl/worksheets/sheet1.xml").decode()
        assert "long_entry" in sheet
        assert "realized_pnl" in sheet
        assert "<v>110.0</v>" in sheet  # numeric exit price cell

    def test_row_count(self):
        blob = events_to_xlsx(make_events())
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            sheet = z.read("xl/worksheets/sheet1.xml").decode()
        assert sheet.count("<row") == 4  # header + 3 events

    def test_escapes_xml(self):
        evs = make_events()
        evil = event_from_fill(
            Position("X"),
            Fill(order_id="9", symbol="X", side=Side.BUY, price=1.0, quantity=1.0),
            strategy="a<b>&\"c\"", reason="r",
        )
        blob = events_to_xlsx([evil])
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            sheet = z.read("xl/worksheets/sheet1.xml").decode()
        assert "a&lt;b&gt;&amp;" in sheet


def order_row(**kw) -> SimpleNamespace:
    defaults = dict(
        id="o1", exchange="bingx", symbol="BTC-USDT", side="buy", quantity=1.0,
        order_type="limit", status="filled", limit_price=100.0, stop_price=None,
        filled_quantity=1.0, strategy="emf", reason="r",
        created_at=T0,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class TestEventsFromOrderRows:
    def test_long_roundtrip(self):
        rows = [
            order_row(id="1", side="buy", limit_price=100.0, created_at=T0),
            order_row(id="2", side="sell", limit_price=110.0,
                      created_at=T0 + timedelta(hours=1)),
        ]
        events = events_from_order_rows(rows)
        assert [e.state for e in events] == [TradeState.LONG_ENTRY, TradeState.LONG_EXIT]
        assert events[1].realized_pnl == pytest.approx(10.0)
        assert events[1].pct_return == pytest.approx(0.10)

    def test_add_and_short_states(self):
        rows = [
            order_row(id="1", side="buy", limit_price=100.0, created_at=T0),
            order_row(id="2", side="buy", limit_price=102.0, created_at=T0 + timedelta(hours=1)),
            order_row(id="3", symbol="ETH-USDT", side="sell", limit_price=50.0,
                      created_at=T0 + timedelta(hours=2)),
            order_row(id="4", symbol="ETH-USDT", side="buy", limit_price=45.0,
                      created_at=T0 + timedelta(hours=3)),
        ]
        events = events_from_order_rows(rows)
        states = [(e.symbol, e.state) for e in events]
        assert states == [
            ("BTC-USDT", TradeState.LONG_ENTRY),
            ("BTC-USDT", TradeState.LONG_ADD),
            ("ETH-USDT", TradeState.SHORT_ENTRY),
            ("ETH-USDT", TradeState.SHORT_EXIT),
        ]
        assert events[3].realized_pnl == pytest.approx(5.0)

    def test_skips_non_filled(self):
        rows = [order_row(status="open", filled_quantity=0.0)]
        assert events_from_order_rows(rows) == []

    def test_unknown_price_zeroes_pnl(self):
        rows = [
            order_row(id="1", side="buy", limit_price=None, stop_price=None,
                      order_type="market", created_at=T0),
            order_row(id="2", side="sell", limit_price=None, stop_price=None,
                      order_type="market", created_at=T0 + timedelta(hours=1)),
        ]
        events = events_from_order_rows(rows)
        assert len(events) == 2
        assert events[1].state is TradeState.LONG_EXIT
        assert events[1].realized_pnl == 0.0
        assert events[1].pct_return == 0.0


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        for model in (BacktestResultRow, OrderRow, ApiKeyRow):
            await s.execute(delete(model))
        await s.commit()
        yield s


class TestPersistence:
    async def test_save_backtest_with_events(self, session):
        store = TaskResultStore(session)
        events = [e.as_dict() for e in make_events()]
        row = await store.save_backtest("emf", "BTC-USDT", {"sharpe": 1.0}, trades=events)
        assert row.id is not None
        loaded = json.loads(row.trades_json)
        assert [e["state"] for e in loaded] == ["long_entry", "long_add", "long_exit"]

    async def test_save_backtest_without_events_defaults_empty(self, session):
        store = TaskResultStore(session)
        row = await store.save_backtest("emf", "X", {"sharpe": 1.0})
        assert json.loads(row.trades_json) == []


class TestExportEndpoints:
    async def test_backtest_export_csv_and_xlsx(self, session):
        import httpx

        from trading.main import app

        store = TaskResultStore(session)
        row = await store.save_backtest(
            "emf", "BTC-USDT", {"sharpe": 1.0},
            trades=[e.as_dict() for e in make_events()],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/v1/auth/token",
                             json={"username": "admin", "password": "admin"})
            headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

            resp = await c.get(f"/api/v1/export/backtest/{row.id}/trades.csv", headers=headers)
            assert resp.status_code == 200
            assert "text/csv" in resp.headers["content-type"]
            assert "long_entry" in resp.text

            resp = await c.get(f"/api/v1/export/backtest/{row.id}/trades.xlsx", headers=headers)
            assert resp.status_code == 200
            assert resp.content[:2] == b"PK"

            assert (await c.get("/api/v1/export/backtest/99999/trades.csv",
                                headers=headers)).status_code == 404

    async def test_live_export_requires_auth_and_classifies(self, session):
        import httpx

        from trading.main import app

        session.add(OrderRow(
            id="live-1", exchange="bingx", symbol="BTC-USDT", side="buy",
            quantity=1.0, order_type="limit", status="filled", limit_price=100.0,
            stop_price=None, filled_quantity=1.0, strategy="emf", reason="entry",
        ))
        session.add(OrderRow(
            id="live-2", exchange="bingx", symbol="BTC-USDT", side="sell",
            quantity=1.0, order_type="limit", status="filled", limit_price=108.0,
            stop_price=None, filled_quantity=1.0, strategy="emf", reason="exit",
        ))
        await session.commit()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            assert (await c.get("/api/v1/export/live-trades.csv")).status_code == 401
            r = await c.post("/api/v1/auth/token",
                             json={"username": "admin", "password": "admin"})
            headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
            resp = await c.get("/api/v1/export/live-trades.csv", headers=headers)
            assert resp.status_code == 200
            assert "long_entry" in resp.text
            assert "long_exit" in resp.text
            resp = await c.get("/api/v1/export/live-trades.xlsx", headers=headers)
            assert resp.status_code == 200 and resp.content[:2] == b"PK"
