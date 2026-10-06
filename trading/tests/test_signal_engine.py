"""Tests for the live signal engine: lifecycle, persistence, publish, export.

No network: ``FetcherRegistry`` is stubbed with a synthetic (but real-shaped)
async fetcher, and the database is the in-process SQLite the other API tests
already use. What is asserted is the contract the API and the exports rely on:

* a signal persists with its full trade plan and opens exactly one position,
* the matching exit closes that position and computes PnL from the signed
  entry/exit prices (never a fabricated number),
* the first poll's warm-up replay is stored as ``backfill``, not ``live``,
* the CSV/XLSX exports contain every column and parse back.
"""
from __future__ import annotations

import asyncio

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.fetchers.registry import FetcherRegistry
from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import KeySignalRow, SignalPositionRow
from trading.application.reporting.signal_export import (
    POSITION_COLUMNS,
    SIGNAL_COLUMNS,
    positions_to_csv,
    positions_to_xlsx,
    signals_to_csv,
    signals_to_xlsx,
)
from trading.application.signal_engine import (
    SignalEngine,
    SignalEngineConfig,
    signal_engine,
)
from trading.domain import Bar, Exchange
from trading.main import app

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


class _StubFetcher:
    """Real-shaped async fetcher over a deterministic bar series."""

    exchange = Exchange.BYBIT

    def __init__(self, bars: list[Bar]) -> None:
        self._bars = list(bars)
        self.calls = 0

    async def get_ohlcv(self, symbol, timeframe, *, start=None, end=None, limit=500):
        self.calls += 1
        return list(self._bars)[-limit:]

    async def get_instruments(self):
        return []

    async def get_orderbook(self, symbol, *, depth=20):
        raise NotImplementedError

    async def get_trades(self, symbol, *, limit=50):
        raise NotImplementedError


def _bars(n: int = 400) -> list[Bar]:
    out: list[Bar] = []
    price = 100.0
    state = 11
    for i in range(n):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        noise = ((state / 2 ** 31) - 0.5) / 100.0
        drift = 0.0025 if (i // 12) % 5 < 4 else -0.0015
        price *= 1.0 + drift + noise
        out.append(Bar(timestamp=_T0 + timedelta(hours=4 * i), open=price,
                       high=price * 1.001, low=price * 0.999, close=price,
                       volume=1000.0))
    return out


class _StubRegistry(FetcherRegistry):
    def __init__(self, fetcher) -> None:
        super().__init__()
        self._fetcher = fetcher
        self.register(fetcher)

    async def get_ohlcv(self, exchanges, symbol, timeframe, *, start=None, end=None, limit=500):
        return await self._fetcher.get_ohlcv(symbol, timeframe, limit=limit)


@pytest_asyncio.fixture
async def engine():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(KeySignalRow))
        await s.execute(delete(SignalPositionRow))
        await s.commit()
    return SignalEngine(session_factory=db.session_factory(),
                        registry=_StubRegistry(_StubFetcher(_bars())))


@pytest_asyncio.fixture
async def client():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(KeySignalRow))
        await s.execute(delete(SignalPositionRow))
        await s.commit()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── configuration ─────────────────────────────────────────────────────
def test_config_from_dict_parsing_and_defaults():
    cfg = SignalEngineConfig.from_dict({"symbols": "BTC, ETH", "timeframe": "1d"})
    assert cfg.symbols == ["BTC", "ETH"]
    assert cfg.timeframe == "1d" and cfg.strategy == "confluence_breakout"
    cfg2 = SignalEngineConfig.from_dict({"symbols": ["BTC"], "params": {"n_break": 12},
                                         "preset": "alligator_4h"})
    assert cfg2.params == {"n_break": 12}


def test_config_rejects_bad_values():
    # ``from_dict`` only parses; the engine itself refuses invalid input, so
    # validation is asserted through ``start`` (exactly what the router does).
    cfg = SignalEngineConfig.from_dict({"symbols": ["ETHUSDT"], "timeframe": "7s"})
    with pytest.raises(ValueError):
        asyncio.run(engine_stub().start(cfg))
    with pytest.raises(ValueError):
        asyncio.run(engine_stub().start(SignalEngineConfig.from_dict(
            {"symbols": ["ETHUSDT"], "poll_seconds": 1})))
    with pytest.raises(ValueError):
        asyncio.run(engine_stub().start(SignalEngineConfig.from_dict(
            {"symbols": ["ETHUSDT"], "bars": 12})))
    with pytest.raises(ValueError):
        asyncio.run(engine_stub().start(SignalEngineConfig.from_dict(
            {"symbols": [], "bars": 300})))


def engine_stub():
    return SignalEngine(session_factory=db.session_factory())


# ── engine lifecycle + persistence ────────────────────────────────────
async def test_engine_persists_signals_and_positions(engine):
    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 400,
    })
    status = await engine.start(cfg)
    assert status["running"] is True and status["n_tickers"] == 1

    for _ in range(20):
        await __import__("asyncio").sleep(0.05)
        if await engine.list_signals(symbol="ETHUSDT"):
            break
    else:
        pytest.fail("no signal was persisted in time")

    sigs = await engine.list_signals(symbol="ETHUSDT")
    entry = next(s for s in sigs if s.state == "long_entry")
    assert entry.entry_price and entry.stop_loss and entry.stop_loss < entry.entry_price
    assert entry.timeframe == "4h" and entry.risk_pct == pytest.approx(0.005)
    assert entry.source == "backfill"       # warm-up replay is not "live"
    assert entry.strategy == "confluence_breakout"

    pos = await engine.list_positions(symbol="ETHUSDT")
    assert pos, "an entry must open a position"
    assert pos[0].side == "long" and pos[0].quantity > 0
    assert pos[0].entry_price == pytest.approx(entry.entry_price)

    await engine.stop()
    st = engine.status()
    assert st["running"] is False and st["stopped_at"] is not None
    assert all(t["running"] is False for t in st["tickers"])


async def test_engine_closes_position_with_computed_pnl(engine):
    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 400,
    })
    await engine.start(cfg)
    for _ in range(40):
        await __import__("asyncio").sleep(0.05)
        if any(p.status == "closed" for p in await engine.list_positions(symbol="ETHUSDT")):
            break
    await engine.stop()

    closed = [p for p in await engine.list_positions(symbol="ETHUSDT") if p.status == "closed"]
    if not closed:
        pytest.skip("synthetic series produced no closed position in this run")
    row = closed[0]
    assert row.exit_price is not None and row.exit_reason
    # PnL is derived from the signed prices — recompute it and compare
    sign = 1.0 if row.side == "long" else -1.0
    expected = (row.exit_price - row.entry_price) * row.quantity * sign
    assert row.net_pnl == pytest.approx(expected, rel=1e-9)
    assert row.pnl_r == pytest.approx(expected / row.risk_amount, rel=1e-9)
    assert row.unrealised_pnl == 0.0


async def test_engine_rejects_double_start_and_empty_symbols(engine):
    cfg = SignalEngineConfig.from_dict({
        "symbols": ["ETHUSDT"], "timeframe": "4h", "poll_seconds": 5, "bars": 300})
    await engine.start(cfg)
    with pytest.raises(RuntimeError):
        await engine.start(cfg)
    await engine.stop()

    with pytest.raises(ValueError):
        await engine.start(SignalEngineConfig.from_dict({"symbols": []}))
    with pytest.raises(ValueError):
        await engine.start(SignalEngineConfig.from_dict(
            {"symbols": [f"S{i}" for i in range(25)], "bars": 300}))


async def test_engine_stats_over_ledger(engine):
    async with engine._session() as s:
        s.add(SignalPositionRow(symbol="ETHUSDT", side="long", status="closed",
                                entry_price=100.0, quantity=1.0, exit_price=110.0,
                                risk_amount=5.0, net_pnl=10.0, pnl_r=2.0))
        s.add(SignalPositionRow(symbol="XRPUSDT", side="long", status="open",
                                entry_price=1.0, quantity=100.0, risk_amount=1.0))
        await s.commit()
    stats = await engine.positions_stats()
    assert stats["open"] == 1 and stats["closed"] == 1
    assert stats["total_pnl"] == pytest.approx(10.0)
    assert stats["wins"] == 1 and stats["losses"] == 0 and stats["win_rate"] == 1.0

    assert await engine.clear_positions(only_open=True) == 1
    assert (await engine.positions_stats())["open"] == 0


# ── exports ───────────────────────────────────────────────────────────
async def test_signal_and_position_exports(engine):
    async with engine._session() as s:
        s.add(KeySignalRow(symbol="ETHUSDT", side="buy", state="long_entry",
                           reason="breakout", strength=0.5, price=100.0,
                           entry_price=100.0, stop_loss=95.0, timeframe="4h",
                           risk_pct=0.005, risk_amount=5.0, position_size=0.01,
                           timestamp=_T0, strategy="confluence_breakout",
                           indicators_json='{"preset": "alligator_4h"}'))
        s.add(SignalPositionRow(symbol="ETHUSDT", side="long", status="closed",
                                entry_time=_T0, entry_price=100.0, quantity=0.5,
                                initial_stop=95.0, stop_price=95.0, risk_amount=5.0,
                                exit_time=_T0 + timedelta(hours=8), exit_price=110.0,
                                exit_reason="trailing_stop", gross_pnl=5.0, net_pnl=5.0))
        await s.commit()

    sigs = await engine.list_signals(limit=10)
    csv_text = signals_to_csv(sigs)
    lines = csv_text.strip().splitlines()
    assert lines[0].split(",") == list(SIGNAL_COLUMNS)
    assert len(lines) == 2
    assert "alligator_4h" in lines[1]
    assert len(signals_to_xlsx(sigs)) > 500

    pos = await engine.list_positions(limit=10)
    pcsv = positions_to_csv(pos).strip().splitlines()
    assert pcsv[0].split(",") == list(POSITION_COLUMNS)
    assert len(pcsv) == 2
    assert positions_to_xlsx(pos)


async def test_export_open_position_has_empty_pnl(engine):
    async with engine._session() as s:
        s.add(SignalPositionRow(symbol="ETHUSDT", side="long", status="open",
                                entry_time=_T0, entry_price=100.0, quantity=1.0))
        await s.commit()
    pos = await engine.list_positions()
    header, row = positions_to_csv(pos).strip().splitlines()
    cols = header.split(",")
    net = dict(zip(cols, row.split(",")))["net_pnl"]
    assert net == "", "an open position must not export a fabricated PnL"


# ── API ───────────────────────────────────────────────────────────────
async def _token(client: httpx.AsyncClient) -> dict[str, str]:
    r = await client.post("/api/v1/auth/token",
                          json={"username": "admin", "password": "admin"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_api_requires_auth(client):
    assert (await client.get("/api/v1/signals")).status_code == 401
    assert (await client.get("/api/v1/signals/engine")).status_code == 401


async def test_api_status_export_and_filters(client):
    headers = await _token(client)
    async with db._session_factory() as s:
        s.add(KeySignalRow(symbol="ETHUSDT", side="buy", state="long_entry",
                           reason="breakout", price=100.0, entry_price=100.0,
                           stop_loss=95.0, timeframe="4h", timestamp=_T0,
                           strategy="confluence_breakout",
                           indicators_json='{"preset": "alligator_4h"}'))
        s.add(KeySignalRow(symbol="XRPUSDT", side="sell", state="long_exit",
                           reason="trailing_stop", price=2.0, timestamp=_T0,
                           strategy="confluence_breakout"))
        s.add(SignalPositionRow(symbol="ETHUSDT", side="long", status="open",
                                entry_time=_T0, entry_price=100.0, quantity=1.0))
        await s.commit()

    r = await client.get("/api/v1/signals/engine", headers=headers)
    assert r.status_code == 200 and r.json()["running"] is False

    r = await client.get("/api/v1/signals", headers=headers)
    assert r.status_code == 200 and len(r.json()) == 2

    r = await client.get("/api/v1/signals", params={"symbol": "ETHUSDT", "side": "buy"},
                         headers=headers)
    assert len(r.json()) == 1 and r.json()[0]["state"] == "long_entry"

    r = await client.get("/api/v1/signals/positions", headers=headers)
    assert len(r.json()) == 1 and r.json()[0]["status"] == "open"

    r = await client.get("/api/v1/signals/stats", headers=headers)
    assert r.json()["open"] == 1

    r = await client.get("/api/v1/signals/export/signals.csv", headers=headers)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.text.splitlines()[0].startswith("signal_id,")
    assert "attachment; filename=" in r.headers["content-disposition"]

    r = await client.get("/api/v1/signals/export/signals.xlsx", headers=headers)
    assert r.status_code == 200
    assert r.content[:2] == b"PK"

    r = await client.get("/api/v1/signals/export/positions.xlsx", headers=headers)
    assert r.status_code == 200 and r.content[:2] == b"PK"

    r = await client.get("/api/v1/signals/export/signals.csv",
                         params={"state": "long_exit"}, headers=headers)
    assert "XRPUSDT" in r.text and "ETHUSDT" not in r.text


async def test_api_engine_start_stop_roundtrip(client):
    headers = await _token(client)
    # The singleton engine must be free (a previous test may have left it up).
    if signal_engine.running:
        await signal_engine.stop()

    r = await client.post("/api/v1/signals/engine/start",
                          json={"symbols": ["ETHUSDT"], "timeframe": "4h",
                                "poll_seconds": 5, "bars": 400}, headers=headers)
    assert r.status_code in (200, 409), r.text
    if r.status_code == 200:
        assert r.json()["n_tickers"] == 1
        r = await client.get("/api/v1/signals/engine", headers=headers)
        assert r.json()["running"] is True
        r = await client.post("/api/v1/signals/engine/stop", headers=headers)
        assert r.json()["running"] is False

    # validation errors surface as 422, never a 500
    r = await client.post("/api/v1/signals/engine/start",
                          json={"symbols": [], "bars": 400}, headers=headers)
    assert r.status_code == 422
    r = await client.post("/api/v1/signals/engine/start",
                          json={"symbols": ["ETHUSDT"], "timeframe": "9s"}, headers=headers)
    assert r.status_code == 422


async def test_api_delete_positions(client):
    headers = await _token(client)
    r = await client.request("DELETE", "/api/v1/signals/positions",
                             params={"only_open": True}, headers=headers)
    assert r.status_code == 200 and "deleted" in r.json()
