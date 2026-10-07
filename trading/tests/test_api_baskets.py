"""Tests for basket export / deploy endpoints and per-ticker signal keys.

Covers the "complete basket transfer" workflow (T02/T03):

* ``POST /baskets/export`` — per-ticker validation (one error never swallows
  another ticker's diagnosis) and a parameter-lossless payload;
* **round-trip** — the export payload, with its provenance fields stripped,
  POSTed back to ``/backtest/portfolio`` reproduces the exact same effective
  per-ticker params;
* ``POST /baskets/deploy`` — validated basket → signal key with **pinned**
  per-ticker params (live pipeline replays exactly what was backtested);
* ``SignalKeyService.generate()`` — per-ticker strategy with a backwards
  compatible fallback to the key-level global strategy (legacy configs).
"""
from __future__ import annotations

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
from trading.application.presets import PresetService
from trading.application.signal_keys import SignalKeyError, SignalKeyService
from trading.domain import Bar
from trading.main import app

#: Provenance fields that are ignored when a payload is fed back into
#: ``/backtest/portfolio`` (the shared round-trip mapping).
PROVENANCE_FIELDS = frozenset({
    "strategy_name", "preset_id", "preset_version", "preset_source",
    "backtest_ref", "assignment", "optimized", "warnings", "capital",
})


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


async def _token(client) -> dict:
    r = await client.post("/api/v1/auth/token",
                          json={"username": "admin", "password": "admin"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _strip_provenance(payload: dict) -> dict:
    """Map an export payload onto a PortfolioBacktestRequest body.

    ``costs.*`` → top-level same-name fields; each ticker keeps only the
    ``TickerConfig`` fields (provenance is audit-only and dropped).
    """
    return {
        **payload["costs"],
        "tickers": [
            {k: v for k, v in t.items() if k not in PROVENANCE_FIELDS}
            for t in payload["tickers"]
        ],
    }


# ── auth ───────────────────────────────────────────────────────────────


@pytest.mark.real_auth
async def test_export_and_deploy_require_auth(client):
    assert (await client.post(
        "/api/v1/baskets/export", json={"tickers": [{"symbol": "AAA"}]})
    ).status_code == 401
    assert (await client.post(
        "/api/v1/baskets/deploy",
        json={"basket": {"tickers": [{"symbol": "AAA"}]}, "exchange": "tbank"})
    ).status_code == 401


# ── export validation (per-ticker, errors never swallow each other) ────


async def test_export_reports_every_ticker_error_with_code(client, session):
    headers = await _token(client)
    basket = {
        "tickers": [
            {"symbol": "AAA"},                                                    # strategy_missing
            {"symbol": "BBB", "preset_id": 987654},                               # preset_not_found
            {"symbol": "CCC", "strategy": "sma_crossover"},                       # params_missing
            {"symbol": "DDD", "strategy": "sma_crossover",
             "params": {"fast": 50, "slow": 20}},                                 # build_failed
        ],
    }
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "validation_failed"
    codes = {e["symbol"]: e["code"] for e in detail["errors"]}
    assert codes == {
        "AAA": "strategy_missing",
        "BBB": "preset_not_found",
        "CCC": "params_missing",
        "DDD": "build_failed",
    }
    assert all(e["message"] for e in detail["errors"])


async def test_export_allows_adhoc_params_with_warning(client, session):
    """No preset but complete explicit params → exportable, flagged as non-optimized."""
    headers = await _token(client)
    basket = {"tickers": [{
        "symbol": "AAA", "strategy": "sma_crossover",
        "params": {"fast": 5, "slow": 20}, "source": "synthetic", "limit": 400,
    }]}
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["schema_version"] == "1"
    assert d["costs"]["initial_cash"] == 100_000.0
    t = d["tickers"][0]
    assert t["assignment"] == "adhoc"
    assert t["optimized"] is False
    assert t["warnings"], "adhoc ticker must carry an explicit non-optimized warning"
    # effective params are the engine's folded set (defaults filled in)
    assert t["params"] == {"fast": 5, "slow": 20, "period": 20}


async def test_export_resolves_params_from_preset_store_not_client_echo(client, session):
    """With a preset, params come from the store server-side (no client echo needed)."""
    headers = await _token(client)
    saved = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover", params={"fast": 5, "slow": 15},
        source="backtest", backtest_ref="backtest:9",
    )
    basket = {"tickers": [{
        "symbol": "AAA", "preset_id": saved.id, "source": "synthetic", "limit": 400,
    }]}
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 200, r.text
    t = r.json()["tickers"][0]
    assert t["strategy"] == "sma_crossover"
    assert t["assignment"] == "preset"
    assert t["optimized"] is True
    assert t["preset_id"] == saved.id
    assert t["preset_version"] == saved.version
    assert t["preset_source"] == "backtest"
    assert t["backtest_ref"] == "backtest:9"
    # server-resolved stored params (plus the fold defaults for lossless re-import)
    assert t["params"]["fast"] == 5
    assert t["params"]["slow"] == 15


# ── round-trip: payload → /backtest/portfolio reproduces the params ────


async def test_export_round_trip_reproduces_effective_params(client, session):
    """Strip provenance from the payload, POST it back, and every ticker's
    effective params must equal the exported params exactly."""
    headers = await _token(client)
    saved = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover", params={"fast": 5, "slow": 15}
    )
    basket = {
        "tickers": [
            {"symbol": "AAA", "preset_id": saved.id, "weight": 0.6,
             "source": "synthetic", "limit": 400},
            {"symbol": "BBB", "strategy": "momentum", "params": {"period": 10},
             "weight": 0.4, "source": "synthetic", "limit": 400},
        ],
        "initial_cash": 80_000, "fee_rate": 0.001, "slippage": 0.0005,
    }
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 200, r.text
    payload = r.json()

    req = _strip_provenance(payload)
    r2 = await client.post("/api/v1/backtest/portfolio", json=req, headers=headers)
    assert r2.status_code == 200, r2.text
    by_symbol = {t["symbol"]: t for t in r2.json()["tickers"]}

    for t in payload["tickers"]:
        assert by_symbol[t["symbol"]]["params"] == t["params"], (
            f"round-trip params diverged for {t['symbol']}"
        )


# ── deploy: validated basket → signal key (pinned per-ticker params) ───


async def test_deploy_rejects_invalid_basket(client, session):
    headers = await _token(client)
    body = {"basket": {"tickers": [{"symbol": "AAA"}]}, "exchange": "tbank"}
    r = await client.post("/api/v1/baskets/deploy", json=body, headers=headers)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "validation_failed"


async def test_deploy_creates_key_and_replays_per_ticker_strategies(client, session):
    headers = await _token(client)
    saved_aaa = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover", params={"fast": 5, "slow": 15}
    )
    saved_ccc = await PresetService(session).save(
        symbol="CCC", strategy="momentum", params={"period": 10}
    )
    basket = {
        "tickers": [
            {"symbol": "AAA", "preset_id": saved_aaa.id,
             "source": "synthetic", "limit": 400},
            {"symbol": "CCC", "preset_id": saved_ccc.id,
             "source": "synthetic", "limit": 400},
        ],
        "initial_cash": 80_000,
    }
    r = await client.post(
        "/api/v1/baskets/deploy",
        json={"basket": basket, "exchange": "tbank", "label": "basket-live"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["key"].startswith("sk_")
    assert out["exchange"] == "tbank"
    assert out["label"] == "basket-live"
    assert out["dashboard"] == f"/API_KEY/{out['key']}"

    # the key config pins strategy + params + preset per ticker
    svc = SignalKeyService(session)
    row = await svc.get_by_key(out["key"])
    config = json.loads(row.config_json)
    cfg = {t["symbol"]: t for t in config["tickers"]}
    assert cfg["AAA"]["strategy"] == "sma_crossover"
    assert cfg["AAA"]["preset_id"] == saved_aaa.id
    assert cfg["AAA"]["params"] is not None and cfg["AAA"]["params"]["fast"] == 5
    assert cfg["CCC"]["strategy"] == "momentum"
    assert cfg["CCC"]["preset_id"] == saved_ccc.id

    # live generation replays exactly the pinned per-ticker strategies
    report = await svc.generate(
        key=row, refresh=False,
        bars_by_symbol={"AAA": _bars(400), "CCC": _bars(400, seed=60.0)},
    )
    assert report["key"] == out["key"]
    summary = svc.summary(row.id)
    per = {p["symbol"]: p["strategy"] for p in summary.per_ticker}
    assert per == {"AAA": "sma_crossover", "CCC": "momentum"}


async def test_create_from_basket_accepts_dict_payload(session):
    """The service works on the payload's dict form too (external callers)."""
    svc = SignalKeyService(session)
    payload = {
        "costs": {"initial_cash": 50_000.0, "fee_rate": 0.001, "slippage": 0.0005,
                  "position_fraction": 0.95, "periods_per_year": 252},
        "tickers": [
            {"symbol": "AAA", "strategy": "sma_crossover",
             "params": {"fast": 5, "slow": 15}, "preset_id": 7,
             "source": "synthetic", "timeframe": "1d", "limit": 400, "enabled": True},
            {"symbol": "BBB", "strategy": "momentum", "params": {"period": 10},
             "preset_id": None, "source": "synthetic", "timeframe": "1d",
             "limit": 400, "enabled": True},
        ],
    }
    row, _warnings = await svc.create_from_basket(
        exchange="tbank", label="dict", payload=payload
    )
    config = json.loads(row.config_json)
    assert config["initial_cash"] == 50_000.0
    cfg = {t["symbol"]: t for t in config["tickers"]}
    assert cfg["AAA"]["strategy"] == "sma_crossover"
    assert cfg["AAA"]["params"] == {"fast": 5, "slow": 15}
    assert cfg["AAA"]["preset_id"] == 7
    assert cfg["BBB"]["strategy"] == "momentum"
    assert cfg["BBB"]["preset_id"] is None


async def test_create_from_basket_requires_valid_exchange_and_keeps_enabled_flag(session):
    svc = SignalKeyService(session)
    payload = {"costs": {}, "tickers": [{"symbol": "AAA", "strategy": "momentum",
                                         "params": {"period": 10}, "enabled": False}]}
    with pytest.raises(SignalKeyError):
        await svc.create_from_basket(exchange="coinbase", payload=payload)

    # a disabled ticker is kept in the config (the portfolio engine filters
    # it out at generation time, exactly like a backtest) instead of being
    # silently dropped or rejected
    row, _ = await svc.create_from_basket(exchange="tbank", payload=payload)
    config = json.loads(row.config_json)
    assert config["tickers"][0]["enabled"] is False
    assert config["tickers"][0]["strategy"] == "momentum"


# ── generate(): backwards-compatible per-ticker strategy fallback ──────


async def test_generate_legacy_config_falls_back_to_global_strategy(session):
    """Old key configs (no per-ticker strategy) keep working unchanged."""
    svc = SignalKeyService(session)
    row, _ = await svc.create(exchange="bingx", tickers=["AAA", "BBB"],
                              strategy="momentum")
    config = json.loads(row.config_json)
    assert all("strategy" not in t for t in config["tickers"])  # legacy shape

    await svc.generate(key=row, refresh=False,
                       bars_by_symbol={"AAA": _bars(300), "BBB": _bars(300, seed=40.0)})
    summary = svc.summary(row.id)
    assert {p["strategy"] for p in summary.per_ticker} == {"momentum"}


async def test_generate_honours_per_ticker_strategy(session):
    """New configs: each ticker runs its own strategy."""
    svc = SignalKeyService(session)
    row, _ = await svc.create(exchange="bingx", tickers=["AAA"], strategy="momentum")

    # hand-upgrade the config the way create_from_basket does
    config = json.loads(row.config_json)
    config["tickers"][0]["strategy"] = "sma_crossover"
    config["tickers"][0]["params"] = {"fast": 5, "slow": 15}
    config["tickers"].append({
        "symbol": "BBB", "params": None, "preset_id": None,
        "strategy": "momentum", "source": "synthetic", "timeframe": "1d",
        "limit": 300, "enabled": True,
    })
    row.config_json = json.dumps(config)
    await session.commit()

    await svc.generate(key=row, refresh=False,
                       bars_by_symbol={"AAA": _bars(300), "BBB": _bars(300, seed=40.0)})
    summary = svc.summary(row.id)
    per = {p["symbol"]: p["strategy"] for p in summary.per_ticker}
    assert per == {"AAA": "sma_crossover", "BBB": "momentum"}
