"""QA acceptance verification for the basket-transfer batch (commits
19103b2 + 3c117b1) — independent round-trip / error-matrix / contract checks
that do not rely on the engineer's own test expectations.

Scope (task T07-D):

1. Round-trip: export a mixed basket (preset-bound + adhoc), strip the
   provenance fields, POST the payload back to ``/backtest/portfolio`` and
   require every ticker's effective params to equal the exported params
   key-for-key.
2. Deploy: the created signal key pins per-ticker params/preset_id/source and
   keeps an ``enabled=False`` ticker in the config.
3. 422 error matrix: strategy_missing / params_missing / preset_not_found /
   build_failed in one request — per-ticker, never short-circuiting; a
   preset-less ticker with complete explicit params is allowed with a warning.
4. Contract: /backtest/portfolio reports the four assignment shapes
   (preset / explicit-override / default / adhoc), preset-missing → 400 with
   the ticker and the id in the message, optimized only with provenance.
"""
from __future__ import annotations

import json

import httpx
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import (
    SignalKeyRow,
    StrategyPresetRow,
)
from trading.application.presets import PresetService
from trading.application.signal_keys import SignalKeyService
from trading.main import app

#: Audit-only fields dropped before replaying an export payload.
PROVENANCE = frozenset({
    "strategy_name", "preset_id", "preset_version", "preset_source",
    "backtest_ref", "assignment", "optimized", "warnings", "capital",
})


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(SignalKeyRow))
        await s.execute(delete(StrategyPresetRow))
        await s.commit()
        yield s


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        # Round-2: the /baskets + /backtest routers are JWT-guarded.
        r = await c.post(
            "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
        )
        assert r.status_code == 200, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        yield c


async def _token(client) -> dict:
    r = await client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── 2. round-trip (independent construction) ───────────────────────────


async def test_qa_round_trip_params_lossless(client, session):
    """Export → strip provenance → /backtest/portfolio: per-ticker params
    in the response must equal the exported params exactly (every key)."""
    headers = await _token(client)
    saved = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover",
        params={"fast": 7, "slow": 33}, strategy_name="qa-preset",
        source="optimizer", optimizer_run_id="run-qa-1",
    )
    basket = {
        "tickers": [
            # preset-bound (with an explicit override on top of the store)
            {"symbol": "AAA", "preset_id": saved.id, "params": {"slow": 33},
             "weight": 0.7, "source": "synthetic", "limit": 400},
            # adhoc: explicit params only
            {"symbol": "BBB", "strategy": "momentum", "params": {"period": 12},
             "weight": 0.3, "source": "synthetic", "limit": 400},
        ],
        "initial_cash": 60_000, "fee_rate": 0.002, "slippage": 0.001,
    }
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 200, r.text
    payload = r.json()

    req = {
        **payload["costs"],
        "tickers": [
            {k: v for k, v in t.items() if k not in PROVENANCE}
            for t in payload["tickers"]
        ],
    }
    r2 = await client.post("/api/v1/backtest/portfolio", json=req)
    assert r2.status_code == 200, r2.text
    by_symbol = {t["symbol"]: t for t in r2.json()["tickers"]}

    for t in payload["tickers"]:
        got = by_symbol[t["symbol"]]["params"]
        assert got == t["params"], (
            f"round-trip diverged for {t['symbol']}:\n"
            f"  exported : {t['params']}\n  effective: {got}"
        )


# ── 2b. deploy: pinned config + disabled ticker retained ───────────────


async def test_qa_deploy_pins_config_and_keeps_disabled(client, session):
    headers = await _token(client)
    saved = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover", params={"fast": 7, "slow": 33}
    )
    basket = {
        "tickers": [
            {"symbol": "AAA", "preset_id": saved.id,
             "source": "synthetic", "timeframe": "1d", "limit": 400},
            {"symbol": "CCC", "strategy": "momentum", "params": {"period": 9},
             "source": "synthetic", "timeframe": "1h", "limit": 300,
             "enabled": False},
        ],
        "initial_cash": 70_000,
    }
    r = await client.post(
        "/api/v1/baskets/deploy",
        json={"basket": basket, "exchange": "tbank", "label": "qa-deploy"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["key"].startswith("sk_")
    assert out["dashboard"] == f"/API_KEY/{out['key']}"

    row = await SignalKeyService(session).get_by_key(out["key"])
    config = json.loads(row.config_json)
    cfg = {t["symbol"]: t for t in config["tickers"]}

    assert cfg["AAA"]["strategy"] == "sma_crossover"
    assert cfg["AAA"]["preset_id"] == saved.id
    assert cfg["AAA"]["params"]["fast"] == 7
    assert cfg["AAA"]["params"]["slow"] == 33
    assert cfg["AAA"]["source"] == "synthetic"
    assert cfg["AAA"]["enabled"] is True

    # disabled ticker: kept verbatim, not dropped, not resurrected
    assert cfg["CCC"]["enabled"] is False
    assert cfg["CCC"]["strategy"] == "momentum"
    assert cfg["CCC"]["params"] == {"period": 9, "fast": 20, "slow": 50}
    assert cfg["CCC"]["source"] == "synthetic"
    assert cfg["CCC"]["timeframe"] == "1h"

    assert config["initial_cash"] == 70_000.0


# ── 3. 422 error matrix (mixed, non-short-circuiting) ──────────────────


async def test_qa_422_matrix_reports_every_ticker(client, session):
    headers = await _token(client)
    basket = {
        "tickers": [
            {"symbol": "AAA"},                                                   # strategy_missing
            {"symbol": "BBB", "preset_id": 999999},                              # preset_not_found
            {"symbol": "CCC", "strategy": "sma_crossover"},                      # params_missing
            {"symbol": "DDD", "strategy": "sma_crossover",
             "params": {"fast": 50, "slow": 20}},                                # build_failed
            # valid adhoc ticker — must not mask the others, and the whole
            # request still 422s (export is all-or-nothing)
            {"symbol": "EEE", "strategy": "momentum", "params": {"period": 10}},
        ],
    }
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "validation_failed"
    by_symbol = {e["symbol"]: e for e in detail["errors"]}
    assert set(by_symbol) == {"AAA", "BBB", "CCC", "DDD"}  # EEE has no error
    assert by_symbol["AAA"]["code"] == "strategy_missing"
    assert by_symbol["BBB"]["code"] == "preset_not_found"
    assert by_symbol["CCC"]["code"] == "params_missing"
    assert by_symbol["DDD"]["code"] == "build_failed"
    assert "999999" in by_symbol["BBB"]["message"]
    assert all(e["message"] for e in detail["errors"])


async def test_qa_presetless_complete_params_pass_with_warning(client, session):
    headers = await _token(client)
    basket = {"tickers": [{
        "symbol": "AAA", "strategy": "sma_crossover",
        "params": {"fast": 5, "slow": 20}, "source": "synthetic", "limit": 400,
    }]}
    r = await client.post("/api/v1/baskets/export", json=basket, headers=headers)
    assert r.status_code == 200, r.text
    t = r.json()["tickers"][0]
    assert t["assignment"] == "adhoc" and t["optimized"] is False
    assert t["warnings"], "preset-less export must carry a warning"
    assert t["params"] == {"fast": 5, "slow": 20, "period": 20}  # fold-closed


# ── 4. contract: four assignment shapes + 400 + optimized provenance ───


async def test_qa_contract_four_assignment_shapes(client, session):
    await _token(client)
    saved = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover", params={"fast": 7, "slow": 33},
        strategy_name="qa-shape", source="backtest", backtest_ref="backtest:77",
    )
    payload = {
        "tickers": [
            # preset assignment (with real-run provenance → optimized)
            {"symbol": "AAA", "preset_id": saved.id,
             "source": "synthetic", "limit": 400},
            # preset assignment with an explicit override
            {"symbol": "BBB", "preset_id": saved.id, "params": {"slow": 41},
             "source": "synthetic", "limit": 400},
            # default: no preset, no explicit params
            {"symbol": "CCC", "strategy": "momentum",
             "source": "synthetic", "limit": 400},
            # adhoc: explicit params, no preset
            {"symbol": "DDD", "strategy": "momentum", "period": 14,
             "source": "synthetic", "limit": 400},
        ],
        "initial_cash": 90_000,
    }
    r = await client.post("/api/v1/backtest/portfolio", json=payload)
    assert r.status_code == 200, r.text
    by = {t["symbol"]: t for t in r.json()["tickers"]}

    aaa = by["AAA"]
    assert aaa["assignment"] == "preset"
    assert aaa["strategy_name"] == "qa-shape"
    assert aaa["preset_id"] == saved.id
    assert aaa["preset_version"] == saved.version
    assert aaa["optimized"] is True  # backtest:77 provenance
    assert aaa["params"]["fast"] == 7 and aaa["params"]["slow"] == 33

    bbb = by["BBB"]
    assert bbb["assignment"] == "preset"
    assert bbb["optimized"] is True
    assert bbb["params"]["slow"] == 41  # explicit override applied
    assert bbb["params"]["fast"] == 7   # preset baseline kept

    ccc = by["CCC"]
    assert ccc["assignment"] == "default"
    assert ccc["optimized"] is False
    assert ccc["preset_id"] is None and ccc["preset_version"] is None

    ddd = by["DDD"]
    assert ddd["assignment"] == "adhoc"
    assert ddd["optimized"] is False
    assert ddd["params"]["period"] == 14


async def test_qa_contract_preset_missing_is_400_with_context(client, session):
    payload = {
        "tickers": [{"symbol": "XYZ", "preset_id": 424242,
                     "source": "synthetic", "limit": 300}],
        "initial_cash": 50_000,
    }
    r = await client.post("/api/v1/backtest/portfolio", json=payload)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "XYZ" in detail and "424242" in detail


async def test_qa_contract_manual_preset_is_not_optimized(client, session):
    """A manual preset has no real-run provenance → optimized stays False."""
    saved = await PresetService(session).save(
        symbol="AAA", strategy="sma_crossover", params={"fast": 7, "slow": 33},
    )
    payload = {
        "tickers": [{"symbol": "AAA", "preset_id": saved.id,
                     "source": "synthetic", "limit": 400}],
        "initial_cash": 50_000,
    }
    r = await client.post("/api/v1/backtest/portfolio", json=payload)
    assert r.status_code == 200, r.text
    t = r.json()["tickers"][0]
    assert t["assignment"] == "preset"
    assert t["optimized"] is False
