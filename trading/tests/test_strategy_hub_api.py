"""Strategy Hub API tests: versioning endpoints, go-live gate payloads,
preset_id-driven backtest/optimize runs.

No network: bar data comes from the deterministic synthetic feed and inline
request bars; the database is the in-process SQLite the other API tests use.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import StrategyPresetRow

_METRICS = {
    "total_return": 0.21, "sharpe": 1.42, "max_drawdown": 0.07,
    "win_rate": 0.58, "n_trades": 14,
}


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(StrategyPresetRow))
        await s.commit()
        yield s


@pytest_asyncio.fixture
async def client():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(StrategyPresetRow))
        await s.commit()
    transport = httpx.ASGITransport(app=__import__("trading.main", fromlist=["app"]).app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        # Round-2: the /presets + /backtest routers are JWT-guarded.
        r = await c.post(
            "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
        )
        assert r.status_code == 200, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        yield c


def _valid_preset_body(**over):
    body = {
        "symbol": "BTC", "strategy": "trend_confluence_pine",
        "strategy_name": "momentum", "params": {"zone_atr": 0.5, "min_confluence": 2},
        "source": "optimizer", "optimizer_run_id": "run-1",
        "metrics": _METRICS, "timeframe": "1d",
    }
    body.update(over)
    return body


# ── create / list / latest / versions ─────────────────────────────────


async def test_create_returns_v2_fields_and_increments_versions(client):
    r = await client.post("/api/v1/presets", json=_valid_preset_body())
    assert r.status_code == 201
    out = r.json()
    assert out["strategy_name"] == "momentum"
    assert out["version"] == 1 and out["status"] == "backtest_only"
    assert out["metrics"]["total_return"] == pytest.approx(0.21)
    assert out["backtest_ref"] == "optimizer:run-1"
    assert out["timeframe"] == "1d" and out["is_default"] is True

    r2 = await client.post("/api/v1/presets", json=_valid_preset_body(
        params={"zone_atr": 0.8}))
    assert r2.status_code == 201 and r2.json()["version"] == 2

    # a different name is a different group → its own version counter
    r3 = await client.post("/api/v1/presets", json=_valid_preset_body(
        strategy_name="mean-revert"))
    assert r3.json()["version"] == 1


async def test_create_refuses_metrics_without_provenance(client):
    body = _valid_preset_body()
    del body["optimizer_run_id"]
    r = await client.post("/api/v1/presets", json=body)
    assert r.status_code == 400


async def test_latest_and_versions_endpoints(client):
    await client.post("/api/v1/presets", json=_valid_preset_body())
    await client.post("/api/v1/presets", json=_valid_preset_body(
        params={"zone_atr": 0.8}))
    await client.post("/api/v1/presets", json=_valid_preset_body(
        strategy_name="other"))

    r = await client.get("/api/v1/presets/latest",
                         params={"symbol": "BTC", "strategy": "trend_confluence_pine"})
    assert r.status_code == 200
    latest = r.json()
    assert len(latest) == 2  # one row per (symbol, strategy, strategy_name)
    by_name = {row["strategy_name"]: row for row in latest}
    assert by_name["momentum"]["version"] == 2
    assert by_name["other"]["version"] == 1

    r = await client.get("/api/v1/presets/versions",
                         params={"symbol": "BTC", "strategy": "trend_confluence_pine",
                                 "name": "momentum"})
    assert r.status_code == 200
    versions = r.json()
    assert [v["version"] for v in versions] == [2, 1]  # newest first

    r = await client.get("/api/v1/presets/versions")
    assert r.status_code == 400  # missing symbol/strategy


# ── validate / promote / rollback / demote ───────────────────────────


async def test_validate_returns_structured_reasons(client):
    r = await client.post("/api/v1/presets", json={
        "symbol": "BTC", "strategy": "trend_confluence_pine",
        "params": {"zone_atr": 0.5},
    })
    preset_id = r.json()["id"]
    r = await client.get(f"/api/v1/presets/{preset_id}/validate")
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert {reason["code"] for reason in out["reasons"]} == {"no_backtest_evidence"}

    r = await client.get("/api/v1/presets/999999/validate")
    assert r.status_code == 404


async def test_promote_gate_failure_is_structured_422(client):
    r = await client.post("/api/v1/presets", json=_valid_preset_body(
        params={}))  # empty params → params_missing
    preset_id = r.json()["id"]
    r = await client.post(f"/api/v1/presets/{preset_id}/promote")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "validation_failed"
    codes = {reason["code"] for reason in detail["reasons"]}
    assert "params_missing" in codes


async def test_promote_rollback_demote_lifecycle(client):
    r = await client.post("/api/v1/presets", json=_valid_preset_body())
    v1 = r.json()["id"]
    r = await client.post("/api/v1/presets", json=_valid_preset_body(
        params={"zone_atr": 0.8}))
    v2 = r.json()["id"]

    # promote v1 → live + group default
    r = await client.post(f"/api/v1/presets/{v1}/promote")
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "live_enabled" and out["is_default"] is True

    # already live → 409
    r = await client.post(f"/api/v1/presets/{v1}/promote")
    assert r.status_code == 409

    # rollback to v2 (re-runs the gate) → v2 live, v1 demoted
    r = await client.post(f"/api/v1/presets/{v2}/rollback")
    assert r.status_code == 200 and r.json()["status"] == "live_enabled"
    r = await client.get("/api/v1/presets/versions",
                         params={"symbol": "BTC", "strategy": "trend_confluence_pine",
                                 "name": "momentum"})
    rows = {v["id"]: v for v in r.json()}
    assert rows[v1]["status"] == "backtest_only"

    # rollback to the already-active version → 409
    r = await client.post(f"/api/v1/presets/{v2}/rollback")
    assert r.status_code == 409

    # delete while live → 409; demote first
    r = await client.delete(f"/api/v1/presets/{v2}")
    assert r.status_code == 409
    r = await client.post(f"/api/v1/presets/{v2}/demote")
    assert r.status_code == 200 and r.json()["status"] == "backtest_only"
    r = await client.post(f"/api/v1/presets/{v2}/demote")
    assert r.status_code == 409  # not live any more
    r = await client.delete(f"/api/v1/presets/{v2}")
    assert r.status_code == 204


async def test_patch_creates_a_new_version(client):
    r = await client.post("/api/v1/presets", json=_valid_preset_body())
    v1 = r.json()
    r = await client.patch(f"/api/v1/presets/{v1['id']}", json={
        "params": {"zone_atr": 0.9, "min_confluence": 3}, "notes": "tightened",
    })
    assert r.status_code == 200
    v2 = r.json()
    assert v2["id"] != v1["id"] and v2["version"] == 2
    assert v2["source"] == "manual" and v2["is_default"] is True
    assert v2["params"] == {"zone_atr": 0.9, "min_confluence": 3}
    # the original version is untouched
    r = await client.get("/api/v1/presets/versions",
                         params={"symbol": "BTC", "strategy": "trend_confluence_pine",
                                 "name": "momentum"})
    rows = {v["id"]: v for v in r.json()}
    assert rows[v1["id"]]["params"] == v1["params"]
    assert rows[v1["id"]]["is_default"] is False


# ── preset_id-driven backtest / optimize (F4) ─────────────────────────


def _inline_bars(n: int = 120):
    import math

    t0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    out = []
    for i in range(n):
        # oscillating with a slight upward drift — guarantees SMA crossovers
        ret = 0.002 + 0.03 * math.sin(i / 2.0)
        prev = price
        price = max(1.0, price * (1 + ret))
        out.append({
            "timestamp": (t0 + timedelta(days=i)).isoformat(),
            "open": prev, "high": max(prev, price) * 1.01,
            "low": min(prev, price) * 0.99, "close": price, "volume": 1000.0,
        })
    return out


async def test_backtest_runs_with_preset_params_verbatim(client):
    r = await client.post("/api/v1/presets", json={
        "symbol": "SYNTH", "strategy": "sma_crossover",
        "params": {"fast": 2, "slow": 5}, "source": "manual",
    })
    preset_id = r.json()["id"]

    r = await client.post("/api/v1/backtest", json={
        "preset_id": preset_id, "bars": _inline_bars(),
    })
    assert r.status_code == 200
    out = r.json()
    assert out["preset_id"] == preset_id
    assert out["strategy"] == "sma_crossover" and out["symbol"] == "SYNTH"
    assert out["n_trades"] > 0  # fast=2/slow=5 trades the synthetic trend

    # explicit request params still override the stored ones per key
    r = await client.post("/api/v1/backtest", json={
        "preset_id": preset_id, "bars": _inline_bars(), "fast": 4,
    })
    assert r.status_code == 200

    r = await client.post("/api/v1/backtest", json={
        "preset_id": 999999, "bars": _inline_bars(),
    })
    assert r.status_code == 404


async def test_optimize_seeds_from_preset_and_saves_versioned_winner(client):
    r = await client.post("/api/v1/presets", json={
        "symbol": "SYNTH", "strategy": "trend_confluence",
        "strategy_name": "lab", "params": {"zone_atr": 0.5},
        "source": "manual",
    })
    preset_id = r.json()["id"]

    r = await client.post("/api/v1/backtest/optimize", json={
        "preset_id": preset_id, "source": "synthetic", "limit": 200,
        "grid": {"zone_atr": [0.5, 0.8]}, "objective": "profit_win",
        "save_preset": True,
    })
    assert r.status_code == 200
    out = r.json()
    assert out["n_candidates"] >= 1
    assert "score_breakdown" in (out.get("best") or {})

    # the winner was saved as the group's next version with metrics + ref
    r = await client.get("/api/v1/presets/versions",
                         params={"symbol": "SYNTH", "strategy": "trend_confluence",
                                 "name": "lab"})
    versions = r.json()
    assert len(versions) == 2
    winner = versions[0]
    assert winner["source"] == "optimizer"
    assert winner["optimizer_run_id"] == out["run_token"]
    assert winner["backtest_ref"] == f"optimizer:{out['run_token']}"
    assert winner["metrics"].get("total_return") is not None  # real snapshot
