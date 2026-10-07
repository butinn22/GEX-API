"""Saved-strategy delete regression tests (Strategy Hub / Strategy lab).

Round-3 bugfix: deleting a saved strategy in the console "did not stick" — the
entry reappeared after refresh/reload. Root causes and the contracts guarded
here:

* a **version** delete (``DELETE /presets/{id}``) removes exactly that row and
  leaves the rest of the group (documented version-scoped semantics);
* a **saved strategy** delete (``DELETE /presets`` by group) removes *every*
  version of the named ``(symbol, strategy, strategy_name)`` group atomically,
  so the group cannot "reappear" as an older version;
* both refuse (409) while any version is ``live_enabled``;
* deleting a missing version/group is a clean 404 (a concurrent delete that
  won the race is also a 404, so two sessions cannot double-delete);
* the row is truly gone from the DB, not merely filtered out of the lists.
"""
from __future__ import annotations

import httpx
import pytest_asyncio
from sqlalchemy import delete, select, update

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import StrategyPresetRow
from trading.main import app

STRAT = "trend_confluence_pine"


@pytest_asyncio.fixture
async def client():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(StrategyPresetRow))
        await s.commit()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
        )
        assert r.status_code == 200, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        yield c


async def _save(client, symbol: str = "DELX", name: str = "grp", **params) -> dict:
    r = await client.post("/api/v1/presets", json={
        "symbol": symbol, "strategy": STRAT, "strategy_version": "1.0.0",
        "strategy_name": name, "params": params or {"ema_fast": 20},
        "source": "manual", "is_default": True,
    })
    assert r.status_code == 201, r.text
    return r.json()


async def _new_version(client, preset_id: int, **params) -> dict:
    r = await client.patch(f"/api/v1/presets/{preset_id}", json={"params": params or {"ema_fast": 21}})
    assert r.status_code == 200, r.text
    return r.json()


async def _ids(client, symbol: str = "DELX") -> list[int]:
    r = await client.get(f"/api/v1/presets?symbol={symbol}")
    assert r.status_code == 200, r.text
    return [p["id"] for p in r.json()]


async def _latest(client, symbol: str = "DELX") -> list[dict]:
    r = await client.get(f"/api/v1/presets/latest?symbol={symbol}")
    assert r.status_code == 200, r.text
    return r.json()


async def _set_status(preset_id: int, status: str) -> None:
    async with db._session_factory() as s:
        await s.execute(
            update(StrategyPresetRow)
            .where(StrategyPresetRow.id == preset_id)
            .values(status=status)
        )
        await s.commit()


def _group_url(symbol: str = "DELX", name: str = "grp") -> str:
    return f"/api/v1/presets?symbol={symbol}&strategy={STRAT}&name={name}"


# ── version delete ────────────────────────────────────────────────────
async def test_delete_version_removes_it_from_both_lists(client):
    p = await _save(client)
    r = await client.delete(f"/api/v1/presets/{p['id']}")
    assert r.status_code == 204
    assert await _ids(client) == []
    assert await _latest(client) == []


async def test_delete_latest_version_exposes_the_previous_version(client):
    v1 = await _save(client)
    v2 = await _new_version(client, v1["id"])
    assert (await client.delete(f"/api/v1/presets/{v2['id']}")).status_code == 204
    # version-scoped delete: the group survives through its earlier version
    assert [p["id"] for p in await _latest(client)] == [v1["id"]]
    assert await _ids(client) == [v1["id"]]


async def test_delete_missing_version_is_404(client):
    assert (await client.delete("/api/v1/presets/999999")).status_code == 404


# ── group (saved-strategy) delete ─────────────────────────────────────
async def test_group_delete_removes_every_version(client):
    v1 = await _save(client)
    await _new_version(client, v1["id"])
    keep = await _save(client, symbol="OTHER", name="keep")

    r = await client.delete(_group_url())
    assert r.status_code == 204
    assert await _ids(client) == []
    assert await _latest(client) == []
    # an unrelated group is untouched
    assert await _ids(client, symbol="OTHER") == [keep["id"]]


async def test_group_delete_last_version_removes_the_group(client):
    await _save(client)
    assert (await client.delete(_group_url())).status_code == 204
    assert await _latest(client) == []


async def test_group_delete_physically_removes_rows(client):
    v1 = await _save(client)
    await _new_version(client, v1["id"])
    assert (await client.delete(_group_url())).status_code == 204
    async with db._session_factory() as s:
        rows = (
            await s.execute(
                select(StrategyPresetRow).where(StrategyPresetRow.symbol == "DELX")
            )
        ).scalars().all()
        assert rows == []


async def test_group_delete_missing_is_404(client):
    assert (await client.delete(_group_url(symbol="NOPE", name="x"))).status_code == 404


async def test_concurrent_group_delete_second_is_404(client):
    await _save(client)
    assert (await client.delete(_group_url())).status_code == 204
    # a second session that lost the race gets a clean 404, not a 500
    assert (await client.delete(_group_url())).status_code == 404


async def test_group_delete_requires_symbol_and_strategy(client):
    assert (await client.delete("/api/v1/presets?symbol=DELX")).status_code == 400


# ── live guard ────────────────────────────────────────────────────────
async def test_delete_refused_while_live_enabled(client):
    p = await _save(client)
    await _set_status(p["id"], "live_enabled")

    assert (await client.delete(f"/api/v1/presets/{p['id']}")).status_code == 409
    assert (await client.delete(_group_url())).status_code == 409
    # a refused delete never removes anything
    assert await _ids(client) == [p["id"]]
