"""Tests for per-ticker strategy presets (repository + service)."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import StrategyPresetRow
from trading.application.presets import PresetService


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(StrategyPresetRow))
        await s.commit()
        yield s


async def test_save_and_get_default(session):
    svc = PresetService(session)
    row = await svc.save(symbol="nvda", params={"zone_atr": 0.4})
    assert row.symbol == "NVDA"  # normalized upper-case
    assert row.source == "manual" and row.is_default

    got = await svc.get_default("NVDA")
    assert got is not None and got.id == row.id
    assert svc.params_of(got) == {"zone_atr": 0.4}

    assert await svc.get_default("MISSING") is None
    assert await svc.resolve_params("MISSING") == {}


async def test_only_one_default_per_symbol_strategy(session):
    svc = PresetService(session)
    first = await svc.save(symbol="AAPL", params={"zone_atr": 0.5})
    second = await svc.save(symbol="aapl", params={"zone_atr": 0.3}, source="optimizer")

    got = await svc.get_default("AAPL")
    assert got.id == second.id
    # the previous default was demoted, not deleted
    old = await svc.get(first.id)
    assert old is not None and not old.is_default


async def test_manual_edit_and_set_default(session):
    svc = PresetService(session)
    row = await svc.save(symbol="BTC", params={"emf_mode": "bonus"})
    updated = await svc.update(row.id, {"emf_mode": "require"})
    assert updated is not None
    assert svc.params_of(updated) == {"emf_mode": "require"}

    other = await svc.save(symbol="BTC", params={"emf_mode": "off"}, is_default=False)
    promoted = await svc.set_default(other.id)
    assert promoted is not None and promoted.is_default
    assert not (await svc.get(row.id)).is_default


async def test_bad_source_rejected(session):
    svc = PresetService(session)
    with pytest.raises(ValueError):
        await svc.save(symbol="X", source="magic")
    with pytest.raises(ValueError):
        await svc.save(symbol="  ", params={})


async def test_params_json_roundtrip(session):
    svc = PresetService(session)
    params = {
        "zone_atr": 0.5,
        "emf_mode": "require",
        "emf": {"atr_tp_mult": 2.5, "length_adl": 20},
        "options": {"enabled": True, "call_wall": 700.0},
        "momentum_period": 10,
    }
    row = await svc.save(symbol="SPY", params=params)
    stored = json.loads(row.params_json)
    assert stored == params  # nested blocks survive the round-trip


async def test_save_optimization_records_run_id(session):
    svc = PresetService(session)
    row = await svc.save_optimization(
        symbol="ETH", best_params={"min_confluence": 3},
        optimizer_run_id="run-42",
    )
    assert row.source == "optimizer"
    assert row.optimizer_run_id == "run-42"
    assert row.is_default
    # v2 fields: every save is a version; optimizer saves carry provenance
    assert row.version >= 1 and row.status == "backtest_only"
    assert row.backtest_ref == "optimizer:run-42"
    assert row.metrics_json == "{}"  # no metrics passed → nothing fabricated


async def test_save_optimization_snapshots_metrics_with_provenance(session):
    svc = PresetService(session)
    row = await svc.save_optimization(
        symbol="ETH", best_params={"min_confluence": 3},
        optimizer_run_id="run-43",
        metrics={"total_return": 0.12345, "sharpe": 1.23456, "max_drawdown": 0.05,
                 "win_rate": 0.55, "n_trades": 12},
        timeframe="1d",
    )
    assert json.loads(row.metrics_json) == {
        "total_return": 0.1235, "sharpe": 1.235, "max_drawdown": 0.05,
        "win_rate": 0.55, "n_trades": 12,
    }
    assert row.timeframe == "1d"


async def test_metrics_without_provenance_refused(session):
    svc = PresetService(session)
    with pytest.raises(ValueError):
        await svc.save_version(
            symbol="ETH", params={"zone_atr": 0.5},
            metrics={"total_return": 0.1, "sharpe": 1.0,
                     "max_drawdown": 0.02, "win_rate": 0.5, "n_trades": 5},
        )


async def test_versions_increment_per_group_and_demote_within_group(session):
    svc = PresetService(session)
    v1 = await svc.save(symbol="AAPL", params={"zone_atr": 0.5})
    v2 = await svc.save(symbol="aapl", params={"zone_atr": 0.4},
                        source="optimizer")
    assert (v1.version, v2.version) == (1, 2)
    assert not v1.is_default and v2.is_default  # group demotion, row kept

    # a *named* strategy is a separate group — independent versions, but the
    # same (symbol, strategy) pair
    named = await svc.save(symbol="AAPL", strategy_name="momentum",
                           params={"zone_atr": 0.9})
    assert named.version == 1
    versions = await svc.list_versions("AAPL", "trend_confluence_unified")
    assert [r.version for r in versions] == [2, 1]
    named_versions = await svc.list_versions(
        "AAPL", "trend_confluence_unified", "momentum")
    assert [r.id for r in named_versions] == [named.id]

    latest = await svc.latest_per_name(symbol="AAPL")
    assert {r.strategy_name for r in latest} == {"", "momentum"}


async def test_update_creates_new_version_not_in_place_edit(session):
    svc = PresetService(session)
    v1 = await svc.save(symbol="BTC", params={"emf_mode": "bonus"})
    v2 = await svc.update(v1.id, {"emf_mode": "require"}, notes="tightened")
    assert v2.id != v1.id and v2.version == 2
    assert v2.source == "manual" and v2.is_default
    old = await svc.get(v1.id)
    assert old is not None and not old.is_default
    assert svc.params_of(old) == {"emf_mode": "bonus"}  # history untouched


async def test_delete_guard_on_live_version(session):
    svc = PresetService(session)
    row = await svc.save(symbol="TSLA", params={"zone_atr": 0.5},
                         optimizer_run_id="run-9",
                         metrics={"total_return": 0.2, "sharpe": 1.5,
                                  "max_drawdown": 0.05, "win_rate": 0.6,
                                  "n_trades": 10})
    assert await svc.delete(row.id) is True

    live = await svc.save(symbol="TSLA", params={"zone_atr": 0.4},
                          optimizer_run_id="run-10",
                          metrics={"total_return": 0.2, "sharpe": 1.5,
                                   "max_drawdown": 0.05, "win_rate": 0.6,
                                   "n_trades": 10})
    promoted = await svc.promote(live.id)
    assert promoted.status == "live_enabled"
    with pytest.raises(ValueError):
        await svc.delete(live.id)
