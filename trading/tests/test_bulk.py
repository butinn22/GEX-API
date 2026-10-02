"""Tests for bulk insert and the task-result store."""
from __future__ import annotations

import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.bulk import TaskResultStore, bulk_insert
from trading.adapters.persistence.models import BacktestResultRow


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(BacktestResultRow))
        await s.commit()
        yield s


async def test_bulk_insert(session):
    n = await bulk_insert(session, BacktestResultRow, [
        {"strategy": "s", "symbol": "X", "metrics_json": "{}"},
        {"strategy": "s", "symbol": "Y", "metrics_json": "{}"},
    ])
    assert n == 2
    assert await bulk_insert(session, BacktestResultRow, []) == 0


async def test_task_result_store(session):
    store = TaskResultStore(session)
    await store.save_backtest("sma", "X", {"sharpe": 1.0})
    await store.save_backtest("sma", "Y", {"sharpe": 2.0})
    assert await store.count() == 2
    recent = await store.recent()
    assert recent[0].symbol == "Y"  # newest first
