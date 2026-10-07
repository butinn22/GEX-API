"""Bulk insert helper + task-result store."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import BacktestResultRow

__all__ = ["bulk_insert", "TaskResultStore"]


async def bulk_insert(session: AsyncSession, model, rows: list[dict]) -> int:
    """Insert many rows in one statement; returns the row count."""
    if not rows:
        return 0
    await session.execute(insert(model).values(rows))
    await session.commit()
    return len(rows)


class TaskResultStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_backtest(
        self, strategy: str, symbol: str, metrics: dict,
        trades: list[dict] | None = None,
    ) -> BacktestResultRow:
        row = BacktestResultRow(
            strategy=strategy, symbol=symbol, metrics_json=json.dumps(metrics),
            trades_json=json.dumps(trades or []),
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def recent(self, limit: int = 20) -> list[BacktestResultRow]:
        result = await self._session.execute(
            select(BacktestResultRow).order_by(BacktestResultRow.id.desc()).limit(limit)
        )
        return list(result.scalars())

    async def count(self) -> int:
        return int((await self._session.execute(select(func.count()).select_from(BacktestResultRow))).scalar())

    async def delete(self, result_id: int) -> bool:
        """Delete one stored backtest result; ``False`` when it does not exist."""
        row = await self._session.get(BacktestResultRow, result_id)
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.commit()
        return True

    async def purge(self, older_than_days: int | None = None) -> int:
        """Delete stored backtest results; when ``older_than_days`` is given,
        only rows older than that window are removed. Returns the count deleted."""
        stmt = delete(BacktestResultRow)
        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            stmt = stmt.where(BacktestResultRow.created_at < cutoff)
        result = await self._session.execute(stmt)
        await self._session.commit()
        return int(result.rowcount or 0)
