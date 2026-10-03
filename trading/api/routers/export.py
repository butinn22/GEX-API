"""Trade export endpoints (CSV / XLSX) — backtest and live trades."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.database import get_session
from trading.adapters.persistence.models import BacktestResultRow, OrderRow
from trading.application.backtest.trade_log import TradeEvent
from trading.application.reporting.trade_export import (
    events_from_order_rows,
    events_to_csv,
    events_to_xlsx,
)

from ..deps import require_auth

router = APIRouter(
    prefix="/export",
    tags=["export"],
    dependencies=[Depends(require_auth)],
)

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _csv_response(events: list[TradeEvent], filename: str) -> Response:
    return Response(
        content=events_to_csv(events),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _xlsx_response(events: list[TradeEvent], filename: str) -> Response:
    return Response(
        content=events_to_xlsx(events),
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _backtest_events(session: AsyncSession, result_id: int) -> list[TradeEvent]:
    row = await session.get(BacktestResultRow, result_id)
    if row is None:
        raise HTTPException(404, "backtest result not found")
    return [TradeEvent.from_dict(d) for d in json.loads(row.trades_json or "[]")]


async def _live_events(session: AsyncSession) -> list[TradeEvent]:
    result = await session.execute(select(OrderRow).order_by(OrderRow.created_at))
    return events_from_order_rows(list(result.scalars()))


@router.get("/backtest/{result_id}/trades.csv")
async def export_backtest_csv(
    result_id: int, session: AsyncSession = Depends(get_session)
) -> Response:
    events = await _backtest_events(session, result_id)
    return _csv_response(events, f"backtest_{result_id}_trades.csv")


@router.get("/backtest/{result_id}/trades.xlsx")
async def export_backtest_xlsx(
    result_id: int, session: AsyncSession = Depends(get_session)
) -> Response:
    events = await _backtest_events(session, result_id)
    return _xlsx_response(events, f"backtest_{result_id}_trades.xlsx")


@router.get("/live-trades.csv")
async def export_live_csv(session: AsyncSession = Depends(get_session)) -> Response:
    return _csv_response(await _live_events(session), "live_trades.csv")


@router.get("/live-trades.xlsx")
async def export_live_xlsx(session: AsyncSession = Depends(get_session)) -> Response:
    return _xlsx_response(await _live_events(session), "live_trades.xlsx")
