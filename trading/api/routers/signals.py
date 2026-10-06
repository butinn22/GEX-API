"""Live signal endpoints — run the signal engine, read signals, export them.

Routes
------
``POST   /api/v1/signals/engine/start``   start a live run (symbols → signals)
``POST   /api/v1/signals/engine/stop``    stop the running engine
``GET    /api/v1/signals/engine``         engine status (per-ticker diagnostics)
``GET    /api/v1/signals``                persisted signals (filterable)
``GET    /api/v1/signals/positions``      position ledger (open + closed)
``GET    /api/v1/signals/stats``          aggregate PnL / win rate over the ledger
``GET    /api/v1/signals/export/signals.csv|signals.xlsx``
``GET    /api/v1/signals/export/positions.csv|positions.xlsx``

Signals are also streamed live on ``WS /ws/signals`` (the same payload the
exports publish) and pushed to every subscribed ``WS /ws/client`` session.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from trading.adapters.persistence.models import KeySignalRow, SignalPositionRow
from trading.application.reporting.signal_export import (
    signals_to_csv,
    signals_to_xlsx,
    positions_to_csv,
    positions_to_xlsx,
)
from trading.application.signal_engine import SignalEngineConfig, signal_engine

from ..deps import require_auth

router = APIRouter(
    prefix="/signals",
    tags=["signals"],
    dependencies=[Depends(require_auth)],
)

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _download(content: str | bytes, filename: str, mime: str) -> Response:
    return Response(
        content=content, media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/engine/start")
async def engine_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Start the live signal engine for ``symbols``.

    Body: :class:`~trading.application.signal_engine.SignalEngineConfig`
    (``symbols``, ``strategy``, ``preset``, ``timeframe``, ``params``, …).
    """
    try:
        config = SignalEngineConfig.from_dict(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        return await signal_engine.start(config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/engine/stop")
async def engine_stop() -> dict[str, Any]:
    return await signal_engine.stop()


@router.get("/engine")
async def engine_status() -> dict[str, Any]:
    return signal_engine.status()


@router.get("")
async def list_signals(
    symbol: str | None = Query(default=None),
    side: str | None = Query(default=None, pattern="^(buy|sell)$"),
    state: str | None = Query(default=None),
    strategy: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    rows = await signal_engine.list_signals(
        symbol=symbol, side=side, state=state, strategy=strategy,
        limit=limit, offset=offset,
    )
    return [_signal_out(r) for r in rows]


@router.get("/positions")
async def list_positions(
    symbol: str | None = Query(default=None),
    status: str | None = Query(default=None, pattern="^(open|closed)$"),
    strategy: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    rows = await signal_engine.list_positions(
        symbol=symbol, status=status, strategy=strategy, limit=limit, offset=offset,
    )
    return [_position_out(r) for r in rows]


@router.get("/stats")
async def positions_stats() -> dict[str, Any]:
    return await signal_engine.positions_stats()


@router.delete("/positions")
async def clear_positions(only_open: bool = Query(default=False)) -> dict[str, Any]:
    deleted = await signal_engine.clear_positions(only_open=only_open)
    return {"deleted": deleted}


# ── exports ───────────────────────────────────────────────────────────
@router.get("/export/signals.csv")
async def export_signals_csv(
    symbol: str | None = Query(default=None),
    side: str | None = Query(default=None),
    state: str | None = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=100000),
) -> Response:
    rows = await signal_engine.list_signals(symbol=symbol, side=side, state=state, limit=limit)
    name = f"signals{symbol and f'_{symbol}' or ''}.csv"
    return _download(signals_to_csv(rows), name, "text/csv")


@router.get("/export/signals.xlsx")
async def export_signals_xlsx(
    symbol: str | None = Query(default=None),
    side: str | None = Query(default=None),
    state: str | None = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=100000),
) -> Response:
    rows = await signal_engine.list_signals(symbol=symbol, side=side, state=state, limit=limit)
    name = f"signals{symbol and f'_{symbol}' or ''}.xlsx"
    return _download(signals_to_xlsx(rows), name, _XLSX_MIME)


@router.get("/export/positions.csv")
async def export_positions_csv(
    symbol: str | None = Query(default=None),
    status: str | None = Query(default=None, pattern="^(open|closed)$"),
    limit: int = Query(default=10000, ge=1, le=100000),
) -> Response:
    rows = await signal_engine.list_positions(symbol=symbol, status=status, limit=limit)
    name = f"positions{symbol and f'_{symbol}' or ''}.csv"
    return _download(positions_to_csv(rows), name, "text/csv")


@router.get("/export/positions.xlsx")
async def export_positions_xlsx(
    symbol: str | None = Query(default=None),
    status: str | None = Query(default=None, pattern="^(open|closed)$"),
    limit: int = Query(default=10000, ge=1, le=100000),
) -> Response:
    rows = await signal_engine.list_positions(symbol=symbol, status=status, limit=limit)
    name = f"positions{symbol and f'_{symbol}' or ''}.xlsx"
    return _download(positions_to_xlsx(rows), name, _XLSX_MIME)


# ``GET /api/v1/signals`` note: ``trading/api/routers/portfolio.py`` declared an
# unauthenticated ``GET /signals`` stub on this same path (that router carries no
# prefix), which shadowed this endpoint and made the ledger look permanently
# empty. The stub is gone — keep it that way.


# ── serialisation ─────────────────────────────────────────────────────
def _signal_out(row: KeySignalRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "key_id": row.key_id,
        "symbol": row.symbol,
        "side": row.side,
        "state": row.state,
        "reason": row.reason,
        "strength": row.strength,
        "price": row.price,
        "entry_price": row.entry_price,
        "stop_loss": row.stop_loss,
        "take_profit": row.take_profit,
        "position_size": row.position_size,
        "risk_pct": row.risk_pct,
        "risk_amount": row.risk_amount,
        "timeframe": row.timeframe,
        "strategy": row.strategy,
        "strategy_version": row.strategy_version,
        "source": row.source,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "bar_time": row.bar_time.isoformat() if row.bar_time else None,
    }


def _position_out(row: SignalPositionRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "key_id": row.key_id,
        "symbol": row.symbol,
        "side": row.side,
        "status": row.status,
        "strategy": row.strategy,
        "strategy_version": row.strategy_version,
        "preset": row.preset,
        "timeframe": row.timeframe,
        "source": row.source,
        "entry_time": row.entry_time.isoformat() if row.entry_time else None,
        "entry_price": row.entry_price,
        "quantity": row.quantity,
        "initial_stop": row.initial_stop,
        "stop_price": row.stop_price,
        "take_profit": row.take_profit,
        "trail_price": row.trail_price,
        "best_price": row.best_price,
        "worst_price": row.worst_price,
        "mfe_r": row.mfe_r,
        "bars_held": row.bars_held,
        "exit_time": row.exit_time.isoformat() if row.exit_time else None,
        "exit_price": row.exit_price,
        "exit_reason": row.exit_reason,
        "risk_amount": row.risk_amount,
        "risk_pct": row.risk_pct,
        "gross_pnl": row.gross_pnl,
        "net_pnl": row.net_pnl,
        "pnl_r": row.pnl_r,
        "pct_return": row.pct_return,
        "unrealised_pnl": row.unrealised_pnl,
        "mark_price": row.mark_price,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
