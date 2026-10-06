"""Signal API-key endpoints (authenticated): create, list, revoke, generate.

The workflow this router serves, in order:

1. optimize the unified strategy per ticker (presets saved),
2. **choose the broker** (``exchange`` — bingx | tbank),
3. **create the signal key** referencing the full optimized configuration,
4. generate signals for the key's tickers,
5. monitor everything on the ``/API_KEY/{key}`` dashboard.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.database import get_session
from trading.adapters.persistence.models import SignalKeyRow
from trading.application.signal_keys import SignalKeyError, SignalKeyService
from trading.domain import DataFetchError

from ..schemas import (
    SignalKeyCreate,
    SignalKeyGenerateReport,
    SignalKeyOut,
)
from ..deps import require_auth

router = APIRouter(
    prefix="/signal-keys",
    tags=["signal-keys"],
    dependencies=[Depends(require_auth)],
)


def _svc(session: AsyncSession = Depends(get_session)) -> SignalKeyService:
    return SignalKeyService(session)


def _to_out(row: SignalKeyRow) -> SignalKeyOut:
    try:
        config = json.loads(row.config_json or "{}")
    except (TypeError, ValueError):
        config = {}
    return SignalKeyOut(
        id=row.id,
        key=row.key,
        exchange=row.exchange,
        label=row.label,
        active=bool(row.active),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
        config=config if isinstance(config, dict) else {},
    )


@router.post("", response_model=SignalKeyOut, status_code=status.HTTP_201_CREATED)
async def create_key(
    body: SignalKeyCreate,
    svc=Depends(_svc),
) -> SignalKeyOut:
    """Create a unique signal key for the selected broker.

    The key references the complete strategy configuration: every ticker with
    its optimized params (or its saved default preset), the data window and
    the cost model. Tickers the broker cannot route orders for are accepted
    with a warning (they still receive signals).
    """
    try:
        row, _warnings = await svc.create(
            exchange=body.exchange,
            label=body.label,
            strategy=body.strategy,
            strategy_version=body.strategy_version,
            tickers=body.tickers,
            params_by_ticker=body.params_by_ticker,
            timeframe=body.timeframe,
            source=body.source,
            limit=body.limit,
            initial_cash=body.initial_cash,
            fee_rate=body.fee_rate,
            slippage=body.slippage,
            position_fraction=body.position_fraction,
        )
    except SignalKeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_out(row)


@router.get("", response_model=list[SignalKeyOut])
async def list_keys(svc=Depends(_svc)) -> list[SignalKeyOut]:
    return [_to_out(r) for r in await svc.list()]


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(key_id: int, svc=Depends(_svc)) -> None:
    """Revoke a key (soft delete — it stays auditable but stops working)."""
    if await svc.revoke(key_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "signal key not found")


@router.patch("/{key_id}", response_model=SignalKeyOut)
async def set_active(
    key_id: int,
    active: bool = True,
    svc=Depends(_svc),
) -> SignalKeyOut:
    """Enable/disable a key without revoking it."""
    try:
        row = await svc.set_active(key_id, active)
    except SignalKeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "signal key not found")
    return _to_out(row)


@router.post("/{key}/generate", response_model=SignalKeyGenerateReport)
async def generate_signals(
    key: str,
    svc=Depends(_svc),
    refresh: bool = True,
) -> SignalKeyGenerateReport:
    """(Re)generate signals for every ticker of this key.

    Idempotent: the key's derived rows are recomputed over the whole
    lookback window on the latest data. The dashboard's Refresh button and
    its auto-refresh both hit this endpoint.
    """
    from datetime import datetime, timezone

    try:
        report = await svc.generate(key=key, refresh=refresh)
    except SignalKeyError as exc:
        code = status.HTTP_404_NOT_FOUND if "not found" in str(exc) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(code, str(exc)) from exc
    except DataFetchError as exc:
        raise HTTPException(status.HTTP_502_BAD_REQUEST, f"data unavailable: {exc}") from exc
    return SignalKeyGenerateReport(**report)
