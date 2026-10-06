"""Basket export / deploy endpoints (authenticated).

The "complete basket transfer" workflow:

* ``POST /baskets/export`` — validate every ticker (per-ticker error
  diagnosis) and return a parameter-lossless payload that can be stored,
  shared, or POSTed back to ``/backtest/portfolio``;
* ``POST /baskets/deploy`` — validate, then create a signal key whose
  config **pins** the per-ticker strategies and params, so the live signal
  pipeline replays exactly what the basket was backtested with.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.database import get_session
from trading.application.basket_export import BasketValidationError, validate_and_build
from trading.application.presets import PresetService
from trading.application.signal_keys import SignalKeyError, SignalKeyService

from ..deps import require_auth
from ..schemas import (
    BasketDeployRequest,
    BasketDeployResponse,
    BasketExportRequest,
    BasketExportResponse,
)

router = APIRouter(
    prefix="/baskets",
    tags=["baskets"],
    dependencies=[Depends(require_auth)],
)


def _validation_failed(exc: BasketValidationError) -> HTTPException:
    return HTTPException(
        422,
        detail={"code": exc.code, "errors": exc.errors},
    )


@router.post("/export", response_model=BasketExportResponse)
async def export_basket(
    body: BasketExportRequest,
    session: AsyncSession = Depends(get_session),
) -> BasketExportResponse:
    """Validate a basket and return its parameter-lossless export payload.

    Every ticker is resolved server-side: preset-bound tickers take their
    complete params from the preset store, the others keep their explicit
    params (flagged with a non-optimized warning). Validation is per-ticker —
    failures return 422 with ``{code: "validation_failed", errors: [...]}``.
    """
    try:
        return await validate_and_build(body, PresetService(session))
    except BasketValidationError as exc:
        raise _validation_failed(exc) from exc


@router.post("/deploy", response_model=BasketDeployResponse)
async def deploy_basket(
    body: BasketDeployRequest,
    session: AsyncSession = Depends(get_session),
) -> BasketDeployResponse:
    """Validate a basket, then inject it into the live signal pipeline.

    The basket must pass the same export validation first. The created key's
    config pins each ticker's strategy + params — live generation replays the
    exact parameters the basket was backtested with, independent of later
    preset store updates (unlike the Deploy tab's follow-store semantics).
    """
    try:
        payload = await validate_and_build(body.basket, PresetService(session))
    except BasketValidationError as exc:
        raise _validation_failed(exc) from exc
    try:
        row, warnings = await SignalKeyService(session).create_from_basket(
            exchange=body.exchange, label=body.label, payload=payload
        )
    except SignalKeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    return BasketDeployResponse(
        id=row.id,
        key=row.key,
        exchange=row.exchange,
        label=row.label,
        warnings=warnings,
        dashboard=f"/API_KEY/{row.key}",
    )
