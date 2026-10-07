"""Per-ticker strategy-preset endpoints — the Strategy Hub REST surface.

The preset is the saved, complete parameter configuration for one ticker under
one strategy — created/validated through the backtest module (source of
truth), refined by the optimizer, editable by hand. Every save is a **new
version** of the group ``(symbol, strategy, strategy_name)``; promote/rollback
run the go-live gate before flipping deployment status, and a successful flip
hot-swaps a running live engine ticker (best-effort).

Route-order note: the literal paths (``/latest``, ``/versions``,
``/default/{symbol}``, ``/from-backtest/{symbol}``) are declared **before**
the ``/{preset_id}`` routes — FastAPI would otherwise let the int path param
shadow the literals and 422 on ``/presets/versions``.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.database import get_session
from trading.adapters.persistence.models import StrategyPresetRow
from trading.application.presets import (
    DEFAULT_STRATEGY,
    DEFAULT_STRATEGY_VERSION,
    PresetService,
    PresetValidationError,
)
from trading.domain import DataFetchError

from ..schemas import (
    PresetCreate,
    PresetFromBacktestRequest,
    PresetOut,
    PresetUpdate,
    PresetValidateOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/presets", tags=["presets"])

#: Gate failure → structured 422 payload.
_VALIDATION_FAILED = "validation_failed"


def _to_out(row: StrategyPresetRow) -> PresetOut:
    try:
        params = json.loads(row.params_json or "{}")
    except (TypeError, ValueError):
        params = {}
    try:
        metrics = json.loads(row.metrics_json or "{}")
    except (TypeError, ValueError):
        metrics = {}
    return PresetOut(
        id=row.id,
        symbol=row.symbol,
        strategy=row.strategy,
        strategy_version=row.strategy_version,
        strategy_name=row.strategy_name or "",
        version=row.version or 1,
        params=params if isinstance(params, dict) else {},
        timeframe=row.timeframe or "",
        metrics=metrics if isinstance(metrics, dict) else {},
        status=row.status or "backtest_only",
        backtest_ref=row.backtest_ref,
        source=row.source,
        optimizer_run_id=row.optimizer_run_id,
        is_default=bool(row.is_default),
        notes=row.notes or "",
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _svc(session: AsyncSession = Depends(get_session)) -> PresetService:
    return PresetService(session)


async def _hot_swap(symbol: str, strategy: str) -> None:
    """Best-effort live-engine reload after a promote/rollback/demote.

    Only acts when the engine is running the same strategy class; any failure
    is logged and swallowed — a store flip must never fail because of the
    engine (the next poll picks the new preset up anyway).
    """
    try:
        from trading.application.signal_engine import signal_engine

        if not signal_engine.running:
            return
        engine_strategy = signal_engine.strategy
        if engine_strategy and engine_strategy != strategy:
            return
        await signal_engine.reload_ticker(symbol)
    except Exception:
        logger.warning("hot-swap of %s/%s failed", symbol, strategy, exc_info=True)


def _not_found(preset_id: int) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"preset {preset_id} not found")


def _gate_failed(exc: PresetValidationError) -> HTTPException:
    return HTTPException(
        422,
        detail={"code": _VALIDATION_FAILED, "reasons": exc.reasons},
    )


@router.post("", response_model=PresetOut, status_code=status.HTTP_201_CREATED)
async def create_preset(body: PresetCreate, svc=Depends(_svc)) -> PresetOut:
    """Create the group's next version (never overwrites a previous one)."""
    try:
        row = await svc.save(
            symbol=body.symbol,
            strategy=body.strategy,
            strategy_version=body.strategy_version or DEFAULT_STRATEGY_VERSION,
            strategy_name=body.strategy_name,
            params=body.params,
            source=body.source,
            optimizer_run_id=body.optimizer_run_id,
            timeframe=body.timeframe,
            metrics=body.metrics,
            backtest_ref=body.backtest_ref,
            is_default=body.is_default,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_out(row)


@router.get("", response_model=list[PresetOut])
async def list_presets(
    symbol: str | None = None,
    strategy: str | None = None,
    name: str | None = None,
    svc=Depends(_svc),
) -> list[PresetOut]:
    """All versions (optionally filtered to one group via ``name``)."""
    rows = await svc.list(symbol=symbol, strategy=strategy, strategy_name=name)
    return [_to_out(r) for r in rows]


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preset_group(
    symbol: str = "",
    strategy: str = "",
    name: str = "",
    svc=Depends(_svc),
) -> None:
    """Delete a whole named saved strategy — **every** version of its group.

    The console's "Saved strategies" hub lists one row per named strategy (a
    group), so deleting the row must remove the group, not only its latest
    version (which would leave an older version to "reappear"). Atomic and
    concurrency-safe: 409 while any version is ``live_enabled`` (demote first),
    404 when the group is already empty (a concurrent delete won the race).
    """
    if not symbol or not strategy:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "both 'symbol' and 'strategy' query parameters are required",
        )
    try:
        deleted = await svc.delete_group(symbol, strategy, name)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not deleted:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no saved strategy {symbol}/{strategy}/{name or 'unnamed'}",
        )


@router.get("/latest", response_model=list[PresetOut])
async def latest_presets(
    symbol: str | None = None,
    strategy: str | None = None,
    svc=Depends(_svc),
) -> list[PresetOut]:
    """Latest version per ``(symbol, strategy, strategy_name)`` — the
    management list the console's "Saved strategies" panel renders."""
    rows = await svc.latest_per_name(symbol=symbol, strategy=strategy)
    return [_to_out(r) for r in rows]


@router.get("/versions", response_model=list[PresetOut])
async def list_versions(
    symbol: str = "",
    strategy: str = "",
    name: str = "",
    svc=Depends(_svc),
) -> list[PresetOut]:
    """Full version history of one group, newest first."""
    if not symbol or not strategy:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "both 'symbol' and 'strategy' query parameters are required",
        )
    rows = await svc.list_versions(symbol, strategy, name)
    return [_to_out(r) for r in rows]


@router.get("/default/{symbol}", response_model=PresetOut | None)
async def get_default_preset(
    symbol: str,
    strategy: str = DEFAULT_STRATEGY,
    name: str = "",
    svc=Depends(_svc),
) -> PresetOut | None:
    """The group's active version (``null`` when none has been saved yet)."""
    row = await svc.get_default(symbol, strategy, name)
    return _to_out(row) if row else None


@router.post("/from-backtest/{symbol}", response_model=PresetOut)
async def preset_from_backtest(
    symbol: str,
    body: PresetFromBacktestRequest,
    svc=Depends(_svc),
) -> PresetOut:
    """Run the backtest module for ``symbol`` and save the default preset.

    The backtest validates the parameter set on real data; its headline
    metrics are recorded in the preset notes for traceability.
    """
    try:
        row, _metrics = await svc.from_backtest(
            symbol=symbol,
            strategy=body.strategy,
            params=body.params,
            notes=body.notes,
        )
    except DataFetchError as exc:
        raise HTTPException(status.HTTP_502_BAD_REQUEST, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_out(row)


@router.get("/{preset_id}/validate", response_model=PresetValidateOut)
async def validate_preset(preset_id: int, svc=Depends(_svc)) -> PresetValidateOut:
    """Run the go-live gate; returns ``{ok, reasons: [{code, message}]}``."""
    if await svc.get(preset_id) is None:
        raise _not_found(preset_id)
    ok, reasons = await svc.validate_for_live(preset_id)
    return PresetValidateOut(ok=ok, reasons=reasons)


@router.post("/{preset_id}/promote", response_model=PresetOut)
async def promote_preset(preset_id: int, svc=Depends(_svc)) -> PresetOut:
    """Gate the version, then make it the (single) live one for its
    ``(symbol, strategy)`` and its group's active version."""
    row = await svc.get(preset_id)
    if row is None:
        raise _not_found(preset_id)
    try:
        row = await svc.promote(preset_id)
    except PresetValidationError as exc:
        raise _gate_failed(exc) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await _hot_swap(row.symbol, row.strategy)
    return _to_out(row)


@router.post("/{preset_id}/rollback", response_model=PresetOut)
async def rollback_preset(preset_id: int, svc=Depends(_svc)) -> PresetOut:
    """Re-activate a prior version — the gate re-runs (the store may have
    migrated since the version was saved)."""
    row = await svc.get(preset_id)
    if row is None:
        raise _not_found(preset_id)
    if row.is_default:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"preset {preset_id} is already the group's active version",
        )
    try:
        row = await svc.rollback(preset_id)
    except PresetValidationError as exc:
        raise _gate_failed(exc) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await _hot_swap(row.symbol, row.strategy)
    return _to_out(row)


@router.post("/{preset_id}/demote", response_model=PresetOut)
async def demote_preset(preset_id: int, svc=Depends(_svc)) -> PresetOut:
    """Flip a live version back to ``backtest_only`` (history kept)."""
    row = await svc.get(preset_id)
    if row is None:
        raise _not_found(preset_id)
    try:
        row = await svc.demote(preset_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await _hot_swap(row.symbol, row.strategy)
    return _to_out(row)


@router.patch("/{preset_id}", response_model=PresetOut)
async def update_preset(
    preset_id: int,
    body: PresetUpdate,
    svc=Depends(_svc),
) -> PresetOut:
    """Edit a preset — a ``params`` edit creates the group's next version."""
    if body.params is not None:
        row = await svc.update(preset_id, body.params, notes=body.notes)
    else:
        row = await svc.get(preset_id)
        if row is None:
            raise _not_found(preset_id)
    if row is None:
        raise _not_found(preset_id)
    if body.set_default:
        row = await svc.set_default(row.id)
        if row is None:
            raise _not_found(preset_id)
    return _to_out(row)


@router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preset(preset_id: int, svc=Depends(_svc)) -> None:
    """Delete one version; refused (409) while it is live_enabled."""
    try:
        deleted = await svc.delete(preset_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not deleted:
        raise _not_found(preset_id)
