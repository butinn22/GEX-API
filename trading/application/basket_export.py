"""Basket export: per-ticker validation + parameter-lossless payload assembly.

The export is the gate every basket must pass before it leaves the system —
either as a portable JSON payload (``POST /baskets/export``) or injected into
the live signal pipeline (``POST /baskets/deploy``). Two guarantees:

* **Diagnose everything.** Validation is per-ticker and never short-circuits:
  one broken ticker cannot hide another one's problem. All failures come back
  as ``{"code": "validation_failed", "errors": [{symbol, code, message}]}``
  with codes ``strategy_missing`` / ``params_missing`` / ``preset_not_found``
  / ``build_failed``.
* **Round-trip fidelity.** Every ticker's ``params`` are resolved
  server-side — from the preset store for preset-bound tickers, from the
  explicit request for the rest (no client echo, no silent default-filling).
  The params are made *fold-closed* (they carry the ergonomic ``fast/slow/
  period`` defaults exactly as the backtest engine folds them), so a payload
  POSTed back to ``/backtest/portfolio`` — with its provenance fields
  stripped — reproduces the exact same effective per-ticker params.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from trading.api.schemas import (
    BasketExportRequest,
    BasketExportResponse,
    BasketTickerIn,
    BasketTickerOut,
)
from trading.application.presets import PresetService
from trading.application.strategy_factory import build_strategy

__all__ = ["BasketValidationError", "validate_and_build", "EXPORT_SCHEMA_VERSION"]

EXPORT_SCHEMA_VERSION = "1"

#: Ergonomic keys ``TickerConfig.folded_params`` injects on re-import. An
#: exported params dict must already carry them (with the same defaults) or
#: the round-trip would grow keys the export never had. They match the
#: strategy-factory defaults for ``fast`` / ``slow``; strategies that read
#: ``period`` with a different default (momentum) always store it explicitly
#: in their presets, so the fallback never fires for a complete preset.
_FOLD_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("fast", 20),
    ("slow", 50),
    ("period", 20),
)


class BasketValidationError(ValueError):
    """One or more tickers failed export validation.

    ``errors`` is the structured per-ticker list
    ``[{"symbol", "code", "message"}]``; the API maps it to HTTP 422 with
    ``detail = {"code": "validation_failed", "errors": [...]}``.
    """

    def __init__(self, errors: list[dict[str, str]]) -> None:
        super().__init__(f"basket validation failed for {len(errors)} ticker(s)")
        self.code = "validation_failed"
        self.errors = list(errors)


class _TickerError(Exception):
    """A single ticker's validation failure (collected, never raised alone)."""

    def __init__(self, symbol: str, code: str, message: str) -> None:
        super().__init__(f"{symbol}: {code}: {message}")
        self.symbol = symbol
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "code": self.code, "message": self.message}


def _clean(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


def _fold_closed(params: dict[str, Any]) -> dict[str, Any]:
    """Add the ergonomic fold defaults the re-import path would inject."""
    out = dict(params)
    for key, default in _FOLD_DEFAULTS:
        out.setdefault(key, default)
    return out


async def validate_and_build(
    req: BasketExportRequest, presets: PresetService
) -> BasketExportResponse:
    """Validate every ticker, then assemble the export payload.

    Raises :class:`BasketValidationError` (per-ticker ``errors``) when any
    ticker fails; a failure never suppresses the diagnosis of the others.
    """
    errors: list[dict[str, str]] = []
    tickers: list[BasketTickerOut] = []
    for t in req.tickers:
        try:
            tickers.append(await _resolve_ticker(t, presets))
        except _TickerError as exc:
            errors.append(exc.as_dict())
    if errors:
        raise BasketValidationError(errors)

    costs = {
        "initial_cash": req.initial_cash,
        "fee_rate": req.fee_rate,
        "slippage": req.slippage,
        "position_fraction": req.position_fraction,
        "periods_per_year": req.periods_per_year,
    }
    return BasketExportResponse(
        schema_version=EXPORT_SCHEMA_VERSION,
        exported_at=datetime.now(timezone.utc),
        costs=costs,
        tickers=tickers,
    )


async def _resolve_ticker(
    t: BasketTickerIn, presets: PresetService
) -> BasketTickerOut:
    """Resolve one ticker against the preset store and dry-run its build."""
    warnings: list[str] = []
    row = None
    if t.preset_id is not None:
        row = await presets.get(int(t.preset_id))
        if row is None:
            raise _TickerError(
                t.symbol, "preset_not_found", f"preset {t.preset_id} not found"
            )

    if row is not None:
        # Preset-bound: the stored params are the baseline, explicit request
        # params still override per key (the existing override convention).
        strategy = row.strategy
        params = PresetService.params_of(row)
        overrides = dict(t.params or {})
        overrides.update(t.explicit_overrides())
        params.update({k: v for k, v in overrides.items() if v is not None})
        if overrides:
            warnings.append(
                "explicit request params override the preset's stored values"
            )
        assignment = "preset"
        optimized = bool(row.optimizer_run_id or row.backtest_ref)
        strategy_name = row.strategy_name or ""
        preset_source: str | None = row.source
        backtest_ref: str | None = row.backtest_ref
        preset_id: int | None = row.id
        preset_version: int | None = row.version
    else:
        # No preset: the explicit params are the only source. A ticker with
        # no strategy at all is unexportable; one with a strategy but no
        # params is unexportable too (never silently default-filled).
        strategy = (t.strategy or "").strip()
        if not strategy:
            raise _TickerError(
                t.symbol, "strategy_missing",
                "no saved strategy assigned and no strategy given",
            )
        explicit = _clean({**dict(t.params or {}), **t.explicit_overrides()})
        if not explicit:
            raise _TickerError(
                t.symbol, "params_missing",
                "no saved strategy assigned and no explicit params provided",
            )
        params = _clean(t.folded_params())
        assignment = "adhoc"
        optimized = False
        strategy_name = ""
        preset_source = None
        backtest_ref = None
        preset_id = None
        preset_version = None
        warnings.append(
            "no saved strategy assigned — exporting explicit params "
            "(not optimized)"
        )

    # Fold-close so the payload re-imported as a plain TickerConfig (with its
    # provenance stripped) reproduces the exact same effective params.
    params = _fold_closed(_clean(params))

    # Dry-run: the params must build a working strategy instance — an export
    # that cannot be replayed must never leave the system.
    try:
        build_strategy(strategy, t.symbol, params)
    except Exception as exc:  # StrategyError and friends
        raise _TickerError(
            t.symbol, "build_failed", f"{type(exc).__name__}: {exc}"
        ) from exc

    return BasketTickerOut(
        symbol=t.symbol,
        strategy=strategy,
        strategy_name=strategy_name,
        preset_id=preset_id,
        preset_version=preset_version,
        preset_source=preset_source,
        backtest_ref=backtest_ref,
        params=params,
        assignment=assignment,
        optimized=optimized,
        weight=t.weight,
        capital=t.capital,
        source=t.source,
        timeframe=t.timeframe,
        limit=t.limit,
        enabled=t.enabled,
        warnings=warnings,
    )
