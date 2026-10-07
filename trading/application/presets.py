"""Per-ticker strategy-preset service — the Strategy Hub's write/read core.

A *preset* is the complete, saved parameter configuration for one ticker under
one strategy — the object the whole optimize → deploy workflow revolves around:

* the **backtest module is the source of truth** for the initial set
  (``from_backtest`` runs a real backtest with the current/default params and
  saves the configuration as the ticker's default preset);
* the **optimizer** refines it (``save_optimization`` records the winning
  params plus the ``optimizer_run_id`` for traceability);
* the user can edit (``save`` / ``update``), re-optimize and delete presets.

Strategy Hub (versioned store): every save — manual, backtest or optimizer —
creates a **new version** of the group ``(symbol, strategy, strategy_name)``
via :meth:`save_version` (the single write path for new versions). Old
versions are never mutated, so history is restorable and auditable.
``is_default`` marks the group's active version; ``status`` ∈
``backtest_only | live_enabled`` with **exactly one live row per
``(symbol, strategy)`` across all names** — flipping to live runs the go-live
gate (:meth:`validate_for_live`) first. Metrics snapshots are only ever
written from real backtest/optimize results: :meth:`save_version` refuses
``metrics`` without provenance (``optimizer_run_id`` or ``backtest_ref``).

The service is storage-agnostic (works on any ``AsyncSession``) and keeps the
JSON round-trip in one place so callers always see ``dict`` params.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.models import StrategyPresetRow
from trading.adapters.persistence.preset_repository import (
    STATUS_BACKTEST_ONLY,
    STATUS_LIVE_ENABLED,
    PresetRepository,
)

__all__ = [
    "PresetService",
    "PresetValidationError",
    "DEFAULT_STRATEGY",
    "DEFAULT_STRATEGY_VERSION",
]

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "trend_confluence_unified"
DEFAULT_STRATEGY_VERSION = "1.0.0"

#: Valid preset provenance values.
SOURCES = ("backtest", "manual", "optimizer")

#: Canonical metrics-snapshot keys (exact shape, never fabricated).
METRIC_KEYS = ("total_return", "sharpe", "max_drawdown", "win_rate", "n_trades")


class PresetValidationError(ValueError):
    """A preset failed the go-live validation gate.

    ``reasons`` is a structured list ``[{"code": …, "message": …}]`` (see
    :meth:`PresetService.validate_for_live` for the codes); the API maps it to
    HTTP 422 with ``detail = {"code": "validation_failed", "reasons": […]}``.
    """

    def __init__(self, message: str, *, reasons: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.code = "validation_failed"
        self.reasons = list(reasons or [])


def _dump(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True, default=str)


def _metrics_json(metrics: Mapping[str, Any] | None) -> str:
    """Serialize the headline metrics snapshot (canonical keys only)."""
    if not metrics:
        return "{}"
    out: dict[str, Any] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        # sharpe at 3 dp, the rest at 4 dp (same convention as the optimizer)
        out[key] = int(value) if key == "n_trades" else round(
            float(value), 3 if key == "sharpe" else 4
        )
    return json.dumps(out, sort_keys=True, default=str)[:1024]


class PresetService:
    """CRUD + workflow helpers for per-ticker parameter presets."""

    #: Deployment statuses (stored in ``strategy_presets.status``).
    STATUSES = (STATUS_BACKTEST_ONLY, STATUS_LIVE_ENABLED)

    def __init__(self, session: AsyncSession) -> None:
        self._repo = PresetRepository(session)

    # ── CRUD (versioned writes) ────────────────────────────────────────
    async def save(
        self,
        *,
        symbol: str,
        strategy: str = DEFAULT_STRATEGY,
        strategy_version: str = DEFAULT_STRATEGY_VERSION,
        params: Mapping[str, Any] | None = None,
        source: str = "manual",
        optimizer_run_id: str | None = None,
        is_default: bool = True,
        notes: str = "",
        strategy_name: str = "",
        timeframe: str = "",
        metrics: Mapping[str, Any] | None = None,
        backtest_ref: str | None = None,
    ) -> StrategyPresetRow:
        """Create a preset version; by default it becomes the group's default.

        Params are stored verbatim (the historical ``save`` contract — the
        round-trip preserves exactly what was given).
        """
        return await self.save_version(
            symbol=symbol,
            strategy=strategy,
            strategy_version=strategy_version,
            strategy_name=strategy_name,
            params=params,
            source=source,
            optimizer_run_id=optimizer_run_id,
            metrics=metrics,
            timeframe=timeframe,
            backtest_ref=backtest_ref,
            notes=notes,
            is_default=is_default,
            expand=False,
        )

    async def save_version(
        self,
        *,
        symbol: str,
        strategy: str = DEFAULT_STRATEGY,
        strategy_version: str = DEFAULT_STRATEGY_VERSION,
        strategy_name: str = "",
        params: Mapping[str, Any] | None = None,
        source: str = "manual",
        optimizer_run_id: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        timeframe: str = "",
        backtest_ref: str | None = None,
        notes: str = "",
        is_default: bool = True,
        expand: bool = True,
    ) -> StrategyPresetRow:
        """The single write path for new versions of a named strategy.

        Computes the group's next version, optionally expands partial params
        into the strategy's complete resolved set (``expand=True``, the
        optimizer convention), dumps the metrics snapshot (``{}`` unless a
        real result is supplied) and always stores ``status=backtest_only``
        (going live is a separate, gated action). When ``is_default=True``
        (the default) the group's previous active version is demoted.

        Anti-fabrication guard: ``metrics`` without provenance
        (``optimizer_run_id`` or ``backtest_ref``) is refused — headline
        numbers only ever come from a real run.
        """
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        if not symbol or not symbol.strip():
            raise ValueError("symbol is required")
        if metrics and not (optimizer_run_id or backtest_ref):
            raise ValueError(
                "metrics require provenance: pass optimizer_run_id or "
                "backtest_ref (metrics are never fabricated)"
            )
        name = (strategy_name or "").strip()[:64]
        if optimizer_run_id and not backtest_ref:
            backtest_ref = f"optimizer:{optimizer_run_id}"[:64]
        resolved = dict(params or {})
        if expand and resolved:
            resolved = self.full_params(strategy, symbol, resolved)
        version = await self._repo.next_version(symbol, strategy, name)
        return await self._repo.create(
            symbol=symbol,
            strategy=strategy,
            strategy_version=strategy_version,
            params_json=_dump(resolved),
            source=source,
            optimizer_run_id=optimizer_run_id,
            is_default=is_default,
            notes=(notes or "")[:250],
            strategy_name=name,
            version=version,
            timeframe=(timeframe or "")[:16],
            metrics_json=_metrics_json(metrics),
            status=STATUS_BACKTEST_ONLY,
            backtest_ref=backtest_ref,
        )

    async def list(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        strategy_name: str | None = None,
    ) -> list[StrategyPresetRow]:
        return await self._repo.list(
            symbol=symbol, strategy=strategy, strategy_name=strategy_name
        )

    async def get(self, preset_id: int) -> StrategyPresetRow | None:
        return await self._repo.get(preset_id)

    async def get_default(
        self, symbol: str, strategy: str = DEFAULT_STRATEGY, strategy_name: str = ""
    ) -> StrategyPresetRow | None:
        """The group's active version (legacy callers resolve the unnamed group)."""
        return await self._repo.get_default(symbol, strategy, strategy_name)

    async def list_versions(
        self, symbol: str, strategy: str, strategy_name: str = ""
    ) -> list[StrategyPresetRow]:
        """Full version history of one group, newest first."""
        return await self._repo.list_versions(symbol, strategy, strategy_name)

    async def latest_per_name(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> list[StrategyPresetRow]:
        """Latest version of every named strategy (the management list)."""
        return await self._repo.latest_per_name(symbol=symbol, strategy=strategy)

    async def update(
        self, preset_id: int, params: Mapping[str, Any], *, notes: str | None = None
    ) -> StrategyPresetRow | None:
        """Edit a preset's parameters by creating the group's **next version**.

        History never lies: a manual edit is a new ``source="manual"`` version
        (the old row stays untouched and restorable), and the new version
        becomes the group's active one.
        """
        row = await self._repo.get(preset_id)
        if row is None:
            return None
        return await self.save_version(
            symbol=row.symbol,
            strategy=row.strategy,
            strategy_version=row.strategy_version,
            strategy_name=row.strategy_name,
            params=params,
            source="manual",
            notes=notes if notes is not None else (row.notes or ""),
            is_default=True,
            expand=False,
        )

    async def set_default(self, preset_id: int) -> StrategyPresetRow | None:
        return await self._repo.set_default(preset_id)

    async def delete(self, preset_id: int) -> bool:
        """Delete a version; refuses (``ValueError``) while it is live_enabled."""
        return await self._repo.delete(preset_id)

    async def delete_group(
        self, symbol: str, strategy: str, strategy_name: str = ""
    ) -> int:
        """Delete a whole saved strategy (every version of one named group).

        Returns the number of versions removed (``0`` when the group is already
        empty); refuses (``ValueError``) while any version is live_enabled.
        """
        return await self._repo.delete_group(symbol, strategy, strategy_name)

    # ── params helpers ──────────────────────────────────────────────────
    @staticmethod
    def params_of(row: StrategyPresetRow) -> dict[str, Any]:
        """Parse a preset row's stored params (empty dict on bad JSON)."""
        try:
            out = json.loads(row.params_json or "{}")
        except (TypeError, ValueError):
            logger.warning("preset %s has unparseable params_json", row.id)
            return {}
        return out if isinstance(out, dict) else {}

    @staticmethod
    def metrics_of(row: StrategyPresetRow) -> dict[str, Any]:
        """Parse a preset row's metrics snapshot (empty dict on bad JSON)."""
        try:
            out = json.loads(row.metrics_json or "{}")
        except (TypeError, ValueError):
            return {}
        return out if isinstance(out, dict) else {}

    @staticmethod
    def full_params(
        strategy: str, symbol: str, overrides: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Expand partial ``overrides`` into the strategy's complete params.

        A saved preset must be the *complete* per-ticker configuration (the
        unified schema with every tunable, defaults filled in) — not just the
        handful of values the optimizer moved. Strategies that can snapshot
        themselves expose ``resolved_params()``; for the others the overrides
        are stored as given (their factory defaults live in code).
        """
        from trading.application.strategy_factory import build_strategy
        from trading.domain import StrategyError

        try:
            strat = build_strategy(strategy, symbol, overrides)
        except (StrategyError, TypeError, ValueError):
            # Unknown strategy or invalid combination — store as given; the
            # caller's validation (or the API) surfaces the real problem.
            return dict(overrides)
        snap = getattr(strat, "resolved_params", None)
        return dict(snap()) if callable(snap) else dict(overrides)

    async def resolve_params(
        self, symbol: str, strategy: str = DEFAULT_STRATEGY
    ) -> dict[str, Any]:
        """The ticker's default preset params, or ``{}`` when none saved yet."""
        row = await self.get_default(symbol, strategy)
        return self.params_of(row) if row else {}

    # ── go-live gate (F6) ───────────────────────────────────────────────
    async def validate_for_live(self, preset_id: int) -> tuple[bool, list[dict[str, str]]]:
        """Run the three go-live gates; returns ``(ok, reasons)``.

        Reasons are structured ``[{"code", "message"}]`` with codes
        ``params_missing`` / ``build_failed`` / ``no_backtest_evidence``.
        Raises ``ValueError`` when the preset does not exist.
        """
        row = await self._repo.get(preset_id)
        if row is None:
            raise ValueError(f"preset {preset_id} not found")

        reasons: list[dict[str, str]] = []

        # Gate 1 — params present and parseable.
        params = self.params_of(row)
        if not params:
            reasons.append({
                "code": "params_missing",
                "message": "params_json is empty or unparseable",
            })

        # Gate 2 — the params build a working strategy instance.
        try:
            from trading.application.strategy_factory import build_strategy

            build_strategy(row.strategy, row.symbol, params)
        except Exception as exc:  # StrategyError and friends
            message = f"{type(exc).__name__}: {exc}"
            if row.strategy == "trend_confluence_pine":
                # Friendlier hint: which schema keys the pine editor expects.
                try:
                    from trading.application.strategies.trend_confluence_pine import (
                        PINE_PARAM_NAMES,
                    )

                    missing = [k for k in PINE_PARAM_NAMES if k not in params]
                    if missing:
                        message += f" (missing pine keys: {', '.join(missing[:12])})"
                except Exception:  # pragma: no cover — hint is best-effort
                    pass
            reasons.append({"code": "build_failed", "message": message})

        # Gate 3 — linked to a real validating run.
        metrics = self.metrics_of(row)
        if not metrics:
            reasons.append({
                "code": "no_backtest_evidence",
                "message": "no metrics snapshot recorded for this version",
            })
        elif not (row.backtest_ref or row.optimizer_run_id):
            reasons.append({
                "code": "no_backtest_evidence",
                "message": (
                    "metrics snapshot lacks provenance "
                    "(backtest_ref / optimizer_run_id missing)"
                ),
            })

        return (not reasons, reasons)

    # ── deployment lifecycle: promote / rollback / demote ───────────────
    async def promote(self, preset_id: int) -> StrategyPresetRow:
        """Gate a version, then make it the (single) live one for its
        ``(symbol, strategy)`` — across all names — and its group's active
        version.

        Raises :class:`PresetValidationError` when the gate fails and
        ``ValueError`` when the version is already live (or missing).
        """
        row = await self._repo.get(preset_id)
        if row is None:
            raise ValueError(f"preset {preset_id} not found")
        ok, reasons = await self.validate_for_live(preset_id)
        if not ok:
            raise PresetValidationError(
                f"preset {preset_id} failed the go-live gate", reasons=reasons
            )
        if row.status == STATUS_LIVE_ENABLED:
            raise ValueError(f"preset {preset_id} is already live_enabled")
        # One live per (symbol, strategy) across all names (Q3).
        await self._repo.demote_live(row.symbol, row.strategy)
        await self._repo.set_status(preset_id, STATUS_LIVE_ENABLED)
        return await self._repo.set_default(preset_id)

    async def rollback(self, preset_id: int) -> StrategyPresetRow:
        """Re-activate a prior version — same gated path as promote.

        The gate re-runs (Q2): params were valid when the version was saved,
        but the store may have migrated since. The router distinguishes
        rollback from promote only for UX (409 when the id already is the
        group's active version).
        """
        return await self.promote(preset_id)

    async def demote(self, preset_id: int) -> StrategyPresetRow:
        """Flip a live version back to ``backtest_only`` (history kept)."""
        row = await self._repo.get(preset_id)
        if row is None:
            raise ValueError(f"preset {preset_id} not found")
        if row.status != STATUS_LIVE_ENABLED:
            raise ValueError(f"preset {preset_id} is not live_enabled")
        return await self._repo.set_status(preset_id, STATUS_BACKTEST_ONLY)

    # ── deployable resolution (the only live-path reader) ───────────────
    async def get_deployable(
        self, symbol: str, strategy: str, preset_id: int | None = None
    ) -> dict[str, Any]:
        """Resolve the params the live paths (engine, signal keys) must serve.

        Resolution order: pinned ``preset_id`` → live_enabled row for
        ``(symbol, strategy)`` → the unnamed group's default. Either way the
        returned params are byte-identical to the stored, backtested
        ``params_json``. Raises ``ValueError`` when nothing resolves.
        """
        row: StrategyPresetRow | None = None
        if preset_id is not None:
            row = await self._repo.get(int(preset_id))
            if row is None:
                raise ValueError(f"preset {preset_id} not found")
        else:
            row = await self._repo.get_live(symbol, strategy)
            if row is None:
                row = await self._repo.get_default(symbol, strategy)
        if row is None:
            raise ValueError(
                f"no deployable preset for {symbol.upper()}/{strategy} "
                "(promote a version or save one first)"
            )
        return {
            "preset_id": row.id,
            "params": self.params_of(row),
            "version": row.version or 1,
            "strategy": row.strategy,
            "symbol": row.symbol,
            "timeframe": row.timeframe or "",
            "status": row.status or STATUS_BACKTEST_ONLY,
        }

    # ── workflow: backtest as the source of truth ──────────────────────
    async def from_backtest(
        self,
        *,
        symbol: str,
        strategy: str = DEFAULT_STRATEGY,
        params: Mapping[str, Any] | None = None,
        validate: bool = True,
        notes: str = "",
    ) -> tuple[StrategyPresetRow, dict[str, Any]]:
        """Validate params via a real backtest run and save the default preset.

        Runs the backtest module over ``symbol`` with ``params`` (falling back
        to the ticker's existing default, then the strategy defaults) and
        stores the configuration together with the backtest's headline
        metrics in ``notes`` — the backtest is the source of truth for the
        initial parameter set of every ticker.

        Returns ``(preset_row, metrics)``; raises ``ValueError``/``DataFetchError``
        from the backtest path when the ticker has no data.
        """
        from trading.application.backtest.engine import BacktestConfig, run_backtest
        from trading.application.backtest.portfolio import TickerSpec, load_bars
        from trading.application.strategy_factory import build_strategy
        from trading.domain import StrategyError

        merged = dict(await self.resolve_params(symbol, strategy))
        if params:
            merged.update(dict(params))
        if validate:
            spec = TickerSpec(symbol=symbol, strategy=strategy, params=merged)
            bars = sorted(
                await load_bars(spec), key=lambda b: b.timestamp
            )
            if len(bars) < 2:
                raise ValueError(f"no data for '{symbol}' ({len(bars)} bars)")
            try:
                strategy_obj = build_strategy(strategy, symbol, merged)
            except StrategyError as exc:
                raise ValueError(str(exc)) from exc
            result = await run_backtest(strategy_obj, bars, BacktestConfig())
            m = result.metrics
            metrics = {
                "total_return": round(m.total_return, 4),
                "sharpe": round(m.sharpe, 3),
                "max_drawdown": round(m.max_drawdown, 4),
                "win_rate": round(m.win_rate, 4),
                "n_trades": len(result.trades),
            }
            # Snapshot the strategy's *complete* resolved params (defaults
            # filled in) — the stored preset is self-contained.
            snap = getattr(strategy_obj, "resolved_params", None)
            merged = dict(snap()) if callable(snap) else merged
        else:
            metrics = {}
            merged = self.full_params(strategy, symbol, merged)
        note = notes or (
            f"backtest: return={metrics.get('total_return')}, "
            f"sharpe={metrics.get('sharpe')}, trades={metrics.get('n_trades')}"
        )
        row = await self.save(
            symbol=symbol,
            strategy=strategy,
            params=merged,
            source="backtest",
            is_default=True,
            notes=note[:250],
        )
        return row, metrics

    # ── workflow: optimizer output ─────────────────────────────────────
    async def save_optimization(
        self,
        *,
        symbol: str,
        strategy: str = DEFAULT_STRATEGY,
        strategy_version: str = DEFAULT_STRATEGY_VERSION,
        best_params: Mapping[str, Any],
        optimizer_run_id: str | None = None,
        notes: str = "",
        metrics: Mapping[str, Any] | None = None,
        strategy_name: str = "",
        timeframe: str = "",
        backtest_ref: str | None = None,
    ) -> StrategyPresetRow:
        """Persist an optimizer winner as the group's **next version**.

        The winner's (partial) overrides are expanded into the strategy's
        complete parameter set before saving, so the stored preset is a
        self-contained per-ticker configuration. When ``metrics`` (the
        optimizer's headline snapshot of the winning run) is supplied the
        version carries it together with its provenance
        (``backtest_ref = "optimizer:<run_token>"``).
        """
        return await self.save_version(
            symbol=symbol,
            strategy=strategy,
            strategy_version=strategy_version,
            strategy_name=strategy_name,
            params=best_params,
            source="optimizer",
            optimizer_run_id=optimizer_run_id,
            metrics=metrics,
            timeframe=timeframe,
            backtest_ref=backtest_ref,
            notes=notes,
            is_default=True,
            expand=True,
        )
