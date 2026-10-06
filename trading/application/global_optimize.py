"""Global (all-tickers) optimization runner with progress tracking.

Runs :func:`~trading.application.backtest.optimize.optimize_strategy` for every
requested ticker, **sequentially and deterministically**, isolating per-ticker
failures: one bad ticker is recorded in ``errors`` and never aborts the run.
The best parameter set of each ticker is saved as its default preset
(source=``optimizer``, carrying the ``run_id`` for traceability).

The runner is an in-process asyncio background job: the HTTP layer starts it,
then the UI polls ``status()`` (current ticker, completed/failed counts, ETA).
State lives in a module-level registry guarded by ``asyncio.Lock`` — simple,
in-process, and enough for the single-worker deployment this platform targets.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from trading.application.backtest.engine import BacktestConfig
from trading.application.backtest.optimize import optimize_strategy
from trading.application.backtest.portfolio import TickerSpec, load_bars
from trading.application.cancellation import CancelToken, RunCancelled
from trading.application.instruments import load_instruments, select_universe
from trading.application.presets import PresetService

__all__ = ["GlobalOptimizeRunner", "global_optimize_runner"]

logger = logging.getLogger(__name__)

_MAX_RUNS_KEPT = 20


@dataclass
class GlobalRunState:
    run_id: str
    state: str = "running"  # running | done | failed | cancelled
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_symbol: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    cancel: CancelToken | None = None
    _durations: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        eta = None
        if self._durations and self.state == "running":
            remaining = self.total - self.completed - self.failed
            eta = round(remaining * (sum(self._durations) / len(self._durations)), 1)
        return {
            "run_id": self.run_id,
            "state": self.state,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "current_symbol": self.current_symbol,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "eta_seconds": eta,
            "results": self.results,
            "errors": self.errors,
        }


class GlobalOptimizeRunner:
    """Starts and tracks global optimization runs (one at a time is typical)."""

    def __init__(self) -> None:
        self._runs: dict[str, GlobalRunState] = {}
        self._lock = asyncio.Lock()

    # ── symbols ─────────────────────────────────────────────────────────
    @staticmethod
    def resolve_symbols(
        symbols: Sequence[str], category: str, n_tickers: int
    ) -> list[str]:
        """Explicit symbol list, or the whole ``category`` universe."""
        if symbols:
            seen: set[str] = set()
            out: list[str] = []
            for s in symbols:
                up = s.strip().upper()
                if up and up not in seen:
                    seen.add(up)
                    out.append(up)
            if not out:
                raise ValueError("no valid symbols given")
            return out
        if n_tickers > 0:
            return [i["symbol"].upper() for i in select_universe(category, n_tickers)]
        instruments = load_instruments()
        if category and category != "all":
            instruments = [i for i in instruments if i["category"] == category]
        seen: set[str] = set()
        out: list[str] = []
        for i in instruments:
            up = i["symbol"].upper()
            if up not in seen:
                seen.add(up)
                out.append(up)
        if not out:
            raise ValueError(f"no instruments in category '{category}'")
        return out

    # ── lifecycle ───────────────────────────────────────────────────────
    async def start(
        self,
        *,
        run_id: str,
        symbols: Sequence[str],
        strategy: str,
        base_params: Mapping[str, Any],
        grid: Mapping[str, Sequence[Any]] | None,
        objective: str,
        source: str,
        timeframe: str,
        limit: int,
        cfg: BacktestConfig,
        refresh: bool,
        save_preset: bool,
        session_factory,
        cancel: CancelToken,
        bars_by_symbol: Mapping[str, Sequence[Any]] | None = None,
    ) -> GlobalRunState:
        async with self._lock:
            self._prune()
            state = GlobalRunState(run_id=run_id, total=len(symbols), cancel=cancel)
            self._runs[run_id] = state
        asyncio.get_running_loop().create_task(
            self._job(
                state, symbols, strategy, base_params, grid, objective, source,
                timeframe, limit, cfg, refresh, save_preset, session_factory, cancel,
                bars_by_symbol=bars_by_symbol,
            )
        )
        return state

    def status(self, run_id: str) -> GlobalRunState | None:
        return self._runs.get(run_id)

    def cancel_run(self, run_id: str) -> bool:
        state = self._runs.get(run_id)
        if state is None or state.state != "running":
            return False
        if state.cancel is not None:
            state.cancel.cancel()
        return True

    def _prune(self) -> None:
        """Keep finished runs bounded (the UI polls the latest one)."""
        finished = [
            rid for rid, st in self._runs.items() if st.state != "running"
        ]
        for rid in finished[:-_MAX_RUNS_KEPT]:
            del self._runs[rid]

    # ── the job ────────────────────────────────────────────────────────
    async def _job(
        self,
        state: GlobalRunState,
        symbols: Sequence[str],
        strategy: str,
        base_params: Mapping[str, Any],
        grid: Mapping[str, Sequence[Any]] | None,
        objective: str,
        source: str,
        timeframe: str,
        limit: int,
        cfg: BacktestConfig,
        refresh: bool,
        save_preset: bool,
        session_factory,
        cancel: CancelToken,
        bars_by_symbol: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        for symbol in symbols:
            if state.state != "running":
                break
            state.current_symbol = symbol
            t0 = time.monotonic()
            try:
                cancel.check()
                injected = (bars_by_symbol or {}).get(symbol)
                if injected is not None:
                    bars = sorted(injected, key=lambda b: b.timestamp)
                else:
                    bars = sorted(
                        await load_bars(
                            TickerSpec(
                                symbol=symbol, source=source,
                                timeframe=timeframe, limit=limit,
                            ),
                            refresh=refresh,
                        ),
                        key=lambda b: b.timestamp,
                    )
                result = await asyncio.to_thread(
                    optimize_strategy,
                    strategy, symbol, bars,
                    base_params=base_params, grid=grid, cfg=cfg,
                    cancel=cancel, objective=objective,
                )
                if save_preset:
                    async with session_factory() as session:
                        await PresetService(session).save_optimization(
                            symbol=symbol,
                            strategy=strategy,
                            best_params=result.best_params,
                            optimizer_run_id=state.run_id,
                            # real metrics snapshot of the winning run (with
                            # its provenance) — the version is promotable
                            metrics=(result.best or {}).get("metrics") or None,
                            timeframe=timeframe,
                        )
                state.results.append({
                    "symbol": symbol,
                    "n_candidates": result.n_candidates,
                    "best_params": result.best_params,
                    "best": result.best,
                    "baseline": result.baseline,
                })
                state.completed += 1
                state._durations.append(time.monotonic() - t0)
            except RunCancelled:
                state.state = "cancelled"
                state.finished_at = datetime.now(timezone.utc)
                logger.info("global optimize %s cancelled at %s", state.run_id, symbol)
                return
            except Exception as exc:  # isolate: one bad ticker never sinks the run
                state.failed += 1
                state.errors.append({
                    "symbol": symbol,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                logger.exception("global optimize %s: ticker %s failed",
                                 state.run_id, symbol)
        if state.state == "running":
            state.state = "done"
            state.finished_at = datetime.now(timezone.utc)
        state.current_symbol = ""
        # Release the run token so a later cancel of the same id is a no-op
        # and the token doesn't linger in the registry's cancelled set.
        from trading.application.cancellation import run_registry

        run_registry.mark_finished(state.run_id)
        run_registry.clear(state.run_id)


#: Module-level singleton used by the API layer.
global_optimize_runner = GlobalOptimizeRunner()
