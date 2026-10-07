"""Real-time signal engine — poll bars, evaluate a strategy, publish + persist.

This is the piece that turns the platform's strategy library into a *live
signal service*:

::

    POST /api/v1/signals/engine/start  {"symbols": [...], "strategy": ...}
      → one asyncio task per ticker
      → fetch bars from the real venue (auto-detected per ticker) on a poll
      → feed only the bars not seen yet (dedup by bar timestamp)
      → the strategy computes a full trade plan
      → persist the signal row (+ open or close its position row)
      → publish the plan to /ws/signals and to every subscribed /ws/client

Design notes
------------
* **Real data only.** Symbols are resolved through
  :func:`trading.application.instruments.resolve_symbol`, so a crypto ticker
  goes to Bybit, a RU name to MOEX and everything else to yFinance (with a
  fallback order per venue). The generative feed is never used here — this is
  the production path.
* **One strategy instance per ticker**, so each ticker owns at most one
  position and the engine can read ``strategy.position`` directly.
* **State is started once**, not per poll — a strategy that resets on ``start()``
  must not be re-started every poll interval or it would forget its position.
* **DB writes are resilient.** A venue, strategy or database failure on one
  ticker is recorded in that ticker's status and the loop keeps polling; a
  failure never takes the engine (or the other tickers) down.
* **Nothing is fabricated.** Only what the strategy emitted is written; PnL for
  a closed position is computed from the entry/exit prices the strategy signed,
  and an open position exports with realised PnL left empty.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from trading.adapters.fetchers.registry import FetcherRegistry
from trading.adapters.persistence.models import KeySignalRow, SignalPositionRow
from trading.application.audit import AuditLog
from trading.application.instruments import resolve_symbol
from trading.application.signal_hub import signal_hub
from trading.application.strategies.confluence_breakout import (
    BREAKOUT_PRESETS,
    DEFAULT_PRESET,
)
from trading.domain import Bar, Exchange, Signal
from trading.observability import STRATEGY_SIGNALS
from trading.ports import Strategy

__all__ = ["SignalEngineConfig", "SignalEngine", "signal_engine"]


#: Venue (from ``resolve_symbol``) → the order bars are fetched from.
_SOURCE_ORDER: dict[str, tuple[str, ...]] = {
    "bybit": ("bybit", "yfinance"),
    "moex": ("moex", "yfinance"),
    "yfinance": ("yfinance",),
    "synthetic": ("yfinance", "bybit"),
}
_DEFAULT_ORDER = ("yfinance", "bybit")

#: Bar intervals the engine accepts.
TIMEFRAMES = ("1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w")

#: Exit reasons the confluence-breakout family can emit (a signal carrying one
#: is a close, not an entry). Used to classify the row without guessing.
_EXIT_REASONS = ("stop_loss", "trailing_stop", "time_stop", "structure_break",
                 "teeth_break", "jaw_break", "lips_break", "ma_exit")


@dataclass
class SignalEngineConfig:
    """Everything needed to start a live signal run.

    ``params`` is forwarded verbatim to the strategy factory, so the request may
    carry a frozen ``preset`` (``alligator_4h`` / ``donchian_1d``), an explicit
    parameter set, or both — the explicit keys win over the preset.
    ``params_by_symbol`` layers per-ticker params on top (a ticker's entry
    wins over the global ``params`` on every key it sets), and ``preset_ids``
    records which saved strategy version each ticker runs — the id is stamped
    on every signal row the engine persists (Strategy Hub provenance).
    """

    symbols: list[str] = field(default_factory=list)
    strategy: str = "confluence_breakout"
    preset: str | None = DEFAULT_PRESET
    timeframe: str = "4h"
    source: str = "auto"                # auto | bybit | moex | yfinance
    params: dict[str, Any] = field(default_factory=dict)
    #: per-ticker params — merged over ``params`` for that symbol only
    params_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: per-ticker saved-strategy-version ids (recorded on the signal rows)
    preset_ids: dict[str, int] = field(default_factory=dict)
    poll_seconds: float = 60.0
    bars: int = 500
    key_id: int | None = None
    strategy_version: str = ""
    initial_equity: float = 100_000.0
    notify_local_clients: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> SignalEngineConfig:
        raw = dict(raw or {})
        symbols = raw.pop("symbols", None) or raw.pop("tickers", None) or []
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        params = raw.pop("params", None) or {}
        if not isinstance(params, Mapping):
            raise ValueError("params must be an object")
        pbs = raw.pop("params_by_symbol", None) or {}
        if not isinstance(pbs, Mapping) or not all(
            isinstance(v, Mapping) for v in pbs.values()
        ):
            raise ValueError("params_by_symbol must map SYMBOL → params object")
        pids = raw.pop("preset_ids", None) or {}
        if not isinstance(pids, Mapping):
            raise ValueError("preset_ids must map SYMBOL → preset id")
        try:
            preset_ids = {str(k).upper(): int(v) for k, v in pids.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("preset_ids must map SYMBOL → preset id") from exc
        return cls(
            symbols=[str(s).upper() for s in symbols],
            strategy=str(raw.pop("strategy", "confluence_breakout")),
            preset=raw.pop("preset", DEFAULT_PRESET),
            timeframe=str(raw.pop("timeframe", "4h")),
            source=str(raw.pop("source", "auto")),
            params=dict(params),
            params_by_symbol={str(k).upper(): dict(v) for k, v in pbs.items()},
            preset_ids=preset_ids,
            poll_seconds=float(raw.pop("poll_seconds", 60.0)),
            bars=int(raw.pop("bars", 500)),
            key_id=raw.pop("key_id", None),
            strategy_version=str(raw.pop("strategy_version", "")),
            initial_equity=float(raw.pop("initial_equity", 100_000.0)),
            notify_local_clients=bool(raw.pop("notify_local_clients", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "strategy": self.strategy,
            "preset": self.preset,
            "timeframe": self.timeframe,
            "source": self.source,
            "params": dict(self.params),
            "params_by_symbol": {k: dict(v) for k, v in self.params_by_symbol.items()},
            "preset_ids": dict(self.preset_ids),
            "poll_seconds": self.poll_seconds,
            "bars": self.bars,
            "key_id": self.key_id,
            "strategy_version": self.strategy_version,
            "initial_equity": self.initial_equity,
            "notify_local_clients": self.notify_local_clients,
        }


class _TickerState:
    """Per-ticker run bookkeeping (survives across polls)."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.source = ""
        self.fetch_symbol = symbol
        self.strategy: Strategy | None = None
        self.task: asyncio.Task | None = None
        #: saved strategy version this ticker runs (stamped on signal rows;
        #: ``None`` when the engine was started with free-form params)
        self.preset_id: int | None = None
        self.seen: set[datetime] = set()
        self.last_bar: datetime | None = None
        self.last_error = ""
        self.polls = 0
        self.signals = 0
        self.started_at = datetime.now(UTC)
        self.running = False
        self.position_open = False
        #: The first poll has to replay the whole fetch window so the strategy
        #: can warm its indicators up; the signals it produces describe bars
        #: that closed *before* the engine started, so they are stored as
        #: ``backfill`` (auditable, clearly distinguishable from live ones)
        #: while still establishing the correct position state.
        self.first_poll = True

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol": self.symbol,
            "running": self.running,
            "source": self.source,
            "fetch_symbol": self.fetch_symbol,
            "preset_id": self.preset_id,
            "last_bar": self.last_bar.isoformat() if self.last_bar else None,
            "bars_seen": len(self.seen),
            "polls": self.polls,
            "signals": self.signals,
            "last_error": self.last_error,
            "started_at": self.started_at.isoformat(),
            "position_open": self.position_open,
        }
        if self.strategy is not None and hasattr(self.strategy, "diagnostics"):
            out["strategy"] = self.strategy.diagnostics()
        return out


class SignalEngine:
    """Live signal service: one background task per configured ticker."""

    #: Mirrors the local-client subscription cap so a run can never wedge the
    #: dispatcher.
    MAX_TICKERS = 20

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        registry: FetcherRegistry | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._registry = registry
        self._audit = audit
        self._tickers: dict[str, _TickerState] = {}
        self._config: SignalEngineConfig | None = None
        self._task: asyncio.Task | None = None
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._error = ""

    # ── lifecycle ───────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def strategy(self) -> str | None:
        """The strategy class the engine is running (``None`` when stopped)."""
        return self._config.strategy if self._config else None

    def status(self) -> dict[str, Any]:
        tickers = [t.status() for t in self._tickers.values()]
        return {
            "running": self.running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "stopped_at": self._stopped_at.isoformat() if self._stopped_at else None,
            "config": self._config.to_dict() if self._config else None,
            "n_tickers": len(tickers),
            "n_running": sum(1 for t in tickers if t["running"]),
            "tickers": tickers,
            "hub_subscribers": signal_hub.subscriber_count,
            "error": self._error,
            "presets": dict(BREAKOUT_PRESETS),
        }

    async def start(self, config: SignalEngineConfig) -> dict[str, Any]:
        """Start (or restart) the engine. Raises :class:`ValueError` on bad input."""
        if self.running:
            raise RuntimeError("signal engine is already running — stop it first")
        config.symbols = [s for s in dict.fromkeys(config.symbols) if s]
        if not config.symbols:
            raise ValueError("at least one symbol is required")
        if len(config.symbols) > self.MAX_TICKERS:
            raise ValueError(f"at most {self.MAX_TICKERS} tickers per run")
        if config.timeframe not in TIMEFRAMES:
            raise ValueError(f"unsupported timeframe {config.timeframe!r} (known: {', '.join(TIMEFRAMES)})")
        if config.poll_seconds < 5:
            raise ValueError("poll_seconds must be >= 5 — venues rate-limit us")
        if config.bars < 250:
            raise ValueError("bars must be >= 250 (strategy warm-up)")

        self._config = config
        self._error = ""
        self._tickers = {}
        for symbol in config.symbols:
            state = _TickerState(symbol)
            state.preset_id = config.preset_ids.get(symbol)
            try:
                info = resolve_symbol(symbol)
                state.source = info["exchange"]
                state.fetch_symbol = info["fetch_symbol"]
                state.strategy = self._build_strategy(symbol)
            except Exception as exc:  # bad symbol or bad params: refuse to boot
                self._error = f"{symbol}: {exc}"
                raise ValueError(f"{symbol}: {exc}") from exc
            self._tickers[symbol] = state

        self._started_at = datetime.now(UTC)
        self._stopped_at = None
        self._task = asyncio.create_task(self._supervise())
        return self.status()

    async def stop(self) -> dict[str, Any]:
        tasks = [t.task for t in self._tickers.values() if t.task is not None]
        for t in self._tickers.values():
            t.running = False
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for t in self._tickers.values():
            t.task = None
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._stopped_at = datetime.now(UTC)
        return self.status()

    def _build_strategy(self, symbol: str) -> Strategy:
        cfg = self._config
        assert cfg is not None
        from trading.application.strategy_factory import build_strategy

        params = dict(cfg.params)
        # per-ticker params win over the global set (Strategy Hub F3/Q1)
        per_symbol = cfg.params_by_symbol.get(symbol)
        if per_symbol:
            params.update(per_symbol)
        params.setdefault("timeframe", cfg.timeframe)
        if cfg.preset:
            params.setdefault("preset", cfg.preset)
        return build_strategy(cfg.strategy, symbol, params)

    async def reload_ticker(self, symbol: str) -> bool:
        """Hot-swap one ticker's strategy from the preset store (best-effort).

        Resolves the ticker's deployable version for the engine's strategy
        class (``PresetService.get_deployable`` — live_enabled first, then the
        group default), rebuilds the strategy, restarts it and sets
        ``first_poll = True`` so the next poll replays the whole fetch window
        and re-establishes indicator/position state (the engine's existing
        warm-up mechanism — no new machinery).

        Returns ``False`` when the engine is stopped, the ticker is not
        tracked, or nothing deployable exists — in the last case the ticker's
        task is stopped with ``last_error="no deployable preset"`` so it never
        signals with stale parameters. A build failure is recorded and leaves
        the previous strategy running.
        """
        cfg = self._config
        sym = str(symbol).upper()
        state = self._tickers.get(sym)
        if cfg is None or not self.running or state is None:
            return False
        from trading.application.presets import PresetService

        try:
            async with self._session() as session:
                deployable = await PresetService(session).get_deployable(
                    sym, cfg.strategy
                )
        except ValueError:
            state.last_error = "no deployable preset"
            if state.task is not None and not state.task.done():
                state.task.cancel()
            state.running = False
            return False

        from trading.application.strategy_factory import build_strategy

        params = dict(deployable.get("params") or {})
        params.setdefault("timeframe", cfg.timeframe)
        if cfg.preset:
            params.setdefault("preset", cfg.preset)
        try:
            strategy = build_strategy(cfg.strategy, sym, params)
        except Exception as exc:  # keep the previous strategy serving
            state.last_error = f"reload failed: {type(exc).__name__}: {exc}"
            return False
        await strategy.start()
        state.strategy = strategy
        state.preset_id = deployable.get("preset_id")
        cfg.preset_ids[sym] = deployable.get("preset_id")
        state.seen.clear()       # replay the fetch window …
        state.first_poll = True  # … so indicator/position state warms up again
        state.last_error = ""
        return True

    # ── supervision ─────────────────────────────────────────────────────
    async def _supervise(self) -> None:
        for state in self._tickers.values():
            state.task = asyncio.create_task(self._run_ticker(state))
            state.running = True
        try:
            await asyncio.gather(*(t.task for t in self._tickers.values()),
                                 return_exceptions=True)
        finally:
            for t in self._tickers.values():
                t.running = False

    async def _run_ticker(self, state: _TickerState) -> None:
        cfg = self._config
        assert cfg is not None and state.strategy is not None
        # ``start()`` resets strategy state, so it is called exactly once —
        # re-starting every poll would make the strategy forget its position.
        await state.strategy.start()
        while True:
            failed = False
            try:
                bars = await self._fetch_bars(state)
                state.polls += 1
                fresh = [b for b in bars if b.timestamp not in state.seen]
                for b in fresh:
                    state.seen.add(b.timestamp)
                if fresh:
                    state.last_bar = fresh[-1].timestamp
                if len(state.seen) > 2000:  # keep memory bounded on long runs
                    for ts in sorted(state.seen)[:1000]:
                        state.seen.discard(ts)
                for bar in fresh:
                    for sig in await state.strategy.on_bar(bar):
                        state.signals += 1
                        state.position_open = self._is_entry(sig)
                        await self._handle_signal(state, sig, backfill=state.first_poll)
                state.first_poll = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # venue/DB hiccup: report, retry next poll
                failed = True
                state.last_error = f"{type(exc).__name__}: {exc}"
            if not failed:
                # Only clear an error when *this* poll succeeded end-to-end —
                # a poll that fetched bars fine but failed to persist a signal
                # must keep the reason visible in the status.
                state.last_error = ""
            await asyncio.sleep(cfg.poll_seconds)

    @staticmethod
    def _is_entry(sig: Signal) -> bool:
        """True when ``sig`` opens a position (an exit closes one)."""
        return (sig.meta or {}).get("exit_reason") is None

    def _source_order(self, state: _TickerState) -> tuple[str, ...]:
        cfg = self._config
        assert cfg is not None
        if cfg.source and cfg.source != "auto":
            return (cfg.source,)
        return _SOURCE_ORDER.get(state.source, _DEFAULT_ORDER)

    async def _fetch_bars(self, state: _TickerState) -> list[Bar]:
        from trading.adapters.fetchers.registry import loop_registry

        registry = self._registry or loop_registry()
        order = tuple(s for s in self._source_order(state) if s in {e.value for e in Exchange})
        return await registry.get_ohlcv(
            [Exchange(s) for s in order], state.fetch_symbol,
            self._config.timeframe, limit=self._config.bars,  # type: ignore[union-attr]
        )

    # ── signal handling ─────────────────────────────────────────────────
    async def _handle_signal(self, state: _TickerState, sig: Signal,
                             *, backfill: bool = False) -> None:
        cfg = self._config
        assert cfg is not None
        meta = dict(sig.meta or {})
        exit_reason = meta.get("exit_reason")
        is_exit = exit_reason is not None
        if is_exit:
            # a long exit is a SELL, a short exit is a BUY
            state_name = "long_exit" if sig.side.value == "sell" else "short_exit"
        else:
            state_name = "long_entry" if sig.side.value == "buy" else "short_entry"

        async with self._session() as session:
            session.add(KeySignalRow(
                key_id=cfg.key_id,
                symbol=sig.symbol,
                side=sig.side.value,
                state=state_name,
                reason=(sig.reason or "")[:255],
                strength=float(sig.strength),
                price=float(meta.get("exit_price") or sig.entry_price or 0.0),
                timestamp=sig.timestamp,
                strategy=sig.strategy,
                strategy_version=cfg.strategy_version,
                preset_id=state.preset_id,
                source="backfill" if backfill else "live",
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                position_size=sig.position_size,
                risk_pct=sig.risk_pct,
                risk_amount=sig.risk_amount,
                timeframe=sig.timeframe,
                bar_time=sig.bar_time,
                indicators_json=json.dumps(meta, default=str)[:4000],
            ))
            if is_exit:
                await self._close_position(session, state, sig, meta)
            else:
                await self._open_position(session, state, sig, meta)
            await session.commit()

        self._publish(state, sig, meta, state_name, exit_reason, backfill)
        STRATEGY_SIGNALS.labels(strategy=sig.strategy, side=sig.side.value).inc()
        if self._audit is not None:
            self._audit.record("live_signal", actor=sig.strategy, symbol=sig.symbol,
                               side=sig.side.value, reason=sig.reason)

    async def _open_position(self, session: AsyncSession, state: _TickerState,
                             sig: Signal, meta: Mapping[str, Any]) -> None:
        cfg = self._config
        assert cfg is not None
        entry = float(sig.entry_price or 0.0)
        stop = float(sig.stop_loss or 0.0)
        risk_amount = float(sig.risk_amount or 0.0)
        stop_dist = abs(entry - stop)
        qty = risk_amount / stop_dist if stop_dist > 0 else 0.0
        session.add(SignalPositionRow(
            key_id=cfg.key_id,
            symbol=sig.symbol,
            strategy=sig.strategy,
            strategy_version=cfg.strategy_version,
            preset=str(meta.get("preset") or cfg.preset or ""),
            timeframe=sig.timeframe or cfg.timeframe,
            source="live",
            side="long" if sig.side.value == "buy" else "short",
            status="open",
            entry_time=sig.bar_time or sig.timestamp,
            entry_price=entry,
            quantity=float(qty),
            initial_stop=stop,
            stop_price=stop,
            take_profit=sig.take_profit,
            trail_price=stop,
            best_price=entry,
            worst_price=entry,
            risk_amount=risk_amount,
            risk_pct=sig.risk_pct,
            mark_price=entry,
        ))

    async def _close_position(self, session: AsyncSession, state: _TickerState,
                              sig: Signal, meta: Mapping[str, Any]) -> None:
        row = await self._open_row(session, sig.symbol)
        if row is None:
            return
        exit_price = float(meta.get("exit_price") or 0.0)
        qty = float(row.quantity or 0.0)
        sign = 1.0 if row.side == "long" else -1.0
        gross = (exit_price - row.entry_price) * qty * sign if (qty and row.entry_price) else 0.0
        risk = abs(float(row.risk_amount or 0.0))
        row.exit_time = sig.bar_time or sig.timestamp
        row.exit_price = exit_price
        row.exit_reason = str(meta.get("exit_reason") or sig.reason or "")[:64]
        row.status = "closed"
        row.mark_price = exit_price
        row.trail_price = float(meta.get("trail") or row.stop_price or 0.0)
        row.bars_held = int(meta.get("bars_held") or 0)
        row.mfe_r = float(meta.get("mfe_r") or 0.0)
        row.gross_pnl = gross
        row.net_pnl = gross
        row.unrealised_pnl = 0.0
        row.pnl_r = (gross / risk) if risk > 0 else None
        row.pct_return = (gross / (row.entry_price * qty) * 100.0) if (qty and row.entry_price) else None

    async def _open_row(self, session: AsyncSession, symbol: str) -> SignalPositionRow | None:
        result = await session.execute(
            select(SignalPositionRow)
            .where(SignalPositionRow.symbol == symbol, SignalPositionRow.status == "open")
            .order_by(SignalPositionRow.id.desc())
        )
        return result.scalars().first()

    def _publish(self, state: _TickerState, sig: Signal, meta: Mapping[str, Any],
                 state_name: str, exit_reason: str | None, backfill: bool = False) -> None:
        cfg = self._config
        assert cfg is not None
        if not cfg.notify_local_clients:
            return  # persistence still happens; only the fan-out is suppressed
        payload = sig.plan_dict()
        payload.update({
            "state": state_name,
            "preset": meta.get("preset"),
            "exit_reason": exit_reason,
            "source": "backfill" if backfill else "live",
            "venue": state.source,
            "event": ("backfill_position_closed" if exit_reason else "backfill_position_opened")
            if backfill else
            ("position_closed" if exit_reason else "position_opened"),
        })
        signal_hub.publish(payload)

    # ── session access ──────────────────────────────────────────────────
    def _session(self):
        if self._session_factory is None:
            from trading.adapters.persistence import database

            self._session_factory = database.session_factory()
        return self._session_factory()

    # ── reads ───────────────────────────────────────────────────────────
    async def list_signals(self, *, symbol: str | None = None, side: str | None = None,
                           state: str | None = None, strategy: str | None = None,
                           limit: int = 200, offset: int = 0) -> list[KeySignalRow]:
        async with self._session() as session:
            stmt = select(KeySignalRow)
            if symbol:
                stmt = stmt.where(KeySignalRow.symbol == symbol.upper())
            if side:
                stmt = stmt.where(KeySignalRow.side == side.lower())
            if state:
                stmt = stmt.where(KeySignalRow.state == state.lower())
            if strategy:
                stmt = stmt.where(KeySignalRow.strategy == strategy)
            stmt = stmt.order_by(KeySignalRow.timestamp.desc()).offset(offset).limit(limit)
            return list((await session.execute(stmt)).scalars())

    async def list_positions(self, *, symbol: str | None = None, status: str | None = None,
                             strategy: str | None = None, limit: int = 500,
                             offset: int = 0) -> list[SignalPositionRow]:
        async with self._session() as session:
            stmt = select(SignalPositionRow)
            if symbol:
                stmt = stmt.where(SignalPositionRow.symbol == symbol.upper())
            if status:
                stmt = stmt.where(SignalPositionRow.status == status.lower())
            if strategy:
                stmt = stmt.where(SignalPositionRow.strategy == strategy)
            stmt = stmt.order_by(SignalPositionRow.id.desc()).offset(offset).limit(limit)
            return list((await session.execute(stmt)).scalars())

    async def positions_stats(self) -> dict[str, Any]:
        """Aggregate over the persisted position ledger."""
        async with self._session() as session:
            out: dict[str, Any] = {"open": 0, "closed": 0, "n_closed": 0,
                                   "total_pnl": 0.0, "total_pnl_r": 0.0, "wins": 0, "losses": 0}
            rows = (await session.execute(
                select(SignalPositionRow.status, func.count())
                .group_by(SignalPositionRow.status)
            )).all()
            for status, count in rows:
                if status in out:
                    out[status] = int(count)
            closed = (await session.execute(
                select(SignalPositionRow)
                .where(SignalPositionRow.status == "closed")
                .order_by(SignalPositionRow.id)
            )).scalars()
            wins = losses = 0
            total = 0.0
            total_r = 0.0
            n_r = 0
            for row in closed:
                pnl = float(row.net_pnl or 0.0)
                total += pnl
                if row.pnl_r is not None:
                    total_r += float(row.pnl_r)
                    n_r += 1
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
            out["n_closed"] = wins + losses
            out["wins"], out["losses"] = wins, losses
            out["total_pnl"] = round(total, 4)
            out["total_pnl_r"] = round(total_r, 4)
            out["win_rate"] = round(wins / (wins + losses), 4) if wins + losses else None
            return out

    async def clear_positions(self, *, only_open: bool = False) -> int:
        async with self._session() as session:
            stmt = delete(SignalPositionRow)
            if only_open:
                stmt = stmt.where(SignalPositionRow.status == "open")
            result = await session.execute(stmt)
            await session.commit()
            return int(result.rowcount or 0)


#: Process-wide singleton — the FastAPI lifespan starts/stops the local-client
#: dispatcher; the signal router drives this engine.
signal_engine = SignalEngine()
