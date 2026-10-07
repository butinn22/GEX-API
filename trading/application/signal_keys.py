"""Signal API keys — key creation, lifecycle and signal generation.

A **signal key** is a unique, securely generated reference to a complete
strategy configuration for one broker: the strategy name + version, the
tickers (each with its own params or a preset reference), the data window and
the cost model. Creating one **requires** the broker to have been selected
first (``bingx`` | ``tbank``); the key then enables signal generation for the
selected tickers and powers the ``/API_KEY/{key}`` dashboard.

Generation is **idempotent**: every run re-executes the same portfolio
backtest engine over the configured lookback window on fresh data, then
replaces the key's derived rows (signals + paper/replay trades). There is no
incremental state to drift: the dashboard is always internally consistent
with the latest market data.
"""
from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.models import (
    KeySignalRow,
    KeyTradeRow,
    SignalKeyRow,
)
from trading.application.backtest.portfolio import (
    PortfolioBacktestConfig,
    TickerSpec,
    run_portfolio_backtest,
)
from trading.application.backtest.trade_log import TradeEvent
from trading.application.instruments import resolve_symbol
from trading.application.presets import PresetService

__all__ = ["SignalKeyService", "SignalKeyError", "VALID_EXCHANGES"]

logger = logging.getLogger(__name__)

VALID_EXCHANGES = ("bingx", "tbank")

#: Directly tradable categories per broker. Everything else (us / fx /
#: sectors) can still generate signals — they just have no order route on the
#: selected broker, which ``create`` reports as a warning.
_BROKER_CATEGORIES: dict[str, set[str]] = {
    "bingx": {"crypto"},
    "tbank": {"ru"},
}


class SignalKeyError(ValueError):
    """Raised for invalid key lifecycle operations (bad broker, revoked …)."""


def _new_key() -> str:
    return "sk_" + secrets.token_urlsafe(32)  # 46 chars, 256 bits of entropy


def _payload_parts(payload: Any) -> tuple[list[Any], dict[str, Any]]:
    """Extract ``(tickers, costs)`` from an export payload (object or dict)."""
    if isinstance(payload, Mapping):
        tickers = list(payload.get("tickers") or [])
        costs = dict(payload.get("costs") or {})
    else:
        tickers = list(getattr(payload, "tickers", None) or [])
        costs = dict(getattr(payload, "costs", None) or {})
    return tickers, costs


@dataclass
class KeySummary:
    """Cached performance snapshot for the dashboard (rebuilt on refresh)."""

    key_id: int
    generated_at: datetime
    metrics: dict[str, Any] = field(default_factory=dict)
    times: list[str] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    per_ticker: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "metrics": self.metrics,
            "times": self.times,
            "equity": self.equity,
            "per_ticker": self.per_ticker,
            "errors": self.errors,
        }


#: In-memory summaries, keyed by key id (single-worker deployment; a cold
#: cache simply regenerates on the next dashboard load).
_summary_cache: dict[int, KeySummary] = {}


def _pair_trades(events: Sequence[TradeEvent]) -> list[dict[str, Any]]:
    """Pair entry/add/exit events into closed trades with full metadata.

    ``entry_time`` / ``exit_time`` come from the event ledger (the engine's
    ``Trade`` rows stamp both sides with the exit bar), adds re-weight the
    average entry price exactly like the position object does.
    """
    out: list[dict[str, Any]] = []
    entry_ts = None
    entry_price = 0.0
    qty = 0.0
    direction = ""
    for ev in events:
        state = ev.state.value if hasattr(ev.state, "value") else str(ev.state)
        if state in ("long_entry", "short_entry"):
            entry_ts, entry_price, qty, direction = ev.timestamp, ev.price, ev.quantity, state.split("_")[0]
        elif state in ("long_add", "short_add") and qty > 0:
            entry_price = (entry_price * qty + ev.price * ev.quantity) / (qty + ev.quantity)
            qty += ev.quantity
        elif state in ("long_exit", "short_exit") and qty > 0:
            sign = 1.0 if direction == "long" else -1.0
            gross = (ev.price - entry_price) * qty * sign
            net = float(ev.realized_pnl or 0.0)
            out.append({
                "symbol": ev.symbol,
                "direction": direction,
                "entry_time": entry_ts,
                "exit_time": ev.timestamp,
                "entry_price": entry_price,
                "exit_price": ev.price,
                "quantity": qty,
                "gross_pnl": gross,
                "net_pnl": net,
                "pct_return": float(ev.pct_return or 0.0) if ev.pct_return is not None else (net / (entry_price * qty) if entry_price and qty else 0.0),
                "holding_seconds": (ev.timestamp - entry_ts).total_seconds() if entry_ts else 0.0,
                "exit_reason": ev.reason or "",
            })
            entry_ts, entry_price, qty, direction = None, 0.0, 0.0, ""
    return out


class SignalKeyService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── lifecycle ───────────────────────────────────────────────────────
    async def create(
        self,
        *,
        exchange: str,
        label: str = "",
        strategy: str = "trend_confluence_unified",
        strategy_version: str = "1.0.0",
        tickers: Sequence[str],
        params_by_ticker: Mapping[str, Mapping[str, Any]] | None = None,
        timeframe: str = "1d",
        source: str = "auto",
        limit: int = 1000,
        initial_cash: float = 100_000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        position_fraction: float = 0.95,
    ) -> tuple[SignalKeyRow, list[str]]:
        """Create a key for ``exchange`` referencing the full config.

        Per-ticker params default to the ticker's saved default preset when
        absent — the optimized presets flow straight into new keys.

        Returns ``(row, warnings)`` where warnings list tickers the broker
        cannot route orders for (they still receive signals).
        """
        if exchange not in VALID_EXCHANGES:
            raise SignalKeyError(
                f"exchange must be one of {VALID_EXCHANGES}, got {exchange!r} — "
                "select the API/broker provider before creating a key"
            )
        seen: set[str] = set()
        symbols: list[str] = []
        for t in tickers:
            up = str(t).strip().upper()
            if up and up not in seen:
                seen.add(up)
                symbols.append(up)
        if not symbols:
            raise SignalKeyError("at least one ticker is required")

        params_map = {str(k).upper(): dict(v) for k, v in (params_by_ticker or {}).items() if isinstance(v, Mapping)}

        warnings: list[str] = []
        ticker_configs: list[dict[str, Any]] = []
        for sym in symbols:
            params = params_map.get(sym)
            # Follow-live by default (Strategy Hub): no embedded params copy,
            # no pinned version — each generate() resolves the ticker's
            # live_enabled/active version from the store, so promote/rollback
            # propagate to the key without re-creating it. Explicit params,
            # when given, are pinned verbatim.
            ticker_configs.append({
                "symbol": sym,
                "params": params,  # None → resolved from the preset store
                "preset_id": None,
            })
            category = resolve_symbol(sym).get("category")
            if category not in _BROKER_CATEGORIES.get(exchange, set()):
                warnings.append(
                    f"{sym}: category '{category}' has no order route on "
                    f"{exchange} — signals will be generated but cannot be "
                    f"executed there"
                )

        config = {
            "strategy": strategy,
            "strategy_version": strategy_version,
            "timeframe": timeframe,
            "source": source,
            "limit": limit,
            "initial_cash": initial_cash,
            "fee_rate": fee_rate,
            "slippage": slippage,
            "position_fraction": position_fraction,
            "periods_per_year": 252,
            "tickers": ticker_configs,
        }
        row = SignalKeyRow(
            key=_new_key(),
            exchange=exchange,
            label=label or f"{strategy}:{','.join(symbols[:3])}",
            config_json=json.dumps(config, default=str),
            active=True,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row, warnings

    async def create_from_basket(
        self,
        *,
        exchange: str,
        label: str = "",
        payload: Any,
    ) -> tuple[SignalKeyRow, list[str]]:
        """Create a key from an exported basket payload (the Transfer flow).

        Unlike :meth:`create` — which follows the preset store so that
        promote/rollback propagate — the per-ticker ``params`` here are
        **pinned verbatim**: the live pipeline must replay exactly the
        parameters the basket was backtested/exported with, even after the
        underlying presets are re-optimized. Per-ticker ``strategy`` /
        ``source`` / ``timeframe`` / ``limit`` are stored alongside so
        :meth:`generate` can run each leg as configured.

        ``payload`` accepts a :class:`~trading.api.schemas.BasketExportResponse`
        or its dict form. Returns ``(row, warnings)`` like :meth:`create`.
        """
        if exchange not in VALID_EXCHANGES:
            raise SignalKeyError(
                f"exchange must be one of {VALID_EXCHANGES}, got {exchange!r} — "
                "select the API/broker provider before creating a key"
            )
        tickers_in, costs = _payload_parts(payload)

        warnings: list[str] = []
        ticker_configs: list[dict[str, Any]] = []
        strategies: list[str] = []
        for t in tickers_in:
            if isinstance(t, Mapping):
                sym = str(t.get("symbol", "")).strip().upper()
                strategy = str(t.get("strategy", "")).strip()
                params = dict(t.get("params") or {})
                preset_id = t.get("preset_id")
                source = str(t.get("source") or "auto")
                timeframe = str(t.get("timeframe") or "1d")
                limit = int(t.get("limit") or 1000)
                enabled = bool(t.get("enabled", True))
            else:
                sym = str(t.symbol).strip().upper()
                strategy = str(t.strategy or "").strip()
                params = dict(t.params or {})
                preset_id = t.preset_id
                source = str(t.source or "auto")
                timeframe = str(t.timeframe or "1d")
                limit = int(t.limit or 1000)
                enabled = bool(t.enabled)
            if not sym or not strategy:
                raise SignalKeyError(
                    f"ticker {sym or '?'}: strategy and params are required"
                )
            ticker_configs.append({
                "symbol": sym,
                "strategy": strategy,
                "params": params,  # pinned — never re-resolved from the store
                "preset_id": preset_id,
                "source": source,
                "timeframe": timeframe,
                "limit": limit,
                "enabled": enabled,
            })
            strategies.append(strategy)
            if not enabled:
                continue
            category = resolve_symbol(sym).get("category")
            if category not in _BROKER_CATEGORIES.get(exchange, set()):
                warnings.append(
                    f"{sym}: category '{category}' has no order route on "
                    f"{exchange} — signals will be generated but cannot be "
                    f"executed there"
                )
        if not ticker_configs:
            raise SignalKeyError("at least one ticker is required")

        head = ticker_configs[0]  # legacy top-level fallbacks for old readers
        config = {
            "strategy": strategies[0],
            "strategy_version": "1.0.0",
            "timeframe": head["timeframe"],
            "source": head["source"],
            "limit": head["limit"],
            "initial_cash": float(costs.get("initial_cash", 100_000.0)),
            "fee_rate": float(costs.get("fee_rate", 0.001)),
            "slippage": float(costs.get("slippage", 0.0005)),
            "position_fraction": float(costs.get("position_fraction", 0.95)),
            "periods_per_year": int(costs.get("periods_per_year", 252)),
            "tickers": ticker_configs,
        }
        row = SignalKeyRow(
            key=_new_key(),
            exchange=exchange,
            label=label or f"{strategies[0]}:{','.join(t['symbol'] for t in ticker_configs[:3])}",
            config_json=json.dumps(config, default=str),
            active=True,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row, warnings

    async def list(self) -> list[SignalKeyRow]:
        result = await self._session.execute(select(SignalKeyRow).order_by(SignalKeyRow.id))
        return list(result.scalars())

    async def get(self, key_id: int) -> SignalKeyRow | None:
        return await self._session.get(SignalKeyRow, key_id)

    async def get_by_key(self, key: str) -> SignalKeyRow | None:
        result = await self._session.execute(
            select(SignalKeyRow).where(SignalKeyRow.key == key)
        )
        return result.scalars().first()

    async def revoke(self, key_id: int) -> SignalKeyRow | None:
        """Soft-revoke: the key stops working but stays auditable."""
        row = await self.get(key_id)
        if row is None:
            return None
        row.active = False
        row.revoked_at = datetime.now(timezone.utc)
        await self._session.commit()
        await self._session.refresh(row)
        # Drop the cached summary so a revoked key can't keep serving a stale
        # dashboard from memory.
        _summary_cache.pop(key_id, None)
        return row

    async def delete_signals(self, key_id: int) -> int:
        """Hard-delete a key's generated ``key_signals`` rows.

        Returns the number deleted, or ``-1`` when the key does not exist (so
        the caller can answer 404 vs ``{"deleted": 0}``). Distinct from
        ``generate`` (which *replaces* rows) — this is the explicit purge the
        API previously lacked.
        """
        if await self.get(key_id) is None:
            return -1
        result = await self._session.execute(
            delete(KeySignalRow).where(KeySignalRow.key_id == key_id)
        )
        await self._session.commit()
        return int(result.rowcount or 0)

    def purge_cache(self, key_id: int | None = None) -> int:
        """Drop the in-process summary cache (one key, or every key when None)."""
        if key_id is None:
            count = len(_summary_cache)
            _summary_cache.clear()
            return count
        return 1 if _summary_cache.pop(key_id, None) is not None else 0

    async def set_active(self, key_id: int, active: bool) -> SignalKeyRow | None:
        row = await self.get(key_id)
        if row is None:
            return None
        if row.revoked_at is not None and active:
            raise SignalKeyError("key is revoked; create a new key instead")
        row.active = active
        await self._session.commit()
        await self._session.refresh(row)
        return row

    # ── signal generation ──────────────────────────────────────────────
    async def generate(
        self,
        *,
        key: str | SignalKeyRow,
        refresh: bool = True,
        bars_by_symbol: Mapping[str, Sequence[Any]] | None = None,
    ) -> dict[str, Any]:
        """(Re)generate signals + paper trades for a key. Idempotent.

        Runs the **same portfolio backtest engine** as the backtest module
        with the key's per-ticker configurations (params resolved from the
        saved presets when the key doesn't pin explicit params), then
        replaces the key's derived rows. Returns a report dict.
        """
        row = key if isinstance(key, SignalKeyRow) else await self.get_by_key(str(key))
        if row is None:
            raise SignalKeyError("signal key not found")
        if row.revoked_at is not None or not row.active:
            raise SignalKeyError("signal key is revoked or disabled")

        try:
            config = json.loads(row.config_json or "{}")
        except (TypeError, ValueError):
            raise SignalKeyError("signal key has a corrupted configuration") from None
        strategy = config.get("strategy", "trend_confluence_unified")
        version = config.get("strategy_version", "")
        presets = PresetService(self._session)

        specs: list[TickerSpec] = []
        preset_ids: dict[str, int | None] = {}
        for t in config.get("tickers", []):
            sym = str(t.get("symbol", "")).upper()
            # Per-ticker strategy with a legacy fallback: basket-deployed keys
            # pin one strategy per ticker, while old configs (and the Deploy
            # tab flow) carry a single key-level strategy — absent per-ticker
            # strategy falls back to it, so old keys need zero migration.
            strategy_t = str(t.get("strategy") or strategy)
            params = t.get("params")
            if not isinstance(params, Mapping):
                params = None
            if params is None:
                # Resolve fresh from the preset store so promote/rollback
                # propagate: pinned ``preset_id`` first, then the ticker's
                # live_enabled version, then the group default — the served
                # params are always byte-identical to a stored, backtested
                # params_json. With nothing saved at all the strategy's own
                # defaults apply (same as before the Strategy Hub).
                try:
                    deployable = await presets.get_deployable(
                        sym, strategy_t, preset_id=t.get("preset_id")
                    )
                    params = deployable["params"]
                    preset_ids[sym] = deployable["preset_id"]
                except ValueError:
                    logger.warning(
                        "no deployable preset for %s/%s — using strategy defaults",
                        sym, strategy_t,
                    )
                    params = {}
                    preset_ids[sym] = t.get("preset_id")
            else:
                preset_ids[sym] = t.get("preset_id")
            # source/timeframe/limit likewise allow per-ticker overrides
            # (basket keys); legacy configs fall back to the key-level values.
            specs.append(TickerSpec(
                symbol=sym,
                strategy=strategy_t,
                params=dict(params),
                source=str(t.get("source") or config.get("source", "auto")),
                timeframe=str(t.get("timeframe") or config.get("timeframe", "1d")),
                limit=int(t.get("limit") or config.get("limit", 1000)),
                enabled=bool(t.get("enabled", True)),
            ))
        if not specs:
            raise SignalKeyError("signal key has no tickers configured")

        cfg = PortfolioBacktestConfig(
            initial_cash=float(config.get("initial_cash", 100_000.0)),
            fee_rate=float(config.get("fee_rate", 0.001)),
            slippage=float(config.get("slippage", 0.0005)),
            position_fraction=float(config.get("position_fraction", 0.95)),
            periods_per_year=int(config.get("periods_per_year", 252)),
        )
        result = await run_portfolio_backtest(
            specs, cfg, bars_by_symbol=bars_by_symbol, refresh=refresh,
        )

        # Replace the derived rows atomically-ish: delete, insert, commit.
        await self._session.execute(delete(KeySignalRow).where(KeySignalRow.key_id == row.id))
        await self._session.execute(delete(KeyTradeRow).where(KeyTradeRow.key_id == row.id))

        n_signals = 0
        n_trades = 0
        for t in result.tickers:
            pid = preset_ids.get(t.symbol)
            for ev in t.result.events:
                self._session.add(KeySignalRow(
                    key_id=row.id,
                    symbol=ev.symbol,
                    side=ev.side.value if hasattr(ev.side, "value") else str(ev.side),
                    state=ev.state.value if hasattr(ev.state, "value") else str(ev.state),
                    reason=(ev.reason or "")[:255],
                    strength=1.0,
                    price=float(ev.price),
                    timestamp=ev.timestamp,
                    strategy=t.strategy,
                    strategy_version=version,
                    preset_id=pid,
                    source="live",
                ))
                n_signals += 1
            for tr in _pair_trades(t.result.events):
                self._session.add(KeyTradeRow(
                    key_id=row.id,
                    symbol=tr["symbol"],
                    direction=tr["direction"],
                    entry_time=tr["entry_time"],
                    exit_time=tr["exit_time"],
                    entry_price=tr["entry_price"],
                    exit_price=tr["exit_price"],
                    quantity=tr["quantity"],
                    fee=max(0.0, tr["gross_pnl"] - tr["net_pnl"]),
                    gross_pnl=tr["gross_pnl"],
                    net_pnl=tr["net_pnl"],
                    pct_return=tr["pct_return"],
                    holding_seconds=tr["holding_seconds"],
                    exit_reason=(tr["exit_reason"] or "")[:64],
                    strategy=t.strategy,
                    strategy_version=version,
                    preset_id=pid,
                    source="replay",
                ))
                n_trades += 1

        row.last_used_at = datetime.now(timezone.utc)
        await self._session.commit()

        m = result.metrics
        metrics = {
            "total_return": round(float(m.total_return), 4),
            "annualized_return": round(float(m.annualized_return), 4),
            "sharpe": round(float(m.sharpe), 3) if m.sharpe == m.sharpe else None,
            "max_drawdown": round(float(m.max_drawdown), 4),
            "win_rate": round(float(m.win_rate), 4),
            "profit_factor": round(float(m.profit_factor), 3) if m.profit_factor and m.profit_factor == m.profit_factor else None,
            "n_trades": n_trades,
            "n_periods": int(m.n_periods),
            "initial_cash": float(result.initial_cash),
            "final_equity": float(result.equity_curve[-1]) if len(result.equity_curve) else float(result.initial_cash),
        }
        step = max(1, len(result.times) // 1000)
        summary = KeySummary(
            key_id=row.id,
            generated_at=datetime.now(timezone.utc),
            metrics=metrics,
            times=[ts.isoformat() for ts in result.times[::step]],
            equity=[float(x) for x in result.equity_curve[::step]],
            per_ticker=[
                {
                    "symbol": t.symbol,
                    "strategy": t.strategy,
                    "total_return": round(t.total_return, 4),
                    "n_trades": len(t.result.trades),
                    "preset_id": preset_ids.get(t.symbol),
                }
                for t in result.tickers
            ],
            errors=[dict(e) for e in result.errors],
        )
        _summary_cache[row.id] = summary

        return {
            "key": row.key,
            "exchange": row.exchange,
            "generated_at": summary.generated_at,
            "n_signals": n_signals,
            "n_trades": n_trades,
            "metrics": metrics,
            "errors": summary.errors,
        }

    # ── dashboard reads ────────────────────────────────────────────────
    def summary(self, key_id: int) -> KeySummary | None:
        return _summary_cache.get(key_id)

    async def signals(self, key_id: int, *, limit: int = 100) -> list[KeySignalRow]:
        result = await self._session.execute(
            select(KeySignalRow)
            .where(KeySignalRow.key_id == key_id)
            .order_by(KeySignalRow.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars())

    async def trades(self, key_id: int) -> list[KeyTradeRow]:
        result = await self._session.execute(
            select(KeyTradeRow)
            .where(KeyTradeRow.key_id == key_id)
            .order_by(KeyTradeRow.exit_time)
        )
        return list(result.scalars())
