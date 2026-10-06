"""SQLAlchemy models (async). API keys are stored encrypted at rest."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ApiKeyRow(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(16), index=True)
    label: Mapped[str] = mapped_column(String(64), default="")
    api_key_encrypted: Mapped[str] = mapped_column(String(1024))
    api_secret_encrypted: Mapped[str] = mapped_column(String(1024))
    extra_json: Mapped[str] = mapped_column(String(1024), default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    exchange: Mapped[str] = mapped_column(String(16), index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[float]
    order_type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    limit_price: Mapped[float | None]
    stop_price: Mapped[float | None]
    filled_quantity: Mapped[float] = mapped_column(default=0.0)
    strategy: Mapped[str | None]
    reason: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class BacktestResultRow(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32))
    metrics_json: Mapped[str] = mapped_column(String(4096))
    #: Granular per-fill trade events (see backtest.trade_log), JSON list.
    trades_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class StrategyPresetRow(Base):
    """One saved parameter preset for a (symbol, strategy) pair.

    The preset is the per-ticker default configuration the backtest module
    produces and the optimizer refines; at most one row per pair carries
    ``is_default=True`` (enforced by the repository, not a DB constraint, so
    non-default history can accumulate freely).

    Strategy Hub (versioned store): a row is one **version** of a named
    strategy for one ticker. The *group key* is
    ``(symbol, strategy, strategy_name)`` — ``strategy_name`` is the
    user-facing name (``''`` = the legacy/unnamed strategy). Every save
    inserts a new row with ``version = max(version) + 1``; rows are never
    mutated in place (auditability). ``is_default`` marks the group's active
    version, ``status`` ∈ ``backtest_only | live_enabled`` — exactly one row
    per ``(symbol, strategy)`` may be live_enabled, across all names.
    ``metrics_json`` is a headline snapshot written **only** from real
    backtest/optimize results; ``backtest_ref`` records the validating run
    (``"optimizer:<run_token>"`` / ``"backtest:<result_id>"``).
    """

    __tablename__ = "strategy_presets"
    __table_args__ = (
        Index(
            "uq_strategy_presets_group_version",
            "symbol", "strategy", "strategy_name", "version",
            unique=True,
        ),
        Index("ix_strategy_presets_status", "symbol", "strategy", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    strategy: Mapped[str] = mapped_column(String(48), index=True)
    strategy_version: Mapped[str] = mapped_column(String(16), default="")
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    #: user-facing strategy name; '' = legacy/unnamed strategy
    strategy_name: Mapped[str] = mapped_column(String(64), default="")
    #: monotonic per (symbol, strategy, strategy_name) — assigned by the service
    version: Mapped[int] = mapped_column(default=1)
    #: bar interval the version was validated on (audit/UX only)
    timeframe: Mapped[str] = mapped_column(String(16), default="")
    #: {total_return, sharpe, max_drawdown, win_rate, n_trades} — never fabricated
    metrics_json: Mapped[str] = mapped_column(String(1024), default="{}")
    #: backtest_only | live_enabled
    status: Mapped[str] = mapped_column(String(16), default="backtest_only")
    #: "optimizer:<run_token>" | "backtest:<backtest_results.id>"
    backtest_ref: Mapped[str | None] = mapped_column(String(64), default=None)
    #: backtest | manual | optimizer
    source: Mapped[str] = mapped_column(String(16), default="manual")
    optimizer_run_id: Mapped[str | None]
    is_default: Mapped[bool] = mapped_column(default=False)
    notes: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SignalKeyRow(Base):
    """A signal-subscription API key — a reference to a full strategy config.

    Distinct from ``api_keys`` (which stores *broker credentials*): this row
    points at the complete unified-strategy configuration (tickers, presets,
    costs) used to generate signals and populate the ``/API_KEY/{key}``
    dashboard.
    """

    __tablename__ = "signal_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    exchange: Mapped[str] = mapped_column(String(16), index=True)  # bingx | tbank
    label: Mapped[str] = mapped_column(String(64), default="")
    #: Full key configuration: strategy, version, tickers (+ per-ticker params
    #: or preset ids), timeframe, source, limit, costs, initial cash.
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[datetime | None]
    revoked_at: Mapped[datetime | None]


class KeySignalRow(Base):
    """One signal generated for a signal key (auditable, exportable)."""

    __tablename__ = "key_signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    #: NULL for signals produced directly by the live engine (no subscription
    #: key); a signal key's generated signals carry its id.
    key_id: Mapped[int | None] = mapped_column(index=True, default=None)
    symbol: Mapped[str] = mapped_column(String(32))
    #: buy | sell
    side: Mapped[str] = mapped_column(String(8))
    #: long_entry | long_add | long_exit | short_entry | short_add | short_exit
    state: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(String(255), default="")
    strength: Mapped[float] = mapped_column(default=1.0)
    price: Mapped[float] = mapped_column(default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    strategy: Mapped[str] = mapped_column(String(48))
    strategy_version: Mapped[str] = mapped_column(String(16), default="")
    preset_id: Mapped[int | None]
    #: live | replay
    source: Mapped[str] = mapped_column(String(16), default="live")
    indicators_json: Mapped[str] = mapped_column(Text, default="{}")
    # ── trade plan (filled by strategies that emit a full plan) ──
    #: entry / stop / target levels the strategy wants executed
    entry_price: Mapped[float | None] = mapped_column(default=None)
    stop_loss: Mapped[float | None] = mapped_column(default=None)
    take_profit: Mapped[float | None] = mapped_column(default=None)
    #: intended exposure: fraction of equity, units, and currency risked
    position_size: Mapped[float | None] = mapped_column(default=None)
    risk_pct: Mapped[float | None] = mapped_column(default=None)
    risk_amount: Mapped[float | None] = mapped_column(default=None)
    #: bar interval the signal was computed on and the closed bar that made it
    timeframe: Mapped[str | None] = mapped_column(String(16), default=None)
    bar_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class SignalPositionRow(Base):
    """Full lifecycle of one signal position — the row behind every export.

    Written by the live signal engine (never by a backtest): an entry signal
    opens a row, the matching exit signal closes it. Because the engine observes
    both prices and the planned risk, a row carries the entry, the initial and
    final stop, the trail, the realised PnL in currency and R, and every exit
    cause — which is exactly what "all data about each position" means.
    """

    __tablename__ = "signal_positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    #: optional link to the signal key that owns the subscription
    key_id: Mapped[int | None] = mapped_column(index=True, default=None)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    strategy: Mapped[str] = mapped_column(String(48), default="")
    strategy_version: Mapped[str] = mapped_column(String(16), default="")
    preset: Mapped[str] = mapped_column(String(32), default="")
    timeframe: Mapped[str] = mapped_column(String(16), default="")
    source: Mapped[str] = mapped_column(String(16), default="live")
    # ── direction / lifecycle ──
    side: Mapped[str] = mapped_column(String(8), default="")       # long | short
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    # ── entry ──
    entry_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    entry_price: Mapped[float] = mapped_column(default=0.0)
    quantity: Mapped[float] = mapped_column(default=0.0)
    initial_stop: Mapped[float | None] = mapped_column(default=None)
    # ── management ──
    stop_price: Mapped[float | None] = mapped_column(default=None)
    take_profit: Mapped[float | None] = mapped_column(default=None)
    trail_price: Mapped[float | None] = mapped_column(default=None)
    best_price: Mapped[float | None] = mapped_column(default=None)
    worst_price: Mapped[float | None] = mapped_column(default=None)
    mfe_r: Mapped[float] = mapped_column(default=0.0)
    bars_held: Mapped[int] = mapped_column(default=0)
    # ── exit ──
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    exit_price: Mapped[float | None] = mapped_column(default=None)
    exit_reason: Mapped[str] = mapped_column(String(64), default="")
    # ── PnL ──
    risk_amount: Mapped[float] = mapped_column(default=0.0)
    risk_pct: Mapped[float | None] = mapped_column(default=None)
    gross_pnl: Mapped[float | None] = mapped_column(default=None)
    net_pnl: Mapped[float | None] = mapped_column(default=None)
    pnl_r: Mapped[float | None] = mapped_column(default=None)
    pct_return: Mapped[float | None] = mapped_column(default=None)
    unrealised_pnl: Mapped[float | None] = mapped_column(default=None)
    #: last mark seen for an open row (0.0 when never marked)
    mark_price: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KeyTradeRow(Base):
    """One closed trade in a signal key's paper/replay ledger."""

    __tablename__ = "key_trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key_id: Mapped[int] = mapped_column(index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    direction: Mapped[str] = mapped_column(String(8))  # long | short
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float]
    exit_price: Mapped[float]
    quantity: Mapped[float]
    fee: Mapped[float] = mapped_column(default=0.0)
    gross_pnl: Mapped[float]
    net_pnl: Mapped[float]
    pct_return: Mapped[float]
    holding_seconds: Mapped[float] = mapped_column(default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(64), default="")
    strategy: Mapped[str] = mapped_column(String(48))
    strategy_version: Mapped[str] = mapped_column(String(16), default="")
    preset_id: Mapped[int | None]
    #: live | replay
    source: Mapped[str] = mapped_column(String(16), default="live")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
