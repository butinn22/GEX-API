"""Event-driven backtest engine.

Replays bars through a strategy, executing signals at the *next* bar's open (no
lookahead), applying slippage + fees, and producing an equity curve, a trade
ledger, and the full metric set.

Execution model
---------------
* Signal at bar ``t`` close → order intent → filled at bar ``t+1`` open.
* BUY pays ``price * (1 + slippage)``, SELL receives ``price * (1 - slippage)``.
* Fee is ``fee_rate * notional`` per side.
* Position sizing: ``position_fraction * strength * equity / price``.
* Positions may be long or short (the engine is side-agnostic; strategies decide).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np

from trading.domain import (
    Bar,
    Fill,
    OrderIntent,
    OrderType,
    Portfolio,
    PositionSide,
    Quantity,
    Side,
    Signal,
)
from trading.application.risk import PositionSizer
from trading.ports import Strategy

from .match_engine import (
    CommissionModel,
    MatchEngine,
    PercentCommissionModel,
    PercentSlippageModel,
    SlippageModel,
)
from .metrics import BacktestMetrics, compute_metrics
from .trade_log import TradeEvent, event_from_fill

__all__ = ["BacktestConfig", "Trade", "BacktestResult", "run_backtest"]


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.001  # per side, fraction of notional
    slippage: float = 0.0005  # per side, fraction of price
    position_fraction: float = 0.95  # fraction of equity deployed per entry
    periods_per_year: int = 252
    min_signal_strength: float = 0.0
    sizer: PositionSizer | None = None  # overrides the fixed position_fraction
    commission_model: CommissionModel | None = None  # defaults to PercentCommissionModel(fee_rate)
    slippage_model: SlippageModel | None = None  # defaults to PercentSlippageModel(slippage)


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: Side  # direction of the position that was (partially) closed
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    entry_time: datetime
    exit_time: datetime


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    trades: tuple[Trade, ...]
    metrics: BacktestMetrics
    #: Per-fill ledger (entry/add/exit states) for granular trade reporting.
    events: tuple[TradeEvent, ...] = ()

    @property
    def trade_pnls(self) -> tuple[float, ...]:
        return tuple(t.realized_pnl for t in self.trades)


def _closed_quantity(pos_side: PositionSide, pos_qty: float, fill: Fill) -> float:
    """Units closed by ``fill`` (0 when opening or adding)."""
    if pos_side is PositionSide.FLAT:
        return 0.0
    if pos_side.sign * fill.side.sign > 0:  # same direction → adding
        return 0.0
    return min(pos_qty, fill.quantity)


def _size_signal(sig: Signal, portfolio: Portfolio, price: float, cfg: BacktestConfig) -> OrderIntent | None:
    if sig.strength < cfg.min_signal_strength:
        return None
    equity = max(portfolio.equity({sig.symbol: price}), 0.0)
    if cfg.sizer is not None:
        qty = cfg.sizer.size(equity=equity, price=price, signal_strength=sig.strength)
    else:
        qty = equity * cfg.position_fraction * sig.strength / price
    if qty <= 0 or price <= 0:
        return None
    return OrderIntent(
        symbol=sig.symbol,
        side=sig.side,
        quantity=Quantity(qty),
        order_type=OrderType.MARKET,
        strategy=sig.strategy,
        reason=sig.reason,
        timestamp=sig.timestamp,
    )


async def run_backtest(
    strategy: Strategy,
    bars: Sequence[Bar],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Replay ``bars`` through ``strategy`` and return the full result."""
    cfg = config or BacktestConfig()
    bars = sorted(bars, key=lambda b: b.timestamp)
    if not bars:
        raise ValueError("backtest requires at least one bar")

    # Batch strategies get one chance to precompute over the whole replay (O(n))
    # instead of recomputing on every bar (O(n^2)). No-op for streaming strategies.
    await strategy.prepare(bars)

    portfolio = Portfolio(cash=cfg.initial_cash)
    equity = np.empty(len(bars), dtype=float)
    trades: list[Trade] = []
    events: list[TradeEvent] = []
    pending: list[OrderIntent] = []

    match_engine = MatchEngine(
        cfg.commission_model or PercentCommissionModel(cfg.fee_rate),
        cfg.slippage_model or PercentSlippageModel(cfg.slippage),
    )

    for i, bar in enumerate(bars):
        # 1. Execute pending orders at this bar's open (signals from t-1 close).
        for intent in pending:
            fill = match_engine.execute(
                intent, bar, order_id=f"{intent.symbol}:{i}:{bar.timestamp.isoformat()}"
            )
            pos = portfolio.position_for(intent.symbol)
            portfolio = portfolio.apply_fill(fill)
            closed_qty = _closed_quantity(pos.side, pos.quantity, fill)
            events.append(
                event_from_fill(pos, fill, strategy=intent.strategy, reason=intent.reason)
            )
            if closed_qty > 0:  # only a real close is a trade (opens/adds are not)
                gross = (fill.price - pos.average_entry_price) * closed_qty * pos.side.sign
                fee_share = fill.fee * (closed_qty / fill.quantity)
                trades.append(
                    Trade(
                        symbol=intent.symbol,
                        side=pos.side,
                        entry_price=pos.average_entry_price,
                        exit_price=fill.price,
                        quantity=closed_qty,
                        realized_pnl=gross - fee_share,
                        entry_time=bar.timestamp,
                        exit_time=bar.timestamp,
                    )
                )
        pending.clear()

        # 2. Generate signals from this bar's close.
        for sig in await strategy.on_bar(bar):
            intent = _size_signal(sig, portfolio, bar.close, cfg)
            if intent is not None:
                pending.append(intent)

        # 3. Mark-to-market equity at this bar's close.
        marks = {p.symbol: bar.close for p in portfolio.positions}
        equity[i] = portfolio.equity(marks)

    metrics = compute_metrics(
        equity,
        [t.realized_pnl for t in trades],
        periods_per_year=cfg.periods_per_year,
    )
    return BacktestResult(
        equity_curve=equity, trades=tuple(trades), metrics=metrics, events=tuple(events)
    )
