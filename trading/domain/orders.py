"""Order/execution domain: Signal, OrderIntent, Order, Fill, Position, Portfolio.

Design notes (see ARCHITECTURE.md ADR-5/ADR-6):

* ``Signal`` = what a strategy *thinks* (direction, reason, strength).
* ``OrderIntent`` = what it *wants executed* (side, qty, type, prices) after
  risk + sizing; carries ``to_order()``.
* ``Order`` = an accepted intent in flight; a mutable entity with an explicit
  state machine (PENDING → OPEN → PARTIAL → terminal).
* ``Fill`` = an immutable execution report.
* ``Position`` / ``Portfolio`` = immutable; ``apply_fill`` returns a **new**
  instance (functional style — safe to share across backtest paths).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import utcnow
from .enums import OrderStatus, OrderType, PositionSide, Side, TimeInForce
from .errors import InvalidStateError
from .money import Price, Quantity

__all__ = [
    "Signal",
    "OrderIntent",
    "Order",
    "Fill",
    "Position",
    "Portfolio",
    "Account",
]

_TERMINAL = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
)

# Legal state transitions for the Order state machine.
_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset({OrderStatus.OPEN, OrderStatus.REJECTED, OrderStatus.CANCELLED}),
    OrderStatus.OPEN: frozenset(
        {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
    ),
    OrderStatus.PARTIAL: frozenset(
        {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
    ),
}


# ── Signal ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Signal:
    """A strategy's opinion: direction + reason + strength (0..1), optionally sized.

    Beyond direction/reason/strength the signal carries the **complete trade
    plan** the strategy wants executed. Those fields are optional so existing
    simple strategies stay untouched, but the confluence-breakout family fills
    them all in — that is what makes a signal self-describing enough to hand to
    a broker (or to export as a per-position row) without re-deriving anything.

    ``entry_price`` / ``stop_loss`` / ``take_profit`` are the levels the risk
    model needs; ``position_size`` / ``risk_pct`` / ``risk_amount`` describe the
    intended exposure (fraction of equity, and the currency amount risked
    between entry and stop); ``timeframe`` is the bar interval the signal was
    computed on; ``bar_time`` is the timestamp of the *closed bar* that produced
    it (which can lag ``timestamp``, the emission time).
    """

    symbol: str
    side: Side
    strategy: str
    reason: str
    timestamp: datetime = field(default_factory=utcnow)
    strength: float = 1.0
    price: Price | None = None
    quantity: Quantity | None = None
    # ── trade plan ──
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    timeframe: str | None = None
    risk_pct: float | None = None
    risk_amount: float | None = None
    position_size: float | None = None
    bar_time: datetime | None = None
    #: Free-form extras (indicator snapshot, exit cause, trail level, …).
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError("signal strength must be in [0, 1]")

    def plan_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the trade plan (``None`` entries dropped)."""
        out = {
            "symbol": self.symbol,
            "side": self.side.value,
            "strategy": self.strategy,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "strength": self.strength,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timeframe": self.timeframe,
            "risk_pct": self.risk_pct,
            "risk_amount": self.risk_amount,
            "position_size": self.position_size,
            "bar_time": self.bar_time.isoformat() if self.bar_time else None,
        }
        if self.price is not None:
            out["price"] = self.price.value if hasattr(self.price, "value") else float(self.price)
        if self.quantity is not None:
            out["quantity"] = self.quantity.value if hasattr(self.quantity, "value") else float(self.quantity)
        if self.meta:
            out["meta"] = dict(self.meta)
        return {k: v for k, v in out.items() if v is not None}


# ── OrderIntent ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrderIntent:
    """What to execute: fully specified, sized, rounded — ready for a broker."""

    symbol: str
    side: Side
    quantity: Quantity
    order_type: OrderType
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    strategy: str | None = None
    reason: str | None = None
    timestamp: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.quantity.value <= 0:
            raise ValueError("order quantity must be > 0")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires stop_price")

    def to_order(self, order_id: str) -> Order:
        """Create a PENDING order from this intent."""
        return Order(
            id=order_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity.value,
            order_type=self.order_type,
            limit_price=self.limit_price.value if self.limit_price else None,
            stop_price=self.stop_price.value if self.stop_price else None,
            time_in_force=self.time_in_force,
            strategy=self.strategy,
            reason=self.reason,
        )


# ── Order ─────────────────────────────────────────────────────────────


@dataclass
class Order:
    """An order accepted into the lifecycle. Mutable entity with a state machine."""

    id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    strategy: str | None = None
    reason: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def _transition(self, target: OrderStatus) -> None:
        if target not in _TRANSITIONS.get(self.status, frozenset()):
            raise InvalidStateError(
                f"illegal order transition {self.status.value} → {target.value}"
            )
        self.status = target
        self.updated_at = utcnow()

    def mark_open(self) -> None:
        self._transition(OrderStatus.OPEN)

    def mark_rejected(self) -> None:
        self._transition(OrderStatus.REJECTED)

    def mark_cancelled(self) -> None:
        self._transition(OrderStatus.CANCELLED)

    def mark_expired(self) -> None:
        self._transition(OrderStatus.EXPIRED)

    def apply_fill(self, fill: Fill) -> None:
        """Accumulate a fill and advance status (OPEN→PARTIAL/FILLED, PARTIAL→…)."""
        if self.status.is_terminal:
            raise InvalidStateError(f"cannot fill a {self.status.value} order")
        if fill.quantity <= 0:
            raise ValueError("fill quantity must be > 0")
        prev_notional = (self.average_fill_price or 0.0) * self.filled_quantity
        self.filled_quantity += fill.quantity
        if self.filled_quantity > self.quantity + 1e-12:
            raise InvalidStateError("fills exceed order quantity")
        self.average_fill_price = (prev_notional + fill.price * fill.quantity) / self.filled_quantity
        if self.filled_quantity >= self.quantity - 1e-12:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
        self.updated_at = utcnow()

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


# ── Fill ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Fill:
    """An execution report: a quantity executed at a price, with a fee."""

    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    timestamp: datetime = field(default_factory=utcnow)
    fee: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("fill quantity must be > 0")
        if self.price < 0:
            raise ValueError("fill price must be >= 0")


# ── Position ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Position:
    """Current exposure on one symbol. Average-cost basis; immutable.

    ``quantity`` is always non-negative and ``side`` says direction (FLAT ⇒ qty 0).
    """

    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("position quantity must be >= 0")
        if self.side is PositionSide.FLAT and self.quantity != 0.0:
            raise ValueError("flat position must have zero quantity")
        if self.side is not PositionSide.FLAT and self.quantity == 0.0:
            raise ValueError("long/short position must have positive quantity")

    @property
    def signed_quantity(self) -> float:
        return self.quantity * self.side.sign

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side is PositionSide.FLAT:
            return 0.0
        return (mark_price - self.average_entry_price) * self.quantity * self.side.sign

    def apply_fill(self, fill: Fill) -> Position:
        """Return the position after ``fill`` executes (add / reduce / close / flip)."""
        if fill.symbol != self.symbol:
            raise ValueError("fill symbol does not match position symbol")
        side_sign = self.side.sign
        signed_qty = self.quantity * side_sign  # signed current exposure
        dq = fill.quantity * fill.side.sign  # signed delta of this fill
        fee = fill.fee

        # Opening from flat
        if side_sign == 0:
            new_side = PositionSide.LONG if dq > 0 else PositionSide.SHORT
            return Position(self.symbol, new_side, abs(dq), fill.price, self.realized_pnl - fee)

        # Same direction → scale in, average cost moves, no realized PnL.
        if (signed_qty > 0 and dq > 0) or (signed_qty < 0 and dq < 0):
            total_qty = abs(signed_qty) + abs(dq)
            new_avg = (self.average_entry_price * abs(signed_qty) + fill.price * abs(dq)) / total_qty
            return Position(self.symbol, self.side, total_qty, new_avg, self.realized_pnl - fee)

        # Opposite direction → close some/all, possibly flip.
        closed = min(abs(signed_qty), abs(dq))
        realized_delta = (fill.price - self.average_entry_price) * closed * side_sign
        new_realized = self.realized_pnl + realized_delta - fee
        remaining = signed_qty + dq  # signed

        if abs(remaining) < 1e-12:
            return Position(self.symbol, PositionSide.FLAT, 0.0, 0.0, new_realized)
        if (remaining > 0) == (signed_qty > 0):
            # Partial reduction, basis unchanged.
            return Position(
                self.symbol, self.side, abs(remaining), self.average_entry_price, new_realized
            )
        # Flip: the excess opens a new position at this fill's price.
        new_side = PositionSide.LONG if remaining > 0 else PositionSide.SHORT
        return Position(self.symbol, new_side, abs(remaining), fill.price, new_realized)


# ── Portfolio ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Portfolio:
    """Account-level view: cash + open positions + cumulative realized PnL.

    Accounting model:
      * ``cash`` moves by trade notional on every fill (settled), so realized
        PnL is *already reflected in cash*;
      * ``realized_pnl`` is a reporting accumulator (closed/flipped PnL + fees)
        kept separate so it survives a position going flat;
      * ``equity = cash + Σ(signed_qty · mark)`` — mark-to-market, not cost basis.
    """

    cash: float = 0.0
    positions: tuple[Position, ...] = ()
    currency: str = "USD"
    realized_pnl: float = 0.0

    def position_for(self, symbol: str) -> Position:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return Position(symbol)

    def unrealized_pnl(self, marks: dict[str, float]) -> float:
        return sum(
            p.unrealized_pnl(marks[p.symbol]) for p in self.positions if p.symbol in marks
        )

    def market_value(self, marks: dict[str, float]) -> float:
        """Mark-to-market value of open positions (short positions are negative)."""
        return sum(
            p.signed_quantity * marks[p.symbol]
            for p in self.positions
            if p.symbol in marks
        )

    def equity(self, marks: dict[str, float] | None = None) -> float:
        """Cash plus mark-to-market value of open positions."""
        return self.cash + self.market_value(marks or {})

    def apply_fill(self, fill: Fill) -> Portfolio:
        """Apply a fill: update cash, position, and cumulative realized PnL.

        BUY pays price·qty + fee (cash down); SELL receives price·qty − fee
        (cash up). Realized PnL accrues as the difference in the position's
        realized PnL before/after, so it survives flat (closed) positions.
        """
        pos = self.position_for(fill.symbol)
        new_pos = pos.apply_fill(fill)
        delta_realized = new_pos.realized_pnl - pos.realized_pnl
        others = tuple(p for p in self.positions if p.symbol != fill.symbol)
        notional = fill.price * fill.quantity
        cash_delta = (-notional if fill.side is Side.BUY else notional) - fill.fee
        new_cash = self.cash + cash_delta
        if new_pos.side is PositionSide.FLAT:
            return Portfolio(
                cash=new_cash, positions=others, currency=self.currency,
                realized_pnl=self.realized_pnl + delta_realized,
            )
        return Portfolio(
            cash=new_cash, positions=(*others, new_pos), currency=self.currency,
            realized_pnl=self.realized_pnl + delta_realized,
        )


# ── Account ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Account:
    """A broker account: cash + buying power + margin (for risk checks)."""

    id: str
    currency: str = "USD"
    cash: float = 0.0
    buying_power: float = 0.0
    margin_used: float = 0.0
