"""Granular trade-event ledger.

The classic ``Trade`` ledger only records *closes*; reporting needs every
fill classified by its effect on the position:

* ``long_entry`` / ``short_entry`` — flat → position opened;
* ``long_add`` / ``short_add`` — same-direction scale-in;
* ``long_exit`` / ``short_exit`` — opposite-direction reduction/close/flip
  (a flip is recorded as an exit of the old side; the new side's first event
  is the *next* fill).

Exits carry realized PnL and percentage return on the closed quantity;
entries/adds carry zero (nothing realized yet).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from trading.domain import Fill, Position, PositionSide, Side

__all__ = ["TradeState", "TradeEvent", "classify_fill", "event_from_fill", "events_from_fill"]


class TradeState(str, Enum):
    LONG_ENTRY = "long_entry"
    LONG_ADD = "long_add"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_ADD = "short_add"
    SHORT_EXIT = "short_exit"

    @property
    def direction(self) -> str:
        return "long" if self.value.startswith("long") else "short"

    @property
    def is_exit(self) -> bool:
        return self.value.endswith("exit")


@dataclass(frozen=True)
class TradeEvent:
    timestamp: datetime
    symbol: str
    state: TradeState
    direction: str
    side: Side
    price: float
    quantity: float
    realized_pnl: float = 0.0
    pct_return: float = 0.0
    strategy: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "state": self.state.value,
            "direction": self.direction,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "realized_pnl": self.realized_pnl,
            "pct_return": self.pct_return,
            "strategy": self.strategy,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeEvent":
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            symbol=d["symbol"],
            state=TradeState(d["state"]),
            direction=d["direction"],
            side=Side(d["side"]),
            price=float(d["price"]),
            quantity=float(d["quantity"]),
            realized_pnl=float(d.get("realized_pnl", 0.0)),
            pct_return=float(d.get("pct_return", 0.0)),
            strategy=d.get("strategy"),
            reason=d.get("reason"),
        )


def classify_fill(position_before: Position, fill: Fill) -> TradeState:
    """Classify a fill by its effect on the position held before it."""
    if position_before.side is PositionSide.FLAT:
        return TradeState.LONG_ENTRY if fill.side is Side.BUY else TradeState.SHORT_ENTRY
    if position_before.side.sign * fill.side.sign > 0:
        return (
            TradeState.LONG_ADD
            if position_before.side is PositionSide.LONG
            else TradeState.SHORT_ADD
        )
    return (
        TradeState.LONG_EXIT
        if position_before.side is PositionSide.LONG
        else TradeState.SHORT_EXIT
    )


def event_from_fill(
    position_before: Position,
    fill: Fill,
    *,
    strategy: str | None = None,
    reason: str | None = None,
) -> TradeEvent:
    """Build the ledger event for ``fill`` applied to ``position_before``."""
    state = classify_fill(position_before, fill)
    realized = 0.0
    pct = 0.0
    if state.is_exit and position_before.average_entry_price > 0:
        closed = min(position_before.quantity, fill.quantity)
        sign = position_before.side.sign
        realized = (fill.price - position_before.average_entry_price) * closed * sign
        pct = (fill.price - position_before.average_entry_price) / position_before.average_entry_price * sign
    return TradeEvent(
        timestamp=fill.timestamp,
        symbol=fill.symbol,
        state=state,
        direction=state.direction,
        side=fill.side,
        price=fill.price,
        quantity=fill.quantity,
        realized_pnl=realized,
        pct_return=pct,
        strategy=strategy,
        reason=reason,
    )


def events_from_fill(
    position_before: Position,
    fill: Fill,
    *,
    strategy: str | None = None,
    reason: str | None = None,
) -> list[TradeEvent]:
    """Ledger events for ``fill`` — a *flip* splits into exit + entry.

    A flip fill (an opposite-side order larger than the current position)
    closes the old side and opens the new one in a single execution. The
    ledger records **both** halves: the exit (realized PnL on the closed
    quantity) and the entry of the remainder. Without the split, positions
    opened by a flip have no entry event and closed-trade reconstruction
    (e.g. ``_pair_trades`` in the signal-key dashboard) cannot pair them.
    """
    state = classify_fill(position_before, fill)
    flipped = (
        state.is_exit
        and position_before.side is not PositionSide.FLAT
        and fill.quantity > position_before.quantity
    )
    if not flipped:
        return [event_from_fill(position_before, fill, strategy=strategy, reason=reason)]

    closed = position_before.quantity
    remainder = fill.quantity - closed
    exit_state = (
        TradeState.LONG_EXIT
        if position_before.side is PositionSide.LONG
        else TradeState.SHORT_EXIT
    )
    entry_state = TradeState.LONG_ENTRY if fill.side is Side.BUY else TradeState.SHORT_ENTRY
    sign = position_before.side.sign
    avg = position_before.average_entry_price
    realized = (fill.price - avg) * closed * sign if avg > 0 else 0.0
    pct = (fill.price - avg) / avg * sign if avg > 0 else 0.0
    return [
        TradeEvent(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            state=exit_state,
            direction=exit_state.direction,
            side=fill.side,
            price=fill.price,
            quantity=closed,
            realized_pnl=realized,
            pct_return=pct,
            strategy=strategy,
            reason=reason,
        ),
        TradeEvent(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            state=entry_state,
            direction=entry_state.direction,
            side=fill.side,
            price=fill.price,
            quantity=remainder,
            realized_pnl=0.0,
            pct_return=0.0,
            strategy=strategy,
            reason=reason,
        ),
    ]
