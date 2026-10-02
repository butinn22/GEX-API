"""Realized-PnL under different cost-basis methods (FIFO / LIFO / average).

The domain ``Position`` already tracks average-cost PnL; these functions provide
the FIFO/LIFO alternatives used by the PnL calculator and the reporter. Fees are
ignored here (gross of fees) — add fees separately at the ledger level.
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

from trading.domain import Fill, Side

__all__ = ["realized_pnl_fifo", "realized_pnl_lifo", "realized_pnl_avg"]


def realized_pnl_fifo(fills: Sequence[Fill]) -> float:
    return _realized(fills, lifo=False)


def realized_pnl_lifo(fills: Sequence[Fill]) -> float:
    return _realized(fills, lifo=True)


def _realized(fills: Sequence[Fill], *, lifo: bool) -> float:
    lots: deque[tuple[float, float]] = deque()  # (qty, price)
    realized = 0.0
    for f in fills:
        if f.side is Side.BUY:
            lots.append((f.quantity, f.price))
            continue
        qty = f.quantity
        while qty > 1e-12 and lots:
            lot_qty, lot_price = lots.pop() if lifo else lots.popleft()
            closed = min(qty, lot_qty)
            realized += (f.price - lot_price) * closed
            qty -= closed
            if lot_qty > closed:
                rem = lot_qty - closed
                lots.appendleft((rem, lot_price)) if not lifo else lots.append((rem, lot_price))
    return realized


def realized_pnl_avg(fills: Sequence[Fill]) -> float:
    qty = 0.0
    avg = 0.0
    realized = 0.0
    for f in fills:
        if f.side is Side.BUY:
            avg = (avg * qty + f.price * f.quantity) / (qty + f.quantity)
            qty += f.quantity
        else:
            realized += (f.price - avg) * f.quantity
            qty -= f.quantity
    return realized
