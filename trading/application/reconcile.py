"""Position reconciliation: diff broker positions against a local store.

The Celery ``reconcile_positions_task`` calls this. Pure and offline-testable:
the diff is computed from two position maps; applying it (open/close orders) is
the execution engine's job once live credentials are present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trading.domain import Position

__all__ = ["ReconcileDiff", "reconcile_positions"]


@dataclass
class ReconcileDiff:
    to_insert: list[Position] = field(default_factory=list)  # on broker, not local
    to_update: list[Position] = field(default_factory=list)  # quantity differs
    to_close: list[str] = field(default_factory=list)  # local only → close
    unchanged: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.to_insert or self.to_update or self.to_close)


def reconcile_positions(
    broker_positions: list[Position],
    local_positions: dict[str, Position],
    *,
    qty_tol: float = 1e-9,
) -> ReconcileDiff:
    diff = ReconcileDiff()
    broker_map = {p.symbol: p for p in broker_positions}

    for symbol, bp in broker_map.items():
        lp = local_positions.get(symbol)
        if lp is None or lp.side.value != bp.side.value:
            diff.to_insert.append(bp)
        elif abs(lp.quantity - bp.quantity) > qty_tol:
            diff.to_update.append(bp)
        else:
            diff.unchanged += 1

    for symbol in local_positions:
        if symbol not in broker_map:
            diff.to_close.append(symbol)

    return diff
