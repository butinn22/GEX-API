"""Smart-order planning (TWAP / VWAP / iceberg) — child-order scheduling.

Produces child-order quantities only; execution + rate-limiting is the
``ExecutionEngine``'s job.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["SmartOrderPlanner"]


class SmartOrderPlanner:
    def twap(self, qty: float, *, duration_seconds: float, interval_seconds: float) -> list[float]:
        """Equal slices over the duration."""
        if qty <= 0 or interval_seconds <= 0 or duration_seconds <= 0:
            raise ValueError("invalid TWAP parameters")
        n = max(int(duration_seconds // interval_seconds), 1)
        slice_qty = qty / n
        return [slice_qty] * n

    def vwap(self, qty: float, *, volume_profile: Sequence[float]) -> list[float]:
        """Slice proportionally to a historical volume profile."""
        profile = np.asarray(volume_profile, dtype=float)
        if profile.size == 0 or profile.sum() <= 0 or qty <= 0:
            raise ValueError("invalid VWAP parameters")
        weights = profile / profile.sum()
        return (qty * weights).tolist()

    def iceberg(self, qty: float, *, display_qty: float) -> list[float]:
        """Chunks of at most ``display_qty`` (the visible portion of each order)."""
        if qty <= 0 or display_qty <= 0:
            raise ValueError("invalid iceberg parameters")
        chunks: list[float] = []
        remaining = qty
        while remaining > 1e-12:
            chunk = min(display_qty, remaining)
            chunks.append(chunk)
            remaining -= chunk
        return chunks
