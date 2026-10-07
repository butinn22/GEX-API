"""Position sizing and risk management.

``PositionSizer`` turns equity + price (+ risk inputs) into a quantity.
``RiskManager`` gates orders on position limits and drawdown.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from trading.domain import OrderIntent, Portfolio

__all__ = [
    "PositionSizer",
    "FixedFractionSizer",
    "PercentRiskSizer",
    "KellySizer",
    "RiskManager",
]


class PositionSizer(ABC):
    @abstractmethod
    def size(self, *, equity: float, price: float, signal_strength: float = 1.0) -> float:
        """Return the quantity to trade (>= 0)."""


class FixedFractionSizer(PositionSizer):
    """Deploy a fixed fraction of equity per signal."""

    def __init__(self, fraction: float = 0.95) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction

    def size(self, *, equity: float, price: float, signal_strength: float = 1.0) -> float:
        if price <= 0:
            return 0.0
        return max(equity, 0.0) * self.fraction * signal_strength / price


class PercentRiskSizer(PositionSizer):
    """Risk a fixed percentage of equity per trade, sized by the stop distance."""

    def __init__(self, risk_pct: float = 0.01) -> None:
        if not 0 < risk_pct <= 1:
            raise ValueError("risk_pct must be in (0, 1]")
        self.risk_pct = risk_pct

    def size(self, *, equity: float, price: float, stop_price: float | None = None,
             signal_strength: float = 1.0) -> float:
        if price <= 0 or stop_price is None:
            return 0.0
        distance = abs(price - stop_price)
        if distance <= 0:
            return 0.0
        risk_amount = max(equity, 0.0) * self.risk_pct * signal_strength
        return risk_amount / distance


class KellySizer(PositionSizer):
    """Fractional-Kelly sizing from a win rate and win/loss ratio.

    f* = win_rate - (1 - win_rate) / win_loss_ratio, then scaled by ``fraction``
    (e.g. half-Kelly = 0.5). Capped to [0, max_fraction].
    """

    def __init__(self, win_rate: float, win_loss_ratio: float, fraction: float = 0.5,
                 max_fraction: float = 0.25) -> None:
        if not 0 <= win_rate <= 1 or win_loss_ratio <= 0:
            raise ValueError("invalid Kelly inputs")
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.fraction = fraction
        self.max_fraction = max_fraction

    def _kelly(self) -> float:
        f = self.win_rate - (1 - self.win_rate) / self.win_loss_ratio
        return max(0.0, min(f * self.fraction, self.max_fraction))

    def size(self, *, equity: float, price: float, signal_strength: float = 1.0) -> float:
        if price <= 0:
            return 0.0
        return max(equity, 0.0) * self._kelly() * signal_strength / price


class RiskManager:
    """Central risk gate: position limits + drawdown halt."""

    def __init__(self, *, max_position_pct: float = 1.0, max_drawdown: float = 0.25) -> None:
        if not 0 < max_position_pct <= 1 or not 0 < max_drawdown < 1:
            raise ValueError("invalid risk limits")
        self.max_position_pct = max_position_pct
        self.max_drawdown = max_drawdown
        self._peak_equity = 0.0

    def update_equity(self, equity: float) -> None:
        self._peak_equity = max(self._peak_equity, equity)

    def approve(self, intent: OrderIntent, portfolio: Portfolio, *, mark: float,
                current_equity: float) -> tuple[bool, str]:
        """Return (approved, reason)."""
        # 1. drawdown halt
        if self._peak_equity > 0:
            dd = (self._peak_equity - current_equity) / self._peak_equity
            if dd >= self.max_drawdown:
                return False, f"drawdown {dd:.1%} exceeds {self.max_drawdown:.1%}"
        # 2. position notional limit
        equity = max(current_equity, 0.0)
        notional = intent.quantity.value * mark
        if equity > 0 and notional > equity * self.max_position_pct:
            return False, f"position notional {notional:.0f} exceeds limit"
        return True, "ok"
