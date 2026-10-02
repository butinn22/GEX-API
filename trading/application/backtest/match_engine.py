"""Match engine: execution simulation with pluggable commission + slippage models.

Extracted from the backtest engine so the same fill logic is reusable by the
live execution path and testable in isolation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from trading.domain import Bar, Fill, OrderIntent, Side

__all__ = [
    "CommissionModel", "SlippageModel",
    "PercentCommissionModel", "PercentSlippageModel",
    "MatchEngine",
]


class CommissionModel(ABC):
    @abstractmethod
    def commission(self, *, price: float, quantity: float) -> float:
        """Fee charged for a fill of ``quantity`` at ``price``."""


class SlippageModel(ABC):
    @abstractmethod
    def adjust(self, *, price: float, side: Side) -> float:
        """Return the fill price after slippage (BUY pays up, SELL receives down)."""


class PercentCommissionModel(CommissionModel):
    def __init__(self, rate: float) -> None:
        if rate < 0:
            raise ValueError("commission rate must be >= 0")
        self.rate = rate

    def commission(self, *, price: float, quantity: float) -> float:
        return price * quantity * self.rate


class PercentSlippageModel(SlippageModel):
    def __init__(self, rate: float) -> None:
        if rate < 0:
            raise ValueError("slippage rate must be >= 0")
        self.rate = rate

    def adjust(self, *, price: float, side: Side) -> float:
        return price * (1.0 + self.rate * side.sign)


class MatchEngine:
    def __init__(self, commission: CommissionModel, slippage: SlippageModel) -> None:
        self.commission = commission
        self.slippage = slippage

    def execute(self, intent: OrderIntent, bar: Bar, *, order_id: str) -> Fill:
        price = self.slippage.adjust(price=bar.open, side=intent.side)
        fee = self.commission.commission(price=price, quantity=intent.quantity.value)
        return Fill(
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            price=price,
            quantity=intent.quantity.value,
            timestamp=bar.timestamp,
            fee=fee,
        )
