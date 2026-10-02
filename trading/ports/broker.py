"""Broker port — the execution boundary for every venue (TBANK, BINGX, …).

A ``BrokerAdapter`` translates domain ``OrderIntent``/``Order`` into vendor calls
and maps vendor responses back to domain types (anticorruption layer, ADR-4).
The ``ExecutionEngine`` depends only on this port, so the backtest ``MatchEngine``
can implement the same interface and stand in for a live broker.

Streams (order/position/portfolio updates) are deliberately *not* part of this
ABC's sync surface; they arrive via async iterators provided by concrete
adapters, which the engine subscribes to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from trading.domain import Account, Exchange, Order, OrderIntent, Portfolio, Position

__all__ = ["BrokerAdapter", "Account"]


class BrokerAdapter(ABC):
    """Execution interface for a single venue."""

    exchange: ClassVar[Exchange]

    @abstractmethod
    async def get_accounts(self) -> list[Account]:
        """Accounts available to the authenticated principal."""
        ...

    @abstractmethod
    async def get_portfolio(self) -> Portfolio:
        """Cash + positions for the default/selected account."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Open positions (a subset of the portfolio view)."""
        ...

    @abstractmethod
    async def place_order(self, intent: OrderIntent) -> Order:
        """Submit an order. Returns the broker's order (PENDING/OPEN)."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an open order; returns the updated order."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Latest state of an order (reconciles local view with the venue)."""
        ...
