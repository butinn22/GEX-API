"""Order execution engine.

``BrokerRouter`` maps an exchange to its ``BrokerAdapter``. ``ExecutionEngine``
places/cancels orders with exponential-backoff retry (only for *transient*
broker errors — ``OrderRejectedError``/``InsufficientFundsError`` never retry),
reads positions/portfolio, and enforces a kill switch (halt when drawdown
breaches a threshold).
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping

from trading.application.account_router import (
    AccountOrderResult,
    AccountRouter,
    apply_risk,
)
from trading.domain import (
    BrokerError,
    Exchange,
    InsufficientFundsError,
    Order,
    OrderIntent,
    OrderRejectedError,
    Portfolio,
    Position,
)
from trading.ports import BrokerAdapter

__all__ = ["BrokerRouter", "KillSwitch", "ExecutionEngine"]


class BrokerRouter:
    """Routes orders to the broker adapter for an exchange."""

    def __init__(self) -> None:
        self._brokers: dict[Exchange, BrokerAdapter] = {}

    def register(self, broker: BrokerAdapter) -> None:
        self._brokers[broker.exchange] = broker

    def get(self, exchange: Exchange) -> BrokerAdapter:
        broker = self._brokers.get(exchange)
        if broker is None:
            raise BrokerError(f"no broker configured for {exchange.value}")
        return broker


class KillSwitch:
    """Halt trading once drawdown from peak equity exceeds ``max_drawdown``."""

    def __init__(self, max_drawdown: float = 0.25) -> None:
        if not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        self.max_drawdown = max_drawdown
        self._peak = 0.0

    def update(self, equity: float) -> bool:
        """Record equity; return True when the kill switch should trip."""
        self._peak = max(self._peak, equity)
        if self._peak <= 0:
            return False
        dd = (self._peak - equity) / self._peak
        return dd >= self.max_drawdown


class ExecutionEngine:
    """Centralised order manager over a :class:`BrokerRouter`."""

    def __init__(
        self,
        router: BrokerRouter,
        *,
        max_retries: int = 3,
        base_delay: float = 0.05,
        max_delay: float = 2.0,
        kill_switch: KillSwitch | None = None,
        account_router: AccountRouter | None = None,
    ) -> None:
        self.router = router
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.kill_switch = kill_switch or KillSwitch()
        self.account_router = account_router

    async def _retry_on(self, broker: BrokerAdapter, action):
        """Exponential-backoff retry against one already-resolved broker."""
        delay = self.base_delay
        last: BrokerError | None = None
        for attempt in range(self.max_retries):
            try:
                return await action(broker)
            except (OrderRejectedError, InsufficientFundsError):
                raise  # permanent — never retry
            except BrokerError as exc:
                last = exc
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_delay)
        raise last  # type: ignore[misc]

    async def _with_retry(self, exchange: Exchange, action):
        return await self._retry_on(self.router.get(exchange), action)

    async def place_order(self, exchange: Exchange, intent: OrderIntent) -> Order:
        return await self._with_retry(exchange, lambda b: b.place_order(intent))

    async def place_multi(
        self,
        intent: OrderIntent,
        *,
        equities: Mapping[int, float] | None = None,
        price: float | None = None,
    ) -> list[AccountOrderResult]:
        """Place ``intent`` on every account whose instrument scope covers it.

        Each account gets the engine's full retry policy and its own risk
        sizing (:func:`~trading.application.account_router.apply_risk`); a
        failure on one account never blocks the others.
        """
        if self.account_router is None:
            raise ValueError("no AccountRouter configured for multi-account execution")
        equities = equities or {}
        results: list[AccountOrderResult] = []
        for account, broker in self.account_router.brokers_for(intent.symbol):
            sized = apply_risk(
                intent, account.settings,
                equity=equities.get(account.key_id, 0.0), price=price,
            )
            try:
                order = await self._retry_on(
                    broker, lambda b, _sized=sized: b.place_order(_sized)
                )
            except BrokerError as exc:
                results.append(AccountOrderResult(
                    account.key_id, account.label,
                    error=f"{type(exc).__name__}: {exc}",
                ))
                continue
            results.append(AccountOrderResult(account.key_id, account.label, order=order))
        return results

    async def cancel_order(self, exchange: Exchange, order_id: str) -> Order:
        return await self._with_retry(exchange, lambda b: b.cancel_order(order_id))

    async def get_positions(self, exchange: Exchange) -> list[Position]:
        return await self.router.get(exchange).get_positions()

    async def get_portfolio(self, exchange: Exchange) -> Portfolio:
        return await self.router.get(exchange).get_portfolio()
