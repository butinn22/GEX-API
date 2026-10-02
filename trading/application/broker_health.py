"""Broker health monitor — pings brokers and reflects status into Prometheus."""
from __future__ import annotations

import asyncio

from trading.domain import Exchange
from trading.observability import BROKER_STATUS
from trading.ports import BrokerAdapter

__all__ = ["BrokerHealthMonitor"]


class BrokerHealthMonitor:
    async def check(self, exchange: Exchange, broker: BrokerAdapter) -> bool:
        """Ping a broker; set the ``trading_broker_status`` gauge accordingly."""
        try:
            await asyncio.wait_for(broker.get_accounts(), timeout=5.0)
            BROKER_STATUS.labels(exchange=exchange.value).set(1)
            return True
        except Exception:
            BROKER_STATUS.labels(exchange=exchange.value).set(0)
            return False

    async def check_all(self, brokers: dict[Exchange, BrokerAdapter]) -> dict[Exchange, bool]:
        results = await asyncio.gather(
            *[self.check(ex, br) for ex, br in brokers.items()], return_exceptions=True
        )
        return {ex: bool(ok) for ex, ok in zip(brokers, results)}
