"""TBANK (T-Investments) broker adapter — real SDK wiring.

Uses the official ``tinkoff-invest`` client library (``SandboxSession`` /
``ProductionSession``). SDK calls are synchronous (requests-based) and run via
``asyncio.to_thread`` to keep the async ``BrokerAdapter`` contract. When no
token is set, the adapter degrades to dry-run (empty accounts, PENDING orders)
so the rest of the platform stays runnable.

Quantities are in **lots** (TBANK trades whole lots; ``OrderIntent.quantity`` is
interpreted as lots). Tickers map to FIGI via ``get_instrument_by_ticker``.
"""
from __future__ import annotations

import asyncio

from trading.domain import (
    Account,
    BrokerError,
    Exchange,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionSide,
    Side,
)
from trading.ports import BrokerAdapter

__all__ = ["TbankBroker"]

_STATUS = {
    "PENDING_NEW": OrderStatus.PENDING,
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILL": OrderStatus.PARTIAL,
    "FILL": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "PENDING_CANCEL": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "REPLACED": OrderStatus.OPEN,
    "PENDING_REPLACE": OrderStatus.OPEN,
}


def _sdk_available() -> bool:
    try:
        import tinkoff_invest  # noqa: F401
        return True
    except Exception:
        return False


class TbankBroker(BrokerAdapter):
    exchange = Exchange.TBANK

    def __init__(self, token: str, account_id: str = "", *, sandbox: bool = True) -> None:
        self.token = token
        self.account_id = account_id
        self.sandbox = sandbox
        self._dry_run = not token or not _sdk_available()
        self._session = None

    def _get_session(self):
        """Lazily construct the SDK session (its ctor registers, i.e. hits the API)."""
        if self._session is None:
            if self._dry_run:
                raise BrokerError("tbank dry-run has no session")
            from tinkoff_invest import ProductionSession, SandboxSession

            cls = SandboxSession if self.sandbox else ProductionSession
            self._session = cls(self.token, account_id=self.account_id)
        return self._session

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    # ── Async adapter surface ──────────────────────────────────────────

    async def get_accounts(self) -> list[Account]:
        if self._dry_run:
            return [Account(id=self.account_id or "tbank-dry-run", currency="RUB")]
        return await asyncio.to_thread(self._sync_accounts)

    async def get_portfolio(self) -> Portfolio:
        if self._dry_run:
            return Portfolio(cash=0.0, currency="RUB")
        return await asyncio.to_thread(self._sync_portfolio)

    async def get_positions(self) -> list[Position]:
        if self._dry_run:
            return []
        return await asyncio.to_thread(self._sync_positions)

    async def place_order(self, intent: OrderIntent) -> Order:
        if self._dry_run:
            return Order(
                id=f"dry-{intent.symbol}-{intent.timestamp.timestamp():.0f}",
                symbol=intent.symbol, side=intent.side,
                quantity=intent.quantity.value, order_type=intent.order_type,
                status=OrderStatus.PENDING, strategy=intent.strategy, reason=intent.reason,
            )
        return await asyncio.to_thread(self._sync_place_order, intent)

    async def cancel_order(self, order_id: str) -> Order:
        if self._dry_run:
            return Order(id=order_id, symbol="", side=Side.BUY, quantity=0.0,
                         order_type=OrderType.MARKET, status=OrderStatus.CANCELLED)
        return await asyncio.to_thread(self._sync_cancel_order, order_id)

    async def get_order_status(self, order_id: str) -> Order:
        if self._dry_run:
            raise BrokerError("dry-run has no order history")
        return await asyncio.to_thread(self._sync_order_status, order_id)

    # ── Sync SDK calls ─────────────────────────────────────────────────

    def _sync_accounts(self) -> list[Account]:
        portfolio = self._get_session().get_portfolio()
        cash = sum(c.balance.value for c in portfolio.currencies)
        return [Account(id=self.account_id, currency="RUB", cash=cash, buying_power=cash)]

    def _sync_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._get_session().get_portfolio_positions():
            qty = abs(float(p.balance))
            side = PositionSide.LONG if float(p.balance) >= 0 else PositionSide.SHORT
            avg = float(p.average_price.value) if p.average_price else 0.0
            realized = float(p.expected_yield.value) if p.expected_yield else 0.0
            out.append(Position(symbol=p.ticker or p.figi, side=side, quantity=qty,
                                average_entry_price=avg, realized_pnl=realized))
        return out

    def _sync_portfolio(self) -> Portfolio:
        portfolio = self._get_session().get_portfolio()
        cash = sum(c.balance.value for c in portfolio.currencies)
        return Portfolio(cash=cash, positions=tuple(self._sync_positions()), currency="RUB")

    def _sync_place_order(self, intent: OrderIntent) -> Order:
        from tinkoff_invest.models.types import OperationType

        operation = OperationType.BUY if intent.side is Side.BUY else OperationType.SELL
        session = self._get_session()
        figi = session.get_instrument_by_ticker(intent.symbol).figi
        lots = int(intent.quantity.value)
        if lots <= 0:
            raise BrokerError("tbank order quantity (lots) must be >= 1")

        if intent.order_type is OrderType.MARKET:
            sdk_order = session.create_market_order(operation, figi, lots)
        elif intent.order_type is OrderType.LIMIT and intent.limit_price is not None:
            sdk_order = session.create_limit_order(
                operation, figi, float(intent.limit_price.value), lots
            )
        else:
            raise BrokerError(f"tbank: unsupported order type {intent.order_type.value}")
        return self._map_order(sdk_order, intent.symbol)

    def _sync_cancel_order(self, order_id: str) -> Order:
        self._get_session().cancel_order(order_id)
        return Order(id=order_id, symbol="", side=Side.BUY, quantity=0.0,
                     order_type=OrderType.MARKET, status=OrderStatus.CANCELLED)

    def _sync_order_status(self, order_id: str) -> Order:
        for o in self._get_session().get_orders():
            if o.id == order_id:
                return self._map_order(o, o.figi)
        raise BrokerError(f"tbank: order {order_id} not found")

    @staticmethod
    def _map_order(sdk_order, symbol: str) -> Order:
        status = _STATUS.get(sdk_order.status.name, OrderStatus.OPEN)
        side = Side.BUY if sdk_order.operation.name == "BUY" else Side.SELL
        return Order(
            id=sdk_order.id,
            symbol=symbol,
            side=side,
            quantity=float(sdk_order.requested_lots),
            order_type=OrderType.MARKET if sdk_order.price is None else OrderType.LIMIT,
            status=status,
            limit_price=sdk_order.price,
        )
