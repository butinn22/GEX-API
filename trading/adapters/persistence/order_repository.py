"""Repository for persisted orders."""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from trading.domain import Order

from .models import OrderRow

__all__ = ["OrderRepository"]


class OrderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, order: Order, exchange: str) -> OrderRow:
        row = OrderRow(
            id=order.id,
            exchange=exchange,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            status=order.status.value,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            filled_quantity=order.filled_quantity,
            strategy=order.strategy,
            reason=order.reason,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def list(self) -> list[OrderRow]:
        result = await self._session.execute(
            select(OrderRow).order_by(OrderRow.created_at.desc())
        )
        return list(result.scalars())

    async def get(self, order_id: str) -> OrderRow | None:
        return await self._session.get(OrderRow, order_id)

    async def delete(self, order_id: str) -> bool:
        """Delete one order row; ``False`` when it does not exist."""
        row = await self._session.get(OrderRow, order_id)
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.commit()
        return True

    async def delete_all(self) -> int:
        """Purge every order row; returns the count removed."""
        result = await self._session.execute(delete(OrderRow))
        await self._session.commit()
        return int(result.rowcount or 0)
