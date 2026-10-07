"""Portfolio / positions / orders / signals endpoints.

These are thin read/write surfaces over the broker adapters. Live broker calls
only happen when credentials are configured; otherwise they return empty/zero
views so the API stays usable without live keys.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.brokers.bingx import BingxBroker, BingxClient
from trading.adapters.brokers.tbank import TbankBroker
from trading.adapters.persistence.database import get_session
from trading.adapters.persistence.order_repository import OrderRepository
from trading.application.keys_service import KeysService
from trading.application.signal_hub import order_hub
from trading.config import settings
from trading.domain import (
    BrokerError,
    Order,
    OrderIntent,
    OrderType,
    Price,
    Quantity,
    Side,
)
from trading.observability import ORDERS_TOTAL
from trading.ports import BrokerAdapter

from ..deps import get_keys_service, require_auth
from ..schemas import BulkIdsRequest, OrderCreate, OrderOut

router = APIRouter(tags=["trading"])

#: Idempotency cache for ``POST /orders``: ``Idempotency-Key`` → (expiry, OrderOut).
#: A retried request (network timeout, double-click, client retry) returns the
#: first accepted order instead of placing a **duplicate live order**. The cache
#: is process-local (the deploy ships single-worker — the same assumption the
#: rate limiter documents) and bounded (LRU + TTL).
_ORDER_IDEMPOTENCY: OrderedDict[str, tuple[float, OrderOut]] = OrderedDict()
_ORDER_IDEMPOTENCY_MAX = 10_000
_ORDER_IDEMPOTENCY_TTL = 24 * 3600.0
_order_idem_lock = threading.Lock()
#: Per-idempotency-key serialisation locks. Bounded: the key is a caller-supplied
#: header, so without a cap a client could grow this dict without limit. Only
#: unlocked entries are evicted, so an in-flight request is never de-duplicated
#: away mid-flight.
_order_idem_locks: dict[str, asyncio.Lock] = {}
_ORDER_IDEM_LOCKS_MAX = 10_000


def _idem_lock(key: str) -> asyncio.Lock:
    lock = _order_idem_locks.get(key)
    if lock is not None:
        return lock
    lock = asyncio.Lock()
    _order_idem_locks[key] = lock
    if len(_order_idem_locks) > _ORDER_IDEM_LOCKS_MAX:
        for old_key in list(_order_idem_locks):
            if old_key != key and not _order_idem_locks[old_key].locked():
                _order_idem_locks.pop(old_key, None)
                if len(_order_idem_locks) <= _ORDER_IDEM_LOCKS_MAX:
                    break
    return lock


def _idem_get(key: str) -> OrderOut | None:
    with _order_idem_lock:
        entry = _ORDER_IDEMPOTENCY.get(key)
        if entry is None:
            return None
        expiry, order = entry
        if expiry < time.monotonic():
            _ORDER_IDEMPOTENCY.pop(key, None)
            return None
        _ORDER_IDEMPOTENCY.move_to_end(key)
        return order


def _idem_put(key: str, order: OrderOut) -> None:
    with _order_idem_lock:
        _ORDER_IDEMPOTENCY[key] = (time.monotonic() + _ORDER_IDEMPOTENCY_TTL, order)
        _ORDER_IDEMPOTENCY.move_to_end(key)
        while len(_ORDER_IDEMPOTENCY) > _ORDER_IDEMPOTENCY_MAX:
            _ORDER_IDEMPOTENCY.popitem(last=False)


async def _broker(
    exchange: str, session: AsyncSession, svc: KeysService
) -> BrokerAdapter | None:
    creds = await svc.resolve_credentials(session, exchange)
    if creds is None:
        return None
    if exchange == "bingx":
        return BingxBroker(
            BingxClient(creds["api_key"], creds["api_secret"], base_url=settings.bingx_base_url)
        )
    if exchange == "tbank":
        return TbankBroker(
            creds["api_key"],
            creds["extra"].get("account_id", ""),
            sandbox=settings.tbank_sandbox,
        )
    return None


async def _bingx_broker(session: AsyncSession, svc: KeysService) -> BingxBroker | None:
    broker = await _broker("bingx", session, svc)
    return broker if isinstance(broker, BingxBroker) else None


@router.get("/portfolio")
async def portfolio(
    session: AsyncSession = Depends(get_session),
    svc: KeysService = Depends(get_keys_service),
) -> dict:
    broker = await _bingx_broker(session, svc)
    if broker is None:
        return {"cash": 0.0, "currency": "USDT", "positions": [], "configured": False}
    try:
        pf = await broker.get_portfolio()
    except BrokerError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return {
        "cash": pf.cash,
        "currency": pf.currency,
        "realized_pnl": pf.realized_pnl,
        "positions": [
            {"symbol": p.symbol, "side": p.side.value, "quantity": p.quantity,
             "average_entry_price": p.average_entry_price, "unrealized_pnl": p.unrealized_pnl(0.0)}
            for p in pf.positions
        ],
        "configured": True,
    }


@router.get("/positions")
async def positions(
    session: AsyncSession = Depends(get_session),
    svc: KeysService = Depends(get_keys_service),
) -> list[dict]:
    broker = await _bingx_broker(session, svc)
    if broker is None:
        return []
    try:
        ps = await broker.get_positions()
    except BrokerError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return [
        {"symbol": p.symbol, "side": p.side.value, "quantity": p.quantity,
         "average_entry_price": p.average_entry_price, "realized_pnl": p.realized_pnl}
        for p in ps
    ]


@router.get("/orders", response_model=list[OrderOut])
async def orders(session: AsyncSession = Depends(get_session)) -> list[OrderOut]:
    rows = await OrderRepository(session).list()
    return [
        OrderOut(
            id=r.id, exchange=r.exchange, symbol=r.symbol, side=r.side,
            quantity=r.quantity, order_type=r.order_type, status=r.status,
            strategy=r.strategy, reason=r.reason,
        )
        for r in rows
    ]


@router.delete("/orders", dependencies=[Depends(require_auth)])
async def purge_orders(session: AsyncSession = Depends(get_session)) -> dict:
    """Purge the persisted order history (management action — audit §4)."""
    return {"deleted": await OrderRepository(session).delete_all()}


@router.delete(
    "/orders/{order_id}", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_auth)],
)
async def delete_order(
    order_id: str, session: AsyncSession = Depends(get_session)
) -> Response:
    """Delete one persisted order row (the local audit record, not a broker cancel)."""
    if not await OrderRepository(session).delete(order_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "order not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/orders", status_code=status.HTTP_202_ACCEPTED,
    response_model=OrderOut, dependencies=[Depends(require_auth)],
)
async def place_order(
    body: OrderCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: AsyncSession = Depends(get_session),
    svc: KeysService = Depends(get_keys_service),
) -> OrderOut:
    """Place an order; ``Idempotency-Key`` makes a retry safe (ENG-05).

    With a key, a replayed request returns the first result instead of placing a
    second live order. Without one, behaviour is unchanged.
    """
    if not idempotency_key:
        return await _place_order(body, session, svc)
    cached = _idem_get(idempotency_key)
    if cached is not None:
        return cached
    lock = _idem_lock(idempotency_key)
    async with lock:
        cached = _idem_get(idempotency_key)  # re-check: a concurrent twin may win
        if cached is not None:
            return cached
        out = await _place_order(body, session, svc)
        _idem_put(idempotency_key, out)
        return out


@router.post(
    "/orders/{order_id}/cancel", response_model=OrderOut,
    dependencies=[Depends(require_auth)],
)
async def cancel_order(
    order_id: str,
    session: AsyncSession = Depends(get_session),
    svc: KeysService = Depends(get_keys_service),
) -> OrderOut:
    """Cancel a live broker order (BingX is symbol-scoped).

    Distinct from ``DELETE /orders/{id}``, which only removes the local audit
    record. The order's stored symbol is combined with its id as
    ``"<symbol>:<orderId>"`` — the adapter's documented convention.
    """
    repo = OrderRepository(session)
    row = await repo.get(order_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "order not found")
    broker = await _broker(row.exchange, session, svc)
    if broker is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"no {row.exchange} credentials configured",
        )
    try:
        cancelled = await broker.cancel_order(f"{row.symbol}:{row.id}")
    except BrokerError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    await repo.update_status(row.id, cancelled.status.value)
    return OrderOut(
        id=row.id, exchange=row.exchange, symbol=row.symbol, side=row.side,
        quantity=row.quantity, order_type=row.order_type,
        status=cancelled.status.value, strategy=row.strategy, reason=row.reason,
    )


@router.post("/orders/bulk-delete", dependencies=[Depends(require_auth)])
async def bulk_delete_orders(
    body: BulkIdsRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete several local order records in one transaction (bulk action).

    Does not cancel live broker orders (use the per-order Cancel for that).
    Unknown ids are returned in ``missing``; the batch is idempotent.
    """
    deleted, missing = await OrderRepository(session).delete_many(body.ids)
    return {"deleted": deleted, "missing": missing}


async def _place_order(
    body: OrderCreate, session: AsyncSession, svc: KeysService
) -> OrderOut:
    broker = await _broker(body.exchange, session, svc)
    if broker is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"no {body.exchange} credentials configured",
        )
    intent = OrderIntent(
        symbol=body.symbol,
        side=Side.BUY if body.side == "buy" else Side.SELL,
        quantity=Quantity(body.quantity),
        order_type=OrderType.MARKET if body.order_type == "market" else OrderType.LIMIT,
        limit_price=Price(body.price) if body.price else None,
        strategy=body.strategy,
        reason=body.reason,
    )
    try:
        order: Order = await broker.place_order(intent)
    except BrokerError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    ORDERS_TOTAL.labels(
        exchange=body.exchange, side=order.side.value, status=order.status.value
    ).inc()
    await OrderRepository(session).create(order, body.exchange)
    order_hub.publish({
        "type": "order", "id": order.id, "exchange": body.exchange,
        "symbol": order.symbol, "side": order.side.value, "status": order.status.value,
    })
    return OrderOut(
        id=order.id, exchange=body.exchange, symbol=order.symbol,
        side=order.side.value, quantity=order.quantity,
        order_type=order.order_type.value, status=order.status.value,
        strategy=order.strategy, reason=order.reason,
    )
