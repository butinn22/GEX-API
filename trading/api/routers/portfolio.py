"""Portfolio / positions / orders / signals endpoints.

These are thin read/write surfaces over the broker adapters. Live broker calls
only happen when credentials are configured; otherwise they return empty/zero
views so the API stays usable without live keys.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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
from ..schemas import OrderCreate, OrderOut

router = APIRouter(tags=["trading"])


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


@router.post(
    "/orders", status_code=status.HTTP_202_ACCEPTED,
    response_model=OrderOut, dependencies=[Depends(require_auth)],
)
async def place_order(
    body: OrderCreate,
    session: AsyncSession = Depends(get_session),
    svc: KeysService = Depends(get_keys_service),
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
