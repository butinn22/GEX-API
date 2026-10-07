"""Native exchange order payload mapping for the local signal client API.

Local signal clients push *raw* signal payloads; this module normalises them
into domain :class:`Signal`s and renders the **native order format** of each
exchange, so a client receives a payload it can fire at the exchange verbatim:

* BingX Swap v3 — parameters of ``POST /openApi/swap/v3/trade/order``
  (attached TP/SL are embedded as JSON strings, per the BingX docs).
* TBANK T-Invest API — ``PostOrderRequest`` fields (figi, lots, direction,
  order_type, Quotation price, client order id).

Pure functions, no I/O — easy to unit-test and reuse from the WS endpoint.
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from decimal import ROUND_DOWN, Decimal
from typing import Any

from trading.domain import OrderType, Price, Quantity, Side, Signal

__all__ = [
    "SignalParseError",
    "signal_from_raw",
    "to_bingx_order_payload",
    "to_tbank_order_payload",
    "native_payloads_for",
    "to_quotation",
]

_SIDE_ALIASES = {
    "buy": Side.BUY,
    "long": Side.BUY,
    "sell": Side.SELL,
    "short": Side.SELL,
}

_BINGX_ORDER_TYPES = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_MARKET",
    OrderType.STOP: "STOP",
}

_TBANK_DIRECTION = {
    Side.BUY: "ORDER_DIRECTION_BUY",
    Side.SELL: "ORDER_DIRECTION_SELL",
}

_TBANK_ORDER_TYPES = {
    OrderType.MARKET: "ORDER_TYPE_MARKET",
    OrderType.LIMIT: "ORDER_TYPE_LIMIT",
}


class SignalParseError(ValueError):
    """A raw signal payload could not be parsed or mapped."""


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def signal_from_raw(payload: Any, *, source: str | None = None) -> Signal:
    """Parse a raw client signal payload (TBANK/BINGX/generic) into a Signal.

    Accepts common aliases: ``symbol|ticker|instrument``,
    ``side|action|direction`` (buy/long/sell/short), ``strength|confidence``.
    """
    if not isinstance(payload, Mapping):
        raise SignalParseError("signal payload must be a JSON object")
    symbol = _first(payload, "symbol", "ticker", "instrument")
    if not symbol or not isinstance(symbol, str):
        raise SignalParseError("signal payload requires a symbol/ticker string")
    side_raw = _first(payload, "side", "action", "direction")
    if side_raw is None:
        raise SignalParseError("signal payload requires a side/action")
    side = _SIDE_ALIASES.get(str(side_raw).strip().lower())
    if side is None:
        raise SignalParseError(f"unknown signal side: {side_raw!r}")

    strength_raw = _first(payload, "strength", "confidence")
    price_raw = _first(payload, "price", "limit_price")
    qty_raw = _first(payload, "quantity", "qty", "size")
    return Signal(
        symbol=symbol.strip().upper(),
        side=side,
        strategy=str(_first(payload, "strategy") or source or "local-client"),
        reason=str(_first(payload, "reason", "message") or "client-signal"),
        strength=float(strength_raw) if strength_raw is not None else 1.0,
        price=Price(float(price_raw)) if price_raw is not None else None,
        quantity=Quantity(float(qty_raw)) if qty_raw is not None else None,
    )


# ── BingX Swap v3 ────────────────────────────────────────────────────


def _bingx_conditional(kind: str, stop_price: float) -> str:
    """Attached TP/SL, sent as a URL-encoded JSON string per BingX docs."""
    return json.dumps(
        {"type": kind, "stopPrice": stop_price, "workingType": "MARK_PRICE"},
        separators=(",", ":"),
    )


def to_bingx_order_payload(
    signal: Signal,
    *,
    quantity: float | None = None,
    position_side: str | None = None,
    order_type: OrderType = OrderType.MARKET,
    price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """Render the native BingX v3 ``/trade/order`` parameter set."""
    qty = quantity
    if qty is None and signal.quantity is not None:
        qty = signal.quantity.value
    if qty is None or qty <= 0:
        raise SignalParseError("bingx payload requires a positive quantity")

    bingx_type = _BINGX_ORDER_TYPES.get(order_type)
    if bingx_type is None:
        raise SignalParseError(f"bingx: unsupported order type {order_type.value}")

    params: dict[str, Any] = {
        "symbol": signal.symbol,
        "side": signal.side.value.upper(),
        "positionSide": position_side or ("LONG" if signal.side is Side.BUY else "SHORT"),
        "type": bingx_type,
        "quantity": qty,
    }
    if order_type is OrderType.LIMIT:
        limit = price if price is not None else (
            signal.price.value if signal.price else None
        )
        if limit is None:
            raise SignalParseError("bingx LIMIT payload requires a price")
        params["price"] = limit
        params["timeInForce"] = "GTC"
    if take_profit is not None:
        params["takeProfit"] = _bingx_conditional("TAKE_PROFIT_MARKET", take_profit)
    if stop_loss is not None:
        params["stopLoss"] = _bingx_conditional("STOP_MARKET", stop_loss)
    if client_order_id:
        params["clientOrderID"] = re.sub(r"[^A-Za-z0-9_]", "_", client_order_id)[:40]
    return params


# ── TBANK T-Invest API ───────────────────────────────────────────────


def to_quotation(value: float) -> dict[str, Any]:
    """T-Invest ``Quotation``: integer ``units`` + ``nano`` (1e-9) fraction."""
    d = Decimal(str(value))
    units = int(d.to_integral_value(rounding=ROUND_DOWN))
    nano = int((d - units) * Decimal(1_000_000_000))
    return {"units": str(units), "nano": nano}


def to_tbank_order_payload(
    signal: Signal,
    *,
    account_id: str = "",
    figi: str = "",
    instrument_id: str = "",
    lots: int | None = None,
    order_type: OrderType = OrderType.MARKET,
    price: float | None = None,
) -> dict[str, Any]:
    """Render the native T-Invest ``PostOrderRequest`` field set.

    TBANK trades whole **lots**; fractional quantities are floored and an order
    for less than one lot is rejected.
    """
    if lots is None:
        if signal.quantity is None:
            raise SignalParseError("tbank payload requires a quantity (lots)")
        lots = int(signal.quantity.value)
    if lots < 1:
        raise SignalParseError("tbank order quantity must be >= 1 lot")

    tbank_type = _TBANK_ORDER_TYPES.get(order_type)
    if tbank_type is None:
        raise SignalParseError(f"tbank: unsupported order type {order_type.value}")

    payload: dict[str, Any] = {
        "figi": figi,
        "quantity": lots,
        "direction": _TBANK_DIRECTION[signal.side],
        "account_id": account_id,
        "order_type": tbank_type,
        "order_id": uuid.uuid4().hex,
    }
    if instrument_id:
        payload["instrument_id"] = instrument_id
    if order_type is OrderType.LIMIT:
        limit = price if price is not None else (
            signal.price.value if signal.price else None
        )
        if limit is None:
            raise SignalParseError("tbank LIMIT payload requires a price")
        payload["price"] = to_quotation(limit)
    return payload


def native_payloads_for(signal: Signal, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Render both native payloads for a signal in one call.

    ``kwargs`` are forwarded to the two renderers; split them by prefix where
    they collide (``tbank_*`` kwargs go to the TBANK renderer).
    """
    tbank_kwargs = {k[len("tbank_"):]: v for k, v in kwargs.items() if k.startswith("tbank_")}
    common = {k: v for k, v in kwargs.items() if not k.startswith("tbank_")}
    return {
        "bingx": to_bingx_order_payload(signal, **common),
        "tbank": to_tbank_order_payload(signal, **tbank_kwargs),
    }
