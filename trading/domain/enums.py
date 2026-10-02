"""Trading enums: side, order type, time-in-force, order status, position, exchange.

Pure stdlib ``str`` enums so they serialise cleanly (API, JSON, DB) and compare
as plain strings where a legacy path expects one.
"""
from __future__ import annotations

from enum import Enum

__all__ = [
    "Side",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "PositionSide",
    "Exchange",
]


class Side(str, Enum):
    """Buy or sell. ``sign`` is +1 for BUY, -1 for SELL (PnL math)."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    GTC = "gtc"  # good till cancelled
    IOC = "ioc"  # immediate or cancel
    FOK = "fok"  # fill or kill
    DAY = "day"  # good till end of session


class OrderStatus(str, Enum):
    """Order lifecycle. ``is_terminal`` marks states that never transition further."""

    PENDING = "pending"  # accepted locally, not yet acknowledged by broker
    OPEN = "open"  # resting at the exchange
    PARTIAL = "partial"  # partially filled
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_active(self) -> bool:
        return self in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL)


class PositionSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """+1 long, -1 short, 0 flat (signed-quantity convention)."""
        return {PositionSide.FLAT: 0, PositionSide.LONG: 1, PositionSide.SHORT: -1}[self]


class Exchange(str, Enum):
    """Venues. Used to route fetchers/brokers and to key per-exchange rate rules."""

    MOEX = "moex"
    BYBIT = "bybit"
    BINGX = "bingx"
    TBANK = "tbank"
    YFINANCE = "yfinance"
    WEBULL = "webull"
