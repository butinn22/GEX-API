"""Trading domain ring — pure entities and value objects.

Rules (same spirit as ``gex/domain``):
  * allowed: stdlib only (``dataclasses``, ``enum``, ``datetime``, ``math``);
  * no numpy/pandas/network/db — keep the core importable and testable anywhere;
  * no imports from ``trading.ports``, ``trading.adapters``, ``trading.application``.

Vectorised/numpy representations of Bars live in the *data layer*, not here: a
domain ``Bar`` is one candle, a value object; the backtest engine works on
``pandas``/``numpy`` frames built *from* these.
"""

from .base import Bar, BookLevel, Instrument, OrderBook, Tick
from .enums import Exchange, OrderStatus, OrderType, PositionSide, Side, TimeInForce
from .errors import (
    BrokerError,
    DataFetchError,
    InsufficientFundsError,
    InvalidStateError,
    OrderRejectedError,
    RateLimitExceededError,
    RiskLimitError,
    StrategyError,
    TradingError,
)
from .money import Money, Price, Quantity
from .orders import Account, Fill, Order, OrderIntent, Portfolio, Position, Signal

__all__ = [
    # base
    "Bar",
    "BookLevel",
    "Instrument",
    "OrderBook",
    "Tick",
    # enums
    "Exchange",
    "OrderStatus",
    "OrderType",
    "PositionSide",
    "Side",
    "TimeInForce",
    # errors
    "BrokerError",
    "DataFetchError",
    "InsufficientFundsError",
    "InvalidStateError",
    "OrderRejectedError",
    "RateLimitExceededError",
    "RiskLimitError",
    "StrategyError",
    "TradingError",
    # money
    "Money",
    "Price",
    "Quantity",
    # orders
    "Account",
    "Fill",
    "Order",
    "OrderIntent",
    "Portfolio",
    "Position",
    "Signal",
]
