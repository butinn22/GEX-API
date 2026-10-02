"""Domain exception hierarchy.

One root (``TradingError``) so callers can catch the whole family, with
meaningful subclasses per failure mode. The API layer maps these to HTTP codes;
broker adapters translate vendor errors into these types so nothing vendor-
specific leaks past the anticorruption layer.
"""
from __future__ import annotations

__all__ = [
    "TradingError",
    "DataFetchError",
    "BrokerError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "RateLimitExceededError",
    "StrategyError",
    "InvalidStateError",
]


class TradingError(Exception):
    """Root of the trading domain exception hierarchy."""


class DataFetchError(TradingError):
    """A market-data source could not be fetched (network, provider, malformed)."""


class BrokerError(TradingError):
    """A broker call failed (rejected, timeout, auth, vendor error)."""


class OrderRejectedError(BrokerError):
    """The broker refused an order (invalid price/qty, market closed, ...)."""


class InsufficientFundsError(BrokerError):
    """Not enough balance / buying power / margin to place the order."""


class RateLimitExceededError(TradingError):
    """A rate-limit budget is exhausted and the call must back off."""


class StrategyError(TradingError):
    """A strategy raised (bug in indicator/signal logic)."""


class InvalidStateError(TradingError):
    """An illegal state transition (e.g. order PENDING → FILLED directly)."""
