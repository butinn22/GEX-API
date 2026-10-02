"""Market-data domain objects: Instrument, Bar (OHLCV), Tick, OrderBook.

These are immutable value objects with no exchange-specific fields. Adapters
(fetchers) normalise vendor payloads into these; the backtest/data layers build
numpy/pandas structures from them. Timestamps are timezone-aware UTC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = ["Instrument", "Bar", "Tick", "BookLevel", "OrderBook"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Instrument:
    """A tradable symbol and its trading rules."""

    symbol: str
    exchange: "str"  # Exchange value; kept str to avoid an import cycle in annotations
    base_asset: str = ""
    quote_asset: str = ""
    tick_size: float = 0.0  # 0 = exchange does not constrain (crypto quotes)
    lot_size: float = 0.0  # 0 = fractional sizes allowed
    min_quantity: float = 0.0
    is_active: bool = True


@dataclass(frozen=True)
class Bar:
    """One OHLCV candle. ``timestamp`` is the bar's open time (UTC, tz-aware)."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= open/close/low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= open/close/high")

    @property
    def ts_ms(self) -> int:
        """Epoch milliseconds — convenient for storage/vectorised code."""
        return int(self.timestamp.timestamp() * 1000)


@dataclass(frozen=True)
class Tick:
    """A single trade/quote tick."""

    timestamp: datetime
    price: float
    volume: float = 0.0
    side: str = ""  # "buy"/"sell" when the feed distinguishes aggressor side


@dataclass(frozen=True, order=True)
class BookLevel:
    """One price level of an order book."""

    price: float
    quantity: float


@dataclass(frozen=True)
class OrderBook:
    """Snapshot of an order book. ``bids`` desc, ``asks`` asc (best first)."""

    bids: tuple[BookLevel, ...] = ()
    asks: tuple[BookLevel, ...] = ()
    timestamp: datetime = field(default_factory=utcnow)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    @property
    def spread(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a - b
