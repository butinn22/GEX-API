"""TBANK WebSocket streams via the tinkoff-invest SDK.

The SDK's streaming methods take a ``BaseStrategy`` whose ``on_candle`` /
``on_order_book`` / ``on_instrument_info`` callbacks receive decoded models.
We bridge those to domain types. The live path needs ``TBANK_TOKEN``; without it
the adapter is in dry-run and ``subscribe_*`` raise a clear error.

The mapping functions are pure and offline-testable; the SDK model field access
is verified against ``tinkoff-invest`` 1.0.5.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from trading.domain import Bar, BookLevel, DataFetchError, OrderBook

__all__ = ["candle_to_bar", "orderbook_to_domain", "TbankStream"]


def _to_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def candle_to_bar(candle: Any) -> Bar:
    """SDK ``Candle`` → domain ``Bar``."""
    return Bar(
        timestamp=_to_dt(candle.time),
        open=float(candle.open_price),
        high=float(candle.highest_price),
        low=float(candle.lowest_price),
        close=float(candle.close_price),
        volume=float(candle.volume),
    )


def orderbook_to_domain(order_book: Any) -> OrderBook:
    """SDK ``OrderBook`` → domain ``OrderBook`` (bids/asks are [price, qty] pairs)."""
    bids = tuple(BookLevel(float(l[0]), float(l[1])) for l in order_book.bids)
    asks = tuple(BookLevel(float(l[0]), float(l[1])) for l in order_book.asks)
    return OrderBook(bids=bids, asks=asks)


class TbankStream:
    """Subscribe to TBANK market streams; bridges SDK callbacks to domain callbacks."""

    def __init__(self, broker) -> None:
        self.broker = broker

    def _require_live(self) -> None:
        if self.broker.dry_run:
            raise DataFetchError("tbank stream requires a live token (dry-run has no streams)")

    def _strategy(self, on_bar=None, on_book=None, on_status=None):
        from tinkoff_invest.base_strategy import BaseStrategy

        class _Bridge(BaseStrategy):
            def on_candle(self, candle):
                if on_bar is not None:
                    on_bar(candle_to_bar(candle))

            def on_order_book(self, order_book):
                if on_book is not None:
                    on_book(orderbook_to_domain(order_book))

            def on_instrument_info(self, status):
                if on_status is not None:
                    on_status(status)

        return _Bridge()

    def subscribe_candles(self, figi: str, interval, on_bar: Callable[[Bar], None]) -> None:
        self._require_live()
        self.broker._get_session().subscribe_to_candles(figi, interval, self._strategy(on_bar=on_bar))

    def subscribe_order_book(self, figi: str, depth: int, on_book: Callable[[OrderBook], None]) -> None:
        self._require_live()
        self.broker._get_session().subscribe_to_order_book(figi, depth, self._strategy(on_book=on_book))
