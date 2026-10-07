"""Buy-and-hold: a single BUY at the first bar, hold to the end."""
from __future__ import annotations

from collections.abc import Sequence

from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = ["BuyAndHold"]


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._bought = False

    async def start(self) -> None:
        self._bought = False

    async def on_bar(self, bar: Bar) -> list[Signal]:
        if not self._bought:
            self._bought = True
            return [
                Signal(
                    self.symbol,
                    Side.BUY,
                    self.name,
                    "initial_entry",
                    strength=1.0,
                    price=Price(bar.close),
                    timestamp=bar.timestamp,
                )
            ]
        return []

    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        if not bars:
            return []
        return [
            Signal(
                self.symbol,
                Side.BUY,
                self.name,
                "initial_entry",
                strength=1.0,
                price=Price(bars[0].close),
                timestamp=bars[0].timestamp,
            )
        ]
