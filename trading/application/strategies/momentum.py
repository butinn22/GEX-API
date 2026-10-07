"""Momentum: long when the rate-of-change is positive, flat when negative."""
from __future__ import annotations

from collections.abc import Sequence

from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = ["Momentum"]


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, symbol: str, period: int = 10) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self.symbol = symbol
        self.period = period
        self._closes: list[float] = []
        self._long = False

    async def start(self) -> None:
        self._closes = []
        self._long = False

    async def on_bar(self, bar: Bar) -> list[Signal]:
        self._closes.append(bar.close)
        if len(self._closes) < self.period + 1:
            return []
        roc = bar.close - self._closes[-self.period - 1]
        if roc > 0 and not self._long:
            self._long = True
            return [Signal(self.symbol, Side.BUY, self.name, "positive_momentum",
                           strength=1.0, price=Price(bar.close), timestamp=bar.timestamp)]
        if roc < 0 and self._long:
            self._long = False
            return [Signal(self.symbol, Side.SELL, self.name, "negative_momentum",
                           strength=1.0, price=Price(bar.close), timestamp=bar.timestamp)]
        return []

    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        if not bars:
            return []
        closes = [b.close for b in bars]
        signals: list[Signal] = []
        long = False
        for i in range(len(closes)):
            if i < self.period:
                continue
            roc = closes[i] - closes[i - self.period]
            if roc > 0 and not long:
                long = True
                signals.append(Signal(self.symbol, Side.BUY, self.name, "positive_momentum",
                                      strength=1.0, price=Price(closes[i]), timestamp=bars[i].timestamp))
            elif roc < 0 and long:
                long = False
                signals.append(Signal(self.symbol, Side.SELL, self.name, "negative_momentum",
                                      strength=1.0, price=Price(closes[i]), timestamp=bars[i].timestamp))
        return signals
