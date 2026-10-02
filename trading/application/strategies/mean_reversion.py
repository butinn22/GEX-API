"""Mean-reversion: buy below the lower Bollinger band, exit at the middle band."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from trading.application.indicators import bollinger
from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = ["MeanReversion"]


class MeanReversion(Strategy):
    name = "mean_reversion"

    def __init__(self, symbol: str, period: int = 20, num_std: float = 2.0) -> None:
        self.symbol = symbol
        self.period = period
        self.num_std = num_std
        self._closes: list[float] = []
        self._long = False

    async def start(self) -> None:
        self._closes = []
        self._long = False

    async def on_bar(self, bar: Bar) -> list[Signal]:
        self._closes.append(bar.close)
        if len(self._closes) < self.period:
            return []
        middle, upper, lower = bollinger(self._closes, self.period, self.num_std)
        m, l = middle[-1], lower[-1]
        if np.isnan(m) or np.isnan(l):
            return []
        if not self._long and bar.close < l:
            self._long = True
            return [Signal(self.symbol, Side.BUY, self.name, "below_lower_band",
                           strength=1.0, price=Price(bar.close), timestamp=bar.timestamp)]
        if self._long and bar.close >= m:
            self._long = False
            return [Signal(self.symbol, Side.SELL, self.name, "mean_reverted",
                           strength=1.0, price=Price(bar.close), timestamp=bar.timestamp)]
        return []

    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        if not bars:
            return []
        closes = np.array([b.close for b in bars], dtype=float)
        middle, _, lower = bollinger(closes, self.period, self.num_std)
        signals: list[Signal] = []
        long = False
        for i in range(len(closes)):
            if np.isnan(lower[i]):
                continue
            if not long and closes[i] < lower[i]:
                long = True
                signals.append(Signal(self.symbol, Side.BUY, self.name, "below_lower_band",
                                      strength=1.0, price=Price(closes[i]), timestamp=bars[i].timestamp))
            elif long and closes[i] >= middle[i]:
                long = False
                signals.append(Signal(self.symbol, Side.SELL, self.name, "mean_reverted",
                                      strength=1.0, price=Price(closes[i]), timestamp=bars[i].timestamp))
        return signals
