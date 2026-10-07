"""SMA-crossover: long when the fast SMA crosses above the slow SMA, flat below.

Crossings are detected with *lagged* moving averages (the previous bar's SMA
pair vs the current pair) so the signal never uses future data.
"""
from __future__ import annotations

from collections.abc import Sequence

from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = ["SmaCrossover"]


class SmaCrossover(Strategy):
    name = "sma_crossover"

    def __init__(self, symbol: str, fast: int = 20, slow: int = 50) -> None:
        if not (1 <= fast < slow):
            raise ValueError("require 1 <= fast < slow")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self._closes: list[float] = []
        self._long = False

    async def start(self) -> None:
        self._closes = []
        self._long = False

    async def on_bar(self, bar: Bar) -> list[Signal]:
        self._closes.append(bar.close)
        if len(self._closes) < self.slow + 1:  # need prev + current SMA pair
            return []
        fast_ma = sum(self._closes[-self.fast:]) / self.fast
        slow_ma = sum(self._closes[-self.slow:]) / self.slow
        prev_fast = sum(self._closes[-self.fast - 1 : -1]) / self.fast
        prev_slow = sum(self._closes[-self.slow - 1 : -1]) / self.slow

        crossed_up = prev_fast <= prev_slow and fast_ma > slow_ma
        crossed_down = prev_fast >= prev_slow and fast_ma < slow_ma

        if crossed_up and not self._long:
            self._long = True
            return [Signal(self.symbol, Side.BUY, self.name, "fast_cross_above",
                           strength=1.0, price=Price(bar.close), timestamp=bar.timestamp)]
        if crossed_down and self._long:
            self._long = False
            return [Signal(self.symbol, Side.SELL, self.name, "fast_cross_below",
                           strength=1.0, price=Price(bar.close), timestamp=bar.timestamp)]
        return []

    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        """Vectorised signal list (same logic as the incremental path)."""
        if not bars:
            return []
        closes = [b.close for b in bars]
        signals: list[Signal] = []
        long = False
        for i in range(len(closes)):
            if i < self.slow:
                continue
            fast_ma = sum(closes[i - self.fast + 1 : i + 1]) / self.fast
            slow_ma = sum(closes[i - self.slow + 1 : i + 1]) / self.slow
            prev_fast = sum(closes[i - self.fast : i]) / self.fast
            prev_slow = sum(closes[i - self.slow : i]) / self.slow
            crossed_up = prev_fast <= prev_slow and fast_ma > slow_ma
            crossed_down = prev_fast >= prev_slow and fast_ma < slow_ma
            if crossed_up and not long:
                long = True
                signals.append(Signal(self.symbol, Side.BUY, self.name, "fast_cross_above",
                                      strength=1.0, price=Price(bars[i].close), timestamp=bars[i].timestamp))
            elif crossed_down and long:
                long = False
                signals.append(Signal(self.symbol, Side.SELL, self.name, "fast_cross_below",
                                      strength=1.0, price=Price(bars[i].close), timestamp=bars[i].timestamp))
        return signals
