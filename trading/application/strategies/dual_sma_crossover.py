"""Dual SMA crossover — independent long and short legs.

Each side has its own enabled flag, fast/slow windows, and signal strength, so
long and short can be tuned (or disabled) independently. The legs are mutually
exclusive: a long entry only fires when flat, a short entry only when flat.

Signals:
  long leg:  fast crosses above slow → BUY (enter long); crosses below → SELL (exit long)
  short leg: fast crosses below slow → SELL (enter short); crosses above → BUY (exit short)
"""
from __future__ import annotations

from typing import Sequence

from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = ["DualSmaCrossover"]


class DualSmaCrossover(Strategy):
    name = "sma_crossover_ls"

    def __init__(
        self,
        symbol: str,
        *,
        long_enabled: bool = True,
        long_fast: int = 10,
        long_slow: int = 20,
        long_strength: float = 1.0,
        short_enabled: bool = True,
        short_fast: int = 10,
        short_slow: int = 20,
        short_strength: float = 1.0,
    ) -> None:
        for name, fast, slow in (("long", long_fast, long_slow), ("short", short_fast, short_slow)):
            if not (1 <= fast < slow):
                raise ValueError(f"{name} windows require 1 <= fast < slow")
        self.symbol = symbol
        self.long_enabled = long_enabled
        self.long_fast, self.long_slow, self.long_strength = long_fast, long_slow, long_strength
        self.short_enabled = short_enabled
        self.short_fast, self.short_slow, self.short_strength = short_fast, short_slow, short_strength
        self._closes: list[float] = []
        self._long = False
        self._short = False

    async def start(self) -> None:
        self._closes = []
        self._long = False
        self._short = False

    def _signals_at(self, closes: Sequence[float], i: int, timestamp) -> list[Signal]:
        n = i + 1
        signals: list[Signal] = []
        price = Price(float(closes[i]))

        if self.long_enabled and n >= self.long_slow + 1:
            fast_ma = sum(closes[i - self.long_fast + 1 : i + 1]) / self.long_fast
            slow_ma = sum(closes[i - self.long_slow + 1 : i + 1]) / self.long_slow
            prev_fast = sum(closes[i - self.long_fast : i]) / self.long_fast
            prev_slow = sum(closes[i - self.long_slow : i]) / self.long_slow
            if prev_fast <= prev_slow and fast_ma > slow_ma and not self._long and not self._short:
                self._long = True
                signals.append(Signal(self.symbol, Side.BUY, self.name, "long_entry",
                                      strength=self.long_strength, price=price, timestamp=timestamp))
            elif prev_fast >= prev_slow and fast_ma < slow_ma and self._long:
                self._long = False
                signals.append(Signal(self.symbol, Side.SELL, self.name, "long_exit",
                                      strength=self.long_strength, price=price, timestamp=timestamp))

        if self.short_enabled and n >= self.short_slow + 1:
            fast_ma = sum(closes[i - self.short_fast + 1 : i + 1]) / self.short_fast
            slow_ma = sum(closes[i - self.short_slow + 1 : i + 1]) / self.short_slow
            prev_fast = sum(closes[i - self.short_fast : i]) / self.short_fast
            prev_slow = sum(closes[i - self.short_slow : i]) / self.short_slow
            if prev_fast >= prev_slow and fast_ma < slow_ma and not self._short and not self._long:
                self._short = True
                signals.append(Signal(self.symbol, Side.SELL, self.name, "short_entry",
                                      strength=self.short_strength, price=price, timestamp=timestamp))
            elif prev_fast <= prev_slow and fast_ma > slow_ma and self._short:
                self._short = False
                signals.append(Signal(self.symbol, Side.BUY, self.name, "short_exit",
                                      strength=self.short_strength, price=price, timestamp=timestamp))
        return signals

    async def on_bar(self, bar: Bar) -> list[Signal]:
        self._closes.append(bar.close)
        return self._signals_at(self._closes, len(self._closes) - 1, bar.timestamp)

    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        if not bars:
            return []
        closes = [b.close for b in bars]
        signals: list[Signal] = []
        for i in range(len(closes)):
            signals.extend(self._signals_at(closes, i, bars[i].timestamp))
        return signals
