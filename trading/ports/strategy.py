"""Strategy port — the pluggable contract for every trading strategy.

A strategy consumes market data (bars/ticks) and emits ``Signal`` objects. It
never talks to a broker directly: the engine takes the signals, applies risk +
sizing to form ``OrderIntent``s, and routes those. Lifecycle hooks
(``start``/``shutdown``) let stateful strategies warm up indicators or flush
state.

The existing Pine→Python strategy in ``gex.strategy`` will be re-issued behind
this ABC so it can run in the same engine as the example strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

from trading.domain import Bar, Signal, Tick

__all__ = ["Strategy"]


class Strategy(ABC):
    """Pluggable trading strategy."""

    name: ClassVar[str]

    @abstractmethod
    async def on_bar(self, bar: Bar) -> list[Signal]:
        """Process one closed bar; return any signals it triggered."""
        ...

    @abstractmethod
    async def on_tick(self, tick: Tick) -> list[Signal]:
        """Process one tick (intra-bar, for tick-driven strategies)."""
        ...

    @abstractmethod
    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        """Process a batch of bars (used by backtest and warm-up)."""
        ...

    async def prepare(self, bars: Sequence[Bar]) -> None:
        """Optional batch pre-computation over the *whole* replay.

        The backtest engine calls this exactly once, with the full sorted bar
        sequence, before the event loop starts feeding bars to ``on_bar``.
        Strategies whose indicators are batch-oriented can compute their feature
        frame here once and turn ``on_bar`` into an O(1) lookup, instead of
        recomputing over the whole accumulated history on every bar.

        Why this matters: a batch strategy that recomputes on each bar is
        O(n²) over a replay. At 1,500 bars the EMF+ADL port took ~7 minutes;
        with ``prepare`` it is a single O(n) pass.

        Default: no-op, so streaming/tick strategies are unaffected and the
        same code path keeps working when there is no prior knowledge of the
        future bars (live trading simply never calls this).
        """

    async def shutdown(self) -> None:
        """Flush/cleanup hook. Default no-op."""
