"""Signal generation pipeline: filters → transformers → aggregator.

A strategy emits raw signals; the pipeline gates and normalises them before they
reach sizing/execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from trading.domain import Signal

__all__ = ["SignalFilter", "MinStrengthFilter", "SignalPipeline"]


class SignalFilter(ABC):
    @abstractmethod
    def apply(self, signals: Sequence[Signal]) -> list[Signal]:
        """Return the subset of signals that pass this filter."""


class MinStrengthFilter(SignalFilter):
    def __init__(self, min_strength: float) -> None:
        if not 0 <= min_strength <= 1:
            raise ValueError("min_strength must be in [0, 1]")
        self.min_strength = min_strength

    def apply(self, signals: Sequence[Signal]) -> list[Signal]:
        return [s for s in signals if s.strength >= self.min_strength]


Aggregator = Callable[[list[Signal]], list[Signal]]


def dedupe_by_key(signals: list[Signal]) -> list[Signal]:
    """Keep only the latest signal per (symbol, side, reason)."""
    seen: dict[tuple, Signal] = {}
    for s in signals:
        seen[(s.symbol, s.side, s.reason)] = s
    return list(seen.values())


class SignalPipeline:
    def __init__(self, filters: Sequence[SignalFilter] | None = None,
                 aggregator: Aggregator | None = None) -> None:
        self.filters = list(filters or [])
        self.aggregator = aggregator or dedupe_by_key

    def run(self, signals: Sequence[Signal]) -> list[Signal]:
        out = list(signals)
        for f in self.filters:
            out = f.apply(out)
        return self.aggregator(out)
