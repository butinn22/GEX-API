"""Strategy registry + live runner.

``StrategyRegistry`` maps a name to a factory (+ its tunable params) and is the
single source of truth for the strategy catalogue. ``StrategyRunner`` drives a
strategy bar-by-bar and fans generated signals out to the signal hub.
"""
from __future__ import annotations

from typing import Callable, Sequence

from trading.application.signal_hub import signal_hub
from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.application.strategies.dual_sma_crossover import DualSmaCrossover
from trading.application.strategies.mean_reversion import MeanReversion
from trading.application.strategies.momentum import Momentum
from trading.application.strategies.sma_crossover import SmaCrossover
from trading.domain import Bar, Signal
from trading.ports import Strategy

__all__ = ["StrategyRegistry", "STRATEGY_REGISTRY", "StrategyRunner"]

Factory = Callable[[str], Strategy]


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Factory] = {}
        self._params: dict[str, list[str]] = {}

    def register(self, name: str, factory: Factory, params: list[str] | None = None) -> None:
        self._registry[name] = factory
        self._params[name] = params or []

    def build(self, name: str, symbol: str, **params) -> Strategy:
        factory = self._registry.get(name)
        if factory is None:
            raise KeyError(f"strategy '{name}' not registered")
        return factory(symbol, **params)

    def names(self) -> list[str]:
        return sorted(self._registry)

    def params(self, name: str) -> list[str]:
        return list(self._params.get(name, []))


def _sma(symbol: str, **p) -> Strategy:
    return SmaCrossover(symbol, fast=p.get("fast", 20), slow=p.get("slow", 50))


def _mean_reversion(symbol: str, **p) -> Strategy:
    return MeanReversion(symbol, period=p.get("period", 20))


def _momentum(symbol: str, **p) -> Strategy:
    return Momentum(symbol, period=p.get("period", 10))


def _dual(symbol: str, **p) -> Strategy:
    return DualSmaCrossover(
        symbol,
        long_enabled=p.get("long_enabled", True),
        long_fast=p.get("long_fast", 10), long_slow=p.get("long_slow", 20),
        short_enabled=p.get("short_enabled", True),
        short_fast=p.get("short_fast", 10), short_slow=p.get("short_slow", 20),
    )


def _gex(symbol: str, **p) -> Strategy:
    from trading.application.strategies.gex_emf import GexEMFStrategy

    settings = {k: v for k, v in p.items() if v is not None}
    return GexEMFStrategy(symbol, settings=settings or None)


def _trend_confluence(symbol: str, **p) -> Strategy:
    from trading.application.strategies.trend_confluence import TrendConfluenceStrategy

    params = {k: v for k, v in p.items() if v is not None}
    return TrendConfluenceStrategy(symbol, params=params or None)


def _trend_confluence_unified(symbol: str, **p) -> Strategy:
    from trading.application.strategies.trend_confluence_unified import (
        UnifiedTrendStrategy,
    )

    params = {k: v for k, v in p.items() if v is not None}
    return UnifiedTrendStrategy(symbol, params=params or None)


STRATEGY_REGISTRY = StrategyRegistry()
STRATEGY_REGISTRY.register("sma_crossover", _sma, params=["fast", "slow"])
STRATEGY_REGISTRY.register("buy_and_hold", lambda symbol, **p: BuyAndHold(symbol))
STRATEGY_REGISTRY.register("mean_reversion", _mean_reversion, params=["period"])
STRATEGY_REGISTRY.register("momentum", _momentum, params=["period"])
STRATEGY_REGISTRY.register(
    "sma_crossover_ls", _dual,
    params=["long_fast", "long_slow", "short_fast", "short_slow"],
)
STRATEGY_REGISTRY.register(
    "gex_emf", _gex,
    params=["tp_percent", "trailing_percent", "atr_length", "rsi_length",
            "length_adl", "verification_threshold",
            "use_risk_exits", "use_atr_stops", "atr_tp_mult", "atr_sl_mult",
            "use_take_profit", "use_trailing"],
)
STRATEGY_REGISTRY.register(
    "trend_confluence", _trend_confluence,
    params=["ema_fast", "ema_mid", "ema_slow", "zone_atr", "min_confluence",
            "pullback_lookback", "use_trendlines", "trendline_refresh",
            "use_options_walls", "gamma_flip_filter", "respect_call_wall",
            "exit_at_call_wall", "allow_range", "range_size_mult",
            "allow_long", "allow_short", "atr_trail_mult", "options"],
)


def _unified_param_names() -> list[str]:
    from trading.application.strategies.trend_confluence_unified import (
        UNIFIED_PARAM_NAMES,
    )

    return list(UNIFIED_PARAM_NAMES)


STRATEGY_REGISTRY.register(
    "trend_confluence_unified", _trend_confluence_unified,
    params=_unified_param_names(),
)


def _trend_confluence_pine(symbol: str, **p) -> Strategy:
    from trading.application.strategies.trend_confluence_pine import (
        PineConfluenceStrategy,
    )

    params = {k: v for k, v in p.items() if v is not None}
    return PineConfluenceStrategy(symbol, params=params or None)


def _pine_param_names() -> list[str]:
    from trading.application.strategies.trend_confluence_pine import (
        PINE_PARAM_NAMES,
    )

    return list(PINE_PARAM_NAMES)


STRATEGY_REGISTRY.register(
    "trend_confluence_pine", _trend_confluence_pine,
    params=_pine_param_names(),
)


def _confluence_breakout(symbol: str, **p) -> Strategy:
    from trading.application.strategies.confluence_breakout import (
        ConfluenceBreakoutStrategy,
    )

    params = {k: v for k, v in p.items() if v is not None}
    preset = params.pop("preset", None)
    return ConfluenceBreakoutStrategy(symbol, params=params or None, preset=preset)


def _confluence_breakout_param_names() -> list[str]:
    from trading.application.strategies.confluence_breakout import (
        ConfluenceBreakoutParams,
    )

    return list(ConfluenceBreakoutParams.FIELD_NAMES)


STRATEGY_REGISTRY.register(
    "confluence_breakout", _confluence_breakout,
    params=_confluence_breakout_param_names(),
)


class StrategyRunner:
    """Drives a strategy over a bar stream and publishes signals to the hub."""

    def __init__(self, strategy: Strategy, *, publish: bool = True) -> None:
        self.strategy = strategy
        self.publish = publish

    async def run_bars(self, bars: Sequence[Bar]) -> list[Signal]:
        await self.strategy.start()
        signals: list[Signal] = []
        for bar in bars:
            for sig in await self.strategy.on_bar(bar):
                signals.append(sig)
                if self.publish:
                    signal_hub.publish({
                        "type": "signal", "strategy": sig.strategy, "symbol": sig.symbol,
                        "side": sig.side.value, "reason": sig.reason,
                        "timestamp": sig.timestamp.isoformat(),
                    })
        await self.strategy.shutdown()
        return signals
