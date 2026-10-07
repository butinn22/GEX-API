"""Unified trend strategy — Trend Confluence core × EMF+ADL × Momentum.

The merge requested for the unified configuration: **Trend Confluence stays the
primary decision framework** (this class subclasses
:class:`~trading.application.strategies.trend_confluence.TrendConfluenceStrategy`,
inheriting regime detection, trendlines, confluence zones, options walls,
invalidation exits and the ATR trailing stop unchanged) and two restored
components are integrated on top of it:

* **EMF + ADL** (the ``gex_emf`` / ``EMAFilterTrendStrategy`` engine) — its full
  ``StrategySettings`` parameter set lives in the nested ``emf`` block, its
  combined entry columns gate or boost TC entries (``emf_mode``), its combined
  exit columns add exits (``use_emf_exits``) and its take-profit / trailing
  risk maths closes positions (``use_risk_exits``).
* **Momentum** — the rate-of-change filter from the standalone ``momentum``
  strategy (``momentum_period``): a direction gate (``momentum_mode="gate"``)
  or a strength bonus (``"bonus"``).

Why gates are applied in the overrides and not inside a rewritten state
machine: every override wraps the parent method and only *vetoes or re-sizes*
its result — so the parent's logic (including its position bookkeeping) stays
the single source of truth and cannot drift out of sync.

Batch vs streaming
------------------
Like the parent, the heavy work happens once in ``prepare`` (TC frame + EMF
feature frame + ROC array); ``on_bar`` is an O(1) lookup. The streaming path
(live trading) recomputes both frames over the accumulated prefix per bar —
expensive, but the bar cadence of live trading absorbs it (the parent already
does this for its own frame).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from trading.domain import Bar, Price, Side, Signal

from .gex_emf import MIN_BARS as EMF_MIN_BARS
from .gex_emf import _flag
from .trend_confluence import (
    OptionsWalls,
    TrendConfluenceParams,
    TrendConfluenceStrategy,
    _compute_frame,
)

__all__ = [
    "UnifiedTrendParams",
    "UnifiedTrendStrategy",
    "UNIFIED_STRATEGY_NAME",
    "UNIFIED_STRATEGY_VERSION",
    "UNIFIED_PARAM_NAMES",
]

UNIFIED_STRATEGY_NAME = "trend_confluence_unified"
UNIFIED_STRATEGY_VERSION = "1.0.0"

#: How the EMF+ADL combined entry interacts with a Trend-Confluence entry.
EMF_MODES = ("require", "bonus", "off")
#: How the momentum rate-of-change filter interacts with entries.
MOMENTUM_MODES = ("gate", "bonus", "off")


@dataclass(frozen=True)
class UnifiedTrendParams(TrendConfluenceParams):
    """Unified tunables — TC fields (inherited) + EMF+ADL + Momentum knobs.

    The full EMF+ADL ``StrategySettings`` field set is carried in the nested
    ``emf`` mapping (validated by ``StrategySettings.from_dict``); this keeps
    every EMF+ADL parameter available without name collisions with the TC
    fields (both strategies, e.g., define a ``use_trailing``).
    """

    # ── EMF + ADL integration ─────────────────────────────────────────
    use_emf: bool = True
    #: ``bonus`` (default) keeps every TC entry and sizes it up when EMF+ADL
    #: agrees — the unified strategy is then strictly an *enhanced* TC and can
    #: never go quiet because the two entry logics disagree. ``require`` is the
    #: strict intersection (higher precision, fewer trades) and is best left to
    #: the optimizer to select per ticker.
    emf_mode: str = "bonus"  # require | bonus | off
    emf_bonus: float = 0.25  # strength bonus when EMF agrees ("bonus" mode)
    use_emf_exits: bool = True  # EMF combined_*_exit columns as exits
    use_risk_exits: bool = True  # EMF take-profit / trailing overlay
    emf: Mapping[str, Any] = field(default_factory=dict)

    # ── Momentum integration ───────────────────────────────────────────
    use_momentum: bool = True
    momentum_period: int = 10  # the standalone strategy's ``period``
    momentum_mode: str = "bonus"  # gate | bonus | off
    momentum_bonus: float = 0.15  # strength bonus when ROC agrees

    def __post_init__(self) -> None:
        if self.emf_mode not in EMF_MODES:
            raise ValueError(f"emf_mode must be one of {EMF_MODES}, got {self.emf_mode!r}")
        if self.momentum_mode not in MOMENTUM_MODES:
            raise ValueError(
                f"momentum_mode must be one of {MOMENTUM_MODES}, got {self.momentum_mode!r}"
            )
        if self.momentum_period <= 0:
            raise ValueError("momentum_period must be > 0")
        if not self.emf_bonus >= 0.0 or not self.momentum_bonus >= 0.0:
            raise ValueError("emf_bonus / momentum_bonus must be >= 0")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> UnifiedTrendParams:
        if not raw:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k not in fields or v is None:
                continue
            kwargs[k] = v
        return cls(**kwargs)


#: Parameter names accepted by the unified strategy (registry/UI surface).
UNIFIED_PARAM_NAMES: list[str] = (
    [*list(TrendConfluenceParams.__dataclass_fields__), "use_emf", "emf_mode", "emf_bonus", "use_emf_exits", "use_risk_exits", "use_momentum", "momentum_period", "momentum_mode", "momentum_bonus", "emf", "options"]
)


def _to_ohlcv(bars: Sequence[Bar]) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
    })


class UnifiedTrendStrategy(TrendConfluenceStrategy):
    """Trend Confluence core, confirmed by EMF+ADL and sized by momentum."""

    name = UNIFIED_STRATEGY_NAME
    version = UNIFIED_STRATEGY_VERSION

    def __init__(
        self,
        symbol: str,
        *,
        params: Mapping[str, Any] | None = None,
        walls: OptionsWalls | None = None,
    ) -> None:
        raw = dict(params or {})
        self._up = UnifiedTrendParams.from_dict(raw)
        # The raw ``options`` block round-trips into ``resolved_params`` so a
        # full-config snapshot stays faithful to what was configured (the
        # *resolved* walls may embed live market data that must not be baked
        # into a stored preset).
        self._raw_options = dict(raw["options"]) if isinstance(raw.get("options"), Mapping) else None
        # The parent constructor filters to its own TrendConfluenceParams
        # fields (pops the nested ``options`` block itself).
        super().__init__(symbol, params=raw, walls=walls)

        from gex.strategy.settings import StrategySettings
        from gex.strategy.trading_algorithm import EMAFilterTrendStrategy

        emf_block = raw.get("emf")
        emf_block = dict(emf_block) if isinstance(emf_block, Mapping) else {}
        self._emf_settings = StrategySettings.from_dict(emf_block) if emf_block else StrategySettings()
        self._gex = EMAFilterTrendStrategy(settings=self._emf_settings)

        # EMF frame availability knobs: entries/exits need it only when used.
        self._needs_emf = (
            (self._up.use_emf and self._up.emf_mode != "off")
            or self._up.use_emf_exits
        )
        self._needs_risk = self._up.use_risk_exits
        self.min_bars = max(
            self.min_bars, EMF_MIN_BARS, self._up.momentum_period + 1,
        )
        self._reset_state()

    # ── configuration snapshot ──────────────────────────────────────────
    def resolved_params(self) -> dict[str, Any]:
        """The complete resolved parameter set (the full unified schema).

        Every tunable of the unified configuration — all Trend-Confluence
        fields, the integration knobs and the *resolved* EMF+ADL
        ``StrategySettings`` block — with defaults filled in. Presets and
        API-key configurations persist this snapshot so a saved preset is a
        complete, self-contained per-ticker config (requirements 5/9), not
        just the deltas that produced it. Re-feeding the result into
        ``UnifiedTrendStrategy(params=...)`` reproduces the same strategy.
        """
        import dataclasses

        out = dataclasses.asdict(self._up)
        # Replace the (possibly empty/partial) raw ``emf`` block with the
        # fully-resolved StrategySettings the adapter actually runs with.
        out["emf"] = dataclasses.asdict(self._emf_settings)
        if self._raw_options is not None:
            out["options"] = dict(self._raw_options)
        return out

    # ── state ───────────────────────────────────────────────────────────
    def _reset_state(self) -> None:
        super()._reset_state()
        self._emf_features: pd.DataFrame | None = None
        self._roc: np.ndarray | None = None
        self._entry_atr: float | None = None
        self.emf_skip_count = 0  # bars where a "require" gate passed blind

    async def start(self) -> None:
        self._reset_state()

    # ── batch path ─────────────────────────────────────────────────────
    async def prepare(self, bars: Sequence[Bar]) -> None:
        """Precompute TC frame (parent) + EMF feature frame + ROC once."""
        await super().prepare(bars)
        bars = list(bars)
        self._roc = self._roc_array(bars)
        if self._needs_emf and len(bars) >= EMF_MIN_BARS:
            try:
                self._emf_features = self._gex.calculate(
                    _to_ohlcv(bars), include_decorative=False
                )
            except Exception:  # an EMF failure must not kill TC entries
                self._emf_features = None

    def _roc_array(self, bars: Sequence[Bar]) -> np.ndarray:
        """Rate-of-change over ``momentum_period`` bars (0 during warm-up)."""
        p = self._up.momentum_period
        close = np.array([float(b.close) for b in bars], dtype=float)
        roc = np.zeros(len(close), dtype=float)
        if len(close) > p:
            roc[p:] = close[p:] - close[:-p]
        return roc

    # ── bar feeding ────────────────────────────────────────────────────
    async def on_bar(self, bar: Bar) -> list[Signal]:
        if self._frame is not None and self._cursor < len(self._ts):
            if self._matches_prepared(self._cursor, bar):
                # Parent advances the cursor and calls _signal_at (our
                # overrides); the EMF frame computed in prepare aligns by idx.
                return await super().on_bar(bar)
            self.fallback_count += 1
            self._frame = None
            self._closes = None
            self._highs = None
            self._lows = None
            self._cursor = 0
            self._emf_features = None

        # Streaming path (live): recompute both frames on the accumulated
        # prefix — same convention as the parent's streaming fallback.
        self._bars.append(bar)
        self._ts.append(bar.timestamp)
        if len(self._bars) < self.min_bars:
            return []
        highs = np.array([b.high for b in self._bars], dtype=float)
        lows = np.array([b.low for b in self._bars], dtype=float)
        self._frame = _compute_frame(
            np.array([b.open for b in self._bars], dtype=float),
            highs, lows,
            np.array([b.close for b in self._bars], dtype=float),
            self._p,
        )
        self._closes = np.array([float(b.close) for b in self._bars], dtype=float)
        self._highs = highs
        self._lows = lows
        self._roc = self._roc_array(self._bars)
        if self._needs_emf and len(self._bars) >= EMF_MIN_BARS:
            try:
                self._emf_features = self._gex.calculate(
                    _to_ohlcv(self._bars), include_decorative=False
                )
            except Exception:
                self._emf_features = None
        return self._drain(len(self._bars) - 1)

    # ── EMF + ADL integration ──────────────────────────────────────────
    def _emf_agrees(self, i: int, side: str) -> bool:
        """Does the EMF+ADL combined entry fire for ``side`` at bar ``i``?"""
        if self._emf_features is None or i >= len(self._emf_features):
            self.emf_skip_count += 1
            return True  # no confirmation available → do not block TC
        return _flag(self._emf_features.iloc[i], f"{side}_entry_signal")

    def _emf_gate(self, i: int, side: str) -> bool:
        if not self._up.use_emf or self._up.emf_mode != "require":
            return True
        return self._emf_agrees(i, side)

    def _emf_mult(self, i: int, side: str) -> float:
        if (
            self._up.use_emf
            and self._up.emf_mode == "bonus"
            and self._emf_agrees(i, side)
        ):
            return 1.0 + self._up.emf_bonus
        return 1.0

    # ── momentum integration ────────────────────────────────────────────
    def _momentum_gate(self, i: int, side: str) -> bool:
        if not self._up.use_momentum or self._up.momentum_mode != "gate":
            return True
        if self._roc is None or i >= len(self._roc):
            return True
        roc = float(self._roc[i])
        return roc > 0 if side == "long" else roc < 0

    def _momentum_mult(self, i: int, side: str) -> float:
        if not self._up.use_momentum or self._up.momentum_mode != "bonus":
            return 1.0
        if self._roc is None or i >= len(self._roc):
            return 1.0
        roc = float(self._roc[i])
        agrees = roc > 0 if side == "long" else roc < 0
        return 1.0 + self._up.momentum_bonus if agrees else 1.0

    # ── entry overrides: veto / re-size the parent's entries ────────────
    def _try_long(
        self, i: int, close: float, low_i: float, atr: float, tol: float,
        wall_tol: float, regime: int, size_mult: float, price: Price, timestamp,
    ) -> Signal | None:
        snapshot = self._state_snapshot()
        sig = super()._try_long(
            i, close, low_i, atr, tol, wall_tol, regime, size_mult, price, timestamp
        )
        return self._post_entry(sig, i, "long", atr, snapshot)

    def _try_short(
        self, i: int, close: float, high_i: float, atr: float, tol: float,
        wall_tol: float, regime: int, size_mult: float, price: Price, timestamp,
    ) -> Signal | None:
        snapshot = self._state_snapshot()
        sig = super()._try_short(
            i, close, high_i, atr, tol, wall_tol, regime, size_mult, price, timestamp
        )
        return self._post_entry(sig, i, "short", atr, snapshot)

    def _post_entry(
        self, sig: Signal | None, i: int, side: str, atr: float, snapshot
    ) -> Signal | None:
        """Apply the EMF/momentum gates to a parent entry, or veto it."""
        if sig is None:
            return None
        if not self._emf_gate(i, side) or not self._momentum_gate(i, side):
            # The parent already booked the position (_open); undo it so a
            # vetoed entry leaves the strategy exactly as it was.
            self._state_restore(snapshot)
            return None
        self._entry_atr = atr if atr and math.isfinite(atr) else None
        strength = min(1.0, sig.strength * self._emf_mult(i, side) * self._momentum_mult(i, side))
        strength = round(strength, 3)
        self._entry_strength = strength
        return replace(sig, strength=strength)

    # ── exit overrides: add EMF indicator + risk exits ───────────────────
    def _exit_check(
        self, i: int, close: float, atr: float, price: Price, timestamp, *, regime: int
    ) -> Signal | None:
        sig = super()._exit_check(i, close, atr, price, timestamp, regime=regime)
        if sig is not None:
            return sig
        if self._side == "flat" or self._entry_price is None:
            return None
        long = self._side == "long"
        s = self._entry_strength
        exit_side = Side.SELL if long else Side.BUY

        # 4. EMF+ADL combined indicator exit.
        if self._up.use_emf_exits and self._emf_features is not None and i < len(self._emf_features):
            row = self._emf_features.iloc[i]
            if _flag(row, "long_exit_signal" if long else "short_exit_signal"):
                self._close_position()
                return Signal(self.symbol, exit_side, self.name, "emf_indicator_exit",
                              strength=s, price=price, timestamp=timestamp, reduce_only=True)

        # 5. EMF+ADL take-profit / trailing-stop risk exit (same maths as the
        #    standalone gex_emf adapter; ATR frozen at entry).
        if self._needs_risk:
            risk = self._risk_exit(close, price, timestamp, long)
            if risk is not None:
                return risk
        return None

    def _risk_exit(self, close: float, price: Price, timestamp, long: bool) -> Signal | None:
        entry = self._entry_price
        if entry is None:  # pragma: no cover - defensive
            return None
        settings = self._emf_settings
        atr = self._entry_atr  # None → _RiskMixin falls back to percentages
        direction = "long" if long else "short"
        exit_side = Side.SELL if long else Side.BUY

        if long:
            stop = self._gex.trailing_stop_price(
                entry, direction, highest_price=self._best, atr_value=atr
            )
            if settings.use_trailing and stop is not None and close <= stop:
                self._close_position()
                return Signal(self.symbol, exit_side, self.name, "trailing_stop",
                              strength=self._entry_strength, price=price, timestamp=timestamp,
                              reduce_only=True)
            tp = self._gex.take_profit_price(entry, direction, atr)
            if settings.use_take_profit and close >= tp:
                self._close_position()
                return Signal(self.symbol, exit_side, self.name, "take_profit",
                              strength=self._entry_strength, price=price, timestamp=timestamp,
                              reduce_only=True)
        else:
            stop = self._gex.trailing_stop_price(
                entry, direction, lowest_price=self._best, atr_value=atr
            )
            if settings.use_trailing and stop is not None and close >= stop:
                self._close_position()
                return Signal(self.symbol, exit_side, self.name, "trailing_stop",
                              strength=self._entry_strength, price=price, timestamp=timestamp,
                              reduce_only=True)
            tp = self._gex.take_profit_price(entry, direction, atr)
            if settings.use_take_profit and close <= tp:
                self._close_position()
                return Signal(self.symbol, exit_side, self.name, "take_profit",
                              strength=self._entry_strength, price=price, timestamp=timestamp,
                              reduce_only=True)
        return None

    # ── position bookkeeping ────────────────────────────────────────────
    def _close_position(self) -> None:
        super()._close_position()
        self._entry_atr = None

    # ``on_tick`` / ``generate_signals`` are inherited: the parent's
    # ``generate_signals`` awaits ``self.prepare`` (our override) and replays
    # ``self._signal_at`` (our entry/exit overrides) unchanged.
