"""Pine-confluence strategy — unified TC×EMF×ADL×Momentum core with the
TradingView "EMF MF + ADL STRAT" position management grafted on top.

What this adds over :class:`UnifiedTrendStrategy` (all user-facing settings):

* **Take profit (percent, partial)** — ``tp_enabled`` on/off, ``tp_percent``
  distance from entry, ``tp_close_pct`` of the position closed per hit,
  ``tp_cooldown_bars`` between partial closes (Pine: ``qty_percent=50`` with a
  3-bar cooldown; the level stays anchored at the *entry* price, so while price
  keeps trading beyond it the strategy keeps banking slices every cooldown).
* **Trailing stop (percent, independent of TP)** — ``trailing_enabled``
  on/off, ``trailing_percent`` distance. Ratchets off the extreme price since
  entry (``self._best``): longs exit when ``close <= best·(1 − pct/100)``,
  shorts when ``close >= best·(1 + pct/100)``. A separate switch from TP — each
  can be on without the other.
* **Adds (pyramiding)** — ``allow_adds`` (default off), ``add_cooldown_bars``,
  ``add_size_mult``. Add signals come from the EMF+ADL frame's combined
  add columns (the Pine ``combinedLongAdd``/``combinedShortAdd`` logic, already
  ported in :mod:`gex.strategy.signals`).

**Position-state guard (hard requirement):** entries only fire when flat
(parent), adds only fire when a position of the matching side is open, and no
exit (TP slice, trail, or any inherited exit) can be emitted without an open
position — every exit path starts with an explicit flat-check.

Exit priority in ``_exit_check``: percent trail → percent TP slice → the
inherited unified stack (hard ATR stop → walls → invalidation → EMF exits).
The subclass defaults turn the *other* profit-protectors off
(``use_risk_exits=False``, ``use_trailing=False``, ``tp_r=0.0``) so the two
percent knobs above are the authoritative TP/trailing controls; each can be
re-enabled explicitly via params. The ATR hard stop (``stop_atr``) stays on as
the catastrophic-loss guard.

Strength bookkeeping: the engine sizes every order as
``equity × fraction × strength / price``, so after a partial TP the remaining
position is ``(1 − tp_close_pct/100)`` of what an exit signal would close —
``_entry_strength`` is rescaled at each partial TP / add to keep full exits
close-only and partial exits proportional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trading.domain import Price, Side, Signal

from .gex_emf import _flag
from .trend_confluence_unified import (
    UNIFIED_PARAM_NAMES,
    UnifiedTrendParams,
    UnifiedTrendStrategy,
)

__all__ = [
    "PINE_PARAM_NAMES",
    "PineConfluenceParams",
    "PineConfluenceStrategy",
]

#: The strategy name registered in the factory / registry / UI.
PINE_STRATEGY_NAME = "trend_confluence_pine"


@dataclass(frozen=True)
class PineConfluenceParams(UnifiedTrendParams):
    """Unified tunables + the Pine-style position-management knobs."""

    # ── Take profit (percent-based, partial close) ────────────────────
    tp_enabled: bool = True       # Take profit on/off
    tp_percent: float = 2.0       # distance from entry price, in %
    tp_close_pct: float = 50.0    # % of the position closed per TP hit
    tp_cooldown_bars: int = 3     # bars between partial TP closes
    # ── Trailing stop (percent-based, independent of TP) ──────────────
    trailing_enabled: bool = True  # Trailing stop on/off
    trailing_percent: float = 1.0  # trail distance, in % of price
    # ── Adds (pyramiding) ─────────────────────────────────────────────
    allow_adds: bool = False      # add to an open position (in-position only)
    add_cooldown_bars: int = 10   # bars between adds
    add_size_mult: float = 0.25   # add size relative to the entry strength

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tp_percent <= 0:
            raise ValueError("tp_percent must be > 0")
        if not 0.0 < self.tp_close_pct <= 100.0:
            raise ValueError("tp_close_pct must be in (0, 100]")
        if self.tp_cooldown_bars < 0:
            raise ValueError("tp_cooldown_bars must be >= 0")
        if self.trailing_percent <= 0:
            raise ValueError("trailing_percent must be > 0")
        if self.add_cooldown_bars < 0:
            raise ValueError("add_cooldown_bars must be >= 0")
        if self.add_size_mult <= 0:
            raise ValueError("add_size_mult must be > 0")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "PineConfluenceParams":
        if not raw:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k not in fields or v is None:
                continue
            kwargs[k] = v
        return cls(**kwargs)


#: Parameter names accepted by the Pine-confluence strategy (registry/UI).
PINE_PARAM_NAMES: list[str] = list(UNIFIED_PARAM_NAMES) + [
    "tp_enabled", "tp_percent", "tp_close_pct", "tp_cooldown_bars",
    "trailing_enabled", "trailing_percent",
    "allow_adds", "add_cooldown_bars", "add_size_mult",
]


class PineConfluenceStrategy(UnifiedTrendStrategy):
    """Unified confluence core + Pine percent TP/trailing and guarded adds."""

    name = PINE_STRATEGY_NAME

    def __init__(
        self,
        symbol: str,
        *,
        params: Mapping[str, Any] | None = None,
        walls=None,
    ) -> None:
        raw = dict(params or {})
        self._pp = PineConfluenceParams.from_dict(raw)
        # The percent TP/trailing knobs are the authoritative profit protectors
        # here: default the overlapping overlays off (explicit params win).
        raw.setdefault("use_risk_exits", False)   # EMF ATR TP/trailing overlay
        raw.setdefault("use_trailing", False)     # TC ATR trailing stop
        raw.setdefault("tp_r", 0.0)               # TC R-multiple take profit
        super().__init__(symbol, params=raw, walls=walls)

    # ── state ─────────────────────────────────────────────────────────
    def _reset_state(self) -> None:
        super()._reset_state()
        self._last_tp_bar = -10**9
        self._last_add_bar = -10**9

    # ── exits: percent trail → percent TP → inherited stack ───────────
    def _exit_check(
        self, i: int, close: float, atr: float, price: Price, timestamp, *, regime: int
    ) -> Signal | None:
        # Position-state guard: never emit ANY exit without an open position.
        if self._side == "flat" or self._entry_price is None:
            return None
        p = self._pp
        long = self._side == "long"
        exit_side = Side.SELL if long else Side.BUY

        # 1. Percent trailing stop (independent of TP).
        if p.trailing_enabled and self._best is not None:
            band = p.trailing_percent / 100.0
            trail = self._best * (1.0 - band) if long else self._best * (1.0 + band)
            hit = close <= trail if long else close >= trail
            if hit:
                self._close_position()
                return Signal(self.symbol, exit_side, self.name, "trailing_stop_pct",
                              strength=self._entry_strength, price=price,
                              timestamp=timestamp)

        # 2. Percent take profit — partial close with cooldown. The level is
        #    anchored at the entry price (Pine semantics), so successive hits
        #    keep slicing the remainder every cooldown while price holds.
        if p.tp_enabled and p.tp_close_pct > 0:
            band = p.tp_percent / 100.0
            tp = self._entry_price * (1.0 + band) if long \
                else self._entry_price * (1.0 - band)
            beyond = close >= tp if long else close <= tp
            if beyond and (i - self._last_tp_bar) >= p.tp_cooldown_bars:
                self._last_tp_bar = i
                part = p.tp_close_pct / 100.0
                strength = round(min(1.0, self._entry_strength * part), 3)
                # Rescale the booked strength so later full exits match the
                # reduced position and later TP slices stay proportional.
                self._entry_strength = round(
                    min(1.0, self._entry_strength * (1.0 - part)), 3)
                return Signal(self.symbol, exit_side, self.name, "take_profit_pct",
                              strength=strength, price=price, timestamp=timestamp)

        # 3. Inherited unified stack (hard ATR stop, walls, invalidation,
        #    EMF indicator exits).
        return super()._exit_check(i, close, atr, price, timestamp, regime=regime)

    # ── adds: only into an open position of the matching side ─────────
    def _signal_at(self, i: int, timestamp) -> list[Signal]:
        out = super()._signal_at(i, timestamp)
        # Position-state guard: adds never open a position, and never stack
        # onto a bar that already produced an exit.
        if self._pp.allow_adds and self._side != "flat" and not out:
            sig = self._try_add(i, timestamp)
            if sig is not None:
                out.append(sig)
        return out

    def _try_add(self, i: int, timestamp) -> Signal | None:
        p = self._pp
        # Position-state guard, enforced here too (not only in _signal_at) so a
        # direct call can never add to a flat book.
        if self._side == "flat" or self._entry_price is None:
            return None
        if (i - self._last_add_bar) < p.add_cooldown_bars:
            return None
        if self._emf_features is None or i >= len(self._emf_features):
            return None
        assert self._closes is not None
        long = self._side == "long"
        row = self._emf_features.iloc[i]
        if not _flag(row, "long_add_signal" if long else "short_add_signal"):
            return None
        self._last_add_bar = i
        # Grow the booked strength so a later full exit still closes (close to)
        # the whole pyramided position.
        self._entry_strength = round(
            min(1.0, self._entry_strength * (1.0 + p.add_size_mult)), 3)
        return Signal(
            self.symbol, Side.BUY if long else Side.SELL, self.name,
            "add_long" if long else "add_short",
            strength=round(min(1.0, self._entry_strength
                               * (p.add_size_mult / (1.0 + p.add_size_mult))), 3),
            price=Price(float(self._closes[i])), timestamp=timestamp,
        )
