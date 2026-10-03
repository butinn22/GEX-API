"""Trend-confluence strategy — EMA regime + trendlines + options walls.

Implements the confluence pullback logic:

1. **Trend regime** — BULLISH when price is above EMA50, above EMA200 and above
   the relevant active trendline from :mod:`gex.domain.trendlines` (the existing
   Pinescript port — trendlines are *not* recalculated from scratch here);
   BEARISH is the mirror. When the evidence conflicts the regime is RANGE and
   the strategy either stays flat (``allow_range=False``) or trades at reduced
   size (``range_size_mult``).

2. **Add-long** — in a bullish regime, when price pulls back into a confluence
   zone made of at least ``min_confluence`` of: EMA20 / EMA50 / EMA200, an
   active trendline support, a high-OI options wall, the PUT wall, the CALL
   wall (only when gamma exposure supports continuation) or the gamma flip
   level acting as support.

3. **Add-short** — the exact mirror in a bearish regime.

Options walls
-------------
Historical option chains are not available to the backtest data sources, so
walls arrive as a **snapshot** via the ``options`` param block::

    {"enabled": true, "call_wall": 700.0, "put_wall": 600.0,
     "gamma_flip": 650.0,
     "walls": [{"strike": 620.0, "oi": 50000.0, "kind": "put"}],
     "min_wall_oi": 10000.0}

Wall helpers themselves (:func:`gex.domain.options.levels.gamma_flip_cumulative`
etc.) are reused when a caller passes raw strike/GEX arrays instead of ready
levels — see :func:`resolve_walls`.

Performance
-----------
Like :class:`~trading.application.strategies.gex_emf.GexEMFStrategy` the heavy
work happens once in ``prepare`` (O(n) EMAs + one trendline pass every
``trendline_refresh`` bars; lines are extrapolated forward between refreshes,
which is exactly what the Pine original does with ``extend=right``). ``on_bar``
is then an O(1) row lookup with a streaming fallback for live trading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from trading.domain import Bar, Price, Side, Signal, Tick
from trading.ports import Strategy

__all__ = [
    "OptionsWalls",
    "TrendConfluenceParams",
    "TrendConfluenceStrategy",
    "resolve_walls",
]

#: Minimum history the strategy needs before it will trade (EMA200 warm-up).
DEFAULT_MIN_BARS = 220


# ────────────────────────────────────────────────────────────────────── #
#  Configuration
# ────────────────────────────────────────────────────────────────────── #
@dataclass(frozen=True)
class Wall:
    """One significant options strike (high-OI cluster or single-strike wall)."""

    strike: float
    oi: float
    kind: str = "cluster"  # "call" | "put" | "cluster"


@dataclass(frozen=True)
class OptionsWalls:
    """Options-market snapshot used for confluence.

    ``call_wall`` / ``put_wall`` / ``gamma_flip`` are the headline levels;
    ``walls`` holds extra significant strikes (high-OI clusters). Any level may
    be ``None`` — missing data simply removes that confluence candidate.
    """

    enabled: bool = False
    call_wall: float | None = None
    put_wall: float | None = None
    gamma_flip: float | None = None
    walls: tuple[Wall, ...] = ()
    min_wall_oi: float = 0.0

    def significant_walls(self) -> list[Wall]:
        return [w for w in self.walls if w.oi >= self.min_wall_oi and w.strike > 0]


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def resolve_walls(raw: Mapping[str, Any] | None) -> OptionsWalls:
    """Build :class:`OptionsWalls` from the API params block.

    Accepts either ready levels (``call_wall``/``put_wall``/``gamma_flip``) or
    raw chain aggregates (``strikes`` + ``gex_net`` [+ ``oi_call``/``oi_put``]),
    in which case the canonical helpers from
    :mod:`gex.domain.options.levels` derive the levels.
    """
    if not raw:
        return OptionsWalls()
    enabled = bool(raw.get("enabled", True))
    call_wall = _f(raw.get("call_wall"))
    put_wall = _f(raw.get("put_wall"))
    gamma_flip = _f(raw.get("gamma_flip"))

    strikes = raw.get("strikes")
    gex_net = raw.get("gex_net")
    if strikes and gex_net and len(strikes) == len(gex_net):
        from gex.domain.options.levels import (
            gamma_flip_cumulative,
            primary_walls_by_gex,
            walls_by_oi,
        )

        if call_wall is None or put_wall is None:
            cw, pw = primary_walls_by_gex(strikes, gex_net)
            call_wall = call_wall or (None if math.isnan(cw) else cw)
            put_wall = put_wall or (None if math.isnan(pw) else pw)
        oi_call, oi_put = raw.get("oi_call"), raw.get("oi_put")
        if oi_call and oi_put and len(oi_call) == len(strikes):
            cw2, pw2 = walls_by_oi(strikes, oi_call, oi_put)
            call_wall = call_wall or (None if math.isnan(cw2) else cw2)
            put_wall = put_wall or (None if math.isnan(pw2) else pw2)
        if gamma_flip is None:
            gamma_flip = gamma_flip_cumulative(strikes, gex_net)

    walls: list[Wall] = []
    for w in raw.get("walls") or []:
        if not isinstance(w, Mapping):
            continue
        strike = _f(w.get("strike"))
        oi = _f(w.get("oi")) or 0.0
        if strike:
            walls.append(Wall(strike=strike, oi=oi, kind=str(w.get("kind") or "cluster")))

    return OptionsWalls(
        enabled=enabled,
        call_wall=call_wall,
        put_wall=put_wall,
        gamma_flip=gamma_flip,
        walls=tuple(walls),
        min_wall_oi=_f(raw.get("min_wall_oi")) or 0.0,
    )


@dataclass(frozen=True)
class TrendConfluenceParams:
    """Tunables for :class:`TrendConfluenceStrategy` (all JSON-serialisable)."""

    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    # confluence zone
    zone_atr: float = 0.5  # half-width of the confluence zone, in ATRs
    min_confluence: int = 2  # independent levels required in the zone
    pullback_lookback: int = 10  # bars that define "a pullback"
    # trendlines (existing gex.domain.trendlines engine)
    use_trendlines: bool = True
    trendline_refresh: int = 5  # re-run the Pine port every N bars
    trendline_history: int = 200  # window handed to analyze_trendlines
    trendline_resolution: int = 6
    # options walls
    use_options_walls: bool = True
    gamma_flip_filter: bool = True  # longs need close > flip, shorts close < flip
    respect_call_wall: bool = True  # do not open longs right under a big CALL wall
    respect_put_wall: bool = True  # do not open shorts right above a big PUT wall
    exit_at_call_wall: bool = True  # cap longs at the CALL wall
    exit_at_put_wall: bool = True  # cap shorts at the PUT wall
    wall_atr_tol: float = 0.5  # "at the wall" tolerance, in ATRs
    # unclear regime
    allow_range: bool = False  # trade RANGE regimes at reduced size
    range_size_mult: float = 0.5
    # sides
    allow_long: bool = True
    allow_short: bool = True
    # exits
    invalidation_atr: float = 0.5  # close beyond EMA200/trendline by this → exit
    use_trailing: bool = True
    atr_trail_mult: float = 3.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TrendConfluenceParams":
        if not raw:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k not in fields or v is None:
                continue
            kwargs[k] = v
        return cls(**kwargs)


# ────────────────────────────────────────────────────────────────────── #
#  Batch frame computation
# ────────────────────────────────────────────────────────────────────── #
def _ema(close: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(close).ewm(span=period, adjust=False).mean().to_numpy()


def _atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


@dataclass
class _Frame:
    """Per-bar precomputed decision data (all causal — no lookahead)."""

    n: int
    regime: np.ndarray  # int8: 1 bull, -1 bear, 0 range
    tl_support: list[list[float]]  # active trendline support prices per bar
    tl_resistance: list[list[float]]
    ema_fast: np.ndarray
    ema_mid: np.ndarray
    ema_slow: np.ndarray
    atr: np.ndarray
    roll_max: np.ndarray  # rolling max of close (pullback reference)
    roll_min: np.ndarray


def _trendline_levels(
    df: pd.DataFrame,
    *,
    refresh: int,
    history: int,
    resolution: int,
) -> tuple[list[list[float]], list[list[float]], np.ndarray]:
    """Run the existing trendline engine causally over the frame.

    Every ``refresh`` bars the Pine port re-runs on the trailing ``history``
    window; between refreshes the lines are simply extrapolated forward by
    their slope (``price = current_price + slope * Δbars``), mirroring
    ``extend=right`` in the original indicator.
    """
    from gex.domain.trendlines import analyze_trendlines

    n = len(df)
    supports: list[list[float]] = [[] for _ in range(n)]
    resistances: list[list[float]] = [[] for _ in range(n)]
    tl_vote = np.zeros(n, dtype=np.int8)

    analysis = None
    computed_at = -1
    for i in range(n):
        if analysis is None or (i - computed_at) >= refresh:
            lo = max(0, i + 1 - history)
            window = df.iloc[lo : i + 1]
            if len(window) >= max(2 * resolution, 30):
                try:
                    analysis = analyze_trendlines(
                        window, timeframe="bt", resolution=resolution,
                        history_bars=history,
                    )
                    computed_at = i
                except ValueError:
                    analysis = None
        if analysis is not None:
            dt = i - computed_at
            supports[i] = [tl.current_price + tl.slope * dt for tl in analysis.support_lines]
            resistances[i] = [tl.current_price + tl.slope * dt for tl in analysis.resistance_lines]
            tl_vote[i] = {"BULLISH": 1, "BEARISH": -1}.get(analysis.trend_direction, 0)
    return supports, resistances, tl_vote


def _compute_frame(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    p: TrendConfluenceParams,
) -> _Frame:
    """Compute every per-bar quantity the state machine needs (vectorised)."""
    n = len(close)
    ema_fast, ema_mid, ema_slow = _ema(close, p.ema_fast), _ema(close, p.ema_mid), _ema(close, p.ema_slow)
    atr = _atr_wilder(high, low, close, p.atr_period)
    roll_max = pd.Series(close).rolling(p.pullback_lookback, min_periods=1).max().shift(1).to_numpy()
    roll_min = pd.Series(close).rolling(p.pullback_lookback, min_periods=1).min().shift(1).to_numpy()

    if p.use_trendlines:
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
        tl_sup, tl_res, tl_vote = _trendline_levels(
            df, refresh=max(1, p.trendline_refresh),
            history=max(30, p.trendline_history), resolution=p.trendline_resolution,
        )
    else:
        tl_sup, tl_res = [[] for _ in range(n)], [[] for _ in range(n)]
        tl_vote = np.zeros(n, dtype=np.int8)

    # Regime: price vs EMA50/EMA200, confirmed by the trendline vote. A
    # contradicting trendline vote downgrades the regime to RANGE (unclear
    # regimes reduce size or stay flat, per the strategy spec).
    above = (close > ema_mid) & (close > ema_slow)
    below = (close < ema_mid) & (close < ema_slow)
    regime = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if above[i]:
            regime[i] = 1 if (not p.use_trendlines or tl_vote[i] >= 0) else 0
        elif below[i]:
            regime[i] = -1 if (not p.use_trendlines or tl_vote[i] <= 0) else 0
    return _Frame(
        n=n, regime=regime, tl_support=tl_sup, tl_resistance=tl_res,
        ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow, atr=atr,
        roll_max=roll_max, roll_min=roll_min,
    )


# ────────────────────────────────────────────────────────────────────── #
#  Confluence evaluation
# ────────────────────────────────────────────────────────────────────── #
def _touched(level: float, close: float, extreme: float, tol: float) -> bool:
    """Is ``level`` inside the zone around this bar (close or wick touch)?"""
    if level <= 0:
        return False
    return abs(close - level) <= tol or abs(extreme - level) <= tol or (extreme <= level <= close) or (close <= level <= extreme)


def _support_levels(f: _Frame, i: int, w: OptionsWalls, close: float) -> list[tuple[str, float]]:
    """Candidate supports for a long pullback zone at bar ``i``."""
    out = [("ema_fast", f.ema_fast[i]), ("ema_mid", f.ema_mid[i]), ("ema_slow", f.ema_slow[i])]
    out += [("trendline", v) for v in f.tl_support[i] if v < close]
    if w.enabled:
        if w.put_wall:
            out.append(("put_wall", w.put_wall))
        if w.gamma_flip and close > w.gamma_flip:
            out.append(("gamma_flip", w.gamma_flip))
        for wall in w.significant_walls():
            if wall.strike < close:
                out.append((f"{wall.kind}_wall", wall.strike))
    return out


def _resistance_levels(f: _Frame, i: int, w: OptionsWalls, close: float) -> list[tuple[str, float]]:
    """Candidate resistances for a short rally zone at bar ``i`` (mirror)."""
    out = [("ema_fast", f.ema_fast[i]), ("ema_mid", f.ema_mid[i]), ("ema_slow", f.ema_slow[i])]
    out += [("trendline", v) for v in f.tl_resistance[i] if v > close]
    if w.enabled:
        if w.call_wall:
            out.append(("call_wall", w.call_wall))
        if w.gamma_flip and close < w.gamma_flip:
            out.append(("gamma_flip", w.gamma_flip))
        for wall in w.significant_walls():
            if wall.strike > close:
                out.append((f"{wall.kind}_wall", wall.strike))
    return out


def _confluence(levels: list[tuple[str, float]], close: float, extreme: float, tol: float) -> list[str]:
    """Names of the *distinct* level families touched by this bar's zone."""
    hits: set[str] = set()
    for name, level in levels:
        if _touched(level, close, extreme, tol):
            hits.add(name.split("_wall")[0] + "_wall" if name.endswith("_wall") else name)
    return sorted(hits)


# ────────────────────────────────────────────────────────────────────── #
#  Strategy
# ────────────────────────────────────────────────────────────────────── #
class TrendConfluenceStrategy(Strategy):
    """Trend-following with confluence pullbacks and options-wall awareness."""

    name = "trend_confluence"

    def __init__(
        self,
        symbol: str,
        *,
        params: Mapping[str, Any] | None = None,
        walls: OptionsWalls | None = None,
    ) -> None:
        self.symbol = symbol
        p = dict(params or {})
        options_block = p.pop("options", None)
        self._p = TrendConfluenceParams.from_dict(p)
        if walls is not None:
            self._walls = walls
        else:
            resolved = resolve_walls(options_block)
            # ``use_options_walls=False`` short-circuits the snapshot entirely.
            if not self._p.use_options_walls:
                resolved = OptionsWalls(enabled=False)
            elif resolved.enabled is False and options_block:
                resolved = OptionsWalls(
                    enabled=True, call_wall=resolved.call_wall, put_wall=resolved.put_wall,
                    gamma_flip=resolved.gamma_flip, walls=resolved.walls,
                    min_wall_oi=resolved.min_wall_oi,
                )
            self._walls = resolved
        self.min_bars = max(DEFAULT_MIN_BARS, self._p.ema_slow + 20)
        self._reset_state()

    # ── state ───────────────────────────────────────────────────────────
    def _reset_state(self) -> None:
        self._bars: list[Bar] = []
        self._frame: _Frame | None = None
        self._ts: list = []
        self._closes: np.ndarray | None = None
        self._highs: np.ndarray | None = None
        self._lows: np.ndarray | None = None
        self._cursor = 0
        self._processed = 0
        self._side = "flat"
        self._entry_price: float | None = None
        self._entry_strength = 1.0
        self._best: float | None = None
        self.fallback_count = 0

    async def start(self) -> None:
        self._reset_state()

    # ── batch path ──────────────────────────────────────────────────────
    async def prepare(self, bars: Sequence[Bar]) -> None:
        """Precompute the full causal frame once; ``on_bar`` becomes O(1)."""
        bars = list(bars)
        self._reset_state()
        if len(bars) < self.min_bars:
            return
        highs = np.array([b.high for b in bars], dtype=float)
        lows = np.array([b.low for b in bars], dtype=float)
        self._frame = _compute_frame(
            np.array([b.open for b in bars], dtype=float),
            highs, lows,
            np.array([b.close for b in bars], dtype=float),
            self._p,
        )
        self._ts = [b.timestamp for b in bars]
        self._closes = np.array([float(b.close) for b in bars], dtype=float)
        self._highs = highs
        self._lows = lows

    async def on_bar(self, bar: Bar) -> list[Signal]:
        if self._frame is not None and self._cursor < len(self._ts):
            if self._matches_prepared(self._cursor, bar):
                idx = self._cursor
                self._cursor += 1
                if idx + 1 < self.min_bars:
                    return []
                return self._drain(idx)
            self.fallback_count += 1
            self._frame = None
            self._closes = None
            self._highs = None
            self._lows = None
            self._cursor = 0

        # Streaming path (live): recompute the frame on the accumulated prefix.
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
        return self._drain(len(self._bars) - 1)

    def _matches_prepared(self, idx: int, bar: Bar) -> bool:
        if self._ts[idx] != bar.timestamp or self._closes is None:
            return False
        expected = self._closes[idx]
        return abs(expected - float(bar.close)) <= 1e-9 * max(1.0, abs(expected))

    def _drain(self, upto: int) -> list[Signal]:
        out: list[Signal] = []
        while self._processed <= upto:
            j = self._processed
            out.extend(self._signal_at(j, self._ts[j]))
            self._processed += 1
        return out

    # ── decision logic ──────────────────────────────────────────────────
    def _signal_at(self, i: int, timestamp) -> list[Signal]:
        f, p, w = self._frame, self._p, self._walls
        assert f is not None
        if i + 1 < self.min_bars:
            # Warm-up: EMAs are defined from bar 0 but are meaningless until
            # the slow window is populated; never trade here. The batch drain
            # replays these rows once the cursor passes the gate, so this
            # check must live inside the row evaluation, not at the drain.
            return []
        assert self._closes is not None and self._highs is not None and self._lows is not None
        close = float(self._closes[i])
        high_i = float(self._highs[i])
        low_i = float(self._lows[i])
        atr = float(f.atr[i]) if math.isfinite(f.atr[i]) else close * 0.01
        tol = p.zone_atr * atr
        wall_tol = p.wall_atr_tol * atr
        price = Price(close)
        out: list[Signal] = []

        regime = int(f.regime[i])
        size_mult = 1.0
        if regime == 0:
            if not p.allow_range:
                # Unclear regime: only manage exits, never open.
                exit_sig = self._exit_check(i, close, atr, price, timestamp, regime=0)
                return [exit_sig] if exit_sig is not None else []
            size_mult = p.range_size_mult

        if self._side == "flat":
            if regime >= 0 and p.allow_long:
                sig = self._try_long(i, close, low_i, atr, tol, wall_tol, regime, size_mult, price, timestamp)
                if sig is not None:
                    out.append(sig)
            if not out and regime <= 0 and p.allow_short:
                sig = self._try_short(i, close, high_i, atr, tol, wall_tol, regime, size_mult, price, timestamp)
                if sig is not None:
                    out.append(sig)
            return out

        exit_sig = self._exit_check(i, close, atr, price, timestamp, regime=regime)
        if exit_sig is not None:
            out.append(exit_sig)
            return out

        # Track extremes after the exit checks (same convention as gex_emf).
        if self._side == "long":
            self._best = high_i if self._best is None else max(self._best, high_i)
        else:
            self._best = low_i if self._best is None else min(self._best, low_i)
        return out

    def _try_long(
        self, i: int, close: float, low_i: float, atr: float, tol: float,
        wall_tol: float, regime: int, size_mult: float, price: Price, timestamp,
    ) -> Signal | None:
        f, p, w = self._frame, self._p, self._walls
        assert f is not None
        # Pullback: price must be off its recent high (a retrace, not a breakout).
        roll_max = f.roll_max[i]
        if math.isfinite(roll_max) and close >= roll_max:
            return None
        # Trend structure intact: no close below EMA200 beyond the invalidation band.
        if close < f.ema_slow[i] - p.invalidation_atr * atr:
            return None
        # Gamma flip must align with the bullish direction.
        if p.gamma_flip_filter and w.enabled and w.gamma_flip and close < w.gamma_flip:
            return None
        # Respect a large CALL wall overhead: do not buy directly into it.
        if p.respect_call_wall and w.enabled and w.call_wall and 0 < w.call_wall - close <= wall_tol:
            return None
        hits = _confluence(_support_levels(f, i, w, close), close, low_i, tol)
        if len(hits) < p.min_confluence:
            return None
        strength = min(1.0, len(hits) / 4.0) * size_mult
        self._open("long", close, strength)
        return Signal(
            self.symbol, Side.BUY, self.name,
            f"add_long_confluence:{','.join(hits)}",
            strength=round(strength, 3), price=price, timestamp=timestamp,
        )

    def _try_short(
        self, i: int, close: float, high_i: float, atr: float, tol: float,
        wall_tol: float, regime: int, size_mult: float, price: Price, timestamp,
    ) -> Signal | None:
        f, p, w = self._frame, self._p, self._walls
        assert f is not None
        roll_min = f.roll_min[i]
        if math.isfinite(roll_min) and close <= roll_min:
            return None
        if close > f.ema_slow[i] + p.invalidation_atr * atr:
            return None
        if p.gamma_flip_filter and w.enabled and w.gamma_flip and close > w.gamma_flip:
            return None
        if p.respect_put_wall and w.enabled and w.put_wall and 0 < close - w.put_wall <= wall_tol:
            return None
        hits = _confluence(_resistance_levels(f, i, w, close), close, high_i, tol)
        if len(hits) < p.min_confluence:
            return None
        strength = min(1.0, len(hits) / 4.0) * size_mult
        self._open("short", close, strength)
        return Signal(
            self.symbol, Side.SELL, self.name,
            f"add_short_confluence:{','.join(hits)}",
            strength=round(strength, 3), price=price, timestamp=timestamp,
        )

    def _exit_check(
        self, i: int, close: float, atr: float, price: Price, timestamp, *, regime: int
    ) -> Signal | None:
        """Exit rules for the open position: wall caps, invalidation, trailing."""
        if self._side == "flat" or self._entry_price is None:
            return None
        f, p, w = self._frame, self._p, self._walls
        assert f is not None
        long = self._side == "long"
        # Exit signals reuse the *entry* strength: the engine sizes every order
        # as ``fraction × strength × equity / price``, so an exit at full
        # strength after a half-strength entry would overshoot the position and
        # silently flip it. Matching strengths keeps exits close-only (up to
        # equity drift).
        s = self._entry_strength

        # 1. Options-wall caps — big walls are magnets/rejection zones.
        if long and p.exit_at_call_wall and w.enabled and w.call_wall and close >= w.call_wall - p.wall_atr_tol * atr:
            self._close_position()
            return Signal(self.symbol, Side.SELL, self.name, "call_wall_cap",
                          strength=s, price=price, timestamp=timestamp)
        if not long and p.exit_at_put_wall and w.enabled and w.put_wall and close <= w.put_wall + p.wall_atr_tol * atr:
            self._close_position()
            return Signal(self.symbol, Side.BUY, self.name, "put_wall_cap",
                          strength=s, price=price, timestamp=timestamp)

        # 2. Trend invalidation — EMA200 and/or the active trendline breaks,
        #    or the regime flips hard against the position.
        inv = p.invalidation_atr * atr
        if long:
            broken = close < f.ema_slow[i] - inv
            if not broken and f.tl_support[i]:
                broken = close < min(f.tl_support[i]) - inv
            if broken or regime < 0:
                self._close_position()
                return Signal(self.symbol, Side.SELL, self.name, "trend_invalidation",
                              strength=s, price=price, timestamp=timestamp)
        else:
            broken = close > f.ema_slow[i] + inv
            if not broken and f.tl_resistance[i]:
                broken = close > max(f.tl_resistance[i]) + inv
            if broken or regime > 0:
                self._close_position()
                return Signal(self.symbol, Side.BUY, self.name, "trend_invalidation",
                              strength=s, price=price, timestamp=timestamp)

        # 3. ATR trailing stop.
        if p.use_trailing and self._best is not None:
            if long and close <= self._best - p.atr_trail_mult * atr:
                self._close_position()
                return Signal(self.symbol, Side.SELL, self.name, "trailing_stop",
                              strength=s, price=price, timestamp=timestamp)
            if not long and close >= self._best + p.atr_trail_mult * atr:
                self._close_position()
                return Signal(self.symbol, Side.BUY, self.name, "trailing_stop",
                              strength=s, price=price, timestamp=timestamp)
        return None

    # ── position bookkeeping ────────────────────────────────────────────
    def _open(self, side: str, close: float, strength: float = 1.0) -> None:
        self._side = side
        self._entry_price = close
        self._entry_strength = strength
        self._best = close

    def _close_position(self) -> None:
        self._side = "flat"
        self._entry_price = None
        self._best = None

    # ── misc Strategy API ───────────────────────────────────────────────
    async def on_tick(self, tick: Tick) -> list[Signal]:
        return []

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        bars = list(bars)
        if len(bars) < self.min_bars:
            return []
        await self.prepare(bars)
        signals: list[Signal] = []
        for i in range(self.min_bars - 1, len(bars)):
            signals.extend(self._signal_at(i, self._ts[i]))
        return signals
