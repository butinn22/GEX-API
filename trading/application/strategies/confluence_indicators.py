"""Causal indicator layer for the confluence-breakout strategy family.

Port of ``quant/alligator/indicators.py`` into the live platform. Everything
here is **causal**: row ``t`` of every returned array is a function of bars
``0..t`` only, so the same code runs identically in a backtest replay and in
the live bar stream (no lookahead, no repainting).

What it computes
----------------
* ``wilder_atr`` — Wilder ATR (the volatility unit used for every stop level).
* ``compute_alligator`` — Bill Williams Alligator: SMMA 13/8/5 of the median
  price, each shifted forward by 8/5/3 bars (lips/teeth/jaw).
* ``compute_emf`` — Ease of Movement (raw volume-normalised midpoint distance,
  SMA-smoothed). Positive ⇒ price advancing on light volume.
* ``compute_adl`` — Chaikin Accumulation/Distribution + its EMA (observe the
  money-flow direction).
* ``compute_structure`` — confirmed swing pivots (3-left / 3-right) and the
  HH/HL (bull) vs LH/LL (bear) structure flags, with the pivot confirmation
  delay applied so a pivot is only visible ``right`` bars after its bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Alligator",
    "Structure",
    "wilder_atr",
    "compute_alligator",
    "compute_emf",
    "compute_adl",
    "compute_structure",
    "compute_frame",
]


def wilder_atr(high, low, close, n: int = 14) -> np.ndarray:
    """Wilder ATR — ``ewm(alpha=1/n)`` over true range, SMA-seeded."""
    h = pd.Series(np.asarray(high, dtype=float))
    l = pd.Series(np.asarray(low, dtype=float))
    c = pd.Series(np.asarray(close, dtype=float))
    prev_close = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy(float)


def _smma(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothed moving average (Bill Williams' SMMA)."""
    return pd.Series(values).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_alligator(high, low) -> Alligator:
    """Alligator jaws/teeth/lips on the median price, shifted 8/5/3 bars."""
    median = (np.asarray(high, dtype=float) + np.asarray(low, dtype=float)) / 2.0
    jaw = _smma(median, 13).shift(8).to_numpy(float)
    teeth = _smma(median, 8).shift(5).to_numpy(float)
    lips = _smma(median, 5).shift(3).to_numpy(float)
    return Alligator(jaw=jaw, teeth=teeth, lips=lips)


def compute_emf(high, low, volume, n: int = 14) -> np.ndarray:
    """Ease of Movement: ``(mid - prev_mid) * (H-L) / volume``, SMA-smoothed."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    vol = np.asarray(volume, dtype=float)
    mid = (h + l) / 2.0
    dm = np.empty_like(mid)
    dm[0] = 0.0
    dm[1:] = mid[1:] - mid[:-1]
    box = (h - l) * dm
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(vol > 0, box / np.where(vol > 0, vol, 1.0), 0.0)
    return pd.Series(raw).rolling(n, min_periods=n).mean().to_numpy(float)


def compute_adl(high, low, close, volume, ema_span: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Chaikin ADL (cumulative money-flow) and its EMA — ``(adl, adl_ema)``."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    rng = h - l
    clv = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1.0), 0.0)
    adl = pd.Series(clv * np.asarray(volume, dtype=float)).cumsum()
    adl_ema = adl.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    return adl.to_numpy(float), adl_ema.to_numpy(float)


def compute_structure(high, low, left: int = 3, right: int = 3) -> Structure:
    """Confirmed higher-high/higher-low market structure (causal).

    A pivot high at bar ``i`` requires ``high[i]`` strictly above every other
    high in ``[i-left, i+right]``; it enters the structure state at bar
    ``i+right``. State at bar ``t`` therefore uses only pivots fully
    determined by bars ``<= t``.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        w = high[i - left:i + right + 1]
        if high[i] == w.max() and np.count_nonzero(w == high[i]) == 1:
            is_ph[i] = True
        w = low[i - left:i + right + 1]
        if low[i] == w.min() and np.count_nonzero(w == low[i]) == 1:
            is_pl[i] = True

    events: list[tuple[int, str, float]] = (
        [(int(i) + right, "H", float(high[i])) for i in np.flatnonzero(is_ph)]
        + [(int(i) + right, "L", float(low[i])) for i in np.flatnonzero(is_pl)]
    )
    if n == 0:
        return Structure(
            bull=np.zeros(0, bool), bear=np.zeros(0, bool),
            last_hh=np.zeros(0), last_hl=np.zeros(0),
            pivot_high_bar=is_ph, pivot_low_bar=is_pl,
        )
    events.sort()

    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    last_hh = np.full(n, np.nan)
    last_hl = np.full(n, np.nan)
    highs: list[float] = []
    lows: list[float] = []
    ei = 0
    for t in range(n):
        while ei < len(events) and events[ei][0] <= t:
            _, typ, price = events[ei]
            ei += 1
            (highs if typ == "H" else lows).append(price)
        if highs:
            last_hh[t] = highs[-1]
        if lows:
            last_hl[t] = lows[-1]
        if len(highs) >= 2 and len(lows) >= 2:
            bull[t] = highs[-1] > highs[-2] and lows[-1] > lows[-2]
            bear[t] = highs[-1] < highs[-2] and lows[-1] < lows[-2]
    return Structure(bull=bull, bear=bear, last_hh=last_hh, last_hl=last_hl,
                     pivot_high_bar=is_ph, pivot_low_bar=is_pl)


def compute_frame(high, low, close, volume, *, pivot_left: int = 3, pivot_right: int = 3,
                  adl_ema_span: int = 20) -> dict[str, np.ndarray]:
    """One causal pass; returns every array the strategy layer needs.

    Accepts plain sequences (lists of floats from ``Bar`` objects) or arrays —
    the live path hands it the accumulated bar window, the backtest path the
    whole replay.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = len(c)
    out: dict[str, np.ndarray] = {
        "high": h, "low": l, "close": c, "volume": v, "n": np.array(n),
    }
    out["atr"] = wilder_atr(h, l, c)
    g = compute_alligator(h, l)
    out["lips"], out["teeth"], out["jaw"] = g.lips, g.teeth, g.jaw
    out["emf"] = compute_emf(h, l, v)
    out["adl"], out["adl_ema"] = compute_adl(h, l, c, v, adl_ema_span)
    st = compute_structure(h, l, pivot_left, pivot_right)
    out["struct_bull"], out["struct_bear"] = st.bull, st.bear
    out["last_hh"], out["last_hl"] = st.last_hh, st.last_hl
    return out


@dataclass
class Alligator:
    """Jaws / teeth / lips. Value at ``t`` uses the bar at ``t - shift``."""

    jaw: np.ndarray    # SMMA13(median) shifted forward 8
    teeth: np.ndarray  # SMMA8(median)  shifted forward 5
    lips: np.ndarray   # SMMA5(median)  shifted forward 3


@dataclass
class Structure:
    """Bull/bear market-structure flags and the last confirmed swing levels."""

    bull: np.ndarray        # bool[t]: confirmed HH + HL
    bear: np.ndarray        # bool[t]: confirmed LH + LL
    last_hh: np.ndarray     # float[t]: last confirmed swing-high (NaN none)
    last_hl: np.ndarray     # float[t]: last confirmed swing-low (NaN none)
    pivot_high_bar: np.ndarray  # bool[t]: bar t is a pivot high
    pivot_low_bar: np.ndarray
