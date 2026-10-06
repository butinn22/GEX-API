"""Causal indicator layer: Alligator, ATR, EMF, Chaikin ADL, swing structure.

Hard invariant: row ``t`` of every output array is a function of bars ``0..t``
only. Swing pivots are detected on a [t-L, t+R] window but become *visible* to
the strategy only R bars after the pivot bar (confirmation shift), so no signal
can know a pivot before it is objectively confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Structure", "Alligator", "compute_all", "wilder_atr"]


def wilder_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """Wilder ATR — same convention as quant.strategy (EMA alpha=1/n, SMA seed)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy(float)


def _smma(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothed moving average (Bill Williams' SMMA)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


@dataclass
class Alligator:
    jaw: np.ndarray    # SMMA13(median) shifted forward 8  -> value at t uses t-8
    teeth: np.ndarray  # SMMA8(median)  shifted forward 5
    lips: np.ndarray   # SMMA5(median)  shifted forward 3


def compute_alligator(df: pd.DataFrame) -> Alligator:
    median = (df["high"] + df["low"]) / 2.0
    jaw = _smma(median, 13).shift(8)
    teeth = _smma(median, 8).shift(5)
    lips = _smma(median, 5).shift(3)
    return Alligator(jaw=jaw.to_numpy(float), teeth=teeth.to_numpy(float),
                     lips=lips.to_numpy(float))


def compute_emf(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """Ease of Movement: ((H+L)/2 - prev(H+L)/2) * (H-L) / volume, SMA-smoothed.

    Positive => price advancing on light volume (path of least resistance up).
    Volume == 0 bars contribute 0 (dm * 0), never NaN/inf.
    """
    mid = (df["high"] + df["low"]) / 2.0
    dm = mid.diff()
    box = (df["high"] - df["low"]) * dm
    vol = df["volume"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(vol > 0, box.to_numpy(float) / np.where(vol > 0, vol, 1.0), 0.0)
    return pd.Series(raw, index=df.index).rolling(n, min_periods=n).mean().to_numpy(float)


def compute_adl(df: pd.DataFrame, ema_span: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Chaikin ADL (cumulative) and its EMA — returns (adl, adl_ema)."""
    h, l, c = df["high"], df["low"], df["close"]
    rng = (h - l).to_numpy(float)
    clv = np.where(rng > 0, ((c - l) - (h - c)).to_numpy(float) / rng, 0.0)
    adl = pd.Series(clv * df["volume"].to_numpy(float), index=df.index).cumsum()
    adl_ema = adl.ewm(span=ema_span, adjust=False, min_periods=ema_span).mean()
    return adl.to_numpy(float), adl_ema.to_numpy(float)


@dataclass
class Structure:
    bull: np.ndarray        # bool[t]: confirmed HH + HL
    bear: np.ndarray        # bool[t]: confirmed LH + LL
    last_hh: np.ndarray     # float[t]: last confirmed swing-high price (NaN none)
    last_hl: np.ndarray     # float[t]: last confirmed swing-low price (NaN none)
    pivot_high_bar: np.ndarray  # bool[t]: bar t IS a pivot high (confirm only at t+R)
    pivot_low_bar: np.ndarray


def compute_structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> Structure:
    """Confirmed higher-high/higher-low (and lower-) market structure, causal.

    A pivot high at bar i requires high[i] strictly above every other high in
    [i-left, i+right]; it enters the structure state at bar i+right. The state
    at bar t therefore uses only pivots fully determined by bars <= t.
    """
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(df)
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


def compute_all(df: pd.DataFrame, pivot_left: int = 3, pivot_right: int = 3,
                adl_ema_span: int = 20) -> dict[str, np.ndarray]:
    """One pass over the frame; returns every array the strategy layer needs."""
    out: dict[str, np.ndarray] = {}
    out["atr"] = wilder_atr(df)
    g = compute_alligator(df)
    out["jaw"], out["teeth"], out["lips"] = g.jaw, g.teeth, g.lips
    out["emf"] = compute_emf(df)
    out["adl"], out["adl_ema"] = compute_adl(df, adl_ema_span)
    st = compute_structure(df, pivot_left, pivot_right)
    out["struct_bull"], out["struct_bear"] = st.bull, st.bear
    out["last_hh"], out["last_hl"] = st.last_hh, st.last_hl
    return out
