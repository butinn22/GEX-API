"""Strategy signal generation — Loop 1 core: Donchian breakout trend following.

All signals are computed from CLOSED bars only. The engine calls
`precompute(df)` once; every array is safe to read at bar t (uses data <= t).
No repainting: the breakout channel at bar t uses the highs of bars
[t-n_break, t-1] (strictly before t), so `close[t] > channel` is a genuine
close-above-prior-channel event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TrendParams", "precompute", "Indicators"]


@dataclass(frozen=True)
class TrendParams:
    n_break: int = 40      # breakout channel lookback (prior bars)
    k_sl: float = 3.0      # initial stop = entry - k_sl * ATR
    k_trail: float = 4.0   # chandelier trail = highest_close - k_trail * ATR
    ma_exit: int = 0       # exit when close < SMA(ma_exit) if > 0
    risk_frac: float = 0.95  # fraction of slice cash deployed per entry


@dataclass
class Indicators:
    entry_signal: np.ndarray   # bool[t]: breakout confirmed at close of t
    exit_signal: np.ndarray   # bool[t]: MA-exit confirmed at close of t
    atr: np.ndarray           # Wilder ATR[t], known at close of t


def _wilder_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing == EMA with alpha = 1/n, seeded with SMA of first n TRs
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return atr.to_numpy(dtype=float)


def precompute(df: pd.DataFrame, p: TrendParams) -> Indicators:
    """Vectorized, strictly causal indicator + signal arrays."""
    # Channel = highest high of the n_break bars BEFORE t (shift(1) => no self-inclusion).
    channel = df["high"].rolling(p.n_break).max().shift(1)
    entry_signal = np.array(df["close"] > channel, dtype=bool)  # writable copy
    entry_signal[: p.n_break + 1] = False  # warm-up: channel undefined

    if p.ma_exit > 0:
        ma = df["close"].rolling(p.ma_exit).mean()
        exit_signal = np.array(df["close"] < ma, dtype=bool)
        exit_signal[: p.ma_exit] = False
    else:
        exit_signal = np.zeros(len(df), dtype=bool)

    return Indicators(entry_signal=entry_signal, exit_signal=exit_signal,
                      atr=_wilder_atr(df))
