"""Strategy rules: Alligator + HH/HL structure + EMF/ADL trend-confluence entries.

Confluence stack (long side, all conditions must hold at the signal bar t):
  1. Alligator aligned and waking:  lips[t] > teeth[t] > jaws[t]  AND  jaws rising.
  2. Smart-money structure: confirmed HH + HL (last two confirmed swing highs
     ascending AND last two confirmed swing lows ascending).
  3. Chaikin ADL confirms accumulation: adl[t] > adl_ema[t].
  4. EMF > 0 (price rising on ease of movement — no volume churn against).

Entry modes:
  breakout  — close[t] closes above the highest high of the prior n_break bars
              (momentum continuation in an established confluence trend).
  pullback  — price dipped to the lips within the last `pullback_window` bars,
              held above the teeth, and closes back above the lips with a up
              close (buy the dip in trend).
  both      — either.

Exits (engine-managed): initial structural+ATR stop, chandelier + structure
ratchet trail, signal exit at close of t -> fill at open of t+1 (same contract
as quant.backtest), and a time stop.

All arrays causal: the breakout channel at t uses bars [t-n_break, t-1]; the
pullback touch window ends at t; pivots are confirmation-shifted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import compute_all

__all__ = ["AlliParams", "Indicators", "precompute"]


@dataclass(frozen=True)
class AlliParams:
    # trend confluence
    pivot_left: int = 3
    pivot_right: int = 3
    adl_ema_span: int = 20
    # entry
    n_break: int = 30            # Donchian channel over prior bars (excl. t)
    entry_mode: str = "both"     # breakout | pullback | both
    pullback_window: int = 8     # bars to look back for a lips touch
    # initial stop
    k_sl_atr: float = 2.5        # ATR-based stop distance if no structure
    k_struct_buf: float = 0.5    # stop = last confirmed HL - buf*ATR
    min_stop_atr: float = 1.0    # stop never closer than this * ATR
    max_stop_atr: float = 4.0    # stop never farther than this * ATR
    # trailing
    k_trail: float = 4.0         # chandelier: highest_close - k*ATR
    use_struct_trail: bool = True
    exit_level: str = "teeth"    # alligator line whose loss ends the trend: teeth|jaw
    # risk / trade management
    risk_pct: float = 0.005      # 0.5% of slice equity risked per trade
    max_hold_bars: int = 45      # time stop if trade went nowhere
    min_mfe_r: float = 1.0       # "went somewhere" = reached +1R at least
    cooldown_bars: int = 3       # no re-entry for N bars after an exit
    warmup: int = 250            # bars before any signal may fire


@dataclass
class Indicators:
    entry_signal: np.ndarray
    exit_signal: np.ndarray
    atr: np.ndarray
    last_hl: np.ndarray
    teeth: np.ndarray
    struct_bull: np.ndarray


def precompute(df: pd.DataFrame, p: AlliParams,
               pre: dict[str, np.ndarray] | None = None) -> Indicators:
    """``pre`` allows reusing a compute_all() result across a parameter grid."""
    a = pre if pre is not None else compute_all(
        df, p.pivot_left, p.pivot_right, p.adl_ema_span)
    n = len(df)
    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    lips, teeth, jaw = a["lips"], a["teeth"], a["jaw"]
    atr = a["atr"]

    finite = np.isfinite(lips) & np.isfinite(teeth) & np.isfinite(jaw) & np.isfinite(atr)
    gator_bull = (lips > teeth) & (teeth > jaw)
    jaws_rising = np.empty(n, dtype=bool)
    jaws_rising[0] = False
    jaws_rising[1:] = jaw[1:] > jaw[:-1]
    adl_bull = a["adl"] > a["adl_ema"]
    emf_ok = np.isfinite(a["emf"]) & (a["emf"] > 0)

    trend_ok = finite & gator_bull & jaws_rising & a["struct_bull"] & adl_bull

    allowed = np.zeros(n, dtype=bool)
    allowed[p.warmup:] = True

    # --- breakout entry: close above the prior n_break bars' highest high ---
    channel = df["high"].rolling(p.n_break).max().shift(1)
    brk = np.array(close > channel.to_numpy(float), dtype=bool)
    brk[: p.n_break + 1] = False
    entry_breakout = brk & trend_ok & emf_ok & allowed

    # --- pullback entry: dipped to lips, held teeth, closed back above lips ---
    touched = (pd.Series(low - lips, index=df.index)
               .rolling(p.pullback_window, min_periods=1).min()
               .to_numpy(float) <= 0)
    held = np.isfinite(teeth) & (close >= teeth)
    back_above = np.zeros(n, dtype=bool)
    back_above[1:] = (close[1:] > lips[1:]) & (close[1:] > close[:-1])
    entry_pullback = (trend_ok & emf_ok & allowed & touched & held & back_above
                      & np.isfinite(lips))

    if p.entry_mode == "breakout":
        entry = entry_breakout
    elif p.entry_mode == "pullback":
        entry = entry_pullback
    elif p.entry_mode == "both":
        entry = entry_breakout | entry_pullback
    else:
        raise ValueError(f"unknown entry_mode {p.entry_mode!r}")

    # --- signal exit: close below the chosen alligator line (trend spent) or
    # structure break (close below last confirmed swing low while bull) ---
    ref_line = teeth if p.exit_level == "teeth" else jaw
    was_bull = np.zeros(n, dtype=bool)
    was_bull[1:] = a["struct_bull"][:-1]
    struct_break = np.isfinite(a["last_hl"]) & was_bull & (close < a["last_hl"])
    exit_signal = (finite & ((close < ref_line) | struct_break)) & allowed

    return Indicators(entry_signal=entry, exit_signal=exit_signal, atr=atr,
                      last_hl=a["last_hl"], teeth=teeth, struct_bull=a["struct_bull"])
