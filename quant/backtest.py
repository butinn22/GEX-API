"""Event-driven single-ticker backtest engine with the mandated execution model.

Execution invariants (Loop spec):
1. Signal confirmed at close of bar t -> entry fills at open of bar t+1
   (slippage + fee applied against the trade).
2. Stop-loss is live from the entry bar onward and may trigger INTRABAR.
   Gap handling: if the bar opens beyond the stop, fill at the (worse) open.
3. Trailing-stop exits are ARMED only after the 4h minimum hold: on a 4h chart
   that is the bar AFTER the entry bar (entry at open, 4h elapses at the entry
   bar's close); on 1d the same rule is applied conservatively (intra-entry-bar
   exit timing is unknowable, so no armed exits inside the entry bar).
4. Signal exits (MA exit) confirmed at close of t -> fill at open of t+1, and
   they execute at the open BEFORE any intrabar stop of that bar can fire
   (market order at open).
5. Pessimistic tie-break: intrabar, the stop is assumed to hit before any
   favorable move of the same bar would have mattered.
6. Exit classification: STOP_LOSS_BEFORE_4H / TAKE_PROFIT_BEFORE_4H / EXIT_AFTER_4H.
7. Bar timestamps are bar OPEN times (exchange convention); a stop inside the
   entry bar is therefore recorded with holding < 4h (conservative under-count).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .strategy import Indicators, TrendParams

__all__ = ["CostModel", "Trade", "EngineResult", "run_single",
           "STOP_BEFORE_4H", "TP_BEFORE_4H", "AFTER_4H"]

STOP_BEFORE_4H = "STOP_LOSS_BEFORE_4H"
TP_BEFORE_4H = "TAKE_PROFIT_BEFORE_4H"
AFTER_4H = "EXIT_AFTER_4H"


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.001        # Bybit spot taker 0.1% per side (conservative)
    slippage: float = 0.0005       # 5 bps per side on fills at open
    stop_slippage: float = 0.0008  # 8 bps adverse extra on intrabar stop fills

    def describe(self) -> str:
        return (f"fee={self.fee_rate:.4%}/side, slippage={self.slippage:.4%}/side, "
                f"stop_slippage={self.stop_slippage:.4%} extra adverse")


@dataclass
class Trade:
    symbol: str
    entry_time: datetime   # bar-open time of the entry fill bar
    exit_time: datetime   # bar-open time of the exit bar
    entry_price: float    # actual fill (slippage applied, fee excluded)
    exit_price: float     # actual fill (slippage applied, fee excluded)
    qty: float
    gross_pnl: float
    fees: float           # both sides
    net_pnl: float
    exit_reason: str      # STOP / TRAIL / SIGNAL / END_OF_DATA
    classification: str
    holding_hours: float
    mae_pct: float
    mfe_pct: float


@dataclass
class EngineResult:
    trades: list[Trade]
    equity: np.ndarray
    exposure: np.ndarray  # bool per bar: position held during bar t
    n_bars: int
    index: pd.DatetimeIndex


def _hours(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def run_single(
    df: pd.DataFrame,
    ind: Indicators,
    p: TrendParams,
    costs: CostModel,
    symbol: str,
    *,
    initial_cash: float = 100_000.0,
    bar_hours: float = 4.0,
    min_hold_hours: float = 4.0,
) -> EngineResult:
    """Bar-by-bar replay for one ticker / one capital slice (long-only spot)."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    idx = df.index
    assert bar_hours >= min_hold_hours, "this engine requires bars >= min hold"

    cash = initial_cash
    equity = np.empty(n)
    exposure = np.zeros(n, dtype=bool)
    trades: list[Trade] = []

    in_pos = False
    qty = entry_price = stop = stop_init = highest_close = 0.0
    entry_fee = 0.0
    entry_time: datetime | None = None
    entry_bar = -1
    trail_moved = False
    mae = mfe = 0.0
    pending_entry = False    # signal at close of t-1 -> fill at open of t
    pending_exit = False     # signal-exit at close of t-1 -> fill at open of t
    sig_bar = -1             # bar whose ATR sized the pending entry stop

    def record_trade(bar: int, fill: float, reason: str) -> None:
        gross = (fill - entry_price) * qty
        exit_fee = fill * qty * costs.fee_rate
        hold_h = _hours(entry_time, idx[bar])
        if hold_h < min_hold_hours - 1e-9:
            cls = STOP_BEFORE_4H if reason in ("STOP", "TRAIL") else TP_BEFORE_4H
        else:
            cls = AFTER_4H
        trades.append(Trade(
            symbol=symbol, entry_time=entry_time, exit_time=idx[bar],
            entry_price=entry_price, exit_price=fill, qty=qty,
            gross_pnl=gross, fees=exit_fee + entry_fee,
            net_pnl=gross - entry_fee - exit_fee, exit_reason=reason,
            classification=cls, holding_hours=hold_h, mae_pct=mae, mfe_pct=mfe,
        ))

    for t in range(n):
        # ── 1. pending SIGNAL exit fills at the open of bar t ──
        if in_pos and pending_exit:
            fill = o[t] * (1 - costs.slippage)
            record_trade(t, fill, "SIGNAL")
            cash += fill * qty * (1 - costs.fee_rate)
            in_pos = pending_exit = False
            qty = 0.0

        # ── 2. pending entry fills at the open of bar t ──
        if not in_pos and pending_entry:
            fill = o[t] * (1 + costs.slippage)
            q = (cash * p.risk_frac) / fill
            entry_fee = fill * q * costs.fee_rate
            cash -= fill * q + entry_fee
            qty = q
            entry_price = fill
            entry_time = idx[t]
            entry_bar = t
            stop = stop_init = fill - p.k_sl * ind.atr[sig_bar]
            highest_close = c[t]
            trail_moved = False
            mae = mfe = 0.0
            in_pos = True
            pending_entry = False

        # ── 3. manage the open position during bar t ──
        if in_pos:
            exposure[t] = True
            mae = min(mae, (l[t] - entry_price) / entry_price)
            mfe = max(mfe, (h[t] - entry_price) / entry_price)
            # min-hold elapses at the CLOSE of the entry bar (entry was at its
            # open), so from the entry bar's close onward the trail may update;
            # it can only TRIGGER from the next bar => holding >= min hold.
            armed = t >= entry_bar

            if l[t] <= stop:  # stop live from the entry bar; pessimistic first
                fill = o[t] * (1 - costs.slippage) if o[t] <= stop \
                    else stop * (1 - costs.stop_slippage)
                reason = "STOP" if not trail_moved else "TRAIL"
                record_trade(t, fill, reason)
                cash += fill * qty * (1 - costs.fee_rate)
                in_pos = False
                qty = 0.0
            elif armed:
                # chandelier update from data up to close of bar t (effective
                # from bar t+1) — strictly causal.
                highest_close = max(highest_close, c[t])
                new_stop = highest_close - p.k_trail * ind.atr[t]
                if np.isfinite(new_stop) and new_stop > stop:
                    stop = new_stop
                    trail_moved = True
            # signal exit confirmed at close of t: fill at open of t+1, which is
            # >= one full bar after entry => >= min hold on both timeframes
            if ind.exit_signal[t]:
                pending_exit = True

        # ── 4. mark to market at the close of bar t ──
        equity[t] = cash + (qty * c[t] if in_pos else 0.0)

        # ── 5. new entry signal at close of bar t ──
        if (not in_pos and not pending_entry and t < n - 1
                and ind.entry_signal[t] and np.isfinite(ind.atr[t])
                and ind.atr[t] > 0):
            pending_entry = True
            sig_bar = t

    # force-close any open position at the final close (flagged, not flattering)
    if in_pos:
        fill = c[n - 1] * (1 - costs.slippage)
        record_trade(n - 1, fill, "END_OF_DATA")
    elif pending_entry:
        pass  # signal never filled (no next bar) — correctly dropped

    return EngineResult(trades=trades, equity=equity, exposure=exposure,
                        n_bars=n, index=idx)
