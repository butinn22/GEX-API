"""Event-driven engine for the Alligator confluence strategy.

Execution model is IDENTICAL in spirit to quant.backtest.run_single (same
no-lookahead contract, so the two engines agree on mechanics):

1. Signal confirmed at close of bar t -> fill at open of bar t+1 (slippage+fee).
2. Stop-loss live from the entry bar onward, may trigger INTRABAR; gap fills at
   the (worse) open; pessimistic tie-break (stop before favorable move).
3. Trailing stop armed only after the min hold (one full bar after entry).
4. Signal/time-stop exits confirmed at close of t -> fill at open of t+1,
   before any intrabar stop of that bar can fire.
5. Entry ONLY when flat (position check before adding a new signal); exit only
   when in position. No averaging, no pyramiding.
6. Cooldown: no re-entry for `cooldown_bars` bars after any exit.

Differences vs quant.backtest (why this engine exists):
  * Risk-based sizing: qty = (equity * risk_pct) / stop_distance, capped at
    95% of available cash (long-only spot). Risk per trade is constant in %.
  * Initial stop blends structure (last confirmed HL - buf*ATR) with ATR.
  * Trail ratchets chandelier AND structure levels.
  * Optional time stop (exit if the trade never reached +1R after N bars).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..backtest import CostModel, STOP_BEFORE_4H, TP_BEFORE_4H, AFTER_4H
from .strategy import AlliParams, Indicators

__all__ = ["Trade", "EngineResult", "run_single"]


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str          # STOP / TRAIL / SIGNAL / TIME / END_OF_DATA
    classification: str
    holding_hours: float
    mae_pct: float
    mfe_pct: float
    risk_pct_at_entry: float  # realised stop distance as % of entry price


@dataclass
class EngineResult:
    trades: list[Trade]
    equity: np.ndarray
    exposure: np.ndarray
    n_bars: int
    index: pd.DatetimeIndex


def _hours(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def run_single(
    df: pd.DataFrame,
    ind: Indicators,
    p: AlliParams,
    costs: CostModel,
    symbol: str,
    *,
    initial_cash: float = 100_000.0,
    bar_hours: float = 4.0,
    min_hold_hours: float = 4.0,
) -> EngineResult:
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
    entry_fee = risk_dist_pct = 0.0
    entry_time: datetime | None = None
    entry_bar = -1
    last_exit_bar = -10**9
    trail_moved = False
    mae = mfe = 0.0
    pending_entry = False
    pending_exit = False
    pending_exit_reason = "SIGNAL"
    sig_bar = -1

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
            risk_pct_at_entry=risk_dist_pct,
        ))

    for t in range(n):
        # ── 1. pending SIGNAL/TIME exit fills at the open of bar t ──
        if in_pos and pending_exit:
            fill = o[t] * (1 - costs.slippage)
            record_trade(t, fill, pending_exit_reason)
            cash += fill * qty * (1 - costs.fee_rate)
            in_pos = pending_exit = False
            qty = 0.0
            last_exit_bar = t

        # ── 2. pending entry fills at the open of bar t ──
        if not in_pos and pending_entry:
            fill = o[t] * (1 + costs.slippage)
            a = ind.atr[sig_bar]
            hl = ind.last_hl[sig_bar]
            # initial stop: tighter of (structure below last confirmed HL) and
            # (k_sl_atr * ATR), floored/capped to sane ATR multiples
            atr_stop = fill - p.k_sl_atr * a
            stop_c = atr_stop
            if np.isfinite(hl) and hl - p.k_struct_buf * a > 0:
                stop_c = max(hl - p.k_struct_buf * a, atr_stop)
            stop_c = min(stop_c, fill - p.min_stop_atr * a)   # distance floor
            stop_c = max(stop_c, fill - p.max_stop_atr * a)   # distance cap
            dist = fill - stop_c
            risk_cash = cash * p.risk_pct
            q = risk_cash / dist if dist > 1e-12 else 0.0
            max_q = cash * 0.95 / fill
            q = min(q, max_q)  # cap: never spend more than 95% of cash
            if q <= 0.0:
                pending_entry = False
            else:
                entry_fee = fill * q * costs.fee_rate
                cash -= fill * q + entry_fee
                qty = q
                entry_price = fill
                entry_time = idx[t]
                entry_bar = t
                stop = stop_init = stop_c
                risk_dist_pct = dist / fill
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
            armed = t >= entry_bar  # min hold elapses at close of entry bar

            if l[t] <= stop:  # intrabar stop, pessimistic first
                fill = o[t] * (1 - costs.slippage) if o[t] <= stop \
                    else stop * (1 - costs.stop_slippage)
                reason = "STOP" if not trail_moved else "TRAIL"
                record_trade(t, fill, reason)
                cash += fill * qty * (1 - costs.fee_rate)
                in_pos = False
                qty = 0.0
                last_exit_bar = t
            elif armed:
                highest_close = max(highest_close, c[t])
                new_stop = highest_close - p.k_trail * ind.atr[t]
                if p.use_struct_trail and np.isfinite(ind.last_hl[t]):
                    new_stop = max(new_stop,
                                   ind.last_hl[t] - p.k_struct_buf * ind.atr[t])
                if np.isfinite(new_stop) and new_stop > stop:
                    stop = new_stop
                    trail_moved = True
                # time stop: trade had N bars and never reached +1R
                if (p.max_hold_bars > 0 and t - entry_bar >= p.max_hold_bars
                        and mfe < p.min_mfe_r * risk_dist_pct):
                    pending_exit = True
                    pending_exit_reason = "TIME"
            if ind.exit_signal[t] and in_pos:
                pending_exit = True
                pending_exit_reason = "SIGNAL"

        # ── 4. mark to market at the close of bar t ──
        equity[t] = cash + (qty * c[t] if in_pos else 0.0)

        # ── 5. new entry signal at close of bar t (only when flat + cooldown) ──
        cooldown_ok = (t - last_exit_bar) >= p.cooldown_bars
        if (not in_pos and not pending_entry and not pending_exit
                and cooldown_ok and t < n - 1
                and ind.entry_signal[t] and np.isfinite(ind.atr[t])
                and ind.atr[t] > 0):
            pending_entry = True
            sig_bar = t

    if in_pos:
        fill = c[n - 1] * (1 - costs.slippage)
        record_trade(n - 1, fill, "END_OF_DATA")
    elif pending_entry:
        pass  # signal never filled — correctly dropped

    return EngineResult(trades=trades, equity=equity, exposure=exposure,
                        n_bars=n, index=idx)
