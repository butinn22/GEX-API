"""Mechanics QA for the engine — hand-checkable scenarios (NOT a backtest).

These verify execution rules only (fill prices, gap handling, min-hold arming,
classification). They use hand-built bars where the correct answer is computed
by hand, so any engine bug shows up as an assertion failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import (AFTER_4H, STOP_BEFORE_4H, CostModel, run_single)
from .strategy import Indicators, TrendParams

COSTS = CostModel(fee_rate=0.0, slippage=0.0, stop_slippage=0.0)  # isolate mechanics


def mk_df(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz="UTC") for r in rows])
    return pd.DataFrame([r[1:] for r in rows],
                        index=idx, columns=["open", "high", "low", "close", "volume"])


def test_stop_hits_intrabar_same_bar_as_entry():
    # entry signal at t=0 close (close 110 > channel), fills at t=1 open 100.
    # ATR[0]=10, k_sl=2 -> stop=80. Bar1 low 75 crosses stop intrabar -> fill at 80.
    rows = [
        ("2020-01-01 00:00", 100, 110, 99, 110, 1),   # t=0: breakout close
        ("2020-01-01 04:00", 100, 105, 75, 90, 1),    # t=1: entry + same-bar stop
        ("2020-01-01 08:00", 90, 95, 85, 92, 1),
    ]
    df = mk_df(rows)
    ind = Indicators(
        entry_signal=np.array([True, False, False]),
        exit_signal=np.zeros(3, dtype=bool),
        atr=np.array([10.0, 10.0, 10.0]),
    )
    p = TrendParams(n_break=1, k_sl=2.0, k_trail=4.0, ma_exit=0, risk_frac=1.0)
    res = run_single(df, ind, p, COSTS, "T", initial_cash=100_000.0, bar_hours=4.0)
    assert len(res.trades) == 1, res.trades
    tr = res.trades[0]
    assert tr.entry_price == 100.0
    assert tr.exit_price == 80.0, tr.exit_price     # filled at the stop
    assert tr.exit_reason == "STOP"
    assert tr.classification == STOP_BEFORE_4H      # exit at bar-1 open = 0h held
    print("OK stop intrabar + classification")


def test_gap_through_stop_fills_at_open():
    # stop=80, bar opens at 70 (gap through) -> fill at 70 (worse), not at 80.
    rows = [
        ("2020-01-01 00:00", 100, 110, 99, 110, 1),
        ("2020-01-01 04:00", 100, 105, 95, 100, 1),  # entry bar
        ("2020-01-01 08:00", 70, 75, 60, 62, 1),     # gaps through the stop
    ]
    df = mk_df(rows)
    ind = Indicators(entry_signal=np.array([True, False, False]),
                     exit_signal=np.zeros(3, dtype=bool),
                     atr=np.array([10.0, 10.0, 10.0]))
    p = TrendParams(n_break=1, k_sl=2.0, k_trail=5.0, ma_exit=0, risk_frac=1.0)
    res = run_single(df, ind, p, COSTS, "T", initial_cash=100_000.0, bar_hours=4.0)
    tr = res.trades[0]
    assert tr.exit_price == 70.0, tr.exit_price
    assert tr.exit_reason == "STOP"
    print("OK gap-through fills at open (worse fill)")


def test_trail_only_armed_after_first_bar():
    # entry at bar1 open 100, stop=80 (k_sl=2, ATR=10).
    # trail = highest_close - k_trail*ATR. If the trail were (incorrectly) armed
    # on bar1, highest_close=105 -> trail=105-50=55 (below stop) — no effect.
    # Make k_trail small so trail matters: k_trail=1 -> trail=highest-10.
    # bar1 close 105 -> trail would be 95 if computed at bar1 close, active bar2.
    # bar2 low 92 <= 95 -> TRAIL exit at 95, holding = 4h+ => AFTER_4H.
    rows = [
        ("2020-01-01 00:00", 100, 110, 99, 110, 1),
        ("2020-01-01 04:00", 100, 106, 95, 105, 1),   # entry bar
        ("2020-01-01 08:00", 104, 106, 92, 93, 1),    # trail hit intrabar
    ]
    df = mk_df(rows)
    ind = Indicators(entry_signal=np.array([True, False, False]),
                     exit_signal=np.zeros(3, dtype=bool),
                     atr=np.array([10.0, 10.0, 10.0]))
    p = TrendParams(n_break=1, k_sl=2.0, k_trail=1.0, ma_exit=0, risk_frac=1.0)
    res = run_single(df, ind, p, COSTS, "T", initial_cash=100_000.0, bar_hours=4.0)
    tr = res.trades[0]
    assert tr.exit_price == 95.0, tr.exit_price
    assert tr.exit_reason == "TRAIL"
    assert tr.classification == AFTER_4H
    assert abs(tr.holding_hours - 4.0) < 1e-9
    print("OK trail armed after min-hold, AFTER_4H classification")


def test_signal_exit_fills_next_open():
    # exit signal at close of entry bar -> fills at open of bar2 (>= 4h held).
    rows = [
        ("2020-01-01 00:00", 100, 110, 99, 110, 1),
        ("2020-01-01 04:00", 100, 105, 95, 105, 1),   # entry bar; exit signal here
        ("2020-01-01 08:00", 108, 109, 100, 101, 1),  # fill at open 108
    ]
    df = mk_df(rows)
    ind = Indicators(entry_signal=np.array([True, False, False]),
                     exit_signal=np.array([False, True, False]),
                     atr=np.array([10.0, 10.0, 10.0]))
    p = TrendParams(n_break=1, k_sl=2.0, k_trail=5.0, ma_exit=0, risk_frac=1.0)
    res = run_single(df, ind, p, COSTS, "T", initial_cash=100_000.0, bar_hours=4.0)
    tr = res.trades[0]
    assert tr.exit_price == 108.0, tr.exit_price
    assert tr.exit_reason == "SIGNAL"
    assert tr.classification == AFTER_4H
    print("OK signal exit at next open")


def test_costs_and_accounting():
    rows = [
        ("2020-01-01 00:00", 100, 110, 99, 110, 1),
        ("2020-01-01 04:00", 100, 105, 95, 105, 1),
        ("2020-01-01 08:00", 108, 109, 100, 101, 1),
    ]
    df = mk_df(rows)
    ind = Indicators(entry_signal=np.array([True, False, False]),
                     exit_signal=np.array([False, True, False]),
                     atr=np.array([10.0, 10.0, 10.0]))
    costs = CostModel(fee_rate=0.001, slippage=0.0, stop_slippage=0.0)
    p = TrendParams(n_break=1, k_sl=2.0, k_trail=5.0, ma_exit=0, risk_frac=1.0)
    res = run_single(df, ind, p, costs, "T", initial_cash=100_000.0, bar_hours=4.0)
    tr = res.trades[0]
    # qty = 100000/100 = 1000 units; fees 0.1% each side = 100 + 108
    assert abs(tr.qty - 1000.0) < 1e-6
    assert abs(tr.fees - (100.0 + 108.0)) < 1e-6, tr.fees
    assert abs(tr.net_pnl - ((108 - 100) * 1000 - 208.0)) < 1e-6
    print("OK fee accounting")


if __name__ == "__main__":
    test_stop_hits_intrabar_same_bar_as_entry()
    test_gap_through_stop_fills_at_open()
    test_trail_only_armed_after_first_bar()
    test_signal_exit_fills_next_open()
    test_costs_and_accounting()
    print("ALL ENGINE MECHANICS TESTS PASSED")
