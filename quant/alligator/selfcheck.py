"""Deterministic self-checks for the Alligator confluence package.

Run: python -m quant.alligator.selfcheck
Everything here is hand-checkable: synthetic frames where the correct answer is
known by construction. If any assertion fails, the research loops must NOT run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import run_single
from .indicators import compute_all, compute_structure
from .strategy import AlliParams, precompute
from ..backtest import CostModel

N = 400


def _frame(n: int = N, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.roll(close, 1); open_[0] = 100.0
    spread = np.abs(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = np.abs(rng.normal(1000, 300, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def check_pivot_confirmation() -> None:
    """A pivot at bar i must be invisible to the structure state before i+right."""
    df = _frame()
    # force one clean, unambiguous pivot high at i=200
    df = df.copy()
    df["high"] = np.minimum(df["high"].to_numpy(float), 150.0)
    df.loc[df.index[200], "high"] = 160.0
    df["low"] = np.maximum(df["low"].to_numpy(float), 50.0)
    st = compute_structure(df, left=3, right=3)
    assert st.pivot_high_bar[200], "synthetic pivot high not detected"
    t_before = 200 + 3 - 1
    t_after = 200 + 3
    assert not np.isfinite(st.last_hh[t_before]) or st.last_hh[t_before] < 160.0, \
        "pivot leaked into state before confirmation"
    assert st.last_hh[t_after] == 160.0, "pivot missing right after confirmation"


def check_causality() -> None:
    """Signals computed on a truncated frame must match the full frame's prefix.

    Recomputing signals on df[:k] and comparing against the first k rows of the
    full-frame signals catches any hidden future leak (rolling/ewm/shift bugs).
    """
    df = _frame()
    p = AlliParams()
    full = precompute(df, p)
    for k in (300, 333, 399):
        part = precompute(df.iloc[:k].copy(), p)
        assert np.array_equal(full.entry_signal[:k], part.entry_signal), \
            f"entry_signal not causal at cut {k}"
        assert np.array_equal(full.exit_signal[:k], part.exit_signal), \
            f"exit_signal not causal at cut {k}"
        assert np.allclose(full.atr[:k], part.atr, equal_nan=True), \
            f"atr not causal at cut {k}"


def check_engine_mechanics() -> None:
    """Hand-computed trade on a scripted frame.

    Frame: flat at 100, one bar whose close triggers the entry signal, then a
    rise then a fall through the stop. Verify fill prices, fees, risk sizing
    cap, and min-hold classification.
    """
    n = 30
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0); c = np.full(n, 100.0)
    h = np.full(n, 100.0); l = np.full(n, 100.0)
    v = np.full(n, 1000.0)
    o[10] = 100.0; h[10] = 100.0; l[10] = 100.0; c[10] = 101.0  # signal bar
    for t in range(11, 16):
        o[t] = c[t - 1]; c[t] = 102.0; h[t] = 103.0; l[t] = o[t]
    # bar 15: intrabar stop at 98.0 (open 102 above stop -> fill AT the stop)
    l[15] = 97.0
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                       "volume": v}, index=idx)

    entry = np.zeros(n, dtype=bool); entry[10] = True
    atr = np.full(n, 2.0)
    ind = type("Ind", (), {
        "entry_signal": entry, "exit_signal": np.zeros(n, dtype=bool),
        "atr": atr, "last_hl": np.full(n, np.nan),
        "teeth": np.full(n, 90.0), "struct_bull": np.zeros(n, dtype=bool),
    })()
    p = AlliParams(risk_pct=0.01, k_sl_atr=2.0, min_stop_atr=1.0,
                   max_stop_atr=4.0, k_trail=4.0, cooldown_bars=0)
    costs = CostModel(fee_rate=0.001, slippage=0.0, stop_slippage=0.0)
    res = run_single(df, ind, p, costs, "TEST", initial_cash=100_000.0,
                     bar_hours=4.0, min_hold_hours=4.0)

    assert len(res.trades) == 1, f"expected 1 trade, got {len(res.trades)}"
    tr = res.trades[0]
    # stop = fill 101 - k_sl 2 * ATR 2 = 97; fill at open of bar 11 = 101.0
    assert tr.entry_price == 101.0, tr.entry_price
    # bar 15 low 97 touches stop 97 exactly: fill AT the stop (open 102 above)
    assert tr.exit_price == 97.0, tr.exit_price
    # sizing: risk 1% of 100k = 1000 / dist 4.0 => qty 250
    assert abs(tr.qty * (tr.entry_price - 97.0) - 1000.0) < 1e-6, tr.qty
    # fees: 0.1% both sides on notional
    exp_fees = 0.001 * (tr.qty * 101.0 + tr.qty * tr.exit_price)
    assert abs(tr.fees - exp_fees) < 1e-6, (tr.fees, exp_fees)
    assert tr.exit_reason == "STOP"
    assert tr.classification == "EXIT_AFTER_4H"  # 5 bars * 4h held


def check_risk_cap() -> None:
    """With an extremely tight stop the notional cap (95% cash) must bind."""
    n = 30
    idx = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0); c = np.full(n, 100.0)
    h = np.full(n, 100.0); l = np.full(n, 100.0)
    v = np.full(n, 1000.0)
    c[10] = 101.0
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                       "volume": v}, index=idx)
    entry = np.zeros(n, dtype=bool); entry[10] = True
    ind = type("Ind", (), {
        "entry_signal": entry, "exit_signal": np.zeros(n, dtype=bool),
        "atr": np.full(n, 0.01), "last_hl": np.full(n, np.nan),
        "teeth": np.full(n, 90.0), "struct_bull": np.zeros(n, dtype=bool),
    })()
    p = AlliParams(risk_pct=0.01, k_sl_atr=2.0, min_stop_atr=1.0,
                   max_stop_atr=4.0, cooldown_bars=0)
    res = run_single(df, ind, p, CostModel(fee_rate=0.0, slippage=0.0,
                                           stop_slippage=0.0), "TEST")
    tr = res.trades[0]
    # raw qty would be 1000/0.01=100k notional > cash -> cap to 95% of 100k
    assert abs(tr.qty * 100.0 - 95_000.0) < 1e-6, tr.qty


def check_position_check_before_add() -> None:
    """A persistent entry signal must produce exactly ONE position at a time:
    the engine never adds a second signal while in position or pending."""
    df = _frame(200)
    n = len(df)
    entry = np.ones(n, dtype=bool)  # signal every bar
    ind = type("Ind", (), {
        "entry_signal": entry, "exit_signal": np.zeros(n, dtype=bool),
        "atr": np.full(n, 2.0), "last_hl": np.full(n, np.nan),
        "teeth": np.full(n, 90.0), "struct_bull": np.zeros(n, dtype=bool),
    })()
    p = AlliParams(risk_pct=0.01, k_sl_atr=3.0, cooldown_bars=0)
    res = run_single(df, ind, p, CostModel(), "TEST")
    # consecutive trades can never overlap in time
    prev_exit = None
    for tr in res.trades:
        if prev_exit is not None:
            assert tr.entry_time >= prev_exit, "overlapping positions!"
        prev_exit = tr.exit_time
    assert len(res.trades) >= 1


def main() -> None:
    check_pivot_confirmation()
    check_causality()
    check_engine_mechanics()
    check_risk_cap()
    check_position_check_before_add()
    print("ALL SELFCHECKS PASSED")


if __name__ == "__main__":
    main()
