"""4H multi-timeframe regime fix + Loop 3 fresh-set confirmation.

Fix hypothesis (a-priori, zero new tuned parameters): 4H breakouts fail mostly
when the DAILY trend is down. Gate 4H entries on the daily regime
(close > SMA200 on the last FULLY CLOSED daily bar — no lookahead).

Loop 3 fresh tickers (never used in Loops 1-2): UNIUSDT, ETCUSDT, XLMUSDT,
ALGOUSDT, CRVUSDT, SANDUSDT (+ availability check; excluded if data missing).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .backtest import run_single
from .data import load_or_fetch
from .final_loop1 import REG
from .run_loop1 import BASE_COSTS, INITIAL, TF_BAR_HOURS
from .strategy import TrendParams, precompute

LOOP3_TICKERS = ["UNIUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT",
                 "CRVUSDT", "SANDUSDT"]


def daily_regime_series(sym: str, sma_n: int = 200) -> pd.Series:
    """UTC-timestamped bool series: True while the last CLOSED daily bar has
    close > SMA(sma_n). Indexed by daily bar CLOSE time (= next day 00:00).

    To avoid lookahead at a 4H close at time T, the caller must use the regime
    value of the last daily bar whose close time <= T (shifted by 1 day).
    """
    df, _ = load_or_fetch(sym, "1d")
    sma = df["close"].rolling(sma_n).mean()
    ok = (df["close"] > sma).astype(float)  # 1.0 known at this bar's close
    # regime becomes usable AFTER the daily bar closes -> index at close time
    ok.index = ok.index + pd.Timedelta(days=1)
    return ok


def run_4h_mtf(sym: str, combo: dict):
    """4H engine run with the daily-regime gate (strictly causal)."""
    df, _ = load_or_fetch(sym, "4h")
    params = TrendParams(**combo)
    ind = precompute(df, params)
    sma4 = df["close"].rolling(REG["4h"]).mean()  # 200d-equivalent on 4H too
    reg4 = np.array(df["close"] > sma4, dtype=bool)
    reg_d = daily_regime_series(sym).reindex(df.index.union(
        daily_regime_series(sym).index), method="ffill").reindex(df.index)
    # value at bar t = regime of the last daily close <= bar t open (bar ts is
    # the open time; signals fire at the close of t — daily bars closing
    # exactly at bar-t open are already fully known, so >= is legitimate)
    reg_daily_ok = np.array(reg_d.to_numpy() > 0.5, dtype=bool)
    ind.entry_signal = ind.entry_signal & reg4 & reg_daily_ok
    return run_single(df, ind, params, BASE_COSTS, sym,
                      initial_cash=INITIAL / 7, bar_hours=4.0)


def eval_set(runs: dict, tf: str, tickers: list[str]):
    from .run_loop1 import _slice_metrics
    import quant.run_loop1 as rl
    saved = rl.TICKERS
    rl.TICKERS = tickers
    try:
        out = {}
        for name, lo, hi in (
            ("train", None, pd.Timestamp("2024-07-01", tz="UTC")),
            ("val", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")),
            ("holdout", pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-10-04", tz="UTC")),
            ("full", None, pd.Timestamp("2026-10-04", tz="UTC")),
        ):
            m, trades, _ = _slice_metrics(runs, tf, lo, hi)
            out[name] = m
            pnl_by = {}
            for s, r in runs.items():
                pnl_by[s] = round(sum(t.net_pnl for t in r.trades
                                      if (lo is None or t.entry_time >= lo.to_pydatetime())
                                      and t.entry_time < hi.to_pydatetime()), 0)
            out[name + "_pnl"] = pnl_by
    finally:
        rl.TICKERS = saved
    return out


COMBO_4H = dict(n_break=30, k_sl=3.0, k_trail=5.0, ma_exit=100)
LOOP1_TICKERS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]
LOOP2_TICKERS = ["BNBUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT"]


def main() -> None:
    sets = {"LOOP1": LOOP1_TICKERS, "LOOP2": LOOP2_TICKERS, "LOOP3": LOOP3_TICKERS}
    # availability check for loop 3
    ok3 = []
    for s in LOOP3_TICKERS:
        try:
            load_or_fetch(s, "1d")
            load_or_fetch(s, "4h")
            ok3.append(s)
        except Exception as e:
            print(f"EXCLUDED {s}: {e}")
    sets["LOOP3"] = ok3
    print("Loop 3 tickers:", ok3)

    for name, tickers in sets.items():
        if not tickers:
            continue
        runs = {s: run_4h_mtf(s, COMBO_4H) for s in tickers}
        res = eval_set(runs, "4h", tickers)
        print(f"\n=== 4H + DAILY-REGIME GATE on {name} ({len(tickers)} tickers) ===")
        for k in ("train", "val", "holdout", "full"):
            m = res[k]
            print(f"  {k:>8}: sharpe={m.sharpe:6.2f} cagr={m.cagr:7.1%} "
                  f"mdd={m.max_dd:6.1%} n={m.n_trades:>4} pf={m.profit_factor:5.2f} "
                  f"win={m.win_rate:5.1%} exp={m.exposure:5.1%}")
        print("  full per-ticker:", res["full_pnl"])

    # also verify the gate does NOT wreck the (already accepted) 1D system — 1D
    # already uses its own 200-bar gate; nothing changes there. Report only 4H.


if __name__ == "__main__":
    main()
