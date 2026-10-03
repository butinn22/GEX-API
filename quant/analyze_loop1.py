"""Validation + improvement analysis for Loop 1.

Commands:
  val      — OOS validation of the top-5 train combos (per timeframe)
  bench    — buy & hold benchmark per split (context, not a strategy)
  regime   — grid WITH the a-priori 200-day regime filter (train + val)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .backtest import CostModel, run_single
from .data import load_or_fetch
from .metrics import portfolio_metrics, regime_split, trade_stats
from .run_loop1 import (BASE_COSTS, INITIAL, RESULTS, TF_BAR_HOURS, TICKERS,
                        TRAIN_END, VAL_END, _run_cached, _slice_metrics, combo_list)
from .strategy import TrendParams, precompute

TOP_4H = [dict(n_break=30, k_sl=3.0, k_trail=5.0, ma_exit=0),
          dict(n_break=30, k_sl=2.5, k_trail=4.0, ma_exit=100),
          dict(n_break=30, k_sl=3.0, k_trail=5.0, ma_exit=100),
          dict(n_break=40, k_sl=3.0, k_trail=5.0, ma_exit=100),
          dict(n_break=30, k_sl=2.5, k_trail=5.0, ma_exit=0)]
TOP_1D = [dict(n_break=55, k_sl=2.5, k_trail=5.0, ma_exit=100),
          dict(n_break=55, k_sl=2.5, k_trail=5.0, ma_exit=0),
          dict(n_break=55, k_sl=3.0, k_trail=5.0, ma_exit=100),
          dict(n_break=20, k_sl=2.5, k_trail=3.0, ma_exit=0),
          dict(n_break=30, k_sl=2.5, k_trail=3.0, ma_exit=0)]


def cmd_val() -> None:
    for tf, top in (("4h", TOP_4H), ("1d", TOP_1D)):
        print(f"\n=== VALIDATION {tf} (2024-07 .. 2025-07) ===")
        for combo in top:
            runs = _run_cached(tf, combo, BASE_COSTS)
            m, trades, _ = _slice_metrics(runs, tf, TRAIN_END, VAL_END)
            ts = trade_stats(trades)
            print(f"n={combo['n_break']:>3} sl={combo['k_sl']} kt={combo['k_trail']} "
                  f"ma={combo['ma_exit']:>3} | sharpe={m.sharpe:6.2f} cagr={m.cagr:7.1%} "
                  f"mdd={m.max_dd:6.1%} pf={m.profit_factor:5.2f} n={m.n_trades:>4} "
                  f"win={m.win_rate:5.1%} best_trade_share={ts['best_trade_share'] if ts['n'] else 0:5.1%}")


def cmd_bench() -> None:
    for tf in TF_BAR_HOURS:
        dfs = {s: load_or_fetch(s, tf)[0] for s in TICKERS}
        for name, lo, hi in (("train", None, TRAIN_END), ("val", TRAIN_END, VAL_END),
                             ("full", None, pd.Timestamp("2026-10-04", tz="UTC"))):
            rets = []
            for s, df in dfs.items():
                d = df[df.index < hi] if lo is None else df[(df.index >= lo) & (df.index < hi)]
                tr = d["close"].iloc[-1] / d["close"].iloc[0] - 1
                rets.append(tr)
            eq = np.prod(1 + np.array(rets) / len(rets))  # equal-weight rebalanced
            print(f"{tf} {name}: equal-weight buy&hold total return {eq - 1:9.1%}")


def _run_regime(tf: str, combo: dict, reg_n: int, costs: CostModel):
    """Grid run with a-priori regime filter: entries only when close > SMA(reg_n)."""
    from .backtest import EngineResult
    params = TrendParams(**combo)
    out: dict[str, EngineResult] = {}
    for sym in TICKERS:
        df, _ = load_or_fetch(sym, tf)
        ind = precompute(df, params)
        sma = df["close"].rolling(reg_n).mean()
        reg_ok = (df["close"] > sma).to_numpy(dtype=bool)
        # copy-on-fix: pandas 3 returns read-only arrays
        es = ind.entry_signal & np.asarray(reg_ok)
        ind.entry_signal = es
        out[sym] = run_single(df, ind, params, costs, sym,
                              initial_cash=INITIAL / len(TICKERS),
                              bar_hours=TF_BAR_HOURS[tf])
    return out


def cmd_regime() -> None:
    combos = combo_list()
    for tf, reg in (("4h", 1200), ("1d", 200)):  # ~200 days, a priori
        rows = []
        for combo in combos:
            runs = _run_regime(tf, combo, reg, BASE_COSTS)
            for split, lo, hi in (("train", None, TRAIN_END), ("val", TRAIN_END, VAL_END)):
                m, trades, _ = _slice_metrics(runs, tf, lo, hi)
                rows.append({"tf": tf, "split": split, **combo,
                             "sharpe": m.sharpe, "calmar": m.calmar, "cagr": m.cagr,
                             "max_dd": m.max_dd, "n_trades": m.n_trades,
                             "pf": m.profit_factor, "win_rate": m.win_rate,
                             "exposure": m.exposure})
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"regime_metrics_{tf}.csv", index=False)
        for split in ("train", "val"):
            sub = df[df.split == split].sort_values("sharpe", ascending=False)
            print(f"\n=== REGIME-FILTERED {tf} {split} (top 6 by Sharpe) ===")
            print(sub.head(6).drop(columns=["tf", "split", "calmar"]).to_string(index=False))


tf_cmd = ""

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "val"
    if cmd == "val":
        cmd_val()
    elif cmd == "bench":
        cmd_bench()
    elif cmd == "regime":
        tf_cmd = sys.argv[2] if len(sys.argv) > 2 else ""
        cmd_regime()
