"""Loop 2 — FRESH ticker set, FROZEN parameters (pure out-of-sample confirmation).

Purpose: test that the Loop-1-selected strategy is stable across a different
set of popular tickers. No parameter is re-optimized here — that would defeat
the purpose. Tickers are a NEW set (never used in Loop 1), all top-liquidity
Bybit spot pairs. Data availability checked before inclusion; anything missing
or thin is EXCLUDED and reported, never substituted.

Loop 2 set (selected by spot volume/liquidity, listed on Bybit before 2022,
still top-100 today — no survivorship consideration applied beyond that):
  BNBUSDT, LINKUSDT, AVAXUSDT, DOTUSDT, TRXUSDT, SHIB1000USDT(?), ATOMUSDT
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .backtest import CostModel, run_single
from .data import load_or_fetch
from .final_loop1 import REG, _metrics, _run
from .run_loop1 import BASE_COSTS, INITIAL, RESULTS, TF_BAR_HOURS, TICKERS
from .strategy import TrendParams, precompute

# candidate set — availability verified at runtime; failures reported + excluded
CANDIDATES = ["BNBUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT",
              "ATOMUSDT", "NEARUSDT", "FILUSDT"]


def run_fresh(tf: str, combo: dict, tickers: list[str]):
    params = TrendParams(**combo)
    out = {}
    for sym in tickers:
        df, _ = load_or_fetch(sym, tf)
        ind = precompute(df, params)
        sma = df["close"].rolling(REG[tf]).mean()
        reg_ok = np.array(df["close"] > sma, dtype=bool)
        ind.entry_signal = ind.entry_signal & reg_ok
        out[sym] = run_single(df, ind, params, BASE_COSTS, sym,
                               initial_cash=INITIAL / len(tickers),
                               bar_hours=TF_BAR_HOURS[tf])
    return out


def main() -> None:
    selected = json.loads((RESULTS / "selected.json").read_text())
    # 1. verify data availability, exclude failures with a report
    ok = []
    for sym in CANDIDATES:
        try:
            df, rep = load_or_fetch(sym, "1d")
            ok.append((sym, rep))
        except Exception as e:
            print(f"EXCLUDED {sym}: {e}")
    tickers = [s for s, _ in ok]
    print(f"Loop 2 tickers: {tickers}")
    for s, rep in ok:
        print(" ", rep.summary())

    # the slice/metrics helpers read the module-level universe — point it at
    # the fresh set for the whole run
    import quant.run_loop1 as rl
    rl.TICKERS = tickers

    report = {}
    for tf, combo in selected.items():
        runs = run_fresh(tf, combo, tickers)
        from .run_loop1 import _slice_metrics
        m, trades, _ = _slice_metrics(runs, tf, None,
                                      pd.Timestamp("2026-10-04", tz="UTC"))
        # split into train/val/holdout as well
        mtr, ttr, _ = _slice_metrics(runs, tf, None,
                                     pd.Timestamp("2024-07-01", tz="UTC"))
        mva, _, _ = _slice_metrics(runs, tf, pd.Timestamp("2024-07-01", tz="UTC"),
                                   pd.Timestamp("2025-07-01", tz="UTC"))
        mho, _, _ = _slice_metrics(runs, tf, pd.Timestamp("2025-07-01", tz="UTC"),
                                   pd.Timestamp("2026-10-04", tz="UTC"))
        pnl_by_sym = {s: round(sum(t.net_pnl for t in r.trades), 0)
                      for s, r in runs.items()}
        report[tf] = {
            "full": m.as_dict(), "train": mtr.as_dict(), "val": mva.as_dict(),
            "holdout": mho.as_dict(), "per_ticker_net_pnl": pnl_by_sym,
        }
        print(f"\n=== LOOP 2 {tf} (frozen params {combo}) ===")
        for k in ("train", "val", "holdout", "full"):
            mm = report[tf][k]
            print(f"  {k:>8}: sharpe={mm['sharpe']:6.2f} cagr={mm['cagr']:7.1%} "
                  f"mdd={mm['max_dd']:6.1%} n={mm['n_trades']:>4} pf={mm['profit_factor']:5.2f} "
                  f"win={mm['win_rate']:5.1%}")
        print("  per-ticker:", pnl_by_sym)

    out = RESULTS.parent / "loop2"
    out.mkdir(exist_ok=True)
    (out / "loop2_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nsaved ->", out / "loop2_report.json")


if __name__ == "__main__":
    main()
