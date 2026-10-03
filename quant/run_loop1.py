"""Loop 1 runner — Donchian/ATR trend-following core, 4H + 1D, Bybit spot.

Deterministic: no randomness anywhere. Ticker set is fixed below (Loop 1 set —
point-in-time top-volume Bybit spot pairs as of 2021-07; never reused).

Splits (by bar timestamp, UTC):
  TRAIN  : history start .. 2024-07-01
  VAL    : 2024-07-01 .. 2025-07-01   (parameter SELECTION happens here)
  HOLDOUT: 2025-07-01 .. end          (touched ONCE, at the very end)

Usage:
  python -m quant.run_loop1 fetch     # download + validate data
  python -m quant.run_loop1 grid      # run full parameter grid (cached)
  python -m quant.run_loop1 select    # train ranking + validation selection
  python -m quant.run_loop1 holdout   # single holdout confirmation + stress
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import AFTER_4H, CostModel, EngineResult, run_single
from .data import load_or_fetch
from .metrics import Metrics, portfolio_metrics, regime_split, trade_stats
from .strategy import TrendParams, precompute

TICKERS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]
TF_BAR_HOURS = {"4h": 4.0, "1d": 24.0}
TRAIN_END = pd.Timestamp("2024-07-01", tz="UTC")
VAL_END = pd.Timestamp("2025-07-01", tz="UTC")
INITIAL = 100_000.0
RESULTS = Path(__file__).resolve().parent / "results" / "loop1"
RESULTS.mkdir(parents=True, exist_ok=True)

GRID = {
    "n_break": [20, 30, 40, 55],
    "k_sl": [2.5, 3.0],
    "k_trail": [3.0, 4.0, 5.0],
    "ma_exit": [0, 100],
}

BASE_COSTS = CostModel()  # fee 0.1%/side, slip 5bps/side, stop slip 8bps extra


def combo_list() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def fetch_all() -> None:
    for sym in TICKERS:
        for tf in TF_BAR_HOURS:
            df, rep = load_or_fetch(sym, tf)
            print(rep.summary())
            if rep.n_gaps > 20:
                print(f"  WARNING: {sym} {tf} has {rep.n_gaps} gaps (max "
                      f"{rep.max_gap_bars} bars) — reported, not filled")


def _run_cached(tf: str, combo: dict, costs: CostModel) -> dict[str, EngineResult]:
    """Run one combo across all tickers (memoized to disk)."""
    tag = f"{tf}_" + "_".join(f"{k}{v}" for k, v in combo.items())
    path = RESULTS / f"run_{tag}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    params = TrendParams(**combo)
    out: dict[str, EngineResult] = {}
    for sym in TICKERS:
        df, _ = load_or_fetch(sym, tf)
        ind = precompute(df, params)
        out[sym] = run_single(df, ind, params, costs, sym,
                               initial_cash=INITIAL / len(TICKERS),
                               bar_hours=TF_BAR_HOURS[tf])
    path.write_bytes(pickle.dumps(out))
    return out


def _slice_metrics(runs: dict[str, EngineResult], tf: str,
                   start: pd.Timestamp | None, end: pd.Timestamp) -> tuple[Metrics, list, dict]:
    """Portfolio metrics for a date slice (equity sliced; trades attributed by entry)."""
    equities, exposures, trades = {}, {}, []
    for sym, res in runs.items():
        s = pd.Series(res.equity, index=res.index)
        if start is not None:
            s = s[s.index >= start]
        s = s[s.index < end]
        if len(s) < 10:
            raise RuntimeError(f"{sym}: only {len(s)} bars in slice")
        equities[sym] = s
        exposures[sym] = np.ones(len(s)) * np.nan  # placeholder, fixed below
        for tr in res.trades:
            if (start is None or tr.entry_time >= start.to_pydatetime()) and \
               tr.entry_time < end.to_pydatetime():
                trades.append(tr)
    grid = equities[TICKERS[0]].index
    total = None
    slice_init = INITIAL / len(TICKERS)
    for sym, s in equities.items():
        # tickers start at different dates: before a ticker exists its slice is
        # uninvested cash (slice_init); after it exists, forward-fill.
        s = s.reindex(grid).ffill().fillna(slice_init)
        total = s if total is None else total + s
    # exposure slice
    for sym, res in runs.items():
        ex = pd.Series(res.exposure, index=res.index)
        if start is not None:
            ex = ex[ex.index >= start]
        ex = ex[ex.index < end].reindex(grid).ffill()
        exposures[sym] = ex.to_numpy(bool)
    m = portfolio_metrics(total.to_numpy(), grid, trades, tf, INITIAL, exposures)
    return m, trades, {"equity": total, "index": grid}


def cmd_grid() -> None:
    combos = combo_list()
    print(f"{len(combos)} combos x {len(TICKERS)} tickers x 2 timeframes")
    for tf in TF_BAR_HOURS:
        for i, combo in enumerate(combos):
            _run_cached(tf, combo, BASE_COSTS)
            if (i + 1) % 12 == 0:
                print(f"  {tf}: {i + 1}/{len(combos)} done")


def cmd_select() -> None:
    combos = combo_list()
    rows = []
    for tf in TF_BAR_HOURS:
        for combo in combos:
            runs = _run_cached(tf, combo, BASE_COSTS)
            m, trades, _ = _slice_metrics(runs, tf, None, TRAIN_END)
            rows.append({"tf": tf, **combo,
                         "sharpe": m.sharpe, "calmar": m.calmar, "cagr": m.cagr,
                         "max_dd": m.max_dd, "n_trades": m.n_trades,
                         "pf": m.profit_factor, "win_rate": m.win_rate,
                         "exposure": m.exposure, "avg_hold_h": m.avg_hold_h})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "train_metrics.csv", index=False)
    for tf in TF_BAR_HOURS:
        sub = df[df.tf == tf].sort_values("sharpe", ascending=False)
        print(f"\n=== TRAIN {tf} (top 10 by Sharpe) ===")
        print(sub.head(10).to_string(index=False))
    print("\nsaved ->", RESULTS / "train_metrics.csv")


def cmd_holdout() -> None:
    # executed only after selection is frozen; placeholder kept explicit
    raise SystemExit("run select first, then edit SELECTED below and re-run")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "grid"
    {"fetch": fetch_all, "grid": cmd_grid, "select": cmd_select,
     "holdout": cmd_holdout}[cmd]()
