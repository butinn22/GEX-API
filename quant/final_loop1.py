"""Selection by neighborhood robustness + ONE holdout run + stress suite.

Selection rule (anti-overfit): among combos in the stable plateau
(n_break 30-55, k_sl 2.5-3.0, k_trail 4-5, ma_exit 100), pick by the MEDIAN
validation Sharpe of each combo's parameter neighbors, not its own max.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .backtest import CostModel, run_single
from .data import load_or_fetch
from .metrics import portfolio_metrics, regime_split, trade_stats
from .run_loop1 import (BASE_COSTS, INITIAL, RESULTS, TF_BAR_HOURS, TICKERS,
                         TRAIN_END, VAL_END, combo_list)
from .strategy import TrendParams, precompute

HOLDOUT_START = VAL_END
REG = {"4h": 1200, "1d": 200}
PLATEAU = [c for c in combo_list()
           if c["n_break"] in (30, 40, 55) and c["k_sl"] in (2.5, 3.0)
           and c["k_trail"] in (4.0, 5.0) and c["ma_exit"] == 100]


def _run(tf: str, combo: dict, costs: CostModel):
    params = TrendParams(**combo)
    out = {}
    for sym in TICKERS:
        df, _ = load_or_fetch(sym, tf)
        ind = precompute(df, params)
        sma = df["close"].rolling(REG[tf]).mean()
        reg_ok = np.array(df["close"] > sma, dtype=bool)
        ind.entry_signal = ind.entry_signal & reg_ok
        out[sym] = run_single(df, ind, params, costs, sym,
                               initial_cash=INITIAL / len(TICKERS),
                               bar_hours=TF_BAR_HOURS[tf])
    return out


def _metrics(runs, tf, lo, hi):
    """Slice metrics for [lo, hi) — same alignment logic as run_loop1."""
    from .run_loop1 import _slice_metrics
    return _slice_metrics(runs, tf, lo, hi)


def _neighbors(combo: dict) -> list[dict]:
    """Plateau combos differing in exactly one parameter by one grid step."""
    steps = {"n_break": [30, 40, 55], "k_sl": [2.5, 3.0],
             "k_trail": [4.0, 5.0], "ma_exit": [0, 100]}
    out = []
    for k, vals in steps.items():
        if combo[k] not in vals:
            return []  # outside plateau — no neighborhood defined
        for v in vals:
            if v != combo[k]:
                c = dict(combo)
                c[k] = v
                out.append(c)
    return out


def cmd_select2() -> None:
    for tf in TF_BAR_HOURS:
        rows = []
        for combo in PLATEAU:
            runs = _run(tf, combo, BASE_COSTS)
            mtr, ttr, _ = _metrics(runs, tf, None, TRAIN_END)
            mva, tva, _ = _metrics(runs, tf, TRAIN_END, VAL_END)
            nb = [c for c in _neighbors(combo) if c in PLATEAU]
            n_sharpes = []
            for c in nb:
                rn = _run(tf, c, BASE_COSTS)
                mn, _, _ = _metrics(rn, tf, TRAIN_END, VAL_END)
                n_sharpes.append(mn.sharpe)
            rows.append({
                **combo,
                "train_sharpe": mtr.sharpe, "train_cagr": mtr.cagr,
                "train_mdd": mtr.max_dd, "train_n": mtr.n_trades,
                "val_sharpe": mva.sharpe, "val_cagr": mva.cagr,
                "val_mdd": mva.max_dd, "val_n": mva.n_trades,
                "nbr_med_val_sharpe": float(np.median(n_sharpes)) if n_sharpes else np.nan,
                "nbr_min_val_sharpe": float(np.min(n_sharpes)) if n_sharpes else np.nan,
            })
        df = pd.DataFrame(rows)
        # joint score: own val sharpe and the neighborhood median must BOTH be solid
        df["score"] = np.minimum(df["val_sharpe"], df["nbr_med_val_sharpe"])
        df = df.sort_values("score", ascending=False)
        print(f"\n=== PLATEAU SELECTION {tf} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        df.to_csv(RESULTS / f"plateau_{tf}.csv", index=False)


SELECTED = {
    "4h": dict(n_break=30, k_sl=3.0, k_trail=5.0, ma_exit=100),
    "1d": dict(n_break=55, k_sl=2.5, k_trail=4.0, ma_exit=100),
}  # placeholder — overwritten by what select2 actually shows


def cmd_holdout() -> None:
    from pathlib import Path
    sel_path = RESULTS / "selected.json"
    if sel_path.exists():
        selected = json.loads(sel_path.read_text())
    else:
        selected = SELECTED
    report: dict = {"selected": selected, "holdout": {}, "stress": {}}
    for tf, combo in selected.items():
        runs = _run(tf, combo, BASE_COSTS)
        m, trades, data = _metrics(runs, tf, HOLDOUT_START,
                                   pd.Timestamp("2026-10-04", tz="UTC"))
        ts = trade_stats(trades)
        # per-ticker attribution on holdout
        attr = {}
        for sym, res in runs.items():
            pnl = sum(t.net_pnl for t in res.trades
                      if t.entry_time >= HOLDOUT_START.to_pydatetime())
            attr[sym] = round(pnl, 0)
        report["holdout"][tf] = {
            "metrics": m.as_dict(), "trade_stats": ts,
            "per_ticker_net_pnl": attr,
            "classifications": {
                "STOP_LOSS_BEFORE_4H": m.n_stop_before_4h,
                "TAKE_PROFIT_BEFORE_4H": m.n_tp_before_4h,
                "EXIT_AFTER_4H": m.n_exit_after_4h,
            },
        }
        print(f"\n=== HOLDOUT {tf} 2025-07 .. 2026-10 (single run, never seen before) ===")
        print(json.dumps(report["holdout"][tf]["metrics"], indent=2, default=str))
        print("per-ticker pnl:", attr)
        print("classifications:", report["holdout"][tf]["classifications"])
        # benchmark
        dfs = {s: load_or_fetch(s, tf)[0] for s in TICKERS}
        rets = []
        for s, d in dfs.items():
            d = d[d.index >= HOLDOUT_START]
            rets.append(d["close"].iloc[-1] / d["close"].iloc[0] - 1)
        print(f"equal-weight buy&hold holdout: {np.mean(rets):.1%}")

    # ── stress suite on the SAME holdout+val+train combined view ──
    for tf, combo in selected.items():
        stress = {}
        # 1. doubled costs
        runs2 = _run(tf, combo, CostModel(fee_rate=0.002, slippage=0.001,
                                          stop_slippage=0.0016))
        m2, _, _ = _metrics(runs2, tf, None, pd.Timestamp("2026-10-04", tz="UTC"))
        stress["2x_costs_full_period"] = {"sharpe": m2.sharpe, "cagr": m2.cagr,
                                           "max_dd": m2.max_dd}
        # 2. full-period metrics at base costs (context for the stress rows)
        runs = _run(tf, combo, BASE_COSTS)
        m, trades, _ = _metrics(runs, tf, None, pd.Timestamp("2026-10-04", tz="UTC"))
        stress["base_full_period"] = {"sharpe": m.sharpe, "cagr": m.cagr,
                                       "max_dd": m.max_dd, "n_trades": m.n_trades}
        # 3. drop best ticker (full period) — align on the union grid like the
        # portfolio slicer does (leading NaNs = uninvested slice cash)
        pnl_by_sym = {s: sum(t.net_pnl for t in r.trades)
                      for s, r in runs.items()}
        best = max(pnl_by_sym, key=pnl_by_sym.get)
        slice_init = INITIAL / len(TICKERS)
        grid = pd.DatetimeIndex(sorted(set().union(*[r.index for s, r in runs.items() if s != best])))
        parts = [pd.Series(r.equity, index=r.index).reindex(grid).ffill().fillna(slice_init)
                 for s, r in runs.items() if s != best]
        total = parts[0]
        for p in parts[1:]:
            total = total + p
        e = total.to_numpy()
        from .metrics import sharpe as _sh, max_drawdown as _mdd, _returns as _ret
        stress["drop_best_ticker"] = {
            "dropped": best, "sharpe": _sh(_ret(e), 365 * 6 if tf == "4h" else 365),
            "total_return": e[-1] / e[0] - 1.0, "max_dd": _mdd(e)}
        # 4. drop best trade (full period)
        pnls = sorted((t.net_pnl for t in trades), reverse=True)
        stress["drop_best_trade"] = {
            "best_trade": pnls[0], "total_net_minus_best": sum(pnls[1:]),
            "n_trades": len(pnls)}
        report["stress"][tf] = stress
        print(f"\n=== STRESS {tf} ===")
        print(json.dumps(stress, indent=2, default=str))

    out = RESULTS / "holdout_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\nsaved ->", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "select2"
    if cmd == "select2":
        cmd_select2()
    elif cmd == "holdout":
        cmd_holdout()
