"""Research runner: grid -> IS selection -> VAL -> HOLDOUT -> walk-forward ->
cost stress -> sensitivity -> fresh-ticker OOS -> regime split.

Discipline (mirrors quant.run_loop1):
  * In-sample universe (Loop-1 set, point-in-time 2021-07 top Bybit spot pairs)
    is the ONLY universe used for parameter selection.
  * The other 14 cached tickers are FRESH — touched once, at the very end.
  * Splits (UTC): TRAIN ..2024-07-01 | VAL 2024-07-01..2025-07-01 |
    HOLDOUT 2025-07-01..end. VAL is a stability check, not a selection pool.

Usage:
  python -m quant.alligator.research grid       # cached full-history runs
  python -m quant.alligator.research select     # train ranking + pick
  python -m quant.alligator.research holdout    # final, once
  python -m quant.alligator.research walkforward
  python -m quant.alligator.research stress     # costs + sensitivity + fresh + regimes
  python -m quant.alligator.research all
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import EngineResult, run_single
from .indicators import compute_all
from .strategy import AlliParams, precompute
from ..backtest import CostModel
from ..data import load_or_fetch
from ..metrics import portfolio_metrics, regime_split, trade_stats

TICKERS_IS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT",
              "ADAUSDT", "DOGEUSDT", "SOLUSDT"]
TICKERS_FRESH = ["ALGOUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "CRVUSDT",
                 "DOTUSDT", "ETCUSDT", "FILUSDT", "LINKUSDT", "NEARUSDT",
                 "SANDUSDT", "TRXUSDT", "UNIUSDT", "XLMUSDT"]
TF_BAR_HOURS = {"4h": 4.0, "1d": 24.0}
TF_MAX_HOLD = {"4h": 45, "1d": 25}          # time-stop, bars (7.5d / ~3.6w)
TRAIN_END = pd.Timestamp("2024-07-01", tz="UTC")
VAL_END = pd.Timestamp("2025-07-01", tz="UTC")
INITIAL = 100_000.0

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = RESULTS / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

GRID = {
    "entry_mode": ["breakout", "pullback", "both"],
    "n_break": [20, 30, 40, 55],
    "k_sl_atr": [2.5, 3.0],
    "k_trail": [3.0, 4.0, 5.0],
    "exit_level": ["teeth", "jaw"],
}
BASE_COSTS = CostModel()  # fee 0.1%/side, slip 5bps/side, stop slip 8bps extra


def combo_list() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def _params(tf: str, combo: dict) -> AlliParams:
    kw = dict(combo)
    kw["max_hold_bars"] = TF_MAX_HOLD[tf]
    return AlliParams(**kw)


def _ind_cache(sym: str, tf: str) -> tuple[pd.DataFrame, dict]:
    df, _ = load_or_fetch(sym, tf)
    return df, compute_all(df)


def run_cached(tf: str, combo: dict, tickers: list[str],
               costs: CostModel, tag: str = "base") -> dict[str, EngineResult]:
    """Full-history runs for one combo (memoized to disk)."""
    key = f"{tag}_{tf}_" + "_".join(f"{k}{v}" for k, v in combo.items())
    key = key.replace("/", "-")
    path = RUNS / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    p = _params(tf, combo)
    out: dict[str, EngineResult] = {}
    for sym in tickers:
        df, pre = _ind_cache(sym, tf)
        ind = precompute(df, p, pre=pre)
        out[sym] = run_single(df, ind, p, costs, sym,
                              initial_cash=INITIAL / len(tickers),
                              bar_hours=TF_BAR_HOURS[tf],
                              min_hold_hours=TF_BAR_HOURS[tf])
    path.write_bytes(pickle.dumps(out))
    return out


def slice_portfolio(runs: dict[str, EngineResult], tf: str,
                    start: pd.Timestamp | None,
                    end: pd.Timestamp | None) -> tuple[object, list, pd.Series]:
    """Portfolio metrics for a date slice (mirror of quant.run_loop1 logic).
    ``None`` start/end means unbounded on that side."""
    slice_init = INITIAL / max(len(runs), 1)
    equities, exposures, trades = {}, {}, []
    for sym, res in runs.items():
        s = pd.Series(res.equity, index=res.index)
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index < end]
        if len(s) < 10:
            raise RuntimeError(f"{sym}: only {len(s)} bars in slice")
        equities[sym] = s
        for tr in res.trades:
            if (start is None or tr.entry_time >= start.to_pydatetime()) and \
               (end is None or tr.entry_time < end.to_pydatetime()):
                trades.append(tr)
    grid = sorted(set().union(*[s.index for s in equities.values()]))
    grid = pd.DatetimeIndex(grid)
    total = None
    for sym, s in equities.items():
        s = s.reindex(grid).ffill().fillna(slice_init)
        total = s if total is None else total + s
    for sym, res in runs.items():
        ex = pd.Series(res.exposure, index=res.index)
        if start is not None:
            ex = ex[ex.index >= start]
        if end is not None:
            ex = ex[ex.index < end]
        ex = ex.reindex(grid).ffill()
        exposures[sym] = ex.to_numpy(bool)
    m = portfolio_metrics(total.to_numpy(), grid, trades, tf, INITIAL, exposures)
    return m, trades, total


def selection_score(m) -> float:
    """Composite IS score. Hard gates first (capital preservation priority),
    then balanced risk-adjusted terms. Documented, deterministic."""
    if m.n_trades < 60 or m.max_dd > 0.45:
        return -1.0
    return (1.0 * m.sortino + 0.5 * m.sharpe + 2.0 * m.calmar
            + 1.0 * m.win_rate + 0.25 * min(m.profit_factor, 4.0))


def cmd_grid() -> None:
    combos = combo_list()
    print(f"{len(combos)} combos x {len(TICKERS_IS)} tickers x 2 timeframes")
    for tf in TF_BAR_HOURS:
        for i, combo in enumerate(combos):
            run_cached(tf, combo, TICKERS_IS, BASE_COSTS)
            if (i + 1) % 18 == 0:
                print(f"  {tf}: {i + 1}/{len(combos)} done")


def cmd_select() -> None:
    rows = []
    for tf in TF_BAR_HOURS:
        for combo in combo_list():
            runs = run_cached(tf, combo, TICKERS_IS, BASE_COSTS)
            m, _, _ = slice_portfolio(runs, tf, None, TRAIN_END)
            mv, _, _ = slice_portfolio(runs, tf, TRAIN_END, VAL_END)
            rows.append({"tf": tf, **combo, "score": selection_score(m),
                         "tr_sharpe": m.sharpe, "tr_sortino": m.sortino,
                         "tr_calmar": m.calmar, "tr_cagr": m.cagr,
                         "tr_max_dd": m.max_dd, "tr_n": m.n_trades,
                         "tr_pf": m.profit_factor, "tr_win": m.win_rate,
                         "val_sharpe": mv.sharpe, "val_max_dd": mv.max_dd,
                         "val_cagr": mv.cagr, "val_n": mv.n_trades})
    df = pd.DataFrame(rows).sort_values(["tf", "score"], ascending=False)
    df.to_csv(RESULTS / "grid_train.csv", index=False)
    for tf in TF_BAR_HOURS:
        sub = df[df.tf == tf].head(8)
        print(f"\n=== TRAIN {tf} top-8 (by composite score) ===")
        print(sub.to_string(index=False))
    # stability: top-5 VAL sharpes must not collapse (report only, gate at holdout)
    print("\nsaved ->", RESULTS / "grid_train.csv")


SELECTED: dict[str, dict] = {}  # filled by research; see selection.json


def _load_selected() -> dict[str, dict]:
    if not SELECTED:
        SELECTED.update(json.loads((RESULTS / "selection.json").read_text()))
    return SELECTED


def cmd_holdout() -> None:
    sel = _load_selected()
    out = {}
    for tf, combo in sel.items():
        runs = run_cached(tf, combo, TICKERS_IS, BASE_COSTS)
        m_tr, _, _ = slice_portfolio(runs, tf, None, TRAIN_END)
        m_va, _, _ = slice_portfolio(runs, tf, TRAIN_END, VAL_END)
        m_ho, trades_ho, eq_ho = slice_portfolio(runs, tf, VAL_END, None)
        btc, _ = load_or_fetch("BTCUSDT", tf)
        regimes = regime_split(eq_ho.to_numpy(), eq_ho.index,
                               btc["close"].reindex(eq_ho.index).ffill())
        out[tf] = {
            "params": combo,
            "train": m_tr.as_dict(), "val": m_va.as_dict(),
            "holdout": m_ho.as_dict(),
            "holdout_regimes": regimes,
            "holdout_trade_stats": trade_stats(trades_ho),
            "holdout_exit_mix": _exit_mix(trades_ho),
            "holdout_expectancy_pct": _expectancy_pct(trades_ho),
        }
        eq_ho.to_csv(RESULTS / f"equity_holdout_{tf}.csv")
        print(f"\n=== {tf} params {combo} ===")
        for split, m in (("TRAIN", m_tr), ("VAL", m_va), ("HOLDOUT", m_ho)):
            print(f"{split:8s} cagr={m.cagr:7.2%} sharpe={m.sharpe:5.2f} "
                  f"sortino={m.sortino:5.2f} maxdd={m.max_dd:6.2%} "
                  f"pf={m.profit_factor:4.2f} win={m.win_rate:5.1%} "
                  f"n={m.n_trades}")
    (RESULTS / "holdout.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved ->", RESULTS / "holdout.json")


def _exit_mix(trades) -> dict:
    if not trades:
        return {}
    reasons = [t.exit_reason for t in trades]
    return {r: reasons.count(r) / len(reasons) for r in sorted(set(reasons))}


def _expectancy_pct(trades) -> float:
    """Mean net pnl per trade as % of the per-symbol slice equity."""
    if not trades:
        return 0.0
    slice_init = INITIAL / len(TICKERS_IS)
    return float(np.mean([t.net_pnl for t in trades]) / slice_init)


def cmd_walkforward() -> None:
    """Anchored walk-forward on 4h: select on [start, test_start), test on the
    following 6 months; step 6 months from 2023-01 through the data end."""
    tf = "4h"
    test_starts = pd.date_range("2023-01-01", "2026-01-01", freq="6MS", tz="UTC")
    windows = []
    chained = None
    for ts in test_starts:
        te = ts + pd.DateOffset(months=6)
        best, best_score = None, -np.inf
        for combo in combo_list():
            runs = run_cached(tf, combo, TICKERS_IS, BASE_COSTS)
            try:
                m, _, _ = slice_portfolio(runs, tf, None, ts)
            except RuntimeError:
                continue
            s = selection_score(m)
            if s > best_score:
                best, best_score = combo, s
        runs = run_cached(tf, best, TICKERS_IS, BASE_COSTS)
        try:
            m, _, eq = slice_portfolio(runs, tf, ts, te)
        except RuntimeError:
            continue
        chained = eq if chained is None else pd.concat([chained, eq])
        windows.append({"test_start": str(ts.date()), "test_end": str(te.date()),
                        "params": best,
                        "cagr": m.cagr, "sharpe": m.sharpe, "sortino": m.sortino,
                        "max_dd": m.max_dd, "pf": m.profit_factor,
                        "win_rate": m.win_rate, "n_trades": m.n_trades})
        print(f"{str(ts.date())} -> {best} | sharpe={m.sharpe:.2f} "
              f"dd={m.max_dd:.2%} n={m.n_trades}")
    chained.to_csv(RESULTS / "equity_walkforward_4h.csv")
    out = {"windows": windows,
           "param_picks": sorted({json.dumps(w["params"]) for w in windows})}
    (RESULTS / "walkforward.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved ->", RESULTS / "walkforward.json")


def cmd_stress() -> None:
    sel = _load_selected()
    # --- cost / slippage robustness on the final params ---
    costs_out = {}
    for tf, combo in sel.items():
        for mult in (0.5, 1.0, 2.0, 3.0):
            c = CostModel(fee_rate=0.001 * mult, slippage=0.0005 * mult,
                          stop_slippage=0.0008 * mult)
            runs = run_cached(tf, combo, TICKERS_IS, c, tag=f"cost{mult}")
            m_full, _, _ = slice_portfolio(runs, tf, None, VAL_END)
            m_ho, _, _ = slice_portfolio(runs, tf, VAL_END, None)
            costs_out[f"{tf}_x{mult}"] = {
                "trainval_cagr": m_full.cagr, "trainval_sharpe": m_full.sharpe,
                "trainval_max_dd": m_full.max_dd,
                "holdout_cagr": m_ho.cagr, "holdout_sharpe": m_ho.sharpe,
                "holdout_max_dd": m_ho.max_dd, "holdout_n": m_ho.n_trades}
    (RESULTS / "costs.json").write_text(json.dumps(costs_out, indent=2, default=str))
    print(json.dumps(costs_out, indent=1))

    # --- risk-per-trade ladder: a portfolio-level decision, NOT a signal
    # parameter; Sharpe/Sortino should be ~invariant, CAGR/DD scale ~linearly ---
    ladder = {}
    for risk in (0.0025, 0.005, 0.01):
        combo = dict(sel["4h"]) | {"risk_pct": risk}
        runs = run_cached("4h", combo, TICKERS_IS, BASE_COSTS, tag=f"risk{risk}")
        m_f, _, eq_f = slice_portfolio(runs, "4h", None, VAL_END)
        m_h, _, _ = slice_portfolio(runs, "4h", VAL_END, None)
        ladder[f"risk={risk:.2%}"] = {
            "trainval": {"cagr": m_f.cagr, "sharpe": m_f.sharpe,
                         "sortino": m_f.sortino, "max_dd": m_f.max_dd,
                         "pf": m_f.profit_factor, "win": m_f.win_rate,
                         "n": m_f.n_trades},
            "holdout": {"cagr": m_h.cagr, "sharpe": m_h.sharpe,
                        "sortino": m_h.sortino, "max_dd": m_h.max_dd,
                        "pf": m_h.profit_factor, "win": m_h.win_rate,
                        "n": m_h.n_trades}}
        eq_f.to_csv(RESULTS / f"equity_trainval_risk{risk}.csv")
    (RESULTS / "risk_ladder.json").write_text(
        json.dumps(ladder, indent=2, default=str))
    print(json.dumps(ladder, indent=1))

    # --- one-at-a-time parameter sensitivity on the FULL train period ---
    rows = []
    for tf, base in sel.items():
        for key, values in GRID.items():
            for v in values:
                if base.get(key) == v:
                    continue
                combo = dict(base) | {key: v}
                runs = run_cached(tf, combo, TICKERS_IS, BASE_COSTS)
                m, _, _ = slice_portfolio(runs, tf, None, TRAIN_END)
                rows.append({"tf": tf, "param": key, "value": str(v),
                             "sharpe": m.sharpe, "sortino": m.sortino,
                             "calmar": m.calmar, "max_dd": m.max_dd,
                             "cagr": m.cagr, "win_rate": m.win_rate,
                             "pf": m.profit_factor, "n": m.n_trades})
    sdf = pd.DataFrame(rows)
    sdf.to_csv(RESULTS / "sensitivity.csv", index=False)
    print("\nsaved ->", RESULTS / "sensitivity.csv")

    # --- fresh tickers (never used in selection), final params ---
    fresh = {}
    for tf, combo in sel.items():
        runs = run_cached(tf, combo, TICKERS_FRESH, BASE_COSTS, tag="fresh")
        m_full, trades_f, eq_f = slice_portfolio(runs, tf, None, None)
        m_ho, _, _ = slice_portfolio(runs, tf, VAL_END, None)
        fresh[tf] = {"full": m_full.as_dict(), "holdout": m_ho.as_dict(),
                     "expectancy_pct": _expectancy_pct_fresh(trades_f),
                     "n_symbols": len(TICKERS_FRESH)}
        eq_f.to_csv(RESULTS / f"equity_fresh_{tf}.csv")
        print(f"\nFRESH {tf}: full cagr={m_full.cagr:.2%} sharpe={m_full.sharpe:.2f} "
              f"dd={m_full.max_dd:.2%} | holdout cagr={m_ho.cagr:.2%} "
              f"sharpe={m_ho.sharpe:.2f} dd={m_ho.max_dd:.2%}")
    (RESULTS / "fresh_tickers.json").write_text(
        json.dumps(fresh, indent=2, default=str))
    print("saved ->", RESULTS / "fresh_tickers.json")


def _expectancy_pct_fresh(trades) -> float:
    if not trades:
        return 0.0
    slice_init = INITIAL / len(TICKERS_FRESH)
    return float(np.mean([t.net_pnl for t in trades]) / slice_init)


def cmd_all() -> None:
    cmd_grid()
    cmd_select()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"grid": cmd_grid, "select": cmd_select, "holdout": cmd_holdout,
     "walkforward": cmd_walkforward, "stress": cmd_stress,
     "all": cmd_all}[cmd]()
