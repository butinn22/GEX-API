"""Adversarial checks on the shortlist, run on the full real sample.

Everything here answers one question: *is the result a property of the strategy, or a
property of one lucky ticker / one lucky trade / one lucky regime?* Each check is
designed to break the result if it is fragile.

Checks
------
1. Pooled full-sample metrics (the headline).
2. Remove the best ticker  -> re-pool without it.
3. Remove the best trade   -> re-run that symbol with that one entry suppressed.
4. Regime segmentation     -> split trades by BTC's own trend state at entry (causal).
5. Drawdown duration       -> the longest stretch below the prior equity high.
6. Exit-class breakdown    -> how many trades close early, and why.

Run:  python -m research.emf_adl.final_check
"""
from __future__ import annotations

import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from research.emf_adl import data as D  # noqa: E402
from research.emf_adl import loop as L  # noqa: E402
from research.emf_adl import metrics as M  # noqa: E402
from research.emf_adl import rules as R  # noqa: E402
from research.emf_adl import run_loops as RL  # noqa: E402
from research.emf_adl.engine import run as run_engine  # noqa: E402

SHORTLIST = ("V1_repaired", "V39_mkt_long_nostop", "V40_lo_struct_mkt",
             "V42_lo_struct_mkt_sl", "V43_mkt_long_atr8")
TIMEFRAMES = ("4H", "1D")


def _run_suppressing(ctx: RL.SeriesContext, v: L.Variant, costs,
                     sym: str, entry_bar: int, exit_bar: int, side: int):
    """Re-run one series with one whole trade blocked.

    This is the "remove the best trade" counterfactual: the same engine, the same costs,
    the same bars — one trade removed.

    Blocking a *single* signal bar is not sufficient and was a real bug here: the EMF+ADL
    entry is a level condition, not an edge, so suppressing one bar merely re-enters on the
    next bar and the trade count is unchanged (2428 -> 2428). To actually remove the trade,
    the entry signal is suppressed from the signal bar through the original trade's exit
    bar, so no replacement position can open inside the same window.
    """
    tf_hours = D.TIMEFRAMES[ctx.timeframe][1]
    ss = ctx.signals[v.repair_hybrid]
    el = ss.entry_long.copy()
    es = ss.entry_short.copy()
    if v.long_only:
        es = np.zeros_like(es)
    if v.short_only:
        el = np.zeros_like(el)
    el, es = RL._apply_gates(ctx, v, el, es)
    # The engine reads the signal at bar j and fills at bar j+1's open, recording
    # ``entry_bar = j + 1``. Suppressing bar ``entry_bar`` would be one bar too late, and
    # suppressing only bar ``j`` lets the trade re-open at ``j+1``.
    sig_bar = entry_bar - 1
    if sig_bar < 0:
        return None
    lo = sig_bar
    hi = min(exit_bar, len(el) - 1)          # signal bars up to and including exit bar
    if side > 0:
        el[lo: hi + 1] = False
    else:
        es[lo: hi + 1] = False
    ss = replace(ss, entry_long=el, entry_short=es)
    res = run_engine(
        ctx.bars, ss, symbol=sym, timeframe=ctx.timeframe, tf_hours=tf_hours,
        costs=costs, stops=RL._stops_for(ctx, v), funding=ctx.funding,
    )
    return res



def _regime_of(symbol: str, timeframe: str, start: int, end: int) -> np.ndarray:
    """BTC trend state per bar: +1 above its own 200-period EMA, -1 below.

    Causal by construction — an EMA at bar i uses bars <= i only. Regimes are assigned
    from BTC and applied to every ticker's trades, so a trade is judged against the
    market backdrop that actually existed when it was opened.
    """
    s = D.load_series("BTCUSDT", timeframe, start, end)
    c = s.bars["close"].to_numpy(float)
    ema = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()
    return np.where(c > ema, 1, -1), s.bars["timestamp"].to_numpy(np.int64)


def _dd_duration(eq: np.ndarray) -> int:
    """Longest number of observations spent below a prior equity peak."""
    if len(eq) < 2:
        return 0
    peak = np.maximum.accumulate(eq)
    under = eq < peak * (1 - 1e-12)
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def _pool(results: dict, tf: str, costs) -> dict:
    pm = L.panel_metrics(results, tf)
    pm["flags"] = L.concentration_flags(pm)
    return pm


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(SHORTLIST))
    args = ap.parse_args(argv)
    shortlist = [n for n in args.variants.split(",") if n]

    uni = RL.load_universe(RL.OUT / "universe.json")
    pool = uni["symbols"]
    start, end = D.ms(*RL.STUDY_START), D.ms(*RL.STUDY_END)
    contexts, exclusions = RL.build_contexts(pool, start, end)
    cat = {v.name: v for v in RL.catalogue()}
    costs = RL.Costs()

    print(f"universe: {len(pool)} symbols; contexts: {len(contexts)}; "
          f"exclusions: {len(exclusions)}")
    for e in exclusions:
        print("   EXCLUDED", e)

    report: dict = {"checks": {}}
    for name in shortlist:
        v = cat[name]
        print("\n" + "=" * 78)
        print(f"VARIANT {name}")
        print(f"  hypothesis: {v.hypothesis}")
        print("=" * 78)
        block: dict = {"variant": name, "hypothesis": v.hypothesis, "by_tf": {}}
        for tf in TIMEFRAMES:
            res = {}
            for (sym, t), ctx in contexts.items():
                if t != tf:
                    continue
                r, _ = RL.evaluate(ctx, v, costs, window="all")
                res[sym] = r
            if not res:
                continue

            base = _pool(res, tf, costs)
            # Exit accounting straight from the ledger, so the report never has to
            # transcribe it by hand.
            ex_reason: dict = {}
            ex_class: dict = {}
            n_pre_stop = n_pre_tp = 0
            for r in res.values():
                for t in r.trades:
                    ex_reason[t.exit_reason] = ex_reason.get(t.exit_reason, 0) + 1
                    ex_class[t.exit_class] = ex_class.get(t.exit_class, 0) + 1
                    if t.exit_class == "STOP_LOSS_BEFORE_4H":
                        n_pre_stop += 1
                    if t.exit_class == "TAKE_PROFIT_BEFORE_4H":
                        n_pre_tp += 1
            print(f"\n--- {tf} full sample ---")
            print(f"  portfolio : ret={base['total_return']:+.2%} "
                  f"cagr={base['cagr']:+.2%} sharpe={base['sharpe']:+.2f} "
                  f"sortino={base['sortino']:+.2f} maxDD={base['max_dd']:.1%} "
                  f"calmar={base['calmar']:+.2f}")
            print(f"  trades    : n={base['total_trades']} PF={base['profit_factor']:.2f} "
                  f"win={base['win_rate']:.1%} exp={base['expectancy_pct']:+.3%} "
                  f"avgHold={base['avg_holding_hours']:.1f}h")
            print(f"  breadth   : pos={base['share_tickers_positive']:.0%} "
                  f"medSharpe={base['median_ticker_sharpe']:+.2f} "
                  f"exposure={base.get('exposure', 0):.1%}")
            print(f"  exits     : stop<4h={base.get('n_stop_before_4h', 0)} "
                  f"tp<4h={base.get('n_tp_before_4h', 0)} "
                  f"after4h={base.get('n_after_4h', 0)} "
                  f"minHoldCompliance={base.get('min_holding_compliance', 1.0):.2f}")
            print(f"  costs     : fees={base.get('total_fees', 0):.1f} "
                  f"funding={base.get('total_funding', 0):+.1f} "
                  f"concentration: bestTrade={base.get('best_trade_share', 0):.0%} "
                  f"top5={base.get('top5_share', 0):.0%}")
            print(f"  flags     : {'; '.join(base['flags']) or 'none'}")

            eq, _ = L.panel_equity(res)
            dd_bars = _dd_duration(eq)
            print(f"  max DD duration: {dd_bars} bars "
                  f"({dd_bars * D.TIMEFRAMES[tf][1] / 24:.0f} days)")

            # --- check: remove the single best ticker ------------------------- #
            by_ret = sorted(((sym, M.compute_metrics(r.equity, r.trades,
                                                     D.TIMEFRAMES[tf][1]).total_return)
                             for sym, r in res.items()), key=lambda kv: kv[1], reverse=True)
            worst_cut = by_ret[0]
            res_wo = {s: r for s, r in res.items() if s != worst_cut[0]}
            wo = _pool(res_wo, tf, costs)
            print(f"  remove best ticker ({worst_cut[0]}, {worst_cut[1]:+.0%}): "
                  f"sharpe={wo['sharpe']:+.2f} ret={wo['total_return']:+.2%} "
                  f"medSharpe={wo['median_ticker_sharpe']:+.2f} "
                  f"pos={wo['share_tickers_positive']:.0%}")

            # --- check: remove the single best trade -------------------------- #
            all_tr = [(sym, t) for sym, r in res.items() for t in r.trades]
            ledger: dict = {}
            wt = None
            removed: dict = {}
            if all_tr:
                bsym, btrade = max(all_tr, key=lambda st: st[1].net_pnl)
                removed = {"symbol": bsym, "net_pnl": float(btrade.net_pnl),
                           "exit_reason": btrade.exit_reason,
                           "entry_bar": int(btrade.entry_bar),
                           "exit_bar": int(btrade.exit_bar)}
                # (a) Ledger-level: pooled trade stats with the single best trade
                #     removed. Exact, no re-simulation, directly answers "does one
                #     trade carry the P&L".
                base_p = np.asarray([t.net_pnl for sym, t in all_tr], float)
                kept = [t for sym, t in all_tr if not (sym == bsym and t is btrade)]
                p = np.asarray([t.net_pnl for t in kept], float)
                gross_w = float(p[p > 0].sum())
                gross_l = float(-p[p < 0].sum())
                pf_ex = gross_w / gross_l if gross_l > 0 else float("inf")
                ledger = {
                    "n_before": int(len(base_p)), "n_after": int(len(p)),
                    "net_before": float(base_p.sum()), "net_after": float(p.sum()),
                    "pf_after": float(pf_ex), "exp_after": float(p.mean()),
                    "win_after": float((p > 0).mean()),
                    "removed_net": float(btrade.net_pnl),
                }
                print(f"  remove best trade (ledger: {bsym} {btrade.net_pnl:+.2f} net, "
                      f"{btrade.exit_reason}): n={len(kept)} net={p.sum():+.2f} "
                      f"PF={pf_ex:.2f} exp={p.mean():+.3f} "
                      f"win={(p > 0).mean():.1%}")
                # (b) Engine-level: block the whole trade window and re-run.
                ctx = contexts[(bsym, tf)]
                r2 = _run_suppressing(ctx, v, costs, bsym,
                                      btrade.entry_bar, btrade.exit_bar, btrade.side)
                res_wt = dict(res)
                if r2 is not None:
                    res_wt[bsym] = r2
                wt = _pool(res_wt, tf, costs)
                delta = wt["total_trades"] - base["total_trades"]
                print(f"  remove best trade (engine: signals {btrade.entry_bar - 1}"
                      f"..{btrade.exit_bar} blocked): sharpe={wt['sharpe']:+.2f} "
                      f"ret={wt['total_return']:+.2%} PF={wt['profit_factor']:.2f} "
                      f"trades {base['total_trades']}->{wt['total_trades']} "
                      f"({delta:+d})")
                if delta == 0:
                    print("     WARNING: trade count unchanged - the suppression did "
                          "not remove a trade")

            # --- regime segmentation ------------------------------------------ #
            regime, ts = _regime_of("BTCUSDT", tf, start, end)
            idx = pd.Series(regime, index=pd.to_datetime(ts, unit="ms", utc=True))
            seg: dict = {}
            for label, keep in (("bull(px>EMA200)", 1), ("bear(px<EMA200)", -1)):
                pnls, notionals = [], []
                for r in res.values():
                    for t in r.trades:
                        et = pd.Timestamp(t.entry_time)
                        col = int(np.searchsorted(idx.index.values,
                                                  np.datetime64(et, "ns"), side="right"))
                        if col <= 0 or col > len(idx):
                            continue
                        if int(idx.iloc[col - 1]) == keep:
                            pnls.append(t.net_pnl)
                            notionals.append(t.notional_at_entry)
                p = np.asarray(pnls, float)
                n = np.asarray(notionals, float)
                if len(p):
                    w = np.divide(p, n, out=np.zeros_like(p), where=n > 0)
                    seg[label] = {
                        "n": int(len(p)),
                        "net": float(p.sum()),
                        "win_rate": float((p > 0).mean()),
                        "avg_pct": float(w.mean()),
                    }
            for label, s_ in seg.items():
                print(f"  regime {label:16s}: n={s_['n']:4d} net={s_['net']:+8.2f} "
                      f"win={s_['win_rate']:.1%} avg={s_['avg_pct']:+.3%}")

            # --- year by year -------------------------------------------------- #
            yrs: dict[int, list] = {}
            for r in res.values():
                for t in r.trades:
                    yrs.setdefault(pd.Timestamp(t.entry_time).year, []).append(t)
            yline = []
            for y in sorted(yrs):
                p = np.asarray([t.net_pnl for t in yrs[y]], float)
                yline.append(f"{y}:n={len(p)}/{'%+.0f' % p.sum()}")
            print("  by year   : " + "  ".join(yline))

            block["by_tf"][tf] = {
                "base": base,
                "remove_best_ticker": wo,
                "removed_ticker": worst_cut[0],
                "remove_best_trade": wt,
                "removed_trade": removed,
                "remove_best_trade_ledger": ledger,
                "dd_duration_bars": dd_bars,
                "regime": seg,
                "exit_reason_counts": ex_reason,
                "exit_class_counts": ex_class,
                "n_pre_stop": n_pre_stop,
                "n_pre_tp": n_pre_tp,
                "per_ticker": L.per_ticker_table(res, tf),
            }
        report["checks"][name] = block

    (RL.OUT / "final_check.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\nwrote", RL.OUT / "final_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
