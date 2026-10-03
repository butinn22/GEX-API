"""Verification battery for the drawdown-first finalist.

Everything here is measurement, not assertion. Each block prints the control alongside the
candidate so the reader can see what the change actually bought, and every number comes
from a real Bybit panel replayed through the same engine.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.emf_adl import data as D  # noqa: E402
from research.emf_adl import loop as L  # noqa: E402
from research.emf_adl import portfolio as P  # noqa: E402
from research.emf_adl import run_loops as RL  # noqa: E402
from research.emf_adl.dd_loops import (  # noqa: E402
    _BARS_PER_YEAR, DdVariant, _signals_for, _stops_for, table,
)
from research.emf_adl.engine import Costs, Sizing  # noqa: E402

OUT = RL.OUT
TIMEFRAMES = ("4H", "1D")

#: The trailing stop is live on BOTH timeframes here — that is the brief's "fully working
#: trailing stop" requirement, satisfied by construction rather than by argument.
_T_4H = {"trail": True, "trail_atr_mult": 4.0, "trail_activate_r": 2.0,
         "breakeven_at_r": 2.0}
_T_1D = {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
         "own_ema_exit": True, "own_ema_span": 50, "breakeven_at_r": 2.0}

#: Recommended operating point of the capital-intensity frontier: 10% risk per trade with
#: per-symbol volatility normalisation to a 30%/yr target. Chosen because it dominates the
#: shipped incumbent on drawdown, Sharpe *and* return simultaneously; the frontier is
#: reported in section 8 rather than this point being presented as uniquely correct.
FINALIST = DdVariant(
    name="FINAL", base="V43_mkt_long_atr8", risk_per_trade=0.10, max_leverage=2.0,
    stop_mode="atr_trail", stop_atr_mult=8.0,
    trail=True, breakeven_at_r=2.0,
    per_tf={"4H": _T_4H, "1D": _T_1D},
    target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
)
#: Same rules and same risk budget, but no trailing stop and no vol normalisation — the
#: attribution control. Whatever FINAL gains over this is attributable to the stop and
#: sizing layers, not to the entry logic.
CONTROL = DdVariant(
    name="CONTROL", base="V43_mkt_long_atr8", risk_per_trade=0.10, max_leverage=2.0,
    stop_mode=None,
)
#: The strategy as originally shipped: 100% of equity per position, no portfolio cap.
#: ``risk_per_trade=0`` is the flat-size path in ``Sizing`` — it is the same switch the
#: I17 arm uses, so this reproduces that artifact exactly rather than approximating it.
INCUMBENT = DdVariant(
    name="INCUMBENT", base="V43_mkt_long_atr8",
    risk_per_trade=0.0, max_leverage=1.0, stop_mode=None,
)


def build(v: DdVariant, contexts, costs: Costs, *, window: str = "all"):
    """Run one variant per timeframe and return {tf: PortfolioRun}.

    Wires the sizing stack exactly as ``dd_loops.run_variant`` does, so a spec reproduces
    bit-for-bit across the two entry points. If these diverge, the verification battery is
    measuring a different strategy from the one the loop selected.
    """
    cat = {x.name: x for x in RL.catalogue()}
    base = cat[v.base]
    runs: dict = {}
    for tf in TIMEFRAMES:
        eff = replace(v, per_tf={}, **v.per_tf.get(tf, {})) if v.per_tf.get(tf) else v
        sub = {k: c for k, c in contexts.items() if k[1] == tf}
        wins = None if window == "all" else (lambda c, w=window: _win_mask(c, w))
        vol_for = (None if eff.target_vol is None
                   else (lambda c, vv=eff, t=tf: P.vol_target_scale(
                       c.bars,
                       target_bar_vol=vv.target_vol / np.sqrt(_BARS_PER_YEAR[t]),
                       window=vv.vol_window, lo=vv.vol_lo, hi=vv.vol_hi)))
        runs[tf] = P.run_portfolio(
            sub,
            signals_for=lambda c, b=base, vv=eff: _signals_for(c, b, vv),
            stops_for=lambda c, b=base, vv=eff: _stops_for(c, b, vv),
            sizing=Sizing(risk_per_trade=eff.risk_per_trade, max_leverage=eff.max_leverage),
            costs=costs, brake=P.NO_BRAKE, windows=wins, vol_for=vol_for,
        )
    return runs


def _win_mask(ctx, window: str) -> np.ndarray:
    sl = ctx.wins[window]
    m = np.zeros(len(ctx.bars), dtype=bool)
    m[sl] = True
    return m


def metrics(pr, tf: str) -> dict:
    trades = [t for r in pr.results.values() for t in r.trades]
    m = L.compute_metrics(pr.equity, trades, D.TIMEFRAMES[tf][1], timestamps=pr.timestamps)
    return {**m.as_dict(), **pr.stats,
            "mean_gross": float(np.nanmean(pr.gross_exposure)),
            "max_gross": float(np.nanmax(pr.gross_exposure)),
            "breadth": float(np.mean([L.compute_metrics(r.equity, r.trades,
                                                        D.TIMEFRAMES[tf][1]).total_return > 0
                                      for r in pr.results.values()])),
            "trail_exits": int(sum(1 for t in trades if t.exit_reason == "trail_stop")),
            "stop_exits": int(sum(1 for t in trades if t.exit_reason == "stop")),
            "n_trades": len(trades)}


def episodes(eq: np.ndarray, ts: np.ndarray | None, top: int = 6) -> list[dict]:
    """The worst N drawdown episodes with calendar dates attached."""
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    out: list[dict] = []
    i = 0
    n = len(dd)
    while i < n and len(out) < 400:
        if dd[i] >= -1e-9:
            i += 1
            continue
        j = i
        while j < n and dd[j] < -1e-9:
            j += 1
        seg = dd[i:j]
        trough = i + int(np.argmin(seg))
        start = int(np.argmax(eq[: i + 1])) if i > 0 else 0
        depth = float(-seg.min())
        if depth > 1e-4:
            out.append({
                "depth": depth,
                "start_bar": start,
                "trough_bar": trough,
                "recover_bar": j,
                "bars": int(j - start),
                "recovery_bars": int(j - trough),
                "start": _dt(ts, start), "trough": _dt(ts, trough), "recover": _dt(ts, j),
            })
        i = j
    out.sort(key=lambda e: -e["depth"])
    return out[:top]


def _dt(ts, i: int) -> str:
    if ts is None or i >= len(ts):
        return "end-of-sample"
    return str(pd.Timestamp(int(ts[i]), unit="ms", tz="UTC"))[:10]


def yearly(eq: np.ndarray, ts: np.ndarray) -> dict:
    idx = pd.to_datetime(ts, unit="ms", utc=True)
    s = pd.Series(eq, index=idx)
    out = {}
    for y, g in s.groupby(s.index.year):
        out[int(y)] = float(g.iloc[-1] / g.iloc[0] - 1.0)
    return out


def main() -> int:
    t0 = time.time()
    uni = RL.load_universe(OUT / "universe.json")
    pool = uni["symbols"]
    start, end = D.ms(*RL.STUDY_START), D.ms(*RL.STUDY_END)
    contexts, exclusions = RL.build_contexts(pool, start, end)
    print(f"universe: {len(pool)} symbols | contexts: {len(contexts)} | exclusions: {len(exclusions)}")
    for e in exclusions:
        print("   EXCLUDED", e)

    rep: dict = {"finalist": asdict(FINALIST), "control": asdict(CONTROL),
                 "incumbent": asdict(INCUMBENT)}
    base_costs = Costs()

    # ---------- 1. headline + equity-curve behaviour ---------- #
    print("\n" + "=" * 78)
    print("1. HEADLINE AND EQUITY-CURVE BEHAVIOUR")
    print("=" * 78)
    # INCUMBENT is the shipped strategy (flat 100% notional). CONTROL is the same rules at
    # the same risk budget as FINAL but stripped of the stop and sizing layers. Both are
    # needed: one shows the total improvement, the other attributes it.
    runs = {INCUMBENT.name: build(INCUMBENT, contexts, base_costs),
            CONTROL.name: build(CONTROL, contexts, base_costs),
            FINALIST.name: build(FINALIST, contexts, base_costs)}
    head: dict = {}
    for name, per_tf in runs.items():
        head[name] = {}
        for tf, pr in per_tf.items():
            met = metrics(pr, tf)
            met["years"] = [yearly(pr.equity, pr.timestamps)]
            met["episodes"] = episodes(pr.equity, pr.timestamps, 6)
            head[name][tf] = met
            print(f"\n{name} {tf}: maxDD={met['max_dd']:.2%} ({met['max_dd_bars']} bars, "
                  f"{met['max_dd_days']:.0f} d) ulcer={met['ulcer']:.4f} "
                  f"Sharpe={met['sharpe']:+.2f} Sortino={met['sortino']:+.2f} "
                  f"Calmar={met['calmar']:+.2f}\n"
                  f"   ret={met['total_return']:+.1%} CAGR={met['cagr']:+.2%} "
                  f"PF={met['profit_factor']:.2f} win={met['win_rate']:.1%} "
                  f"trades={met['n_trades']} breadth={met['breadth']:.0%}\n"
                  f"   time underwater={met['time_underwater']:.0%} "
                  f"longest underwater={met['longest_underwater_bars']} bars "
                  f"({met['longest_underwater_bars'] * D.TIMEFRAMES[tf][1] / 24:.0f} d)\n"
                  f"   frac DD>10%={met['frac_dd_gt_10']:.1%} "
                  f"longest DD>10%={met['longest_dd_gt_10_bars']} bars | "
                  f"gross mean={met['mean_gross']:.2f}x max={met['max_gross']:.2f}x\n"
                  f"   stop exits={met['stop_exits']} trail exits={met['trail_exits']} "
                  f"min-hold compliance={met['min_holding_compliance']:.2f}")
            print(f"   yearly: " + "  ".join(f"{y}:{v:+.1%}" for y, v in met["years"][0].items()))
            for e in met["episodes"][:4]:
                print(f"     DD {e['depth']:6.1%} {e['start']} -> {e['trough']} "
                      f"-> recover {e['recover']}  ({e['bars']} bars)")
    rep["headline"] = head

    # ---------- 2. the long drawdown episode, in detail ---------- #
    print("\n" + "=" * 78)
    print("2. THE PROLONGED STRETCH — is it deep, or just long?")
    print("=" * 78)
    for tf, pr in runs[FINALIST.name].items():
        eq = pr.equity
        ts = pr.timestamps
        peak = np.maximum.accumulate(eq)
        dd = eq / peak - 1.0
        uw = dd < -1e-9
        longest = 0
        cur = 0
        for flag in uw:
            cur = cur + 1 if flag else 0
            longest = max(longest, cur)
        thr = {}
        for t in (0.02, 0.05, 0.10, 0.15, 0.20):
            run_len, best = 0, 0
            for flag in (dd < -t):
                run_len = run_len + 1 if flag else 0
                best = max(best, run_len)
            thr[f"longest_dd_gt_{int(t*100)}pct_bars"] = best
        print(f"{tf}: longest below-peak run = {longest} bars "
              f"({longest * D.TIMEFRAMES[tf][1] / 24:.0f} days)")
        print("   longest run deeper than:", {k: v for k, v in thr.items()})
        print(f"   median depth while underwater = {np.median(dd[uw]):.2%}")
        rep.setdefault("prolonged", {})[tf] = {
            "longest_underwater_bars": int(longest), **thr,
            "median_underwater_depth": float(np.median(dd[uw])),
        }

    # ---------- 3. cost stress ---------- #
    print("\n" + "=" * 78)
    print("3. COST STRESS (2x fees, 4x slippage, +latency, funding on/off)")
    print("=" * 78)
    stress = {
        "base": Costs(),
        "slip_4bps": Costs(slippage_bps=4.0),
        "slip_8bps": Costs(slippage_bps=8.0),
        "fee_8bps": Costs(fee_rate=0.0008),
        "no_funding": Costs(funding_on=False),
        "harsh": Costs(fee_rate=0.0008, slippage_bps=8.0, latency_bps=3.0),
    }
    stress_out: dict = {}
    for label, c in stress.items():
        per_tf = build(FINALIST, contexts, c)
        stress_out[label] = {}
        line = []
        for tf, pr in per_tf.items():
            met = metrics(pr, tf)
            stress_out[label][tf] = {"max_dd": met["max_dd"], "sharpe": met["sharpe"],
                                     "total_return": met["total_return"],
                                     "profit_factor": met["profit_factor"],
                                     "expectancy_pct": met["expectancy_pct"]}
            line.append(f"{tf}: DD={met['max_dd']:.2%} Sharpe={met['sharpe']:+.2f} "
                        f"ret={met['total_return']:+.1%} PF={met['profit_factor']:.2f} "
                        f"exp={met['expectancy_pct']:+.3%}")
        print(f"  {label:12s} " + " | ".join(line))
    rep["cost_stress"] = stress_out

    # ---------- 4. walk-forward windows ---------- #
    print("\n" + "=" * 78)
    print("4. WALK-FORWARD (per-symbol calendar windows, no re-selection)")
    print("=" * 78)
    wf: dict = {}
    for window in ("train", "valid", "holdout", "all"):
        per_tf = build(FINALIST, contexts, base_costs, window=window)
        wf[window] = {}
        line = []
        for tf, pr in per_tf.items():
            if not len(pr.equity):
                continue
            met = metrics(pr, tf)
            wf[window][tf] = {"sharpe": met["sharpe"], "max_dd": met["max_dd"],
                              "total_return": met["total_return"], "n_trades": met["n_trades"]}
            line.append(f"{tf}: Sharpe={met['sharpe']:+.2f} DD={met['max_dd']:.2%} "
                        f"ret={met['total_return']:+.1%} n={met['n_trades']}")
        print(f"  {window:8s} " + " | ".join(line))
    rep["walk_forward"] = wf

    # ---------- 5. adversarial: remove best ticker / best trade ---------- #
    print("\n" + "=" * 78)
    print("5. ADVERSARIAL (remove best ticker; remove single best trade, ledger + engine)")
    print("=" * 78)
    adv: dict = {}
    for name, per_tf in runs.items():
        adv[name] = {}
        for tf, pr in per_tf.items():
            all_tr = [(s, t) for s, r in pr.results.items() for t in r.trades]
            net = np.asarray([t.net_pnl for _, t in all_tr], float)
            notion = np.asarray([t.notional_at_entry for _, t in all_tr], float)
            order = np.argsort(net)[::-1]
            tot = float(net.sum())
            # Per-ticker totals, so "best ticker" is a ticker, not a sum of notionals.
            by_sym = {}
            for s, t in all_tr:
                by_sym[s] = by_sym.get(s, 0.0) + t.net_pnl
            best_sym = max(by_sym, key=by_sym.get)
            without = {s: v for s, v in by_sym.items() if s != best_sym}
            kept = np.delete(net, order[0])
            g_w = float(kept[kept > 0].sum())
            g_l = float(-kept[kept < 0].sum())
            # Concentration in R terms (notionals differ wildly between trades).
            rel = net / np.where(notion > 0, notion, np.nan)
            rel = rel[np.isfinite(rel)]
            order_r = np.sort(rel)[::-1]
            entry = {
                "base_sharpe": metrics(pr, tf)["sharpe"],
                "base_dd": metrics(pr, tf)["max_dd"],
                "base_net": tot,
                "best_ticker": best_sym,
                "best_ticker_net": by_sym[best_sym],
                "best_ticker_share": by_sym[best_sym] / tot if tot > 0 else 0.0,
                "net_without_best_ticker": float(sum(without.values())),
                "worst_ticker": min(by_sym, key=by_sym.get),
                "worst_ticker_net": min(by_sym.values()),
                "best_trade_net": float(net[order[0]]),
                "best_trade_share": float(net[order[0]] / tot) if tot > 0 else 0.0,
                "top3_share": float(order_r[:3].sum() / rel.sum()) if len(rel) else 0.0,
                "net_without_best_trade": float(kept.sum()),
                "pf_without_best_trade": g_w / g_l if g_l > 0 else float("inf"),
                "n_trades": len(net),
            }
            print(f"  {name:8s} {tf}: best ticker {best_sym} = {entry['best_ticker_share']:.1%} "
                  f"of net; net w/o it {entry['net_without_best_ticker']:+.2f} (was {tot:+.2f}) | "
                  f"best trade = {entry['best_trade_share']:.1%}; "
                  f"net w/o it {entry['net_without_best_trade']:+.2f} "
                  f"PF {entry['pf_without_best_trade']:.2f} | top3 R-share {entry['top3_share']:.1%} | "
                  f"worst ticker {entry['worst_ticker']} = {entry['worst_ticker_net']:+.2f}")
            adv[name][tf] = entry
    rep["adversarial"] = adv

    # ---------- 6. proof the trailing stop works ---------- #
    print("\n" + "=" * 78)
    print("6. TRAILING STOP PROVENANCE (does the trail actually trail?)")
    print("=" * 78)
    trail: dict = {}
    for tf, pr in runs[FINALIST.name].items():
        tr = [t for r in pr.results.values() for t in r.trades if t.exit_reason == "trail_stop"]
        if not tr:
            print(f"  {tf}: NO trailing-stop exits — the trail is decoration")
            trail[tf] = {"n": 0}
            continue
        adv_r = np.asarray([abs(t.stop_level - t.initial_stop)
                            / max(abs(t.entry_price - t.initial_stop), 1e-12) for t in tr])
        pnl = np.asarray([t.net_pnl for t in tr])
        hold = np.asarray([t.holding_hours for t in tr])
        pre = int(sum(1 for t in tr if t.exit_class == "STOP_LOSS_BEFORE_4H"))
        print(f"  {tf}: {len(tr)} trail exits of {metrics(pr, tf)['n_trades']} trades "
              f"({len(tr)/metrics(pr, tf)['n_trades']:.1%})")
        print(f"     level advanced {np.median(adv_r):.2f}R (median), "
              f"max {adv_r.max():.2f}R | net {pnl.sum():+.2f} "
              f"mean {pnl.mean():+.3f} win {(pnl>0).mean():.0%}")
        print(f"     holding: median {np.median(hold):.0f}h, min {hold.min():.0f}h | "
              f"before-4h exits: {pre}")
        trail[tf] = {"n": len(tr), "median_advance_r": float(np.median(adv_r)),
                     "max_advance_r": float(adv_r.max()), "net": float(pnl.sum()),
                     "mean": float(pnl.mean()), "win": float((pnl > 0).mean()),
                     "median_hold_h": float(np.median(hold)),
                     "min_hold_h": float(hold.min()), "before_4h": pre}
    rep["trail_provenance"] = trail

    # ---------- 7. exit-class breakdown ---------- #
    print("\n" + "=" * 78)
    print("7. EXIT BREAKDOWN AND MINIMUM-HOLDING COMPLIANCE")
    print("=" * 78)
    ex: dict = {}
    for tf, pr in runs[FINALIST.name].items():
        trades = [t for r in pr.results.values() for t in r.trades]
        reasons: dict = {}
        classes: dict = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            classes[t.exit_class] = classes.get(t.exit_class, 0) + 1
        print(f"  {tf}: reasons={reasons}")
        print(f"     classes={classes} (minimum holding: "
              f"{'4h' if tf == '4H' else '24h'})")
        ex[tf] = {"reasons": reasons, "classes": classes}
    rep["exits"] = ex

    # ---------- 8. risk ladder (the stated preference) ---------- #
    print("\n" + "=" * 78)
    print("8. RISK LADDER — the return/drawdown frontier (Sharpe is scale-flat)")
    print("=" * 78)
    ladder: dict = {}
    for rp in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
        v = replace(FINALIST, name=f"r{int(rp*100)}", risk_per_trade=rp)
        per_tf = build(v, contexts, base_costs)
        ladder[f"{rp:.2f}"] = {}
        line = []
        for tf, pr in per_tf.items():
            met = metrics(pr, tf)
            ladder[f"{rp:.2f}"][tf] = {
                "max_dd": met["max_dd"], "max_dd_bars": met["max_dd_bars"],
                "max_dd_days": met["max_dd_days"], "sharpe": met["sharpe"],
                "calmar": met["calmar"], "total_return": met["total_return"],
                "cagr": met["cagr"], "ulcer": met["ulcer"],
                "mean_gross": met["mean_gross"], "max_gross": met["max_gross"],
                "time_underwater": met["time_underwater"],
            }
            line.append(f"{tf}: DD={met['max_dd']:5.2%}({met['max_dd_days']:4.0f}d) "
                        f"Sharpe={met['sharpe']:+.2f} Calmar={met['calmar']:+.2f} "
                        f"CAGR={met['cagr']:+6.2%} gross={met['mean_gross']:.2f}x "
                        f"uw={met['time_underwater']:.0%}")
        print(f"  risk {rp:4.0%}  " + " | ".join(line))
    rep["risk_ladder"] = ladder

    # ---------- 9. capital-intensity frontier + the brake's verdict ---------- #
    # Sourced from the iteration-21 artifact so the two entry points are cross-checked
    # against each other rather than re-derived under a possibly different code path.
    print("\n" + "=" * 78)
    print("9. CAPITAL-INTENSITY FRONTIER (identical rules, leverage as the dial)")
    print("=" * 78)
    frontier: dict = {}
    src = OUT / "dd_stage_i21.json"
    if not src.exists():
        print("  (dd_stage_i21.json absent — run `--iter 21` first)")
    else:
        i21 = json.loads(src.read_text(encoding="utf-8"))

        def g(b, k, dflt=None):
            v = b.get(k)
            return dflt if v is None else v

        order = ["I21_trail_base", "I21_trail_r10", "I21_trail_r20", "I21_trail_r30",
                 "I21_trail_r45", "I21_trail_r30_brake", "I21_nosh4trail_r30"]
        print(f"  {'arm':22s} {'tf':3s} {'maxDD':>8s} {'Sharpe':>7s} {'CAGR':>8s} "
              f"{'ret':>9s} {'gross':>6s} {'d5bars':>7s}")
        for name in order:
            blk = i21.get(name)
            if not blk:
                continue
            frontier[name] = {}
            for tf, b in blk["by_tf"].items():
                if not b.get("n_tickers"):
                    continue
                frontier[name][tf] = {
                    "max_dd": g(b, "max_dd"), "sharpe": b["sharpe"],
                    "cagr": g(b, "cagr"), "total_return": b["total_return"],
                    "mean_gross": g(b, "mean_gross"),
                    "longest_dd_gt_5_bars": g(b, "longest_dd_gt_5_bars"),
                    "n_trail_exits": g(b, "n_trail_exits"),
                    "breadth": g(b, "breadth"),
                }
                print(f"  {name:22s} {tf:3s} {g(b,'max_dd')*100:7.2f}% "
                      f"{b['sharpe']:+7.2f} {g(b,'cagr')*100:+7.2f}% "
                      f"{b['total_return']*100:+8.1f}% {g(b,'mean_gross'):6.2f} "
                      f"{g(b,'longest_dd_gt_5_bars'):7d}")
        # The brake's verdict: only meaningful where a drawdown exists for it to act on.
        a = i21.get("I21_trail_r30", {}).get("by_tf", {}).get("4H")
        b_ = i21.get("I21_trail_r30_brake", {}).get("by_tf", {}).get("4H")
        if a and b_:
            d_dd = (b_["max_dd"] - a["max_dd"]) * 100
            d_sh = b_["sharpe"] - a["sharpe"]
            d_ret = (b_["total_return"] - a["total_return"]) * 100
            verdict = ("earns its complexity" if d_sh > 0 else "does NOT earn its complexity")
            print(f"\n  brake verdict @30% risk, 4H: DD {a['max_dd']:.2%} -> {b_['max_dd']:.2%} "
                  f"({d_dd:+.2f}pp), Sharpe {a['sharpe']:+.2f} -> {b_['sharpe']:+.2f} "
                  f"({d_sh:+.2f}), ret {a['total_return']:+.1%} -> {b_['total_return']:+.1%} "
                  f"({d_ret:+.1f}pp) => {verdict}")
            rep["brake_verdict"] = {"dd_delta_pp": d_dd, "sharpe_delta": d_sh,
                                    "ret_delta_pp": d_ret, "verdict": verdict}
    rep["frontier"] = frontier

    (OUT / "dd_verify.json").write_text(json.dumps(rep, indent=2, default=str),
                                        encoding="utf-8")
    print(f"\nwrote dd_verify.json | elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
