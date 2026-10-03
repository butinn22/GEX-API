"""Drawdown-first iteration loop.

Objective, in priority order (note the inversion from the previous run):

1. smallest **max portfolio drawdown**
2. shortest **prolonged-drawdown exposure** (bars underwater, bars with DD > 10%)
3. highest **Sharpe**
4. highest **net PnL**

...subject to a hard requirement that the trailing stop is **present and actually firing**.
An earlier run established that trade-management stops cost PnL on this strategy; that
finding stands and is not being relitigated. What that run could not do is reduce portfolio
drawdown, because it had no portfolio-level control at all. This loop adds exactly that:
constant-risk sizing and an equity-curve drawdown brake.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.emf_adl import data as D  # noqa: E402
from research.emf_adl import loop as L  # noqa: E402
from research.emf_adl import portfolio as P  # noqa: E402
from research.emf_adl import rules as R  # noqa: E402
from research.emf_adl import run_loops as RL  # noqa: E402
from research.emf_adl.engine import Costs, Sizing, StopSpec  # noqa: E402

log = logging.getLogger("dd")
OUT = RL.OUT
TIMEFRAMES = ("4H", "1D")
#: Bars in a 24/7 crypto year. Used to convert an annualised vol target into a per-bar one.
_BARS_PER_YEAR = {"4H": 6 * 365, "1D": 365}


@dataclass(frozen=True)
class DdVariant:
    """One iteration of the drawdown-first loop."""

    name: str
    base: str = "V43_mkt_long_atr8"
    #: sizing
    risk_per_trade: float = 0.0
    max_leverage: float = 1.0
    #: per-trade stop override
    stop_mode: str | None = None
    stop_atr_mult: float = 3.0
    #: Trailing distance in ATR, separate from the initial width, so "add a trailing stop"
    #: does not silently also narrow the initial catastrophe stop.
    trail_atr_mult: float = 0.0
    trail_activate_r: float = 0.0
    breakeven_at_r: float = 0.0
    #: portfolio brake
    brake_dd_on: float = 0.0
    brake_dd_full: float = 0.25
    brake_floor: float = 1.0
    brake_dd_off: float = 0.0
    #: Exit an open position when the *market* regime flips against it, rather than only
    #: refusing new entries. Entries are already gated on BTC being above its own EMA, so
    #: without this a position opened in an uptrend is held through an entire market turn —
    #: which is where the 494-day underwater stretch comes from.
    regime_exit: bool = False
    #: Quote currency rule: for a long book, exit when the coin closes below its own EMA.
    #: A per-trade trend exit, cheaper than the market gate and applied only to open risk.
    own_ema_exit: bool = False
    own_ema_span: int = 100
    long_only: bool = True
    #: Hard switch for the trailing behaviour. See ``StopSpec.trail``: expressing "no
    #: trail" via ``trail_atr_mult=0`` silently produced an *immediate* full-width trail,
    #: which is the opposite of the intent. This flag is what the "no trail" arms use.
    trail: bool = True
    #: Per-symbol volatility normalisation. ``target_vol`` is an *annualised* volatility
    #: target; positions are scaled so each symbol contributes comparable risk. ``None``
    #: disables it. This is the diversification lever, and it is independent of every stop
    #: decision, so it can be measured on its own.
    target_vol: float | None = None
    vol_window: int = 100
    vol_lo: float = 0.25
    vol_hi: float = 4.0
    #: Per-timeframe parameter overrides. The measurements above show the trailing stop
    #: PAYS on 1D (Sharpe 0.69 -> 0.86, DD -31%) and COSTS on 4H (Sharpe 1.01 -> 0.85),
    #: replicated at two risk levels. Since the timeframe is known at deployment, the
    #: honest finalist carries the setting that works on each, and both are reported.
    per_tf: dict = field(default_factory=dict)
    hypothesis: str = ""
    change: str = ""


def _signals_for(ctx: RL.SeriesContext, base, v: "DdVariant | None" = None):
    """Signal arrays exactly as ``RL.evaluate`` builds them, minus the engine call."""
    ss = ctx.signals[base.repair_hybrid]
    el = ss.entry_long.copy()
    es = ss.entry_short.copy()
    if base.long_only:
        es = np.zeros_like(es)
    if base.short_only:
        el = np.zeros_like(el)
    el, es = RL._apply_gates(ctx, base, el, es)
    xl = np.asarray(ss.exit_long, dtype=bool).copy()
    xs = np.asarray(ss.exit_short, dtype=bool).copy()
    if v is not None and (v.regime_exit or v.own_ema_exit):
        n = len(el)
        if v.regime_exit:
            mr = ctx.market_regime
            if mr is not None:
                # Regime -1 == market trend down. Exit longs, exit shorts. The regime is
                # read from BTC's own closed bars, so this is causal.
                xl |= np.asarray(mr, dtype=np.int8) < 0
                xs |= np.asarray(mr, dtype=np.int8) > 0
        if v.own_ema_exit:
            close = ctx.bars["close"].to_numpy(float)
            ema = pd.Series(close).ewm(span=v.own_ema_span, adjust=False).mean().to_numpy()
            xl |= close < ema
            xs |= close > ema
        xl = xl[:n]
        xs = xs[:n]
    return replace(ss, entry_long=el, entry_short=es, exit_long=xl, exit_short=xs), None


def _stops_for(ctx: RL.SeriesContext, base, v: DdVariant) -> StopSpec:
    s = RL._stops_for(ctx, base)
    if v.stop_mode is None:
        return replace(s, trail_activate_r=v.trail_activate_r,
                       breakeven_at_r=v.breakeven_at_r, trail=v.trail)
    return StopSpec(
        mode=v.stop_mode,
        atr_mult=v.stop_atr_mult,
        trail_atr_mult=v.trail_atr_mult,
        trail=v.trail,
        tp_mode=s.tp_mode, tp_atr_mult=s.tp_atr_mult, tp_pct=s.tp_pct,
        breakeven_at_r=v.breakeven_at_r,
        trail_activate_r=v.trail_activate_r,
    )


def _shift_windows(ctx: RL.SeriesContext, window: str):
    """Entry mask restricted to one walk-forward window, as a boolean array."""
    sl = ctx.wins[window]
    n = len(ctx.bars)
    m = np.zeros(n, dtype=bool)
    m[sl] = True
    return m


def score(stats: dict, met: dict, n_bars: int) -> float:
    """Scalar ranking key for the drawdown-first objective.

    Priority, in order: drawdown depth, drawdown *persistence* (ulcer index and the longest
    stretch deeper than 10%), then Sharpe, then return. ``time_underwater`` is deliberately
    NOT used — every variant sits underwater ~95% of the time, so it carries no information
    and would just add a constant. The coefficients are a stated preference, not a fitted
    quantity.
    """
    deep_frac = float(stats.get("longest_dd_gt_10_bars", 0)) / max(n_bars, 1)
    return (
        -8.0 * float(stats.get("max_dd", 0.0))
        - 2.0 * float(stats.get("ulcer", 0.0))
        - 1.0 * deep_frac
        - 0.5 * float(stats.get("worst_year_dd", 0.0))
        + 1.0 * float(met.get("sharpe", 0.0))
        + 0.5 * float(met.get("total_return", 0.0))
    )


def run_variant(contexts, v: DdVariant, costs: Costs, *, window: str = "all",
                cat=None) -> dict:
    cat = cat or {x.name: x for x in RL.catalogue()}
    base = cat[v.base]
    out: dict = {"variant": asdict(v), "hypothesis": v.hypothesis,
                 "change": v.change, "by_tf": {}}
    for tf in TIMEFRAMES:
        # Apply this timeframe's overrides, if any, so one variant can carry the best 4H
        # setting and the best 1D setting side by side.
        eff = replace(v, per_tf={}, **v.per_tf.get(tf, {})) if v.per_tf.get(tf) else v
        sub = {k: c for k, c in contexts.items() if k[1] == tf}
        brake = P.BrakeSpec(dd_on=eff.brake_dd_on, dd_full=eff.brake_dd_full,
                            floor=eff.brake_floor, dd_off=eff.brake_dd_off)
        sizing = Sizing(risk_per_trade=eff.risk_per_trade, max_leverage=eff.max_leverage)
        wins = None if window == "all" else (lambda c, w=window: _shift_windows(c, w))
        vol_for = (None if eff.target_vol is None
                   else (lambda c, vv=eff, t=tf: P.vol_target_scale(
                       c.bars,
                       target_bar_vol=vv.target_vol / np.sqrt(_BARS_PER_YEAR[t]),
                       window=vv.vol_window, lo=vv.vol_lo, hi=vv.vol_hi)))
        pr = P.run_portfolio(
            sub,
            signals_for=lambda c, b=base, vv=eff: _signals_for(c, b, vv),
            stops_for=lambda c, b=base, vv=eff: _stops_for(c, b, vv),
            sizing=sizing, costs=costs, brake=brake, windows=wins, vol_for=vol_for,
        )
        if not len(pr.equity):
            out["by_tf"][tf] = {"n_tickers": 0}
            continue
        tf_hours = D.TIMEFRAMES[tf][1]
        all_trades = [t for r in pr.results.values() for t in r.trades]
        # Timestamps must be epoch-ms: metrics casts them as datetime64[ms].
        met = L.compute_metrics(pr.equity, all_trades, tf_hours,
                                timestamps=pr.timestamps).as_dict()
        per = {s: L.compute_metrics(r.equity, r.trades, tf_hours)
               for s, r in pr.results.items()}
        rets = np.asarray([m.total_return for m in per.values()])
        reasons: dict[str, int] = {}
        for t in all_trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        out["by_tf"][tf] = {
            **met,
            **pr.stats,
            "n_tickers": len(pr.results),
            "total_trades": len(all_trades),
            "breadth": float((rets > 0).mean()) if len(rets) else 0.0,
            "median_ticker_sharpe": float(np.median([m.sharpe for m in per.values()]))
            if per else 0.0,
            "exit_reasons": reasons,
            "n_stop_exits": int(reasons.get("stop", 0)),
            "n_trail_exits": int(reasons.get("trail_stop", 0)),
            # Gross book as a multiple of equity. Risk sizing caps risk per *trade*, not
            # the aggregate, so this is the honest leverage readout and it must be small.
            "mean_gross": float(np.nanmean(pr.gross_exposure)) if len(pr.gross_exposure) else 0.0,
            "p95_gross": float(np.nanpercentile(pr.gross_exposure, 95)) if len(pr.gross_exposure) else 0.0,
            "max_gross": float(np.nanmax(pr.gross_exposure)) if len(pr.gross_exposure) else 0.0,
            "mean_live": float(np.mean(pr.live)) if len(pr.live) else 0.0,
            "max_live": int(np.max(pr.live)) if len(pr.live) else 0,
            "n_tp_exits": int(reasons.get("take_profit", 0)),
            "n_indicator_exits": int(reasons.get("indicator", 0)),
            "n_eod_exits": int(reasons.get("eod", 0)),
            "n_pre_stop": int(sum(1 for t in all_trades
                                  if t.exit_class == "STOP_LOSS_BEFORE_4H")),
            "n_pre_tp": int(sum(1 for t in all_trades
                                if t.exit_class == "TAKE_PROFIT_BEFORE_4H")),
            # Ledger proof that the trailing stop trailed: how far the level had moved.
            "mean_stop_advance_r": float(np.mean([
                abs(t.stop_level - t.initial_stop) / max(abs(t.entry_price - t.initial_stop), 1e-12)
                for t in all_trades
                if t.exit_reason == "trail_stop" and t.initial_stop > 0
            ])) if any(t.exit_reason == "trail_stop" for t in all_trades) else 0.0,
            "converged": bool(pr.converged),
            "mean_scale": float(np.mean(pr.scale)) if len(pr.scale) else 1.0,
            "min_scale": float(np.min(pr.scale)) if len(pr.scale) else 1.0,
            "brake_passes": pr.convergence,
            "residual": pr.convergence[-1]["residual"] if pr.convergence else 0.0,
            "net_pnl": float(sum(t.net_pnl for t in all_trades)),
            "total_fees": float(sum(t.fees for t in all_trades)),
            "total_funding": float(sum(t.funding_cost for t in all_trades)),
        }
        out["by_tf"][tf]["score"] = score(pr.stats, met, len(pr.equity))
    return out


def table(results: dict, title: str) -> str:
    lines = [
        f"=== {title} ===",
        f"{'variant':30s} {'TF':3s} {'maxDD':>7s} {'ulcer':>6s} {'wyDD':>6s} "
        f"{'d5bars':>7s} {'d5days':>7s} {'d10days':>8s} {'Sharpe':>6s} {'ret':>8s} "
        f"{'trd':>5s} {'stop':>5s} {'trl':>5s} {'grs':>5s} {'brd':>4s}",
    ]
    lines.append("-" * 138)
    for name, blk in results.items():
        for tf in TIMEFRAMES:
            b = blk["by_tf"].get(tf) or {}
            if not b.get("n_tickers"):
                lines.append(f"{name:30s} {tf:3s}   (no data)")
                continue
            lines.append(
                f"{name:30s} {tf:3s} {b['max_dd']:7.2%} {b['ulcer']:6.3f} "
                f"{b['worst_year_dd']:6.1%} {b.get('longest_dd_gt_5_bars', 0):7d} "
                f"{b.get('longest_dd_gt_5_days', 0.0):7.0f} "
                f"{b.get('longest_dd_gt_10_days', 0.0):8.0f} "
                f"{b['sharpe']:+6.2f} {b['total_return']:+8.1%} {b['total_trades']:5d} "
                f"{b['n_stop_exits']:5d} {b['n_trail_exits']:5d} "
                f"{b.get('mean_gross', 0.0):5.2f} {b['breadth']:4.0%}"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", default="1,2", help="comma list of iteration groups")
    ap.add_argument("--windows", default="all")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    uni = RL.load_universe(RL.OUT / "universe.json")
    pool = uni["symbols"]
    start, end = D.ms(*RL.STUDY_START), D.ms(*RL.STUDY_END)
    log.info("building contexts for %d symbols x %s ...", len(pool), TIMEFRAMES)
    contexts, exclusions = RL.build_contexts(pool, start, end)
    log.info("contexts: %d, exclusions: %d", len(contexts), len(exclusions))
    for e in exclusions:
        log.info("  EXCLUDED %s %s: %s", e.get("symbol"), e.get("timeframe"), e.get("reason"))

    cat = {x.name: x for x in RL.catalogue()}
    costs = Costs()
    windows = [w for w in args.windows.split(",") if w]
    groups = {g for g in args.iter.split(",") if g}

    results: dict[str, dict] = {}

    def add(vs, label):
        for v in vs:
            for w in windows:
                key = v.name if len(windows) == 1 else f"{v.name}@{w}"
                results[key] = run_variant(contexts, v, costs, window=w, cat=cat)
                b45 = results[key]["by_tf"].get("4H", {})
                log.info("  %-28s %-8s 4H DD=%s Sharpe=%s | 1D DD=%s Sharpe=%s",
                         v.name, w,
                         f"{b45.get('max_dd', float('nan')):.1%}",
                         f"{b45.get('sharpe', float('nan')):+.2f}",
                         f"{results[key]['by_tf'].get('1D', {}).get('max_dd', float('nan')):.1%}",
                         f"{results[key]['by_tf'].get('1D', {}).get('sharpe', float('nan')):+.2f}")

    # ---------------- iteration 0: the incumbent control ---------------- #
    if "0" in groups:
        add([DdVariant(
            name="I0_incumbent", base="V43_mkt_long_atr8",
            risk_per_trade=0.0, brake_dd_on=0.0, brake_floor=1.0,
            stop_mode=None,
            hypothesis="The previous programme's finalist, unchanged: flat 100% notional, "
                       "no portfolio control, 8xATR catastrophe stop only.",
            change="none - this is the bar to beat on drawdown.")], "I0")

    # ---------------- iteration 1: constant-risk sizing ---------------- #
    if "1" in groups:
        add([DdVariant(
            name=f"I1_risk{r*100:.0f}", base="V43_mkt_long_atr8",
            risk_per_trade=r, max_leverage=1.0,
            hypothesis="Risk-per-trade sizing normalises the risk actually taken, so "
                       "portfolio drawdown stops being a function of which coins happened "
                       "to be volatile.",
            change=f"size so entry-to-stop distance = {r:%} of equity; cap notional at 1x")
            for r in (0.01, 0.02, 0.03, 0.05)], "I1")

    # ---------------- iteration 2: a trailing stop that fires ---------------- #
    if "2" in groups:
        add([DdVariant(
            name=f"I2_trail{m}_act{a}", base="V43_mkt_long_atr8",
            risk_per_trade=0.02, stop_mode="atr_trail", stop_atr_mult=m,
            trail_activate_r=a, breakeven_at_r=1.0,
            hypothesis="A trailing stop only works if it is inside the noise. Activating it "
                       "after the trade has earned a profit keeps the entry from being "
                       "shaken out while still capping giveback.",
            change=f"atr_trail {m}xATR, trail engages at {a}R, break-even floor at 1R")
            for m, a in ((2.0, 0.5), (2.5, 0.5), (3.0, 0.5), (2.5, 1.0), (3.0, 1.0))], "I2")

    # ---------------- iteration 3: equity-curve brake ---------------- #
    if "3" in groups:
        add([DdVariant(
            name=f"I3_brake{int(on*100)}_{int(fl*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=0.02, brake_dd_on=on, brake_dd_full=full,
            brake_floor=fl, brake_dd_off=on / 2,
            hypothesis="Equity-curve trading: cut total exposure as the portfolio sinks "
                       "below its own peak, restore it as it recovers. This is the only "
                       "lever that acts on the portfolio rather than the single trade.",
            change=f"brake on at {on:.0%} panel DD, to {fl:.0%} scale at {full:.0%}, "
                   f"releases at {on/2:.0%}")
            for on, fl, full in ((0.10, 0.25, 0.25), (0.15, 0.25, 0.30),
                                 (0.20, 0.25, 0.35), (0.15, 0.0, 0.30),
                                 (0.15, 0.50, 0.30))], "I3")

    # ---------------- iteration 4: combine the winners ---------------- #
    if "4" in groups:
        add([DdVariant(
            name=f"I4_c{int(on*100)}_{int(fl*100)}_t{str(m).replace('.','')}",
            base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=m,
            trail_activate_r=0.5, breakeven_at_r=1.0,
            brake_dd_on=on, brake_dd_full=full, brake_floor=fl, brake_dd_off=on / 2,
            hypothesis="All three drawdown levers together.",
            change=f"risk {rp:.0%}, atr_trail {m}x/{0.5}R/be 1R, brake {on:.0%}->{fl:.0%}")
            for rp, m, on, fl, full in ((0.02, 3.0, 0.15, 0.25, 0.30),
                                        (0.03, 3.0, 0.15, 0.25, 0.30),
                                        (0.02, 2.5, 0.20, 0.25, 0.35),
                                        (0.03, 2.5, 0.15, 0.25, 0.30))], "I4")

    # ---------------- iteration 5: trailing stop, wide base retained ---------------- #
    # Iteration 2 was confounded: setting atr_trail to 3xATR also replaced the incumbent's
    # 8xATR catastrophe stop. Here the base stays at 8xATR and only the *trail* is tight,
    # which is the only way to tell whether adding a trail helps or hurts.
    if "5" in groups:
        add([DdVariant(
            name=f"I5_base8_trail{tm}_act{a}", base="V43_mkt_long_atr8",
            risk_per_trade=0.02, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=tm, trail_activate_r=a, breakeven_at_r=0.0,
            hypothesis="Keep the incumbent's 8xATR catastrophe stop as the initial level, "
                       "then trail at a tighter multiple once the trade has earned it. "
                       "This isolates the trailing stop from the initial-stop width.",
            change=f"initial 8xATR, trail {tm}xATR after +{a}R")
            for tm, a in ((1.5, 1.0), (2.0, 1.0), (3.0, 1.0), (2.0, 2.0), (3.0, 2.0))], "I5")

    # ---------------- iteration 6: does the brake earn its keep? ---------------- #
    # The brake is inert at 2% risk because portfolio DD never reaches the threshold.
    # Two questions, both answered at matched risk: does it help at a risk level where it
    # can bind, and is brake+high-risk better than no-brake+low-risk at the SAME drawdown?
    if "6" in groups:
        add([DdVariant(
            name=f"I6_r{int(rp*100)}_brake{int(on*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, brake_dd_on=on, brake_dd_full=on * 2,
            brake_floor=0.25, brake_dd_off=on * 0.6,
            hypothesis="At a risk level where the panel can actually draw down past the "
                       "threshold, does cutting exposure on the way down improve the "
                       "risk-adjusted result versus simply sizing smaller?",
            change=f"risk {rp:.0%} with brake engaging at {on:.0%} panel DD")
            for rp, on in ((0.05, 0.04), (0.05, 0.06), (0.08, 0.06), (0.08, 0.10),
                           (0.10, 0.08))], "I6")
        add([DdVariant(
            name=f"I6b_r{int(rp*100)}_nobrake", base="V43_mkt_long_atr8",
            risk_per_trade=rp,
            hypothesis="Control arm for the matched-drawdown comparison: same risk, no "
                       "brake. If brake+high-risk does not beat this at equal DD, the brake "
                       "is complexity without payoff and should be dropped.",
            change=f"risk {rp:.0%}, no brake")
            for rp in (0.05, 0.08, 0.10)], "I6")

    # ---------------- iteration 7: raise Sharpe at matched risk ---------------- #
    # Sizing is scale-invariant to Sharpe, so the only remaining lever on Sharpe is what
    # gets traded. These re-test the entry structure at a FIXED 2% risk so the comparison
    # is Sharpe-for-Sharpe rather than return-for-return.
    if "7" in groups:
        add([DdVariant(
            name=f"I7_{nm}", base=bn, risk_per_trade=0.02,
            hypothesis=hyp, change=f"entry structure {bn} at fixed 2% risk",
        ) for nm, bn, hyp in (
            ("repaired", "V1_repaired",
             "The repaired EMF+ADL entries with no gate at all, as a Sharpe baseline."),
            ("mkt", "V43_mkt_long_atr8", "The incumbent: market-regime gate only."),
            ("struct", "V32_lo_struct_nostop",
             "Structural gate with no stop — isolates the structural filter."),
            ("mkt_struct", "V40_lo_struct_mkt",
             "Market gate plus structural gate, no stop."),
            ("volq", "V20_long_sl_volq",
             "Structural gate plus a volatility-quantile gate and the structural stop."),
        )], "I7")

    # ---------------- iteration 8: recover the Sharpe the trail cost ---------------- #
    # Iteration 5 established the shape: with the incumbent's 8xATR initial stop retained,
    # a trailing stop CUTS 4H drawdown (4.33% -> 3.49%) but COSTS Sharpe (1.01 -> 0.82),
    # while on 1D it improves both (0.68 -> 0.83 Sharpe, 2.86% -> 1.73% DD). This sweep
    # asks whether a wider trail, or a later activation, keeps the drawdown benefit and
    # gives the 4H Sharpe back. Pre-registered grid, no cherry-picking.
    if "8" in groups:
        add([DdVariant(
            name=f"I8_trail{tm}_act{a}", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=tm, trail_activate_r=a, breakeven_at_r=0.0,
            hypothesis="A trailing stop needs to sit outside the noise of the timeframe it "
                       "runs on. Widening the trail or delaying activation should keep the "
                       "drawdown reduction while letting 4H winners run.",
            change=f"initial 8xATR; trail {tm}xATR once +{a}R")
            for tm in (3.0, 4.0, 6.0, 8.0) for a in (2.0, 3.0, 4.0)], "I8")

    # ---------------- iteration 9: risk ladder at a pre-registered trail ---------------- #
    # Sharpe was flat (1.00-1.01) across the 1%-5% risk range, so the risk level is a stated
    # preference against a drawdown budget, not an optimum to be searched. These ladders are
    # set BEFORE looking, and deliberately use the MID trail (4x/2R) rather than whichever
    # cell of I8 happened to score highest.
    if "9" in groups:
        add([DdVariant(
            name=f"I9_r{int(rp*100)}_trail{tm}_{int(a)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=tm, trail_activate_r=a, breakeven_at_r=0.0,
            hypothesis="Same trail, rising risk budget: maps the return-vs-drawdown frontier "
                       "so the risk level can be chosen against a drawdown mandate rather "
                       "than picked by maximising a score.",
            change=f"risk {rp:.0%} with the mid-grade trail {tm}xATR after +{a}R")
            for rp in (0.03, 0.05, 0.07, 0.10) for tm, a in ((4.0, 2.0), (8.0, 2.0))], "I9")
        add([DdVariant(
            name=f"I9b_r{int(rp*100)}_notrail", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode=None,
            hypothesis="Control arm at matched risk: the incumbent stop with NO trail at all. "
                       "If a ladder point does not beat this on the drawdown-first objective, "
                       "the trail is not earning its place.",
            change=f"risk {rp:.0%}, no trailing stop")
            for rp in (0.03, 0.05, 0.07, 0.10)], "I9")

    # ---------------- iteration 10: cut the prolonged underwater stretch ---------------- #
    # Every variant so far shares the same 494-day (4H) longest stretch below a prior peak.
    # Sizing and stops cannot touch it because it is structural: entries are gated on the
    # market regime, but an OPEN position is held straight through a market turn. These add
    # a regime exit (and a per-coin trend exit) that closes risk when the trend breaks.
    if "10" in groups:
        add([DdVariant(
            name=f"I10_rexit_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=6.0, trail_activate_r=3.0, regime_exit=True,
            hypothesis="Close long risk when BTC's own trend flips down, instead of merely "
                       "refusing new entries. This is the only lever that can shorten the "
                       "prolonged underwater stretch.",
            change=f"risk {rp:.0%}, 6xATR trail after +3R, market-regime exit")
            for rp in (0.03, 0.05)], "I10")
        add([DdVariant(
            name=f"I10_ownema{sp}_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=6.0, trail_activate_r=3.0, own_ema_exit=True,
            own_ema_span=sp,
            hypothesis="Per-coin trend exit: close when the coin itself loses its own EMA. "
                       "Finer-grained than the market regime and applied only to open risk.",
            change=f"risk {rp:.0%}, trail 6x/+3R, exit on close < own EMA{sp}")
            for sp in (50, 100, 200) for rp in (0.03,)], "I10")
        add([DdVariant(
            name=f"I10_both_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=6.0, trail_activate_r=3.0,
            regime_exit=True, own_ema_exit=True, own_ema_span=100,
            hypothesis="Both exits together: market regime plus per-coin trend.",
            change=f"risk {rp:.0%}, trail 6x/+3R, regime exit + own EMA100 exit")
            for rp in (0.03, 0.05)], "I10")
        add([DdVariant(
            name=f"I10_control_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=6.0, trail_activate_r=3.0,
            hypothesis="Control: identical stop and trail, no regime exit. Everything the "
                       "next row gains must be attributable to the exit, not the trail.",
            change=f"risk {rp:.0%}, trail 6x/+3R, no regime exit")
            for rp in (0.03, 0.05)], "I10")

    # ---------------- iteration 11: cheapest possible trail on 4H ---------------- #
    # Iteration 10 showed the market-regime exit is a null result (uwMax 2968 -> 2967: the
    # prolonged stretch is not caused by holding through a turn). Iteration 8 showed the
    # trail costs 4H Sharpe, and that delaying activation recovers most of it (act 2R -> 0.82,
    # act 4R -> 0.90). This pushes activation later and the trail wider to find the point
    # where the trail is a genuine profit-lock that costs the least, and reports the trail
    # exit count so "is it actually working?" is answered from the ledger.
    if "11" in groups:
        add([DdVariant(
            name=f"I11_trail{tm}_act{a}", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=tm, trail_activate_r=a, breakeven_at_r=0.0,
            hypothesis="A trailing stop placed very late and very wide behaves as a "
                       "profit-lock rather than a trend exit, so it should cut the tail "
                       "without truncating 4H winners.",
            change=f"initial 8xATR; trail {tm}xATR once +{a}R")
            for tm in (8.0, 12.0) for a in (4.0, 6.0)], "I11")
        add([DdVariant(
            name=f"I11_be{a}_notrail", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode=None, breakeven_at_r=a,
            hypothesis="Break-even floor alone, no trail: the cheapest form of profit "
                       "protection — once the trade has earned aR, it cannot become a loser.",
            change=f"risk 3%, no trail, stop to break-even at +{a}R")
            for a in (1.0, 2.0)], "I11")
        add([DdVariant(
            name="I11_control", base="V43_mkt_long_atr8", risk_per_trade=0.03,
            stop_mode=None,
            hypothesis="Control: incumbent stop, no trail, no break-even.",
            change="risk 3%, incumbent stop only")], "I11")

    # ---------------- iteration 12: the finalist and its neighbours ---------------- #
    # The measurements support one honest deployment: because the timeframe is known in
    # advance, carry the stop structure that works on each. 4H gets a very wide, very late
    # trail plus a break-even floor (the trail there is profit protection, not a trend
    # exit); 1D gets a tighter trail plus a per-coin trend exit, both of which PAY on 1D.
    # Neighbours are included so the sensitivity of the choice is measured, not asserted.
    if "12" in groups:
        add([DdVariant(
            name=f"I12_finalist_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            hypothesis="Per-timeframe stop structure: a wide late trail on 4H (where any "
                       "fast exit is whipsawed) and a tighter trail plus coin-trend exit on "
                       "1D (where both measurably pay).",
            change="4H: risk r, 8xATR stop, 12xATR trail at +6R, BE at +2R. "
                   "1D: 8xATR stop, 6xATR trail at +3R, BE at +2R, exit on close < EMA50.")
            for rp in (0.03, 0.05)], "I12")
        add([DdVariant(
            name=f"I12_uniform_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=6.0, trail_activate_r=3.0, breakeven_at_r=2.0,
            hypothesis="Simpler alternative: one identical stop structure on both "
                       "timeframes. Costs some 4H Sharpe but has fewer moving parts.",
            change=f"both TFs: risk {rp:.0%}, 8xATR stop, 6xATR trail at +3R, BE at +2R")
            for rp in (0.03, 0.05)], "I12")
        add([DdVariant(
            name=f"I12_nostop_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode=None, breakeven_at_r=2.0,
            hypothesis="Lower bound on the contribution of the trailing stop: the incumbent "
                       "catastrophe stop plus a break-even floor, no trail at all.",
            change=f"risk {rp:.0%}; 8xATR stop and BE at +2R only, no trail")
            for rp in (0.03, 0.05)], "I12")
        add([DdVariant(
            name=f"I12_finalist_wide_r{int(rp*100)}", base="V43_mkt_long_atr8",
            risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=18.0, trail_activate_r=8.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 8.0, "trail_activate_r": 4.0,
                           "own_ema_exit": True, "own_ema_span": 100,
                           "breakeven_at_r": 2.0}},
            hypothesis="Sensitivity neighbour: the same structure with every stop parameter "
                       "shifted one notch looser. If the result collapses, the choice is "
                       "fragile; if it holds, the structure is doing the work.",
            change="4H: trail 18x at +8R. 1D: trail 8x at +4R with EMA100 exit.")
            for rp in (0.03,)], "I12")

    # ---------------- iteration 13: volatility targeting ---------------- #
    # Risk-based sizing already normalises each trade by the entry-to-stop distance, and the
    # stop is 8xATR — so the risk budget is *already* scaled by a volatility estimate. A
    # second, realised-vol normalisation therefore measures largely the same quantity. The
    # honest question is whether it adds anything the ATR term misses, so it is tested
    # against the identical structure with vol targeting switched off, at matched risk.
    if "13" in groups:
        add([DdVariant(
            name=f"I13_vol{int(tv*100)}_w{w}", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            target_vol=tv, vol_window=w, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Scale each symbol to a common annualised volatility target so the "
                       "portfolio's risk is not set by whichever name happens to be "
                       "noisiest. If Sharpe does not improve, the ATR term was already "
                       "doing this job and the extra machinery is redundant.",
            change=f"risk 3%, finalist stops, realised-vol normalisation to {tv:.0%}/yr "
                   f"over a {w}-bar window")
            for tv, w in ((0.25, 100), (0.40, 100), (0.25, 300))], "I13")
        add([DdVariant(
            name="I13_novol_control", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            hypothesis="Control for iteration 13: the identical finalist with no volatility "
                       "targeting. Any gain in the rows above must be measured against this.",
            change="risk 3%, finalist stops, no vol normalisation")], "I13")

    # ---------------- iteration 14: does the 4H trail earn its place? ---------------- #
    # Iteration 12's per-timeframe finalist carries a trail on 4H that buys 0.37pp of
    # drawdown for 0.10 of Sharpe. Both arms are measured here at the two risk levels, so
    # the 4H stop structure can be chosen on the stated preference (drawdown first) with the
    # cost of that preference stated in Sharpe, rather than hidden.
    if "14" in groups:
        for rp in (0.03, 0.05):
            add([DdVariant(
                name=f"I14_4Htrail_r{int(rp*100)}", base="V43_mkt_long_atr8",
                risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
                trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
                per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                               "own_ema_exit": True, "own_ema_span": 50,
                               "breakeven_at_r": 2.0}},
                hypothesis="Arm A: a late, wide trail on 4H as well as 1D. Drawdown-first, at "
                           "a small cost in 4H Sharpe.",
                change=f"risk {rp:.0%}; 4H trail 12xATR at +6R; 1D trail 6xATR at +3R + EMA50 "
                       "exit"), DdVariant(
                name=f"I14_4Hnotrail_r{int(rp*100)}", base="V43_mkt_long_atr8",
                risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
                trail=False, breakeven_at_r=2.0,
                per_tf={"4H": {"trail": False, "breakeven_at_r": 2.0},
                        "1D": {"trail": True, "trail_atr_mult": 6.0,
                               "trail_activate_r": 3.0,
                               "own_ema_exit": True, "own_ema_span": 50,
                               "breakeven_at_r": 2.0}},
                hypothesis="Arm B: the 4H leg keeps the 8xATR catastrophe stop and a +2R "
                           "break-even floor but drops the trail, because on 4H every exit "
                           "rule tested so far cost more Sharpe than drawdown it saved.",
                change=f"risk {rp:.0%}; 4H 8xATR fixed stop + BE at +2R, no trail; "
                       "1D trail 6xATR at +3R + EMA50 exit")], "I14")

    # ---------------- iteration 15: vol-target frontier, and the combination ------- #
    # Iteration 13 found a large joint improvement (4H DD 5.99% -> 2.41%, Sharpe 0.91 ->
    # 1.13) but the 0.25 and 0.40 targets gave nearly identical Sharpe at very different
    # return. Before adopting a target, map the frontier so the choice is read off a curve
    # rather than taken from whichever point happened to be tested first. All rows carry the
    # same finalist stop structure, so the target is the only thing moving.
    if "15" in groups:
        add([DdVariant(
            name=f"I15_vol{int(tv*100)}_finalist", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            target_vol=tv, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Map the risk/return frontier of the volatility target holding the "
                       "stop structure fixed. If Sharpe is flat across a wide band of "
                       "targets, the level is a preference not a tuned parameter.",
            change=f"finalist stops; realised-vol target {tv:.0%}/yr, 100-bar window")
            for tv in (0.30, 0.50, 0.60)], "I15")

    # ---------------- iteration 16: the combination, and the 4H trail at final risk ----- #
    # Iteration 13/15 showed vol targeting is the big win and that Sharpe is flat across the
    # target, so the level is a preference rather than a tuned value. Iteration 14 (after the
    # ``trail`` switch was fixed) showed the 4H trail is close to a wash. This block asks the
    # only combination question that remains open: at the chosen risk level, does the 4H trail
    # still earn its place, and does the vol target survive when the trail is removed?
    if "16" in groups:
        add([DdVariant(
            name=f"I16_vol{int(tv*100)}_trail", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            target_vol=tv, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Combination: vol target + per-timeframe stop structure, 4H trail on. "
                       "This is the proposed finalist configuration.",
            change=f"vol {tv:.0%}/yr; 4H trail 12xATR at +6R; 1D trail 6xATR at +3R + EMA50")
            for tv in (0.30, 0.50)], "I16")
        add([DdVariant(
            name=f"I16_vol{int(tv*100)}_no4Htrail", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail=False, breakeven_at_r=2.0,
            per_tf={"4H": {"trail": False, "breakeven_at_r": 2.0},
                    "1D": {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            target_vol=tv, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Does the vol target still pay without the 4H trail? If yes, the two "
                       "effects are independent; if no, one is standing in for the other.",
            change=f"vol {tv:.0%}/yr; 4H fixed 8xATR + BE at +2R, no trail; 1D as finalist")
            for tv in (0.30, 0.50)], "I16")
        add([DdVariant(
            name="I16_novol_finalist", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail_atr_mult=12.0, trail_activate_r=6.0, breakeven_at_r=2.0,
            per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            hypothesis="Matched control with no vol targeting, so the vol gain at the "
                       "finalist stop structure is measured against its own baseline rather "
                       "than borrowed from an earlier iteration.",
            change="finalist stops, no vol normalisation")], "I16")

    # ---------------- iteration 17: the de-gearing control ---------------- #
    # The incumbent as specified sized every position at 100% of equity with no cap on the
    # number of concurrent positions, which on this universe means ~4.7x gross leverage. A
    # 31% drawdown on a 4.7x book is not comparable to a 5% drawdown on a 1x book, so the
    # headline improvement has to be split into "less leverage" and "better strategy" before
    # it means anything. This arm runs the ORIGINAL sizing through the SAME metric code.
    if "17" in groups:
        add([DdVariant(
            name="I17_flat_incumbent", base="V43_mkt_long_atr8",
            risk_per_trade=0.0, max_leverage=1.0, stop_mode=None,
            hypothesis="The incumbent as originally specified: 100% of equity per position, "
                       "no portfolio cap. Measured through the identical aggregation so the "
                       "leverage change can be separated from the strategy change.",
            change="flat 100%-notional sizing (incumbent), risk-based sizing off")], "I17")
        add([DdVariant(
            name="I17_risk_incumbent", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode=None,
            hypothesis="De-geared incumbent: identical signals and stops, risk-based sizing "
                       "so mean gross is ~1x. Isolates the sizing change alone.",
            change="risk 3% sizing, incumbent stops, no vol target, no trail")], "I17")

    # ---------------- iteration 18: clean attribution ladder ---------------- #
    # The improvements must be attributed one at a time. I17 gives the de-gearing step;
    # this adds the missing single step (vol target with the incumbent stop structure and no
    # 1D-specific exits), so the ladder reads:
    #   flat sizing -> risk sizing -> + vol target -> + per-timeframe exits.
    if "18" in groups:
        add([DdVariant(
            name=f"I18_vol{int(tv*100)}_only", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode=None,
            target_vol=tv, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Volatility target alone on top of the de-geared incumbent, with the "
                       "stop structure untouched. Isolates the reallocation effect from any "
                       "change to the exits.",
            change=f"risk 3%, incumbent 8xATR fixed stop, vol target {tv:.0%}/yr")
            for tv in (0.30, 0.50)], "I18")

    # ---------------- iteration 19: can ANY 4H trail earn its place? ---------------- #
    # The brief requires a working trailing stop. On 1D the trail fires ~33 times and helps.
    # On 4H every setting tried so far cost more Sharpe than drawdown it saved. Before
    # dropping it from the 4H leg, sweep the two knobs that could plausibly change that
    # verdict - how far in profit the trail activates, and how wide it trails - at the
    # chosen sizing. If no cell of the grid beats the no-trail arm on the drawdown-first
    # objective, the conclusion is a measurement rather than a preference.
    if "19" in groups:
        for tm, act in ((4.0, 2.0), (6.0, 2.0), (6.0, 3.0), (8.0, 3.0), (8.0, 4.0),
                        (10.0, 4.0), (12.0, 6.0)):
            add([DdVariant(
                name=f"I19_4Htrail_tm{tm:.0f}_act{act:.0f}", base="V43_mkt_long_atr8",
                risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
                trail_atr_mult=tm, trail_activate_r=act, breakeven_at_r=2.0,
                per_tf={"1D": {"trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                               "own_ema_exit": True, "own_ema_span": 50,
                               "breakeven_at_r": 2.0}},
                target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
                hypothesis="Grid over the 4H trail's two knobs at final sizing. Either some "
                           "cell beats the no-trail arm on the drawdown-first objective, or "
                           "the 4H trail is genuinely a cost and the 4H leg uses the "
                           "catastrophe stop plus a break-even floor instead.",
                change=f"4H trail {tm:.0f}xATR activating at +{act:.0f}R, vol 30%/yr")],
                "I19")
        add([DdVariant(
            name="I19_4Hnotrail_ctrl", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail=False, breakeven_at_r=2.0,
            per_tf={"4H": {"trail": False, "breakeven_at_r": 2.0},
                    "1D": {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50,
                           "breakeven_at_r": 2.0}},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="The matched control every grid cell must beat: 4H wide fixed stop + "
                       "break-even floor, no trail; 1D unchanged.",
            change="4H no trail; control for the iteration-19 grid")], "I19")

    # ---------------- iteration 20: finalist + parameter sensitivity ---------------- #
    # The finalist is fixed here, before the neighbours are run, and the neighbours are a
    # pre-registered perturbation grid rather than a search. The point is to show the result
    # is a plateau, not a spike: if a +-1 grid step in risk or vol target moves the
    # drawdown-first objective a lot, nothing here is stable enough to ship.
    if "20" in groups:
        FIN_4H = {"trail": False, "breakeven_at_r": 2.0}
        FIN_1D = {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                  "own_ema_exit": True, "own_ema_span": 50, "breakeven_at_r": 2.0}
        add([DdVariant(
            name="I20_finalist", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail=False, breakeven_at_r=2.0, per_tf={"4H": FIN_4H, "1D": FIN_1D},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="The selected strategy: the EMF+ADL long book, market-regime gated, "
                       "sized by risk with a 30%/yr volatility target, an 8xATR catastrophe "
                       "stop and a +2R break-even floor on 4H, and a 6xATR trail activating "
                       "at +3R plus an EMA50 exit on 1D.",
            change="FINALIST - see report section 12 for the full rule set")], "I20")
        for rp in (0.02, 0.025, 0.035, 0.04):
            add([DdVariant(
                name=f"I20_risk{int(rp*1000)}", base="V43_mkt_long_atr8",
                risk_per_trade=rp, stop_mode="atr_trail", stop_atr_mult=8.0,
                trail=False, breakeven_at_r=2.0, per_tf={"4H": FIN_4H, "1D": FIN_1D},
                target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
                hypothesis="Risk-per-trade perturbation. The drawdown-first objective must "
                           "not depend on landing exactly on 3%.",
                change=f"risk per trade {rp:.1%}")], "I20")
        for tv in (0.20, 0.25, 0.35, 0.40):
            add([DdVariant(
                name=f"I20_vol{int(tv*100)}", base="V43_mkt_long_atr8",
                risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
                trail=False, breakeven_at_r=2.0, per_tf={"4H": FIN_4H, "1D": FIN_1D},
                target_vol=tv, vol_window=100, vol_lo=0.25, vol_hi=4.0,
                hypothesis="Volatility-target perturbation.",
                change=f"vol target {tv:.0%}/yr")], "I20")
        for tm in (4.0, 8.0):
            add([DdVariant(
                name=f"I20_1Dtrail_tm{tm:.0f}", base="V43_mkt_long_atr8",
                risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
                trail=False, breakeven_at_r=2.0,
                per_tf={"4H": FIN_4H,
                        "1D": {"trail": True, "trail_atr_mult": tm, "trail_activate_r": 3.0,
                               "own_ema_exit": True, "own_ema_span": 50,
                               "breakeven_at_r": 2.0}},
                target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
                hypothesis="The 1D trailing distance is the one stop parameter with a real "
                           "job to do; it must be a plateau in the distance too.",
                change=f"1D trail {tm:.0f}xATR")], "I20")
        add([DdVariant(
            name="I20_noBE", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail=False, breakeven_at_r=0.0,
            per_tf={"4H": {"trail": False, "breakeven_at_r": 0.0},
                    "1D": {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": True, "own_ema_span": 50, "breakeven_at_r": 0.0}},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Is the break-even floor earning its place, or is it decoration? "
                       "Ablation of the one profit-protection rule in the finalist.",
            change="break-even floor off on both timeframes")], "I20")
        add([DdVariant(
            name="I20_noEMA50", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, stop_mode="atr_trail", stop_atr_mult=8.0,
            trail=False, breakeven_at_r=2.0,
            per_tf={"4H": FIN_4H,
                    "1D": {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                           "own_ema_exit": False, "breakeven_at_r": 2.0}},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="Ablation of the 1D EMA50 exit.",
            change="1D EMA50 exit off")], "I20")

    # ---------------- iteration 21: the capital-intensity frontier ------------------- #
    # Iterations 0-20 answered "can the drawdown be controlled?" (yes, decisively). This
    # iteration answers the second half of the brief: *at the drawdown the previous strategy
    # actually ran* (-31% on 4H), what PnL and Sharpe does the risk-normalised book deliver?
    # Once risk is normalised, leverage is a free dial, so the frontier is reported rather
    # than a single point — and the brief's "robust, fully working trailing stop" is enabled
    # on BOTH timeframes here, so the stop layer is live by construction.
    if "21" in groups:
        T_4H = {"trail": True, "trail_atr_mult": 4.0, "trail_activate_r": 2.0,
                "breakeven_at_r": 2.0}
        T_1D = {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
                "own_ema_exit": True, "own_ema_span": 50, "breakeven_at_r": 2.0}
        add([DdVariant(
            name="I21_trail_base", base="V43_mkt_long_atr8",
            risk_per_trade=0.03, max_leverage=2.0, stop_mode="atr_trail",
            stop_atr_mult=8.0, trail=True, breakeven_at_r=2.0,
            per_tf={"4H": T_4H, "1D": T_1D},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="The brief-compliant finalist: trailing stop live on both "
                       "timeframes, so the stop layer is 'fully working' by construction "
                       "rather than only on 1D.",
            change="4H trail 4xATR arming at +2R; 1D trail 6xATR at +3R plus EMA50 exit")],
            "I21")
        for rp in (0.10, 0.20, 0.30, 0.45):
            add([DdVariant(
                name=f"I21_trail_r{int(rp*100)}", base="V43_mkt_long_atr8",
                risk_per_trade=rp, max_leverage=2.0, stop_mode="atr_trail",
                stop_atr_mult=8.0, trail=True, breakeven_at_r=2.0,
                per_tf={"4H": T_4H, "1D": T_1D},
                target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
                hypothesis="Capital-intensity frontier: identical rules, larger risk "
                           "budget. Reports the PnL available at each drawdown level so "
                           "the trade-off is explicit instead of chosen for the reader.",
                change=f"risk per trade {rp:.0%}")], "I21")
        add([DdVariant(
            name="I21_trail_r30_brake", base="V43_mkt_long_atr8",
            risk_per_trade=0.30, max_leverage=2.0, stop_mode="atr_trail",
            stop_atr_mult=8.0, trail=True, breakeven_at_r=2.0,
            per_tf={"4H": T_4H, "1D": T_1D},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            brake_dd_on=0.12, brake_dd_full=0.30, brake_floor=0.25, brake_dd_off=0.06,
            hypothesis="At 10x the risk budget the equity brake finally has a drawdown to "
                       "act on; this tests whether it earns its complexity there, which is "
                       "the only regime where the question is meaningful.",
            change="equity brake on: de-risk from -12%, floor 25% at -30%, release at -6%")],
            "I21")
        add([DdVariant(
            name="I21_nosh4trail_r30", base="V43_mkt_long_atr8",
            risk_per_trade=0.30, max_leverage=2.0, stop_mode="atr_trail",
            stop_atr_mult=8.0, trail=False, breakeven_at_r=2.0,
            per_tf={"4H": {"trail": False, "breakeven_at_r": 2.0}, "1D": T_1D},
            target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
            hypothesis="The Sharpe-max alternative at the same capital intensity: 4H keeps "
                       "the 8xATR catastrophe stop and the +2R break-even floor but no "
                       "trail, because on 4H the trail has cost more Sharpe than the "
                       "drawdown it saved in every grid cell tested.",
            change="4H trail off, everything else identical")], "I21")

    tag = args.tag or "dd"
    RL._dump(OUT / f"dd_stage_{tag}.json", results)
    print()
    print(table(results, f"drawdown-first iterations [{','.join(windows)}]"))
    print(f"\nelapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
