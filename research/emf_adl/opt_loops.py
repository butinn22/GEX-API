"""User-requested optimisation round: HTF filter, EMF+ADL confluence, TP targets,
consecutive-loss circuit breaker, buy-and-hold benchmark, fresh-ticker OOS.

Operating point
---------------
Every arm runs at the previous programme's recommended finalist (`I21_trail_r10`
in dd_loops, called FINAL in the DD report):

* base signal: EMF+ADL (repaired hybrid), long only, gated by BTC>EMA200 same-TF
* sizing: risk 10% of equity per trade, notional cap 2x, 30%/yr vol target
* stops: 8xATR(14) catastrophe; 4H trail 4xATR arming at +2R; 1D trail 6xATR
  arming at +3R plus EMA50 exit; break-even floor at +2R
* costs: 5.5 bps taker, 3 bps slippage, real funding

Pre-registered arms (no tuning, each isolates one requested change):

O0_FINAL        control, unchanged incumbent
O1_HTF          4H entries additionally require BTC **1D** close > 1D EMA200,
                using only fully-closed daily bars (true higher-timeframe filter).
                The 1D book is unchanged: its own gate is already the same rule on
                its own timeframe, and a weekly EMA200 would consume the entire
                5.25y sample as warm-up (disclosed, not hidden).
O2_CONFLUENCE   entries require the EMF leg AND the ADL leg (the shipped entry is
                an OR of the two; this replaces OR with AND)
O3_TP{3,5,8}    take-profit at 3/5/8 x ATR(14) at entry
O4_CB{2,3}      consecutive-loss circuit breaker: after K consecutive losing
                trades on a symbol, halve the size of every entry until the next
                winner. Implemented as a two-pass causal simulation (pass 1
                produces the trade ledger, the scale is derived from closed
                trades only, pass 2 re-runs with the scale; trade sequence is
                verified identical between passes).

Buy-and-hold benchmark (never computed by the earlier programmes): per symbol,
enter one bar after warm-up at the open with the same costs, pay real funding
while held, close at the last bar. Equal-weight panel average, identical to the
strategy's aggregation. The BTC-only variant is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.emf_adl import data as D  # noqa: E402
from research.emf_adl import loop as L  # noqa: E402
from research.emf_adl import portfolio as P  # noqa: E402
from research.emf_adl import rules as R  # noqa: E402
from research.emf_adl import run_loops as RL  # noqa: E402
from research.emf_adl import dd_loops as DD  # noqa: E402
from research.emf_adl.engine import (  # noqa: E402
    Costs, Result, Sizing, SignalSet, StopSpec, run,
)

log = logging.getLogger("opt")
OUT = RL.OUT
TIMEFRAMES = ("4H", "1D")
_BARS_PER_YEAR = {"4H": 6 * 365, "1D": 365}
#: Fresh, currently-liquid, never-used-in-this-programme tickers (all listed on
#: Bybit perps in 2023, so the train window is short — disclosed in the report).
FRESH = ["ARBUSDT", "SUIUSDT", "1000PEPEUSDT", "WLDUSDT", "TAOUSDT"]

START, END = RL.STUDY_START, RL.STUDY_END


# --------------------------------------------------------------------------- #
# Operating point
# --------------------------------------------------------------------------- #
def op_variant(name: str = "O0_FINAL", **kw) -> DD.DdVariant:
    """I21_trail_r10 verbatim — the incumbent every arm is measured against."""
    t4 = {"trail": True, "trail_atr_mult": 4.0, "trail_activate_r": 2.0,
          "breakeven_at_r": 2.0}
    t1 = {"trail": True, "trail_atr_mult": 6.0, "trail_activate_r": 3.0,
          "own_ema_exit": True, "own_ema_span": 50, "breakeven_at_r": 2.0}
    base = dict(
        name=name, base="V43_mkt_long_atr8",
        risk_per_trade=0.10, max_leverage=2.0,
        stop_mode="atr_trail", stop_atr_mult=8.0, trail=True, breakeven_at_r=2.0,
        per_tf={"4H": t4, "1D": t1},
        target_vol=0.30, vol_window=100, vol_lo=0.25, vol_hi=4.0,
    )
    base.update(kw)
    return DD.DdVariant(**base)


# --------------------------------------------------------------------------- #
# Higher-timeframe regime: BTC 1D close vs 1D EMA200, closed-bar projection
# --------------------------------------------------------------------------- #
_HTF_CACHE: dict = {}


def htf_daily_regime():
    """BTC 1D trend state on its own daily bars. ``+1``/``-1`` per daily bar."""
    key = ("1D", START, END, 200)
    hit = _HTF_CACHE.get(key)
    if hit is not None:
        return hit
    s = D.load_series("BTCUSDT", "1D", D.ms(*START), D.ms(*END))
    c = s.bars["close"].to_numpy(float)
    ema = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()
    reg = np.where(c > ema, 1, -1).astype(np.int8)
    _HTF_CACHE[key] = (s.bars["timestamp"].to_numpy(np.int64), reg)
    return _HTF_CACHE[key]


def htf_gate_for_4h(bars: pd.DataFrame) -> np.ndarray:
    """Project the *closed* daily regime onto 4H bars.

    A daily bar closes at ``open_ts + 24h``; a 4H bar at ``t`` may only see daily
    bars that closed at or before ``t``. No partial-day information is used.
    """
    dts, reg = htf_daily_regime()
    close_ms = dts + 86_400_000
    own = bars["timestamp"].to_numpy(np.int64)
    pos = np.searchsorted(close_ms, own, side="right") - 1
    out = np.full(len(own), -1, dtype=np.int8)  # no closed daily bar yet -> no trade
    ok = pos >= 0
    out[ok] = reg[pos[ok]]
    return out


# --------------------------------------------------------------------------- #
# Confluence legs (from the repaired feature frame)
# --------------------------------------------------------------------------- #
def adl_leg(ctx: RL.SeriesContext) -> np.ndarray:
    """The ADL-side agreement: AD line above its 50 and 200 averages and its own
    prior value, with the ADL MACD and its signal both positive. This is the ADL
    subset of the shipped strategy-B entry conditions (the tp_f / tl thresholds
    are omitted to keep the leg a single pre-registered form)."""
    f = ctx.frames[True]
    adl = f["adline"].to_numpy(float)
    out = (
        (adl > f["adl50"].to_numpy(float))
        & (adl > f["ad"].to_numpy(float))
        & (adl > f["adl200"].to_numpy(float))
        & (f["adl_macd"].to_numpy(float) > 0)
        & (f["adl_signal"].to_numpy(float) > 0)
    )
    return np.nan_to_num(out.astype(float), nan=0).astype(bool)


# --------------------------------------------------------------------------- #
# Circuit breaker: causal per-bar scale from a closed-trade ledger
# --------------------------------------------------------------------------- #
def cb_scale_array(trades: list, n: int, k: int, scale_val: float) -> np.ndarray:
    """Scale applied to entries after ``k`` consecutive net losses.

    Derived only from trades already closed: after each trade closes the streak
    updates, and the new scale applies from the next bar onward. Because sizing
    cannot change the signal or the stop levels, the trade *sequence* is
    invariant to the scale, which the caller verifies.
    """
    scale = np.ones(n, dtype=float)
    streak = 0
    for t in sorted(trades, key=lambda x: x.entry_bar):
        streak = streak + 1 if t.net_pnl < 0 else 0
        nxt = t.exit_bar + 1
        if nxt < n:
            scale[nxt:] = scale_val if streak >= k else 1.0
    return scale


def _seq_key(trades: list) -> list:
    return [(t.entry_bar, t.exit_bar, t.exit_reason) for t in trades]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _vol_scale_for(ctx: RL.SeriesContext, tf: str, target: float):
    if target is None:
        return None
    return P.vol_target_scale(
        ctx.bars, target_bar_vol=target / math.sqrt(_BARS_PER_YEAR[tf]),
        window=100, lo=0.25, hi=4.0,
    )


def _window_mask(ctx: RL.SeriesContext, window: str) -> np.ndarray | None:
    if window == "all":
        return None
    m = np.zeros(len(ctx.bars), dtype=bool)
    m[ctx.wins[window]] = True
    return m


def run_book(
    contexts: dict,
    *,
    tf: str,
    variant: DD.DdVariant,
    costs: Costs,
    cat: dict,
    window: str = "all",
    confluence: bool = False,
    htf_4h: bool = False,
    tp_atr_mult: float | None = None,
    cb_k: int = 0,
    cb_val: float = 0.5,
) -> dict:
    """One arm on one timeframe. Mirrors dd_loops.run_variant but supports the
    confluence / HTF signal edits, a TP override, and the circuit breaker."""
    base = cat[variant.base]
    eff = replace(variant, per_tf={}, **variant.per_tf.get(tf, {}))
    sub = {k: c for k, c in contexts.items() if k[1] == tf}
    results: dict[str, Result] = {}
    cb_info = {"symbols_braked": 0, "sequence_stable": True}

    for (sym, tfx), ctx in sorted(sub.items()):
        sig, _ = DD._signals_for(ctx, base, eff)
        if confluence:
            sig = replace(sig, entry_long=sig.entry_long & adl_leg(ctx))
        if htf_4h and tfx == "4H":
            sig = replace(sig, entry_long=sig.entry_long & (htf_gate_for_4h(ctx.bars) > 0))
        m = _window_mask(ctx, window)
        if m is not None:
            sig = replace(sig, entry_long=sig.entry_long & m)
        stops = DD._stops_for(ctx, base, eff)
        if tp_atr_mult is not None:
            stops = replace(stops, tp_mode="atr", tp_atr_mult=float(tp_atr_mult))
        vs = _vol_scale_for(ctx, tf, eff.target_vol)
        s1 = Sizing(risk_per_trade=eff.risk_per_trade, max_leverage=eff.max_leverage,
                    vol_scale=vs)
        r = run(ctx.bars, sig, symbol=sym, timeframe=tfx, tf_hours=ctx.tf_hours,
                costs=costs, stops=stops, funding=ctx.funding, sizing=s1)
        if cb_k > 0:
            scale = cb_scale_array(r.trades, len(ctx.bars), cb_k, cb_val)
            if (scale < 1.0).any():
                cb_info["symbols_braked"] += 1
            s2 = Sizing(risk_per_trade=eff.risk_per_trade, max_leverage=eff.max_leverage,
                        vol_scale=vs, bar_scale=scale)
            r2 = run(ctx.bars, sig, symbol=sym, timeframe=tfx, tf_hours=ctx.tf_hours,
                     costs=costs, stops=stops, funding=ctx.funding, sizing=s2)
            if _seq_key(r.trades) != _seq_key(r2.trades):
                cb_info["sequence_stable"] = False
            r = r2
        results[sym] = r

    grid = P.panel_grid(sub)
    if len(grid) == 0 or not results:
        return {"n_tickers": 0}
    eq, grid, live = P.panel_equity(results, grid)
    ts_ms = P.to_ms(grid)
    ts = np.concatenate([[ts_ms[0] - 1 if len(ts_ms) else 0], ts_ms]).astype(np.int64)
    trades_all = [t for r in results.values() for t in r.trades]
    tf_hours = D.TIMEFRAMES[tf][1]
    met = L.compute_metrics(eq, trades_all, tf_hours, timestamps=ts).as_dict()
    per = {s: L.compute_metrics(r.equity, r.trades, tf_hours) for s, r in results.items()}
    rets = np.asarray([m.total_return for m in per.values()])
    reasons: dict[str, int] = {}
    for t in trades_all:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # gross exposure, identical construction to portfolio.run_portfolio
    notional_sum = np.zeros(len(grid), dtype=float)
    for r in results.values():
        pos = np.searchsorted(grid, P.to_ms(r.timestamps))
        for t in r.trades:
            a, b = int(t.entry_bar), int(t.exit_bar)
            if b <= a:
                continue
            seg = pos[slice(max(a, 0), min(b, len(pos)))]
            ok = (seg >= 0) & (seg < len(grid))
            np.add.at(notional_sum, seg[ok], float(t.notional_at_entry))
    gross = notional_sum / np.where(eq[1:] == 0, np.nan, eq[1:])

    return {
        **met, **P.drawdown_stats(eq, ts),
        "n_tickers": len(results),
        "total_trades": len(trades_all),
        "breadth": float((rets > 0).mean()) if len(rets) else 0.0,
        "median_ticker_sharpe": float(np.median([m.sharpe for m in per.values()])),
        "exit_reasons": reasons,
        "mean_gross": float(np.nanmean(gross)) if len(gross) else 0.0,
        "max_gross": float(np.nanmax(gross)) if len(gross) else 0.0,
        "n_pre_stop": int(sum(1 for t in trades_all
                               if t.exit_class == "STOP_LOSS_BEFORE_4H")),
        "n_pre_tp": int(sum(1 for t in trades_all
                            if t.exit_class == "TAKE_PROFIT_BEFORE_4H")),
        "cb": cb_info,
        "per_ticker": {s: {"sharpe": round(m.sharpe, 3), "total_return": round(m.total_return, 4)}
                       for s, m in per.items()},
    }


# --------------------------------------------------------------------------- #
# Buy-and-hold benchmark
# --------------------------------------------------------------------------- #
def bh_signals(ctx: RL.SeriesContext) -> SignalSet:
    n = len(ctx.bars)
    el = np.zeros(n, dtype=bool)
    el[R.WARMUP] = True  # fills at the open of WARMUP+1
    return SignalSet(entry_long=el, entry_short=np.zeros(n, dtype=bool),
                     exit_long=np.zeros(n, dtype=bool),
                     exit_short=np.zeros(n, dtype=bool),
                     atr=ctx.signals[True].atr, warmup=R.WARMUP, label="buy_hold")


def buy_and_hold(contexts: dict, costs: Costs) -> dict:
    """Equal-weight buy-and-hold on the same panel, same aggregation as the strategy.

    Model, stated rather than hidden: one unit of notional is bought at the open
    of the first post-warm-up bar (fee + slippage paid), real funding settlements
    are paid **by trimming the position** (``qty *= 1 - rate`` per settlement —
    a long *receives* when the rate is negative, which the same formula gives),
    and the book is liquidated at the last close (fee + slippage). Trimming is
    the honest treatment: on a perpetual, funding paid from cash would drive a
    5-year hold's account equity negative, which no real account survives —
    the exchange would liquidate it long before.
    """
    out = {}
    slip, fee = costs.slip, costs.fee_rate
    for tf in TIMEFRAMES:
        sub = {k: c for k, c in contexts.items() if k[1] == tf}
        results = {}
        for (sym, tfx), ctx in sorted(sub.items()):
            bars = ctx.bars
            ts = bars["timestamp"].to_numpy(np.int64)
            o = bars["open"].to_numpy(float)
            c = bars["close"].to_numpy(float)
            n = len(bars)
            # entry at the open of WARMUP+1 (the strategy's own first fill bar)
            ei = min(R.WARMUP + 1, n - 1)
            fill = o[ei] * (1 + slip)
            qty = (1.0 / (1.0 + fee)) / fill  # notional 1.0, fee on top
            # funding settlements bucketed per bar, as the engine does
            fund = ctx.funding
            if len(fund):
                ft = fund["timestamp"].to_numpy(np.int64)
                fr = fund["rate"].to_numpy(float)
                idx = np.searchsorted(ts, ft, side="right") - 1
                ok = (idx >= ei) & (idx < n)
                for k, rate in zip(idx[ok], fr[ok]):
                    qty *= (1.0 - float(rate))
            # equity path: mark to close, trim model keeps it non-negative
            eq = np.empty(n, dtype=float)
            eq[:ei] = 1.0
            eq[ei:] = qty * c[ei:]
            eq[-1] = qty * c[-1] * (1 - slip) / (1 + fee)  # liquidation at close
            res = Result(symbol=sym, timeframe=tfx, equity=eq, timestamps=ts,
                         trades=[], costs=costs, label="buy_hold")
            results[sym] = res
        grid = P.panel_grid(sub)
        peq, grid, live = P.panel_equity(results, grid)
        ts_ms = P.to_ms(grid)
        ts2 = np.concatenate([[ts_ms[0] - 1], ts_ms]).astype(np.int64)
        met = L.compute_metrics(peq, [], D.TIMEFRAMES[tf][1], timestamps=ts2).as_dict()
        btc = results.get("BTCUSDT")
        btc_met = (L.compute_metrics(btc.equity, [], D.TIMEFRAMES[tf][1],
                                     timestamps=btc.timestamps.astype("int64")).as_dict()
                   if btc is not None else {})
        out[tf] = {"panel": {k: met[k] for k in
                             ("total_return", "cagr", "sharpe", "max_dd",
                              "win_rate", "profit_factor", "n_trades")},
                   "btc_only": {k: btc_met.get(k) for k in
                                ("total_return", "cagr", "sharpe", "max_dd")}}
    return out


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def diagnose(contexts: dict, costs: Costs, cat: dict) -> dict:
    out = {}
    for tf in TIMEFRAMES:
        sub = {k: c for k, c in contexts.items() if k[1] == tf}
        variant = op_variant("diag")
        eff = replace(variant, per_tf={}, **variant.per_tf.get(tf, {}))
        rows = []
        held_bear_bars = 0
        total_bars = 0
        for (sym, tfx), ctx in sorted(sub.items()):
            base = cat[variant.base]
            sig, _ = DD._signals_for(ctx, base, eff)
            stops = DD._stops_for(ctx, base, eff)
            vs = _vol_scale_for(ctx, tf, eff.target_vol)
            r = run(ctx.bars, sig, symbol=sym, timeframe=tfx, tf_hours=ctx.tf_hours,
                    costs=costs, stops=stops, funding=ctx.funding,
                    sizing=Sizing(risk_per_trade=eff.risk_per_trade,
                                  max_leverage=eff.max_leverage, vol_scale=vs))
            mr = ctx.market_regime
            total_bars += len(ctx.bars)
            for t in r.trades:
                a, b = int(t.entry_bar), int(t.exit_bar)
                bear = int((mr[a:b] < 0).sum())
                held_bear_bars += bear
                risk_usd = abs(t.entry_price - t.initial_stop) * t.quantity
                realized_r = t.net_pnl / risk_usd if risk_usd > 0 else 0.0
                rows.append({
                    "sym": sym, "regime_entry": int(mr[a]) if a < len(mr) else 0,
                    "bear_bars_held": bear, "holding_bars": t.holding_bars,
                    "mfe_r": t.mfe, "mae_r": t.mae, "realized_r": realized_r,
                    "net": t.net_pnl, "win": t.net_pnl > 0,
                    "reason": t.exit_reason,
                })
        df = pd.DataFrame(rows)
        n = len(df)
        reg_pos = df[df.regime_entry > 0]
        reg_neg = df[df.regime_entry < 0]
        win = df[df.net > 0]
        losers = df[df.net <= 0]

        def pf(d):
            g, l = d[d.net > 0].net.sum(), -d[d.net <= 0].net.sum()
            return float(g / l) if l > 0 else float("inf")

        dead = df.mfe_r < 0.5
        giveback = (win.mfe_r - win.realized_r)
        out[tf] = {
            "n_trades": int(n),
            "entries_in_btc_downtrend": int(len(reg_neg)),
            "pnl_entries_in_downtrend": float(reg_neg.net.sum()) if len(reg_neg) else 0.0,
            "win_rate_regime_up": float(reg_pos.win.mean()) if len(reg_pos) else 0.0,
            "pnl_regime_up": float(reg_pos.net.sum()) if len(reg_pos) else 0.0,
            "pf_regime_up": pf(reg_pos),
            "held_bars_btc_downtrend": int(held_bear_bars),
            "held_bars_total": int(total_bars),
            "exposure_in_downtrend": float(held_bear_bars / max(total_bars, 1)),
            "dead_entries_lt_0p5R_mfe": int(dead.sum()),
            "dead_entry_rate": float(dead.mean()) if n else 0.0,
            "median_mfe_r": float(df.mfe_r.median()) if n else 0.0,
            "median_mae_r": float(df.mae_r.median()) if n else 0.0,
            "median_giveback_r_winners": float(giveback.median()) if len(win) else 0.0,
            "mean_capture_ratio_winners": (
                float((win.realized_r / win.mfe_r.clip(lower=1e-9)).mean())
                if len(win) else 0.0),
            "win_rate": float(df.win.mean()) if n else 0.0,
            "profit_factor": pf(df),
            "median_holding_bars": float(df.holding_bars.median()) if n else 0.0,
            "exit_reasons": df.reason.value_counts().to_dict(),
        }
    return out


# --------------------------------------------------------------------------- #
# Contexts
# --------------------------------------------------------------------------- #
def load_contexts():
    uni = RL.load_universe(RL.OUT / "universe.json")
    pool = uni["symbols"]
    start, end = D.ms(*START), D.ms(*END)
    t0 = time.time()
    contexts, exclusions = RL.build_contexts(pool, start, end)
    log.info("contexts: %d in %.0fs (exclusions: %d)",
             len(contexts), time.time() - t0, len(exclusions))
    return contexts


def _relaxed_wins(bars: pd.DataFrame, min_bars: int) -> dict[str, slice]:
    """Calendar wins for late-listed symbols: never raise on an empty *train*
    window (a 2023 listing cannot have 2021 bars), but still require the valid
    and holdout windows to be real — a robustness confirmation that cannot
    cover 2024+ is worthless."""
    ts = bars["timestamp"].to_numpy(dtype=np.int64)
    wins: dict[str, slice] = {}
    for name, (s, e) in RL.SPLITS.items():
        a = int(np.searchsorted(ts, D.ms(*s), side="left"))
        b = int(np.searchsorted(ts, D.ms(*e), side="left"))
        if name == "train":
            wins[name] = slice(max(a, min(R.WARMUP, b)), b)  # may be empty
        else:
            if b - a < min_bars:
                raise D.DataError(
                    f"{name} window has {b - a} bars — series too short to confirm")
            wins[name] = slice(a, b)
    wins["all"] = slice(0, len(bars))
    return wins


def load_fresh_contexts():
    start, end = D.ms(*START), D.ms(*END)
    contexts, excl, notes = {}, [], []
    for sym in FRESH:
        for tf in TIMEFRAMES:
            try:
                s = D.load_series(sym, tf, start, end, with_funding=True)
                bars = s.bars.reset_index(drop=True)
                n = len(bars)
                if n - R.WARMUP < 400:
                    raise D.DataError(f"only {n - R.WARMUP} usable bars after warm-up")
                wins = _relaxed_wins(bars, 60 if tf == "4H" else 30)
                signals, frames = {}, {}
                for rep in (False, True):
                    ss, f = R.base_signals(bars, repair_hybrid=rep)
                    signals[rep], frames[rep] = ss, f
                contexts[(sym, tf)] = RL.SeriesContext(
                    symbol=sym, timeframe=tf, bars=bars, funding=s.funding, wins=wins,
                    signals=signals, frames=frames, tf_hours=D.TIMEFRAMES[tf][1],
                    gates={}, struct_stops=None, struct_raw=None,
                    start_ms=start, end_ms=end,
                    market_regime=RL._align_market_regime(bars, tf, start, end),
                    integrity=s.manifest.integrity,
                )
            except D.DataError as e:
                excl.append(f"{sym} {tf}: {e}")
        try:
            s0 = D.load_series(sym, "4H", start, end)
            notes.append({"symbol": sym, "first_bar": str(
                pd.Timestamp(s0.bars["timestamp"].iloc[0], unit="ms", tz="UTC"))})
        except Exception as e:  # noqa: BLE001
            notes.append({"symbol": sym, "error": str(e)})
    log.info("fresh contexts: %d (exclusions: %d)", len(contexts), len(excl))
    for e in excl:
        log.info("  EXCLUDED %s", e)
    return contexts, excl, notes


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
#: (name, variant-kwargs for op_variant, run_book-kwargs). The O5 arms are
#: DD-matched controls: CB3's drawdown saving must beat simply lowering the
#: risk dial, otherwise the mechanism is complexity without payoff — exactly
#: the standard the equity brake was held to (and failed) in the DD programme.
ARMS = [
    ("O0_FINAL", dict(), dict()),
    ("O1_HTF", dict(), dict(htf_4h=True)),
    ("O2_CONFLUENCE", dict(), dict(confluence=True)),
    ("O3_TP3", dict(), dict(tp_atr_mult=3.0)),
    ("O3_TP5", dict(), dict(tp_atr_mult=5.0)),
    ("O3_TP8", dict(), dict(tp_atr_mult=8.0)),
    ("O4_CB2", dict(), dict(cb_k=2)),
    ("O4_CB3", dict(), dict(cb_k=3)),
    ("O5_CTRL_R8", dict(risk_per_trade=0.08), dict()),
    ("O5_CTRL_R9", dict(risk_per_trade=0.09), dict()),
    ("O5_CB3_R8", dict(risk_per_trade=0.08), dict(cb_k=3)),
]


def cmd_arms(contexts, cat, costs) -> dict:
    results = {}
    windows = ["all", "train", "valid", "holdout"]
    for name, vkw, rkw in ARMS:
        v = op_variant(name, **vkw)
        for w in windows:
            key = f"{name}@{w}"
            t0 = time.time()
            results[key] = {
                tf: run_book(contexts, tf=tf, variant=v, costs=costs, cat=cat,
                             window=w, **rkw)
                for tf in TIMEFRAMES}
            b4, b1 = results[key]["4H"], results[key]["1D"]
            log.info("  %-14s %-7s 4H Sh=%+.2f DD=%5.1f%% ret=%+7.1f%% n=%4d | "
                     "1D Sh=%+.2f DD=%5.1f%% ret=%+7.1f%% n=%4d  (%.0fs)",
                     name, w, b4.get("sharpe", float("nan")), b4.get("max_dd", 0) * 100,
                     b4.get("total_return", 0) * 100, b4.get("total_trades", 0),
                     b1.get("sharpe", float("nan")), b1.get("max_dd", 0) * 100,
                     b1.get("total_return", 0) * 100, b1.get("total_trades", 0),
                     time.time() - t0)
    return results


def cmd_fresh(contexts, cat, costs, arms=(("fresh_control", {}),)) -> dict:
    """Run given arms on the fresh set, windows all/valid/holdout (train is
    mostly empty for 2023 listings — disclosed per symbol)."""
    out = {}
    for name, rkw in arms:
        v = op_variant(name)
        for w in ("all", "valid", "holdout"):
            out[f"{name}@{w}"] = {
                tf: run_book(contexts, tf=tf, variant=v, costs=costs, cat=cat,
                             window=w, **rkw)
                for tf in TIMEFRAMES}
            b4, b1 = out[f"{name}@{w}"]["4H"], out[f"{name}@{w}"]["1D"]
            log.info("  %-18s %-7s 4H Sh=%+.2f ret=%+7.1f%% n=%4d | 1D Sh=%+.2f ret=%+7.1f%% n=%4d",
                     name, w, b4.get("sharpe", float("nan")),
                     b4.get("total_return", 0) * 100, b4.get("total_trades", 0),
                     b1.get("sharpe", float("nan")),
                     b1.get("total_return", 0) * 100, b1.get("total_trades", 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["diag", "arms", "fresh", "all"])
    ap.add_argument("--tag", default="opt")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cat = {x.name: x for x in RL.catalogue()}
    costs = Costs()

    if args.stage in ("diag", "all"):
        contexts = load_contexts()
        t0 = time.time()
        diag = diagnose(contexts, costs, cat)
        bh = buy_and_hold(contexts, costs)
        RL._dump(OUT / f"opt_stage_{args.tag}_diag.json",
                 {"diagnosis": diag, "buy_hold": bh})
        log.info("diag + benchmark done in %.0fs", time.time() - t0)
        for tf in TIMEFRAMES:
            d = diag[tf]
            log.info("  [%s] dead-entry rate %.1f%% | median MFE %.2fR | "
                     "giveback(win) %.2fR | held-in-downtrend %.1f%% of bars",
                     tf, d["dead_entry_rate"] * 100, d["median_mfe_r"],
                     d["median_giveback_r_winners"], d["exposure_in_downtrend"] * 100)
            b = bh[tf]["panel"]
            log.info("  [%s] B&H panel: ret %+.1f%% cagr %.1f%% Sharpe %.2f maxDD %.1f%%",
                     tf, b["total_return"] * 100, b["cagr"] * 100, b["sharpe"],
                     b["max_dd"] * 100)

    if args.stage in ("arms", "all"):
        contexts = load_contexts()
        res = cmd_arms(contexts, cat, costs)
        RL._dump(OUT / f"opt_stage_{args.tag}_arms.json", res)

    if args.stage in ("fresh", "all"):
        fctx, excl, notes = load_fresh_contexts()
        res = cmd_fresh(fctx, cat, costs, arms=[
            ("fresh_control", {}),
            ("fresh_CB3", dict(cb_k=3)),
        ])
        RL._dump(OUT / f"opt_stage_{args.tag}_fresh.json",
                 {"fresh_results": res, "exclusions": excl, "listings": notes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
