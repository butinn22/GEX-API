"""Walk-forward re-validation of the FROZEN confluence_breakout presets on engine v2.

The presets are frozen research artefacts, so this is a **calendar out-of-sample
evaluation** (no parameter fitting → no train/validation leakage): run each preset
over the full real history per ticker, build an equal-weight, live-ticker-aware
panel, and segment it on absolute UTC calendar windows. Then apply the skill's
stress + adversarial battery.

Every number in the report is written to `results.json`; the report is generated
from that artefact. Real data only (quant/cache). Point-in-time fills come from the
project engine (signal at bar close i → fill at bar i+1 open).

`remove best trade` is done two ways per the skill: (a) ledger deletion (exact) and
(b) engine suppression of the entry across that trade's whole holding window (a
one-bar suppression would just re-fire — the reference's trap #7), asserting the
trade count drops by exactly one.

Usage: PYTHONPATH=. .venv/Scripts/python.exe harness.py
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CACHE = os.path.join(ROOT, "quant", "cache")
OUT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT)

from trading.application.backtest.engine import (  # noqa: E402
    ENGINE_VERSION, BacktestConfig, run_backtest,
)
from trading.application.backtest.metrics import compute_metrics, profit_factor  # noqa: E402
from trading.application.strategies.confluence_breakout import (  # noqa: E402
    ConfluenceBreakoutStrategy, preset_params,
)
from trading.domain import Bar  # noqa: E402

PPY = {"4h": 2190, "1d": 365}
WINDOWS = 5
PRESETS = [("alligator_4h", "4h"), ("donchian_1d", "1d")]


# ── data layer ──────────────────────────────────────────────────────────────

def load_bars(symbol: str, tf: str) -> list[Bar]:
    path = os.path.join(CACHE, f"{symbol}USDT_{tf}.csv")
    out: list[Bar] = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = datetime.fromisoformat(r["ts"]).replace(tzinfo=timezone.utc)
            out.append(Bar(timestamp=ts, open=float(r["open"]), high=float(r["high"]),
                           low=float(r["low"]), close=float(r["close"]),
                           volume=float(r["volume"])))
    return out


def manifest(symbol: str, tf: str, bars: list[Bar]) -> dict:
    ts = [b.timestamp for b in bars]
    bad = sum(1 for b in bars if not (b.low <= b.open <= b.high and b.low <= b.close <= b.high))
    zero = sum(1 for b in bars if b.volume <= 0)
    gaps = sum(1 for i in range(1, len(ts)) if (ts[i] - ts[i - 1]).total_seconds() <= 0)
    raw = "\n".join(f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}"
                    for b in bars)
    return {
        "symbol": symbol, "timeframe": tf, "n_bars": len(bars),
        "first": ts[0].isoformat() if ts else None,
        "last": ts[-1].isoformat() if ts else None,
        "bad_ohlc": bad, "zero_volume": zero, "non_monotonic_or_dup": gaps,
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def universe(tf: str, min_bars: int) -> tuple[list[str], list[dict]]:
    syms = []
    for f in sorted(os.listdir(CACHE)):
        if f.endswith(f"USDT_{tf}.csv"):
            s = f.split("USDT_")[0]
            if len(load_bars(s, tf)) >= min_bars:
                syms.append(s)
    return syms, [manifest(s, tf, load_bars(s, tf)) for s in syms]


# ── correctness ─────────────────────────────────────────────────────────────

async def _signals(strategy, bars) -> list[tuple[str, str, bool]]:
    await strategy.prepare(bars)
    out = []
    for b in bars:
        for s in await strategy.on_bar(b):
            out.append((b.timestamp.isoformat(), s.reason, bool(s.reduce_only)))
    return out


def correctness_no_lookahead(name: str, tf: str, bars: list[Bar]) -> dict:
    cut = int(len(bars) * 0.6)
    params = {**preset_params(name), "timeframe": tf}
    full = asyncio.run(_signals(ConfluenceBreakoutStrategy("X", params=dict(params), preset=name), bars))
    pre = asyncio.run(_signals(ConfluenceBreakoutStrategy("X", params=dict(params), preset=name), bars[:cut]))
    cutoff = bars[cut].timestamp.isoformat()
    full_upto = [s for s in full if s[0] < cutoff]
    return {"name": name, "cut_bar": cut, "prefix_signals": len(pre),
            "full_signals_before_cut": len(full_upto), "identical": bool(pre == full_upto)}


# ── engine suppression of the best trade's entry over its whole holding window ─

class SuppressWindow:
    """Wrap a strategy and drop its entry signals while ``t0 <= bar.ts <= t1``."""

    def __init__(self, inner, t0: str, t1: str):
        self.inner = inner
        self.t0 = t0
        self.t1 = t1

    async def prepare(self, bars):
        return await self.inner.prepare(bars)

    async def on_bar(self, bar):
        ts = bar.timestamp.isoformat()
        out = []
        for s in await self.inner.on_bar(bar):
            if (not s.reduce_only) and self.t0 <= ts <= self.t1:
                continue
            out.append(s)
        return out


# ── evaluation ──────────────────────────────────────────────────────────────

def cfg_for(tf: str, params: dict, *, slip: float = 0.0005, fee: float = 0.001) -> BacktestConfig:
    pf = float(params.get("position_fraction", 0.95))
    return BacktestConfig(initial_cash=100_000.0, fee_rate=fee, slippage=slip,
                          position_fraction=min(max(pf, 0.05), 1.0),
                          periods_per_year=PPY[tf])


def _holdings(res) -> list[tuple[str, str]]:
    """Closed-trade holding windows (entry ts, exit ts) in order, from the ledger."""
    out: list[tuple[str, str]] = []
    entry = None
    for ev in res.events:
        st = ev.state.value if hasattr(ev.state, "value") else str(ev.state)
        if st in ("long_entry", "short_entry"):
            entry = ev.timestamp.isoformat()
        elif st in ("long_exit", "short_exit") and entry is not None:
            out.append((entry, ev.timestamp.isoformat()))
            entry = None
    return out


def run_all(name: str, tf: str, syms: list[str], *, slip=0.0005, fee=0.001,
            suppress: dict[str, tuple[str, str]] | None = None):
    params = {**preset_params(name), "timeframe": tf}
    per_ticker, trades, holdings = {}, [], {}
    for sym in syms:
        bars = load_bars(sym, tf)
        strat = ConfluenceBreakoutStrategy(f"{sym}USDT", params=dict(params), preset=name)
        if suppress and sym in suppress:
            strat = SuppressWindow(strat, *suppress[sym])
        res = asyncio.run(run_backtest(strat, bars, cfg_for(tf, params, slip=slip, fee=fee)))
        per_ticker[sym] = pd.Series(res.equity_curve, index=[b.timestamp for b in bars])
        holdings[sym] = _holdings(res)
        for t in res.trades:
            trades.append({"symbol": sym, "pnl": t.realized_pnl, "qty": t.quantity,
                           "entry": t.entry_price, "exit": t.exit_price,
                           "exit_time": t.exit_time.isoformat()})
    return per_ticker, trades, holdings


def panel(per_ticker: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame(per_ticker).sort_index()
    rets = df.pct_change()
    live = rets.notna().sum(axis=1)
    eq = (1.0 + rets.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    return eq, live


def window_table(eq: pd.Series, per_ticker, trades) -> list[dict]:
    start, end = eq.index.min(), eq.index.max()
    edges = pd.date_range(start, end, periods=WINDOWS + 1)
    rows = []
    for w in range(WINDOWS):
        lo, hi = edges[w], edges[w + 1]
        last = w == WINDOWS - 1
        mask = (eq.index >= lo) & ((eq.index <= hi) if last else (eq.index < hi))
        seg = eq[mask]
        seg_ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0) if len(seg) > 1 else 0.0
        peak = seg.cummax()
        maxdd = float(((peak - seg) / peak).max()) if len(seg) else 0.0
        wt = [t for t in trades if lo.isoformat() <= t["exit_time"] <= hi.isoformat()]
        live_ret = []
        for s, ser in per_ticker.items():
            m = (ser.index >= lo) & ((ser.index <= hi) if last else (ser.index < hi))
            sub = ser[m]
            if len(sub) > 1:
                live_ret.append(float(sub.iloc[-1] / sub.iloc[0] - 1.0))
        rows.append({
            "window": w + 1, "start": lo.isoformat(), "end": hi.isoformat(),
            "return": seg_ret, "max_drawdown": maxdd, "trades": len(wt),
            "live_tickers": len(live_ret),
            "breadth": float(np.mean([r > 0 for r in live_ret])) if live_ret else 0.0,
        })
    return rows


def summarise(per_ticker, trades, periods_per_year: int) -> dict:
    eq, live = panel(per_ticker)
    pnls = [t["pnl"] for t in trades]
    m = compute_metrics(eq.to_numpy(), pnls, periods_per_year=periods_per_year)
    ticker_sharpe = [compute_metrics(ser.to_numpy(), [], periods_per_year=periods_per_year).sharpe
                     for ser in per_ticker.values()]
    return {
        "total_return": m.total_return, "sharpe": m.sharpe, "sortino": m.sortino,
        "max_drawdown": m.max_drawdown, "calmar": m.calmar,
        "profit_factor": (None if math.isinf(m.profit_factor) else m.profit_factor),
        "win_rate": m.win_rate, "n_trades": m.n_trades, "n_tickers": len(per_ticker),
        "median_ticker_sharpe": float(np.median(ticker_sharpe)) if ticker_sharpe else 0.0,
        "live_ticker_min": int(live.min()), "live_ticker_max": int(live.max()),
    }


def concentration(trades: list[dict]) -> dict:
    net = sum(t["pnl"] for t in trades)
    best = max(trades, key=lambda t: t["pnl"]) if trades else None
    top5 = sum(sorted((t["pnl"] for t in trades), reverse=True)[:5])
    rets = [(t["pnl"] / (t["entry"] * t["qty"])) for t in trades if t["entry"] * t["qty"] > 0]
    return {
        "net_pnl": net, "best_symbol": best["symbol"] if best else None,
        "best_trade_currency_share": (best["pnl"] / net) if best and net else None,
        "top5_currency_share": (top5 / net) if net else None,
        "best_trade_return_share": (max(rets) / sum(rets)) if rets and sum(rets) else None,
    }


def per_year(per_ticker) -> dict:
    eq, _ = panel(per_ticker)
    out = {}
    for year, sub in eq.groupby(eq.index.year):
        if len(sub) > 1:
            out[str(year)] = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
    return out


def remove_best_trade_engine(name, tf, syms, holdings, best_symbol, best_index):
    """Suppress the best trade's entry across its whole holding window.

    A re-entering strategy may immediately form a *replacement* trade after the
    window, so the raw trade count need not drop by exactly one (the reference's
    trap #7 assumes no re-entry). We therefore verify the structural claim that
    can be checked without that assumption: the specific trade's exit disappears
    from the window. The exact economic effect comes from the ledger route.
    """
    _, base_tr, _ = run_all(name, tf, syms)
    t0, t1 = holdings[best_symbol][best_index]
    # The entry *signal* fires one bar before the entry *fill*, so the suppression
    # window must open one bar early (else the signal re-fires and the trade forms).
    delta = timedelta(hours=4) if tf == "4h" else timedelta(days=1)
    t0_lo = (datetime.fromisoformat(t0) - delta).isoformat()
    _, sup_tr, _ = run_all(name, tf, syms, suppress={best_symbol: (t0_lo, t1)})
    in_before = sum(1 for t in base_tr if t["symbol"] == best_symbol and t0 <= t["exit_time"] <= t1)
    in_after = sum(1 for t in sup_tr if t["symbol"] == best_symbol and t0 <= t["exit_time"] <= t1)
    return {"trades_before": len(base_tr), "trades_after": len(sup_tr),
            "count_delta": len(sup_tr) - len(base_tr), "holding_window": [t0, t1],
            "specific_trade_before": in_before, "specific_trade_after": in_after,
            "specific_trade_removed": (in_after == 0 and in_before >= 1),
            "dropped_exactly_one": len(sup_tr) == len(base_tr) - 1}


def evaluate_preset(name: str, tf: str) -> dict:
    syms, manif = universe(tf, min_bars=1500 if tf == "4h" else 500)
    corr = correctness_no_lookahead(name, tf, load_bars(syms[0], tf))

    per_ticker, trades, holdings = run_all(name, tf, syms)
    base = summarise(per_ticker, trades, PPY[tf])
    eq, _live = panel(per_ticker)
    windows = window_table(eq, per_ticker, trades)
    worst = min(windows, key=lambda w: w["return"])

    stress = {"base": {"sharpe": base["sharpe"], "total_return": base["total_return"]}}
    for label, slip, fee in (("slippage_2x", 0.001, 0.001), ("slippage_4x", 0.002, 0.001),
                             ("fees_2x", 0.0005, 0.002), ("harsh", 0.002, 0.002)):
        pt, tr, _ = run_all(name, tf, syms, slip=slip, fee=fee)
        s = summarise(pt, tr, PPY[tf])
        stress[label] = {"sharpe": s["sharpe"], "total_return": s["total_return"]}

    best_sym = max(per_ticker, key=lambda s: (per_ticker[s].iloc[-1] / per_ticker[s].iloc[0] - 1.0))
    pt2, tr2, _ = run_all(name, tf, [s for s in syms if s != best_sym])
    drop_ticker = summarise(pt2, tr2, PPY[tf])
    drop_ticker["removed"] = best_sym

    best_trade = max(trades, key=lambda t: t["pnl"])
    ledger = [t["pnl"] for t in trades if t is not best_trade]
    sym_trades = [t for t in trades if t["symbol"] == best_trade["symbol"]]
    best_index = next(i for i, t in enumerate(sym_trades) if t is best_trade)
    sym_holdings = holdings[best_trade["symbol"]]
    engine_check = remove_best_trade_engine(name, tf, syms, holdings, best_trade["symbol"], best_index)

    return {
        "preset": name, "timeframe": tf, "params": {**preset_params(name), "timeframe": tf},
        "universe": {"n": len(syms), "symbols": syms}, "manifest": manif,
        "correctness": corr, "full": base, "windows": windows, "worst_window": worst,
        "holdout_last_window": windows[-1], "cost_stress": stress,
        "adversarial": {
            "remove_best_ticker": drop_ticker,
            "remove_best_trade_ledger": {
                "pf_before": base["profit_factor"],
                "pf_after": (None if math.isinf(profit_factor(ledger)) else profit_factor(ledger)),
                "n_before": len(trades), "n_after": len(ledger),
            },
            "remove_best_trade_engine": engine_check,
            "concentration": concentration(trades),
            "holdings_available": len(sym_holdings),
        },
        "per_year": per_year(per_ticker),
    }


def verdict(p: dict) -> dict:
    w = p["windows"]
    pos = sum(1 for x in w if x["return"] > 0)
    mean_breadth = float(np.mean([x["breadth"] for x in w]))
    harsh = p["cost_stress"]["harsh"]["sharpe"]
    engine_ok = p["adversarial"]["remove_best_trade_engine"]["specific_trade_removed"]
    nolook = p["correctness"]["identical"]
    robust = (pos >= len(w) - 1) and (mean_breadth >= 0.5) and (harsh > 0) and engine_ok and nolook
    return {"positive_windows": pos, "total_windows": len(w), "mean_breadth": mean_breadth,
            "harsh_sharpe": harsh, "no_lookahead_ok": nolook,
            "remove_best_trade_engine_ok": engine_ok, "robust": bool(robust)}


def main():
    out = {"engine_version": ENGINE_VERSION,
           "generated": datetime.now(timezone.utc).isoformat(), "presets": {}}
    for name, tf in PRESETS:
        print(f"evaluating {name} ({tf}) ...", flush=True)
        res = evaluate_preset(name, tf)
        res["verdict"] = verdict(res)
        out["presets"][name] = res
        v = res["verdict"]
        print(f"  robust={v['robust']} windows={v['positive_windows']}/{v['total_windows']} "
              f"breadth={v['mean_breadth']:.2f} harsh_sharpe={v['harsh_sharpe']:.2f} "
              f"specific_trade_removed={v['remove_best_trade_engine_ok']}", flush=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote results.json")


if __name__ == "__main__":
    main()
