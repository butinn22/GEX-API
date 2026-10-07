"""Generate REPORT.md from results.json. No number is typed by hand."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))


def pct(x):
    return "n/a" if x is None else f"{x * 100:.2f}%"


def num(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


L = []
L.append("# Walk-forward re-validation — frozen `confluence_breakout` presets (engine v2)")
L.append("")
L.append(f"**Engine:** `{D['engine_version']}` · **generated:** {D['generated']} · "
         "**data:** real `quant/cache` (no synthetic).")
L.append("")
L.append("> The presets are **frozen** research artefacts, so this is a calendar "
         "out-of-sample evaluation (no parameter fitting → no train/validation leakage). "
         "Every figure below is read from `results.json`.")
L.append("")

# Verdict up front
L.append("## 0. Verdict")
for name, p in D["presets"].items():
    v = p["verdict"]
    L.append(
        f"- **`{name}` ({p['timeframe']}) — {'ROBUST' if v['robust'] else 'NOT ROBUST'}.** "
        f"Positive OOS windows **{v['positive_windows']}/{v['total_windows']}**, "
        f"mean breadth **{v['mean_breadth'] * 100:.1f}%** (share of tickers profitable per window), "
        f"harsh-cost Sharpe **{num(v['harsh_sharpe'], 2)}**, "
        f"no-lookahead **{'pass' if v['no_lookahead_ok'] else 'FAIL'}**, "
        f"best-trade removal structural check **{'pass' if v['remove_best_trade_engine_ok'] else 'FAIL'}**. "
        f"Full-sample return {pct(p['full']['total_return'])}, Sharpe {num(p['full']['sharpe'], 2)}, "
        f"median ticker Sharpe {num(p['full']['median_ticker_sharpe'], 2)}.")
L.append("")
L.append("The robustness bar (≥4/5 positive windows **and** mean breadth ≥50% **and** "
         "harsh-cost Sharpe > 0 **and** both correctness checks) is **not met by either preset**. "
         "Breadth is the binding failure: returns are carried by a minority of tickers.")

L.append("")
L.append("## 1. Constraints restated")
L.append("- Real bars only (repo cache); no synthetic candles, no fabricated fills.")
L.append("- Point-in-time: the project engine fills at bar `i+1` open off a bar-`i` close signal; "
         "verified by truncation (below).")
L.append("- Frozen presets: **no** parameter search on any window, so the OOS windows are clean.")
L.append("- Equal-weight, live-ticker-aware panel (tickers are averaged only where they have data).")
L.append("- No funding modelled (spot-style cache; perps funding not available in the cache) — "
         "an untested axis.")

for name, p in D["presets"].items():
    tf = p["timeframe"]
    L.append("")
    L.append(f"## `{name}` ({tf})")
    L.append("")
    mf = p["manifest"]
    L.append(f"**Universe:** {p['universe']['n']} USDT perps. Manifest (n bars, first→last, "
             "integrity, sha256 prefix):")
    L.append("")
    L.append(table(
        [(m["symbol"], m["n_bars"], m["first"][:10], m["last"][:10],
          m["bad_ohlc"], m["zero_volume"], m["non_monotonic_or_dup"], m["sha256"][:12])
         for m in mf],
        ["sym", "bars", "first", "last", "badOHLC", "zeroVol", "nonMono", "sha256"]))

    c = p["correctness"]
    L.append("")
    L.append(f"**Correctness (no-lookahead by truncation):** prefix signals {c['prefix_signals']} == "
             f"full signals before the cut {c['full_signals_before_cut']} → "
             f"**{'identical' if c['identical'] else 'DIFFERENT'}**.")
    L.append("")
    L.append("**Walk-forward windows (calendar-aligned):**")
    L.append("")
    L.append(table(
        [(w["window"], w["start"][:10], w["end"][:10], pct(w["return"]), pct(w["max_drawdown"]),
          w["trades"], w["live_tickers"], f"{w['breadth'] * 100:.0f}%") for w in p["windows"]],
        ["#", "start", "end", "return", "maxDD", "trades", "live", "breadth"]))
    L.append("")
    L.append(f"- **Worst window:** #{p['worst_window']['window']} "
             f"{p['worst_window']['start'][:10]}→{p['worst_window']['end'][:10]} "
             f"return {pct(p['worst_window']['return'])}, maxDD {pct(p['worst_window']['max_drawdown'])}.")
    L.append(f"- **Holdout (last window, read once):** return {pct(p['holdout_last_window']['return'])}, "
             f"breadth {p['holdout_last_window']['breadth'] * 100:.0f}%.")

    L.append("")
    L.append("**Cost robustness (Sharpe):**")
    L.append("")
    L.append(table([(k, num(v["sharpe"], 3), pct(v["total_return"]))
                    for k, v in p["cost_stress"].items()],
                   ["scenario", "sharpe", "total_return"]))

    a = p["adversarial"]
    L.append("")
    L.append("**Adversarial:**")
    L.append(f"- Remove best ticker (`{a['remove_best_ticker']['removed']}`): return becomes "
             f"{pct(a['remove_best_ticker']['total_return'])}, Sharpe {num(a['remove_best_ticker']['sharpe'], 2)}.")
    L.append(f"- Remove best trade — ledger (exact): PF {num(a['remove_best_trade_ledger']['pf_before'], 3)} → "
             f"{num(a['remove_best_trade_ledger']['pf_after'], 3)} "
             f"({a['remove_best_trade_ledger']['n_before']}→{a['remove_best_trade_ledger']['n_after']} trades).")
    e = a["remove_best_trade_engine"]
    L.append(f"- Remove best trade — engine: suppress entry over {e['holding_window'][0][:10]}→"
             f"{e['holding_window'][1][:10]}; specific trade present before={e['specific_trade_before']} "
             f"after={e['specific_trade_after']} (**{'removed' if e['specific_trade_removed'] else 'NOT removed'}**); "
             f"raw count {e['trades_before']}→{e['trades_after']} (a re-entering strategy can form a "
             f"replacement trade, so the count need not drop by one).")
    con = a["concentration"]
    L.append(f"- Concentration: best trade = {pct(con['best_trade_currency_share'])} of net P&L, "
             f"top-5 = {pct(con['top5_currency_share'])} (currency); scale-free best-trade "
             f"return-share {pct(con['best_trade_return_share'])}.")

    L.append("")
    L.append("**Per-year panel return:** " + ", ".join(f"{y}:{pct(v)}" for y, v in p["per_year"].items()))

L.append("")
L.append("## 2. Untested axes (what this study does NOT establish)")
L.append("- No funding/borrow cost (perps funding absent from the cache).")
L.append("- No maker/taker fee tiering beyond the 1×/2× sweep; no latency model.")
L.append("- Only the two shipped presets were evaluated (frozen) — the strategy's parameter "
         "neighbours were **not** swept; this is a re-validation, not a re-optimisation.")
L.append("- Panel is USDT-perp crypto only; equities/FX were excluded.")
L.append("- The engine's `Trade.entry_time` stamps the exit bar (a reporting quirk); the event "
         "ledger was used for holding windows, so this does not affect the results.")

L.append("")
L.append("## 3. Conditions where these presets should be disabled")
L.append("- Panel breadth per window falls below 50% (currently already at/below it).")
L.append("- Harsh-cost Sharpe (4× slippage + 2× fees) turns negative.")
L.append("- Two consecutive OOS windows negative.")
L.append("- Best-ticker removal turns the full-sample return negative.")
L.append("- Rolling 6-month panel drawdown exceeds the worst window in this study.")

L.append("")
L.append("## 4. Reproducibility")
L.append("- `harness.py` reads `quant/cache` (hashes in the manifest above) and writes `results.json`.")
L.append("- Command: `PYTHONPATH=. .venv/Scripts/python.exe deliverables/software-company/audit/walkforward/harness.py`")
L.append("- Engine version and all inputs are recorded in `results.json`.")

open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote REPORT.md")
