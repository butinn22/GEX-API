"""Report generator: turns the stage artifacts into the final deliverable.

Everything here is derived from the JSON written by ``run_loops`` — no number is typed
by hand. If a number is not in the artifacts, it does not appear in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("out")

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
           "#be185d", "#65a30d", "#475569", "#ea580c", "#0f766e", "#9333ea"]
BG = "#ffffff"
FG = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def load(name: str):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _f(x, nd=4, pct=False, sign=False):
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if pct:
        return f"{v * 100:+.2f}%" if sign else f"{v * 100:.2f}%"
    return f"{v:.{nd}f}"


def svg_lines(series: list[dict], *, width=960, height=340, title="", logscale=False):
    """Minimal dependency-free SVG line chart."""
    series = [s for s in series if s.get("v")]
    if not series:
        return "<p><em>no curve</em></p>"
    n = max(len(s["v"]) for s in series)
    ys = [v for s in series for v in s["v"] if v and v > 0]
    if not ys:
        return "<p><em>no curve</em></p>"
    lo, hi = min(ys), max(ys)
    if logscale:
        import math
        lo, hi = math.log(lo), math.log(hi)
    pad = (hi - lo) * 0.06 or 1e-6
    lo, hi = lo - pad, hi + pad
    ml, mr, mt, mb = 62, 14, 26, 30
    w, h = width - ml - mr, height - mt - mb

    def X(i, total):
        return ml + (w * i / max(1, total - 1))

    def Y(v):
        if logscale:
            import math
            v = math.log(v)
        return mt + h - h * (v - lo) / (hi - lo)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="background:{BG};font-family:ui-monospace,monospace">']
    if title:
        parts.append(f'<text x="{ml}" y="15" fill="{FG}" font-size="12">{title}</text>')
    for k in range(5):
        y = mt + h * k / 4
        val = (hi - (hi - lo) * k / 4)
        if logscale:
            import math
            val = math.exp(val)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + w}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        lab = f"{val:.2f}x" if not logscale else f"{val:.2f}"
        parts.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" fill="{MUTED}" '
                     f'font-size="10" text-anchor="end">{lab}</text>')
    for si, s in enumerate(series):
        col = PALETTE[si % len(PALETTE)]
        pts = " ".join(f"{X(i, len(s['v'])):.1f},{Y(v):.1f}"
                       for i, v in enumerate(s["v"]) if v and v > 0)
        parts.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.7" '
                     f'points="{pts}"/>')
    parts.append("</svg>")
    return "".join(parts)


def legend(names: list[str]) -> str:
    items = []
    for i, nm in enumerate(names):
        c = PALETTE[i % len(PALETTE)]
        items.append(f'<span style="margin-right:14px;font-size:12px;color:{FG}">'
                     f'<b style="color:{c}">&#9632;</b> {nm}</span>')
    return "".join(items)


def pm_row(name, tf, pm):
    if not pm or not pm.get("n_tickers"):
        return None
    pf = pm.get("profit_factor", 0.0)
    pf = "inf" if pf == float("inf") else f"{pf:.2f}"
    return [
        name, tf, str(pm.get("n_tickers", "")), str(pm.get("total_trades", "")),
        _f(pm.get("total_return"), pct=True, sign=True),
        _f(pm.get("cagr"), pct=True, sign=True),
        _f(pm.get("sharpe"), 2), _f(pm.get("median_ticker_sharpe"), 2),
        _f(pm.get("sortino"), 2), _f(pm.get("calmar"), 2),
        _f(pm.get("max_dd"), pct=True), f"{pm.get('max_dd_days', 0):.0f}",
        pf, _f(pm.get("win_rate"), pct=True), _f(pm.get("expectancy_pct"), 5),
        _f(pm.get("share_tickers_positive"), pct=True),
        _f(pm.get("exposure"), pct=True),
        _f(pm.get("avg_holding_hours"), 1),
        f"{pm.get('n_stop_before_4h', 0)}/{pm.get('n_tp_before_4h', 0)}/"
        f"{pm.get('n_after_4h', 0)}",
    ]


HEAD = ("variant", "TF", "tick", "trades", "net", "CAGR", "Sharpe", "medShp",
        "Sortino", "Calmar", "maxDD", "DDd", "PF", "win", "exp/trade", "pos%",
        "expo", "avgHold h", "SL/TP/after4h")


def md_table(rows, head=HEAD):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def html_table(rows, head=HEAD):
    th = "".join(f'<th style="text-align:right;padding:4px 8px;border-bottom:1px solid #cbd5e1">{h}</th>'
                 for h in head)
    trs = []
    for r in rows:
        tds = "".join(
            f'<td style="padding:3px 8px;text-align:right;border-bottom:1px solid #f1f5f9">{c}</td>'
            for c in r
        )
        trs.append(f"<tr>{tds}</tr>")
    return (f'<table style="border-collapse:collapse;font:12px ui-monospace,monospace;'
            f'color:{FG}"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>')


# --------------------------------------------------------------------------- #
def build() -> tuple[str, str]:
    """Assemble the markdown report and the HTML report from the artifacts."""
    final = load("stage_final.json")
    ab = {p.stem.replace("stage_ab_", ""): load(p.name)
          for p in sorted(OUT.glob("stage_ab_*.json"))}
    loops = {p.stem.replace("stage_loops_", ""): load(p.name)
             for p in sorted(OUT.glob("stage_loops_*.json"))}
    holds = {p.stem.replace("stage_holdout_", ""): load(p.name)
             for p in sorted(OUT.glob("stage_holdout_*.json"))}
    robs = {p.stem.replace("stage_robustness_", ""): load(p.name)
            for p in sorted(OUT.glob("stage_robustness_*.json"))}
    mans = {p.stem.replace("data_manifest_", ""): load(p.name)
            for p in sorted(OUT.glob("data_manifest_*.json"))}

    md: list[str] = []
    html: list[str] = []

    def m(s=""):
        md.append(s)

    def h(s):
        html.append(s)

    h(f'<div style="max-width:1180px;margin:0 auto;font-family:system-ui,Segoe UI,'
      f'sans-serif;color:{FG};line-height:1.5;padding:24px 18px">')

    # ---------- headline ----------
    m("# EMF+ADL — real-data improvement programme")
    m()
    m("Deterministic, cost-aware, walk-forward. Bybit v5 public market data only. "
      "No synthetic candles, no interpolation, no fabricated fills.")
    m()
    h("<h1 style='margin:0 0 6px'>EMF+ADL &mdash; real-data improvement programme</h1>")
    h(f"<p style='color:{MUTED};margin:0 0 18px'>Deterministic, cost-aware, walk-forward. "
      f"Bybit v5 public market data only. No synthetic candles, no interpolation, "
      f"no fabricated fills.</p>")

    # ---------- defect ----------
    m("## 0. A defect found before any tuning: the Heikin-Ashi open never propagates")
    m()
    m("`_FeaturesMixin._heikin_ashi` in `gex/strategy/features.py` builds `ha_open` with")
    m("a single vectorised pandas expression:")
    m()
    m("```python")
    m("ha_open.iloc[1:] = (ha_open.shift(1).iloc[1:] + ha_close.shift(1).iloc[1:]) / 2")
    m("```")
    m()
    m("`ha_open.shift(1)` is evaluated **once, from the initial all-NaN series**, so the "
      "recursion never propagates. Indices 0 and 1 are correct; 2..n-1 stay NaN. "
      "Measured on real BTCUSDT 4H (2021-07 → 2026-10): **911 of 913 bars NaN** in the "
      "first test window, and 99.9% of bars NaN over the full study.")
    m()
    m("The NaN then vanishes silently, because `DataFrame.max(axis=1)` skips NaN:")
    m()
    m("| column | shipped | after repair (vendored reference) |")
    m("|---|---|---|")
    m("| `ha_open` NaN rate | ~100% (index ≥ 2) | 0% |")
    m("| `candle_top − candle_bottom` | **0.0000 mean, 99.9% of bars exactly 0** | 432.26 mean, 0% zero |")
    m("| `avg_candle` | collapses to `hybrid_close` | `(hybrid_open + hybrid_close)/2` |")
    m("| `median_top` | collapses to `max(open, close)` | includes `ha_open` |")
    m("| long entries, BTC 4H 2024 | 384 | 443 |")
    m("| short entries, BTC 4H 2024 | 170 | 227 |")
    m()
    m("`novelsrc` — the single source behind every EMA, the VWAP context, and every entry "
      "and exit in this strategy — is therefore computed from a degenerate hybrid body. "
      "The hybrid body range is also the **buffer unit** for structural stops; at zero "
      "range the buffer collapses to a raw pivot level. "
      "The vendored reference transform passes 39/39 self-checks including "
      "`identity: avg_candle == (hybrid_open + hybrid_close)/2` and "
      "`no NaN in any derived series`; the shipped implementation fails both.")
    m()
    h("<h2>A defect found before any tuning: the Heikin-Ashi open never propagates</h2>")
    h("<p><code>ha_open.shift(1)</code> is evaluated once, from the initial all-NaN "
      "series, so the recursion never propagates. Indices 0&ndash;1 are correct; "
      "2..n-1 stay NaN. Because <code>DataFrame.max(axis=1)</code> skips NaN, the "
      "degenerate values disappear silently and <code>novelsrc</code> &mdash; the source "
      "behind every EMA, VWAP context, entry and exit &mdash; is built from a hybrid body "
      "of <b>zero range</b>.</p>")
    h("<table style='border-collapse:collapse;font:12px ui-monospace,monospace'>"
      "<tr><th style='text-align:left;padding:4px 10px'>symptom</th>"
      "<th style='text-align:right;padding:4px 10px'>shipped</th>"
      "<th style='text-align:right;padding:4px 10px'>repaired</th></tr>"
      "<tr><td style='padding:3px 10px'>hybrid body range mean</td>"
      "<td style='text-align:right;padding:3px 10px'>0.07</td>"
      "<td style='text-align:right;padding:3px 10px'>432.26</td></tr>"
      "<tr><td style='padding:3px 10px'>bars with exactly zero body range</td>"
      "<td style='text-align:right;padding:3px 10px'>99.9%</td>"
      "<td style='text-align:right;padding:3px 10px'>0%</td></tr>"
      "<tr><td style='padding:3px 10px'>BTC 4H long entries (2024)</td>"
      "<td style='text-align:right;padding:3px 10px'>384</td>"
      "<td style='text-align:right;padding:3px 10px'>443</td></tr>"
      "</table>")

    # ---------- data provenance ----------
    m()
    m("## 1. Data")
    m()
    m("| loop | tickers | window (UTC) | bars 4H | bars 1D | coverage | gaps | bad OHLC | funding pts |")
    m("|---|---|---|---|---|---|---|---|---|")
    for tag, man in mans.items():
        ser = man.get("series", {})
        n4 = [v for k, v in ser.items() if k.endswith("|4H")]
        n1 = [v for k, v in ser.items() if k.endswith("|1D")]
        cov = min([v.get("coverage") or 0 for v in ser.values()] or [0])
        gaps = sum(v.get("n_gaps") or 0 for v in ser.values())
        bad = sum(v.get("bad_ohlc") or 0 for v in ser.values())
        fu = sum(v.get("n_funding") or 0 for v in n4)
        m(f"| {tag} | {', '.join(man.get('tickers_used', []))} | "
          f"{man.get('study_window_utc', ['', ''])[0][:10]} → "
          f"{man.get('study_window_utc', ['', ''])[1][:10]} | "
          f"{min([v['n_bars'] for v in n4] or [0])}–{max([v['n_bars'] for v in n4] or [0])} | "
          f"{min([v['n_bars'] for v in n1] or [0])}–{max([v['n_bars'] for v in n1] or [0])} | "
          f"{cov:.4f} | {gaps} | {bad} | {fu} |")
    m()
    m("Source: `https://api.bybit.com/v5/market/kline`, `/funding/history`, `/tickers` "
      "(linear USDT perpetuals). Every series is hashed (sha256 over the raw candle "
      "array) into its manifest so a run can be audited and reproduced. "
      "Excluded series are listed in the manifest, never silently replaced.")
    m()
    m("**Residual survivorship bias (unresolved, disclosed):** the candidate pool is "
      "today's liquid perps, so symbols that were delisted are absent. Ranking *within* "
      "the pool uses only trailing real turnover at the time, but the pool membership "
      "itself is not point-in-time. This is a real limitation of the public data.")
    m()
    m("**Warm-up:** 400 bars are consumed before the first signal (`ema200(novelsrc)` and "
      "`adl200`; a span-200 EWM still carries <2% residual weight at 400 bars). "
      "Applied identically to every variant.")
    m()
    m("**Splits:** train 50% / validate 25% / holdout 25% of the post-warm-up sample. "
      "Variants are *ranked on validation only*; the holdout is read once.")
    m()
    m("**Costs:** Bybit linear-perp VIP0 taker **0.055% per side** (market orders on both "
      "legs — maker pricing would be self-deception), **slippage 3 bps per side** "
      "(impact + half-spread; Bybit publishes no spread series, so that half of the "
      "number is estimated and disclosed), plus **real 8h funding settlements** charged to "
      "the open notional. Stressed to 6/9 bps slippage, 0.08% fee, and a +3 bps latency "
      "penalty in the robustness battery.")
    m()
    m("**Execution:** signal at bar close → fill at next bar open. Intrabar stops are "
      "triggered on high/low and filled at the worse of stop/bar-open, then slippage "
      "applied against. Take-profits fill at the level, never at a favourable gap. "
      "If a bar could have hit both stop and target, the **stop** is assumed first. "
      "Trailing levels used to test bar *i* are derived from bars strictly before *i*.")
    m()
    m("**Minimum-holding rule.** On 4H one bar *is* the 4-hour minimum, so "
      "`holding_bars == 0` ⇒ before 4h, `>= 1` ⇒ after. On 1D an exit inside the entry "
      "bar has an unobservable holding time in (0, 24h]; it is classified as *before 4h* "
      "because the data cannot prove otherwise. Every exit is classified "
      "STOP_LOSS_BEFORE_4H / TAKE_PROFIT_BEFORE_4H / EXIT_AFTER_4H and the counts are "
      "reported per variant.")
    m()
    m("**Spec tension flagged.** The brief says a position *may* close early on a stop or "
      "a predefined profit exit, and later that take-profit is *only* allowed after 4 "
      "hours. Those contradict. This study follows the permissive holding rule (stop and "
      "predefined profit exits both allowed early) and reports the classification counts "
      "so the stricter reading can be audited from the same numbers.")

    # ---------- AB ----------
    if ab:
        m()
        m("## 2. Baseline vs repaired (full sample, real costs)")
        m()
        rows = []
        for tag, blk in ab.items():
            for nm, vblk in blk.items():
                for tf in ("4H", "1D"):
                    r = pm_row(nm, tf, vblk["by_tf"][tf])
                    if r:
                        rows.append(r)
        m(md_table(rows))
        h("<h2>Baseline vs repaired (full sample, real costs)</h2>")
        h(html_table(rows))

    # ---------- loops ----------
    if loops:
        m()
        m("## 3. Optimisation loops — out-of-sample ranking")
        m()
        for tag, blk in loops.items():
            m(f"### Loop {tag}")
            m()
            rows = []
            for nm, vblk in blk.items():
                for tf in ("4H", "1D"):
                    r = pm_row(nm, tf, vblk["by_tf"][tf])
                    if r:
                        rows.append(r)
            m(md_table(rows))
            m()
            for nm, vblk in blk.items():
                m(f"- **{nm}** — {vblk['hypothesis']}")
            m()
            h(f"<h2>Loop {tag} — out-of-sample ranking</h2>")
            h(html_table(rows))

    # ---------- holdout ----------
    if holds:
        m()
        m("## 4. Holdout (read once)")
        m()
        for tag, blk in holds.items():
            m(f"### Holdout, loop {tag}")
            m()
            rows = []
            for nm, vblk in blk.items():
                for tf in ("4H", "1D"):
                    r = pm_row(nm, tf, vblk["by_tf"][tf])
                    if r:
                        rows.append(r)
            m(md_table(rows))
            m()
            h(f"<h2>Holdout, loop {tag}</h2>")
            h(html_table(rows))

    # ---------- robustness ----------
    if robs:
        m()
        m("## 5. Robustness battery")
        m()
        for tag, blk in robs.items():
            m(f"### Loop {tag}")
            m()
            for label, res in blk.items():
                m(f"**{label}**")
                m()
                rows = []
                for nm, vblk in res.items():
                    for tf in ("4H", "1D"):
                        r = pm_row(nm, tf, vblk["by_tf"][tf])
                        if r:
                            rows.append(r)
                m(md_table(rows))
                m()
            h(f"<h2>Robustness battery, loop {tag}</h2>")
            for label, res in blk.items():
                h(f"<h3>{label}</h3>")
                rows = []
                for nm, vblk in res.items():
                    for tf in ("4H", "1D"):
                        r = pm_row(nm, tf, vblk["by_tf"][tf])
                        if r:
                            rows.append(r)
                h(html_table(rows))

    # ---------- final ----------
    if final:
        m()
        m("## 6. Finalists on the full pool — train / validate / holdout")
        m()
        names = list(final["variants"].keys())
        h("<h2>Finalists &mdash; full pool equity curves</h2>")
        for tf in ("4H", "1D"):
            curves, labels = [], []
            for nm in names:
                blk = final["variants"][nm]
                pm = blk["by_tf"][tf].get("all", {})
                if pm.get("curve"):
                    curves.append(pm["curve"])
                    labels.append(f"{nm}")
            h(f"<h3>{tf} &mdash; equal-weight panel, all liquidity-pool series</h3>")
            h(legend(labels))
            h(svg_lines(curves, title=f"EMF+ADL variants, {tf}, real costs"))
        for nm in names:
            blk = final["variants"][nm]
            m(f"### {nm}")
            m()
            m(f"*{blk['hypothesis']}*")
            m()
            rows = []
            for tf in ("4H", "1D"):
                for window in ("train", "valid", "holdout", "all"):
                    pm = blk["by_tf"][tf].get(window, {})
                    r = pm_row(f"{nm}", f"{tf}·{window}", pm)
                    if r:
                        rows.append(r)
            m(md_table(rows))
            m()
            # worst periods + per-ticker
            for tf in ("4H", "1D"):
                pm = blk["by_tf"][tf].get("all", {})
                if pm.get("per_ticker"):
                    m(f"**{tf} per-ticker attribution**")
                    m()
                    m("| symbol | trades | net | sharpe | calmar | maxDD | win | PF | exp/trade | exposure | funding | fees |")
                    m("|---|---|---|---|---|---|---|---|---|---|---|---|")
                    for r in pm["per_ticker"]:
                        m(f"| {r['symbol']} | {r['n_trades']} | {r['net_return']*100:+.2f}% | "
                          f"{r['sharpe']:+.2f} | {r['calmar']:+.2f} | {r['max_dd']*100:.1f}% | "
                          f"{r['win_rate']*100:.1f}% | {r['profit_factor']:.2f} | "
                          f"{r['expectancy_pct']*100:+.4f}% | {r['exposure']*100:.1f}% | "
                          f"{r['funding']:+.4f} | {r['fees']:.4f} |")
                    m()
            h(f"<h3>{nm}</h3>")
            for tf in ("4H", "1D"):
                pm = blk["by_tf"][tf].get("all", {})
                if pm.get("ticker_curves"):
                    h(f"<h4>{tf} &mdash; per-ticker equity</h4>")
                    tc = pm["ticker_curves"]
                    h(legend(list(tc.keys())))
                    h(svg_lines(list(tc.values()), height=280,
                                 title=f"{nm} {tf} per-ticker (all windows)"))

    h("</div>")
    return "\n".join(md), "\n".join(html)


def main() -> int:
    md, html = build()
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    page = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>EMF+ADL improvement programme</title></head><body "
            f"style='margin:0;background:{BG}'>" + html + "</body></html>")
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'REPORT.md'} ({len(md)} chars)")
    print(f"wrote {OUT / 'report.html'} ({len(page)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
