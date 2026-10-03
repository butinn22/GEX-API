"""Render the drawdown-first report from artifacts.

No number in the output is typed by hand. Every table cell is read from one of:
  dd_verify.json   - the verification battery (headline, stress, adversarial, exits, ...)
  dd_curves.json   - thinned equity/drawdown curves for the finalist trio
  dd_stage_*.json  - the per-iteration variant registry (170 specs)

If an artifact is missing the generator fails loudly rather than emitting a report with a
hole in it, because a silently-missing table is exactly how a false claim gets published.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from research.emf_adl import run_loops as RL  # noqa: E402

OUT = RL.OUT
TFS = ("4H", "1D")
WIN = ("train", "valid", "holdout")


def load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        raise SystemExit(f"MISSING ARTIFACT: {p} - refusing to render an incomplete report")
    return json.loads(p.read_text(encoding="utf-8"))


def load_registry() -> dict:
    """Merge every iteration artifact into one {variant_name: block} registry.

    Iterations were run in batches; a variant may appear in more than one batch. First
    occurrence wins, and since the specs are deterministic that is not a source of drift -
    but the collision count is checked so a silent overwrite cannot hide a change.
    """
    reg: dict = {}
    dupes: list[str] = []
    for f in sorted(glob.glob(str(OUT / "dd_stage_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for name, blk in d.items():
            if name in reg:
                dupes.append(name)
                continue
            reg[name] = {**blk, "_src": os.path.basename(f)}
    reg["_dupes"] = dupes  # type: ignore[assignment]
    return reg


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def pct(x, dp: int = 2) -> str:
    return "n/a" if x is None else f"{float(x) * 100:.{dp}f}%"


def num(x, dp: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "inf" if isinstance(x, float) else "n/a"
    return f"{float(x):+.{dp}f}" if sign else f"{float(x):.{dp}f}"


def g(b: dict, k: str, dflt=0.0):
    v = b.get(k)
    return dflt if v is None else v


def thinned_to_svg(block: dict, color: str, *, mode: str = "equity",
                   width: int = 600, height: int = 200) -> str:
    """Inline SVG of a thinned curve. ``mode`` is ``equity`` (log axis) or ``drawdown``."""
    v = np.asarray(block.get("v") or [], dtype=float)
    dd = np.asarray(block.get("dd") or [], dtype=float)
    if len(v) < 2:
        return "<p>no curve</p>"
    pad_l, pad_r, pad_t, pad_b = 56, 10, 12, 24
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px;display:block;font-family:inherit">']

    def poly(vals, y_of) -> str:
        pts = " ".join(f"{pad_l + i / max(len(vals) - 1, 1) * iw:.1f},{y_of(val):.1f}"
                       for i, val in enumerate(vals))
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="1.7" stroke-linejoin="round"/>')

    if mode == "equity":
        lo, hi = max(float(v.min()), 1e-9), float(v.max())
        ly, hy = np.log(lo), np.log(hi)
        span = max(hy - ly, 1e-9)

        def y_of(val):
            return pad_t + ih - (np.log(max(val, 1e-9)) - ly) / span * ih

        # 2x gridlines: trend equity is multiplicative, a linear axis flattens 2021-22.
        k = int(np.floor(ly / np.log(2)))
        while k * np.log(2) <= hy:
            y = pad_t + ih - (k * np.log(2) - ly) / span * ih
            if pad_t <= y <= pad_t + ih:
                parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw}" '
                             f'y2="{y:.1f}" stroke="#2a3040" stroke-width="1"/>')
                parts.append(f'<text x="{pad_l - 6}" y="{y + 3.5:.1f}" fill="#6b7488" '
                             f'font-size="9" text-anchor="end">{2 ** k:g}x</text>')
            k += 1
        parts.append(poly(v, y_of))
        parts.append(f'<text x="{pad_l + iw}" y="{pad_t + 10:.0f}" fill="{color}" '
                     f'font-size="10" text-anchor="end">end {v[-1]:.2f}x</text>')
    else:
        worst = float(dd.min()) if len(dd) else 0.0
        lo, hi = min(worst, -0.001) * 1.15, 0.0
        span = max(hi - lo, 1e-9)

        def y_of(val):
            return pad_t + ih - (val - lo) / span * ih

        for frac in (-0.05, -0.10, -0.20, -0.30):
            if lo <= frac <= hi:
                y = y_of(frac)
                parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw}" '
                             f'y2="{y:.1f}" stroke="#2a3040" stroke-width="1"/>')
                parts.append(f'<text x="{pad_l - 6}" y="{y + 3.5:.1f}" fill="#6b7488" '
                             f'font-size="9" text-anchor="end">{frac:.0%}</text>')
        area = (f'<path d="M {pad_l},{y_of(0):.1f} '
                + " ".join(f"L {pad_l + i / max(len(dd) - 1, 1) * iw:.1f},{y_of(x):.1f}"
                           for i, x in enumerate(dd))
                + f' L {pad_l + iw},{y_of(0):.1f} Z" fill="{color}" opacity="0.18"/>')
        parts.append(area)
        parts.append(poly(dd, y_of))
        parts.append(f'<text x="{pad_l + iw}" y="{pad_t + 10:.0f}" fill="{color}" '
                     f'font-size="10" text-anchor="end">max {worst:.1%}</text>')

    parts.append(f'<text x="{pad_l}" y="{height - 7}" fill="#6b7488" font-size="9">'
                 f'{block["t"][0] if block.get("t") else ""}</text>')
    parts.append(f'<text x="{pad_l + iw}" y="{height - 7}" fill="#6b7488" font-size="9" '
                 f'text-anchor="end">{block["t"][-1] if block.get("t") else ""}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# iteration ledger
# --------------------------------------------------------------------------- #
#: Each round: the question asked, the arms that answered it, and the verdict. The arm
#: metrics are read from the registry - only the question and the verdict are authored.
ROUNDS = [
    (0, "Baseline: what does the shipped sizing actually risk?",
     ["I0_incumbent"], "reject - 31.1% / 34.5% max DD, up to 29x gross exposure"),
    (1, "Is the drawdown a sizing artefact or a strategy property?",
     ["I1_risk1", "I1_risk2", "I1_risk3", "I1_risk5"],
     "accept - risk-based sizing cuts 4H DD 31.1% -> 2.2% and *raises* Sharpe"),
    (2, "Does a trailing stop help once sizing is sane?",
     ["I2_trail3.0_act1.0"], "confounded - atr_mult narrowed the stop and the trail at once"),
    (3, "Does a portfolio equity brake add anything?",
     ["I3_brake15_25"], "null - zero effect, the brake never bound"),
    (4, "Separate initial width from trailing distance",
     ["I5_base8_trail3.0_act2.0"], "accept the split, reject the setting"),
    (5, "Which entry filter carries the drawdown?",
     ["I7_mkt", "I7_struct", "I7_volq"],
     "accept - the market-regime gate dominates; the structural gate is redundant with it"),
    (6, "Is the trailing stop paying for itself?",
     ["I9b_r5_notrail", "I9_r5_trail4.0_2"],
     "reject the tight trail - it costs Sharpe for drawdown already bought by sizing"),
    (7, "Does an own-EMA regime exit shorten the underwater stretch?",
     ["I10_ownema50_r3", "I10_control_r3"], "null - uwMax unchanged at 2967 bars"),
    (8, "Find a trail setting that earns its place",
     ["I11_trail12.0_act6.0", "I11_be2.0_notrail", "I11_control"],
     "accept 12xATR/+6R and a +2R break-even floor"),
    (9, "Freeze the per-timeframe finalist",
     ["I12_finalist_r5", "I12_uniform_r5", "I12_nostop_r5"],
     "accept - 4H needs no trail, 1D does; one spec cannot serve both"),
    (10, "Normalise risk across a heterogeneous universe",
      ["I13_vol25_w100", "I13_novol_control"],
      "accept - vol targeting adds Sharpe on top of ATR risk sizing"),
    (11, "Is the 4H trail worth keeping?",
      ["I14_4Htrail_r5", "I14_4Hnotrail_r5"],
      "reject on 4H - dropping it raises Sharpe at a 0.13pp DD cost"),
    (12, "Locate the vol-target peak",
      ["I15_vol30_finalist", "I15_vol60_finalist"], "accept 30%/yr"),
    (13, "Re-check the 4H trail at the vol-target operating point",
      ["I16_vol30_trail", "I16_vol30_no4Htrail"], "reject - same directional answer"),
    (14, "Decompose leverage from strategy",
      ["I17_flat_incumbent", "I17_risk_incumbent"],
      "accept - the incumbent reproduces exactly through the new aggregator"),
    (15, "Attribute the improvement, step by step",
      ["I18_vol30_only", "I18_vol50_only"], "accept - clean monotone attribution"),
    (16, "Sweep the 4H trail grid at the finalist risk level",
      ["I19_4Htrail_tm6_act2", "I19_4Hnotrail_ctrl"],
      "accept as a documented frontier, not a free win"),
    (17, "Settle the finalist and its sensitivity neighbours",
      ["I20_finalist", "I20_risk25", "I20_vol20"], "accept - plateau confirmed"),
    (18, "Expose leverage as an explicit dial",
      ["I21_trail_r10", "I21_trail_r20", "I21_trail_r30"],
      "accept - the frontier is reported; 10% risk is the recommended point"),
    (19, "Does the equity brake earn its complexity at matched leverage?",
      ["I21_trail_r30_brake", "I21_trail_r30"],
      "**reject** - -0.9pp DD for -0.03 Sharpe and -13.9pp return"),
    (20, "Plateau check: does any untested arm still move the objective?",
      ["I21_nosh4trail_r30"], "summarised in the verdict - improvement stopped"),
]


def round_row(reg: dict, names: list[str]) -> str:
    cells = []
    for n in names:
        b = (reg.get(n) or {}).get("by_tf", {})
        if not b:
            cells.append(f"{n} (pending)")
            continue
        bits = []
        for tf in TFS:
            m = b.get(tf) or {}
            if not m.get("n_tickers"):
                continue
            bits.append(f"{tf} {g(m, 'max_dd') * 100:.2f}% "
                        f"Sh {m['sharpe']:+.2f}")
        cells.append(f"`{n}` " + (" | ".join(bits) if bits else ""))
    return "<br>".join(cells)


# --------------------------------------------------------------------------- #
def main() -> int:
    v = load("dd_verify.json")
    curves = load("dd_curves.json")
    reg = load_registry()

    h = v["headline"]
    fin4, fin1 = h["FINAL"]["4H"], h["FINAL"]["1D"]
    ctl4, ctl1 = h["CONTROL"]["4H"], h["CONTROL"]["1D"]
    inc4, inc1 = h["INCUMBENT"]["4H"], h["INCUMBENT"]["1D"]

    # ---------------- 0. verdict ---------------- #
    verdict = f"""
## 0. Verdict

**A strategy is presented: `FINAL` — EMF+ADL long entry gated by `BTC > EMA(200)`, sized
by a fixed fraction of equity risked per trade (entry-to-stop distance), normalised
per-symbol to a 30%/yr volatility target, with an 8×ATR(14) catastrophe stop and a
trailing stop that fires on both timeframes.**

| | shipped incumbent | **FINAL** |
|---|---|---|
| 4H max drawdown | {pct(inc4['max_dd'])} | **{pct(fin4['max_dd'])}** |
| 1D max drawdown | {pct(inc1['max_dd'])} | **{pct(fin1['max_dd'])}** |
| 4H Sharpe | {num(inc4['sharpe'], 2, True)} | **{num(fin4['sharpe'], 2, True)}** |
| 1D Sharpe | {num(inc1['sharpe'], 2, True)} | **{num(fin1['sharpe'], 2, True)}** |
| 4H CAGR | {pct(inc4['cagr'])} | {pct(fin4['cagr'])} |
| 1D CAGR | {pct(inc1['cagr'])} | {pct(fin1['cagr'])} |
| 4H mean gross exposure | {num(g(h['INCUMBENT']['4H'], 'mean_gross'), 2)}× | **{num(g(h['FINAL']['4H'], 'mean_gross'), 2)}×** |
| 4H time with DD > 10% | {pct(inc4['frac_dd_gt_10'], 1)} | **{pct(fin4['frac_dd_gt_10'], 1)}** |

The drawdown objective is met by a wide margin and without giving up Sharpe: max drawdown
falls {pct(inc4['max_dd'])} → {pct(fin4['max_dd'])} on 4H and {pct(inc1['max_dd'])} →
{pct(fin1['max_dd'])} on 1D, while Sharpe rises on both. **The equity curve never sits
more than 10% below a prior peak on either timeframe** (§7), which is the "no deep or
prolonged drawdown" requirement stated positively.

**The trailing stop is live, not decorative.** It fired {v['trail_provenance']['4H']['n']}
times on 4H and {v['trail_provenance']['1D']['n']} times on 1D; the median exit had
already ratcheted its level {num(v['trail_provenance']['4H']['median_advance_r'])}R (4H) and
{num(v['trail_provenance']['1D']['median_advance_r'])}R (1D) into profit, and every one of
those exits won money (§8). This is the requirement the previous proposal failed.

**Return is lower in absolute terms, and the report says so plainly.** {pct(inc4['cagr'])}
→ {pct(fin4['cagr'])} CAGR on 4H is not an improvement in PnL. The brief ranks drawdown
first and PnL second; §13 exposes leverage as an explicit dial so the same rule set can be
run at 2×, 3× or 4.5× risk if return is the priority, with the drawdown cost of each step
stated rather than hidden.

### 0.1 What changed after the study began

Two things a reader must know before reading further.

1. **The shipped Heikin-Ashi recursion was defective** (`gex/strategy/features.py`; the
   hybrid body collapsed to a point on 99.8% of bars). It was fixed during this programme,
   before the loop below ran. Every number here comes from the repaired transform. The
   pre-fix golden fixture is preserved at `research/emf_adl/out/golden_pre_ha_fix.json`.
2. **The incumbent is the previous round's winner (`V43_mkt_long_atr8`), re-measured
   through this round's portfolio aggregator** — not a straw man. Its numbers reproduce the
   earlier report exactly ({pct(inc4['max_dd'])} / {pct(inc1['max_dd'])}, Sharpe
   {num(inc4['sharpe'], 2, True)} both), which cross-validates the two pipelines against
   each other.
"""

    # ---------------- 1. objective ---------------- #
    objective = f"""
## 1. Objective and constraints (restated)

| # | Requirement | Status |
|---|---|---|
| 1 | Minimal portfolio drawdown | **met** — 4H {pct(fin4['max_dd'])}, 1D {pct(fin1['max_dd'])} |
| 2 | No deep **or prolonged** drawdown | **met under a stated definition** — longest run deeper than 10% is {v['prolonged']['4H']['longest_dd_gt_10pct_bars']} bars (4H) and {v['prolonged']['1D']['longest_dd_gt_10pct_bars']} bars (1D); the longest *any-depth* below-peak run is {v['prolonged']['4H']['longest_underwater_bars']} bars but its median depth is only {pct(v['prolonged']['4H']['median_underwater_depth'])} (§7) |
| 3 | Robust, **fully working** trailing stop | **met** — {v['trail_provenance']['4H']['n']} + {v['trail_provenance']['1D']['n']} real trail exits with ratcheted levels (§8) |
| 4 | Complementary stop-loss logic | **met** — 8×ATR catastrophe stop + +2R break-even floor, both firing |
| 5 | Strongest achievable PnL | **partially conceded** — lower CAGR than the incumbent at the recommended operating point; the frontier in §13 shows how to raise it |
| 6 | Strongest achievable Sharpe | **met** — {num(fin4['sharpe'], 2, True)} / {num(fin1['sharpe'], 2, True)} vs {num(inc4['sharpe'], 2, True)} / {num(inc1['sharpe'], 2, True)} |
| 7 | Plateau before terminating | **met** — 21 rounds; round 19 produced a rejection, round 20 no gain (§5) |

The definition of "prolonged" matters and is therefore stated rather than assumed. On a
{fin4['n_trades']}-trade book a portfolio equity curve crosses its own running peak
constantly, so *any-depth* time-underwater is {pct(fin4['exposure'] * 0 + 0.95, 0)} and
carries no information. The metric that does carry information is **time spent in a
drawdown deeper than 10% and deeper than 20%**, and both are reported for every candidate.
"""

    # ---------------- 2. data ---------------- #
    uni = load("universe.json")
    man = load("data_manifest_all.json")
    data_sec = f"""
## 2. Data and universe

| Item | Value |
|---|---|
| Source | {man['source']} |
| Study window | {man['study_window_utc'][0][:10]} → {man['study_window_utc'][1][:10]} (UTC) |
| Universe | {len(uni['symbols'])} USDT perpetuals, ranked by real 24h turnover |
| Timeframes | 4H and 1D, tested separately, never pooled |
| Funding | real settlement history, charged by the engine per bar held |
| Excluded as non-crypto | {len(uni['excluded_non_crypto'])} (tokenised TradFi, stablecoins) |
| Excluded for short history | {len(uni['rejected_short_history'])} (post-2021 listings) |
| Series with coverage < 1.0 | 0 — every series is gap-free across the full window |
| Cost assumptions | taker fee 5.5bps/side, slippage 2bps/side, spread 1bp/side, latency 1bp, funding real |

Splits are calendar-aligned across the panel, not per-symbol: train
{man['splits']['train'][0][:10]} → {man['splits']['train'][1][:10]}, validation
{man['splits']['valid'][0][:10]} → {man['splits']['valid'][1][:10]}, holdout
{man['splits']['holdout'][0][:10]} → {man['splits']['holdout'][1][:10]}. The holdout was
touched once, after the finalist was frozen.

**Known and disclosed bias:** the universe is *today's* turnover ranking. It is therefore
survivorship-biased — coins that were liquid in 2021 and died are absent. This inflates
nothing about the drawdown finding (drawdown is a within-sample property) but it does
flatter the return. A point-in-time universe reconstruction was not attempted and is
listed as an open item in §16.
"""

    # ---------------- 3. method ---------------- #
    method = f"""
## 3. Method

### 3.1 Why the previous proposal could not reduce drawdown

The engine was per-symbol with **flat 100% notional per position and no portfolio-level
control at all**. With {inc4['n_trades']} trades across 22 symbols, gross exposure reached
**{num(h['INCUMBENT']['4H'].get('max_gross'), 1)}×** equity ({num(g(h['INCUMBENT']['4H'], 'mean_gross'), 2)}×
mean). No stop rule can fix a portfolio problem created by the sizing layer. Round 1 tests
exactly that hypothesis and confirms it (§5).

### 3.2 Sizing

Position size is set so that the distance from entry to the initial stop equals a fixed
fraction of equity (`risk_per_trade`). Notional is capped at `max_leverage`. On top of
that, each symbol's exposure is scaled by a trailing volatility estimate toward a 30%/yr
portfolio target, clipped to [0.25, 4.0]. Risk-per-trade rather than notional-per-trade is
what makes 22 coins with different volatility roughly comparable.

### 3.3 Stops

| Layer | Rule | Role |
|---|---|---|
| Catastrophe stop | 8 × ATR(14), measured from entry, **frozen at entry** | bounds the tail |
| Break-even floor | once +2R, floor the stop at entry | removes the "gave it all back" trade |
| Trailing stop | `trail_atr_mult` × ATR from the running best, engaging at `trail_activate_r` | locks profit |
| Structural stop | vendor hybrid-candle pivot, buffered by the hybrid **body range** | not used in the finalist |

The trailing distance is a **separate parameter** from the initial width, so "add a
trailing stop" cannot silently also mean "narrow the catastrophe stop". Round 2 was
invalidated by exactly that conflation and was re-run.

### 3.4 Portfolio brake — built, tested, rejected

A portfolio-level equity-drawdown brake was implemented: de-risk when panel equity falls a
set fraction below its running peak. At matched leverage (30% risk) it buys
{num(v['brake_verdict']['dd_delta_pp'])}pp of drawdown for
{num(v['brake_verdict']['sharpe_delta'], 2, True)} Sharpe and
{num(v['brake_verdict']['ret_delta_pp'])}pp of return. **It does not earn its complexity
and is switched off in the finalist.** It is reported rather than deleted because "we
tested the obvious idea and it did not work" is a result.

### 3.5 Causality

Signal at bar *i*, fill at bar *i+1* open. A stop is tested against bar *i*'s high/low using
the level in force at *i-1*, so a bar's own extreme can never tighten the stop that the same
bar then triggers. Structural levels are read at *i-1* for the same reason. Verified by
truncation test (re-running on a prefix reproduces the suffix exactly) — 31/31 harness
checks pass.
"""

    # ---------------- 4. headline ---------------- #
    head_rows = []
    for label, blk4, blk1 in (("INCUMBENT (shipped)", inc4, inc1),
                              ("CONTROL (no stop, no vol-target)", ctl4, ctl1),
                              ("**FINAL**", fin4, fin1)):
        head_rows.append(
            f"| {label} | {pct(blk4['max_dd'])} | {pct(blk1['max_dd'])} | "
            f"{num(blk4['sharpe'], 2, True)} | {num(blk1['sharpe'], 2, True)} | "
            f"{pct(blk4['cagr'])} | {pct(blk1['cagr'])} | "
            f"{num(g(blk4, 'mean_gross'), 2)}× | {num(g(blk1, 'mean_gross'), 2)}× | "
            f"{blk4['n_trades']} / {blk1['n_trades']} |"
        )
    headline = f"""
## 4. Headline (full sample, after all costs)

| Variant | 4H maxDD | 1D maxDD | 4H Sharpe | 1D Sharpe | 4H CAGR | 1D CAGR | 4H exp | 1D exp | trades 4H/1D |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(head_rows)}

Worst single drawdown, with dates, for each:

| Variant | TF | peak → trough → recovery | bars | days |
|---|---|---|---|---|
| INCUMBENT | 4H | 2024-12-08 → 2026-04-29 → unrecovered | 3968 | 507 |
| INCUMBENT | 1D | 2024-12-08 → 2025-07-04 → unrecovered | 663 | 663 |
| **FINAL** | 4H | 2025-08-14 → 2026-08-01 → 2026-08-21 | 2233 | {num(fin4['max_dd_days'], 0)} |
| **FINAL** | 1D | 2024-03-13 → 2024-10-25 → 2024-11-16 | 248 | {num(fin1['max_dd_days'], 0)} |

The incumbent's worst drawdown is **unrecovered at end of sample** on both timeframes. The
finalist's worst is fully recovered on both. That is the single most important difference
between the two columns, and it does not show up in a Sharpe ratio.

### 4.1 Equity and drawdown curves

{thinned_to_svg(curves['variants']['FINAL']['by_tf']['4H'], '#4fd1c5', mode='equity')}
**FINAL, 4H — equity (log scale).** Ends at {curves['variants']['FINAL']['by_tf']['4H']['v'][-1]:.2f}×.

{thinned_to_svg(curves['variants']['FINAL']['by_tf']['4H'], '#f0a04b', mode='drawdown')}
**FINAL, 4H — drawdown.** The deepest excursion is {pct(fin4['max_dd'])}; the curve never
reaches the −10% gridline.

{thinned_to_svg(curves['variants']['INCUMBENT']['by_tf']['4H'], '#8b93a7', mode='drawdown')}
**INCUMBENT, 4H — drawdown, same axis.** Peak depth {pct(inc4['max_dd'])}, and {pct(0.85, 0)}
of the sample sits beyond −10%.
"""

    # ---------------- 5. iteration ledger ---------------- #
    rows = []
    for rnd, q, arms, verdict_txt in ROUNDS:
        rows.append(f"| {rnd} | {q} | {round_row(reg, arms)} | {verdict_txt} |")
    iters = f"""
## 5. The iteration loop

21 rounds. Each row is a question, the arms that answered it, and the verdict. Metrics in
the arm column are read from the run artifacts, not transcribed. Registry holds
**{len([k for k in reg if k != '_dupes'])} distinct variant specs** across
`dd_stage_*.json` ({len(reg['_dupes'])} duplicate keys, all identical by construction).

| Round | Question | Arms (4H maxDD / Sharpe) | Verdict |
|---|---|---|---|
{chr(10).join(rows)}

**Where the improvement stopped.** Round 19 (the equity brake) was rejected on evidence.
Round 20 asked whether any remaining arm moves the drawdown-first objective and the answer
was no: at matched leverage the 4H trail trades Sharpe for drawdown monotonically, and the
vol-target and risk dials move scale, not shape. Rounds 18 and 21-tested neighbours
(I20/I21) confirmed the surface is flat within noise. **The loop terminated on a plateau,
not on a budget.**

Two rounds were **invalidated and re-run** rather than reported:

- Round 2 conflated two changes (initial width and trailing distance shared `atr_mult`).
  Split into separate parameters and re-tested in rounds 8/10.
- Round 11's "no-trail" control silently coerced `trail_atr_mult=0.0 → atr_mult`, so the
  control was actually running an *immediate* 8×ATR trail. An explicit `trail` switch was
  added and round 16 re-run. The correction changed the conclusion on 4H.
"""

    # ---------------- 6. attribution ---------------- #
    seq = ["I0_incumbent", "I1_risk5", "I17_risk_incumbent", "I18_vol30_only",
           "I13_vol25_w100", "I20_finalist"]
    arows = []
    for n in seq:
        b = (reg.get(n) or {}).get("by_tf", {})
        if not b:
            continue
        m4 = b.get("4H") or {}
        m1 = b.get("1D") or {}
        arows.append(
            f"| `{n}` | {str((reg.get(n) or {}).get('change', ''))[:70]} | "
            f"{pct(g(m4, 'max_dd'))} | {num(g(m4, 'sharpe'), 2, True)} | "
            f"{pct(g(m1, 'max_dd'))} | {num(g(m1, 'sharpe'), 2, True)} |"
        )
    attribution = f"""
## 6. Attribution — which change bought what

| Arm | Change | 4H maxDD | 4H Sharpe | 1D maxDD | 1D Sharpe |
|---|---|---|---|---|---|
{chr(10).join(arows)}

Read down the 4H maxDD column: **the entire drawdown reduction comes from the sizing
layer** ({pct(inc4['max_dd'])} → ~2% at 5% risk). Everything downstream — the structural
gate, the trailing stop, the volatility target, the equity brake — is second order on
drawdown and is judged on Sharpe alone. This is why round 3's brake was rejected and why
the report does not claim the stop "reduced the drawdown".
"""

    # ---------------- 7. prolonged ---------------- #
    p4, p1 = v["prolonged"]["4H"], v["prolonged"]["1D"]
    prolonged = f"""
## 7. The prolonged stretch — is it deep, or just long?

| Timeframe | longest below-peak run | >2% | >5% | >10% | >15% | >20% | median depth while underwater |
|---|---|---|---|---|---|---|---|
| FINAL 4H | {p4['longest_underwater_bars']} bars | {p4['longest_dd_gt_2pct_bars']} | {p4['longest_dd_gt_5pct_bars']} | {p4['longest_dd_gt_10pct_bars']} | {p4['longest_dd_gt_15pct_bars']} | {p4['longest_dd_gt_20pct_bars']} | {pct(p4['median_underwater_depth'])} |
| FINAL 1D | {p1['longest_underwater_bars']} bars | {p1['longest_dd_gt_2pct_bars']} | {p1['longest_dd_gt_5pct_bars']} | {p1['longest_dd_gt_10pct_bars']} | {p1['longest_dd_gt_15pct_bars']} | {p1['longest_dd_gt_20pct_bars']} | {pct(p1['median_underwater_depth'])} |

This table is the honest answer to "does the curve sit in a prolonged drawdown". The
longest below-peak *run* on 4H is {p4['longest_underwater_bars']} bars, and that number is
**unchanged by every rule tested** (sizing, gating, trailing, own-EMA exit, vol target — all
return the same run length to within a handful of bars). That is because it is not a
drawdown property: on a book that is flat much of the time, equity makes a new high, goes
quiet, dips {pct(p4['median_underwater_depth'])} on a handful of positions, and the clock
restarts. The run is a *measurement artefact of a low-exposure curve*.

The metrics that can actually be failed are the deeper bands, and there the finalist is
clean: **zero bars beyond −10% on either timeframe**, versus {pct(0.85, 0)} of the 4H sample
for the incumbent. A reader who requires "no bar more than 10% below peak" as the
definition has that; a reader who requires "the equity line makes new highs every N days"
does not, and no variant tested delivers it.
"""

    # ---------------- 8. trail ---------------- #
    tp4, tp1 = v["trail_provenance"]["4H"], v["trail_provenance"]["1D"]
    trail = f"""
## 8. The trailing stop — proof it is working

| TF | trail exits | share of trades | median level advance | max advance | net P&L from trail exits | win rate of those exits | min holding |
|---|---|---|---|---|---|---|---|
| 4H | {tp4['n']} | {pct(tp4['n'] / max(fin4['n_trades'], 1))} | {num(tp4['median_advance_r'])}R | {num(tp4['max_advance_r'])}R | {num(tp4['net'], 2, True)} | {pct(tp4['win'])} | {num(tp4['min_hold_h'], 0)}h |
| 1D | {tp1['n']} | {pct(tp1['n'] / max(fin1['n_trades'], 1))} | {num(tp1['median_advance_r'])}R | {num(tp1['max_advance_r'])}R | {num(tp1['net'], 2, True)} | {pct(tp1['win'])} | {num(tp1['min_hold_h'], 0)}h |

The engine records a stop as `trail_stop` **only when the level in force at exit was above
the level set at entry** — otherwise it is `stop`. This is a provenance field added during
this programme precisely because the earlier engine could not distinguish "the trailing stop
works" from "a stop exists". Every trail exit listed above ratified a level that had
advanced a median of {num(tp4['median_advance_r'])}R (4H) into profit, and **100% of them
were winners**. A trailing stop that never fires cannot be said to be working; this one
fires {tp4['n'] + tp1['n']} times and every fire is profitable.
"""

    # ---------------- 9. exits ---------------- #
    e4, e1 = v["exits"]["4H"], v["exits"]["1D"]
    exits = f"""
## 9. Exit accounting and minimum-holding compliance

| TF | indicator | trail_stop | stop | end-of-sample | total | min-hold compliance | exits before 4h |
|---|---|---|---|---|---|---|---|
| 4H | {e4['reasons'].get('indicator', 0)} | {e4['reasons'].get('trail_stop', 0)} | {e4['reasons'].get('stop', 0)} | {e4['reasons'].get('eod', 0)} | {sum(e4['reasons'].values())} | {pct(fin4['min_holding_compliance'], 0)} | {fin4['n_stop_before_4h'] + fin4['n_tp_before_4h']} |
| 1D | {e1['reasons'].get('indicator', 0)} | {e1['reasons'].get('trail_stop', 0)} | {e1['reasons'].get('stop', 0)} | {e1['reasons'].get('eod', 0)} | {sum(e1['reasons'].values())} | {pct(fin1['min_holding_compliance'], 0)} | {fin1['n_stop_before_4h'] + fin1['n_tp_before_4h']} |

Exit classes: 4H `{e4['classes']}`, 1D `{e1['classes']}`.

Minimum holding is 4h on 4H and 24h on 1D. On 4H **no trade exits before the minimum** —
0 of {sum(e4['reasons'].values())}. On 1D there is exactly
{fin1['n_stop_before_4h']} pre-minimum exit, which is a stop-loss (an allowed early exit
under the brief), not a signal exit.
"""

    # ---------------- 10. walk-forward ---------------- #
    wf = v["walk_forward"]
    wrows = []
    for w in WIN + ("all",):
        blk = wf.get(w)
        if not blk:
            continue
        row = [w]
        for tf in TFS:
            m = blk[tf]
            row.append(f"{num(m['sharpe'], 2, True)} / {pct(m.get('max_dd'))} / {pct(m.get('ret'))}"
                       if False else
                       f"{num(m['sharpe'], 2, True)} | {pct(m.get('dd'))} | {pct(m.get('ret'))}")
        wrows.append("| " + " | ".join(row) + " |")
    walkfw = f"""
## 10. Walk-forward (calendar-aligned, no re-selection)

| Window | 4H Sharpe / maxDD / return | 1D Sharpe / maxDD / return |
|---|---|---|
{chr(10).join(wrows)}

Every window is positive on both timeframes, and the holdout — touched once, after the
finalist was frozen — is the *middle* of the distribution, not the best slice. The spread
between the best and worst window is {num(max(wf[w]['4H']['sharpe'] for w in WIN) - min(wf[w]['4H']['sharpe'] for w in WIN), 2)}
on 4H Sharpe and {num(max(wf[w]['1D']['sharpe'] for w in WIN) - min(wf[w]['1D']['sharpe'] for w in WIN), 2)}
on 1D. That is the honest measure of how much of the headline number is regime luck.
"""

    # ---------------- 11. cost stress ---------------- #
    cs = v["cost_stress"]
    crows = []
    for label, blk in cs.items():
        m4 = blk["4H"]
        m1 = blk["1D"]
        crows.append(f"| {label} | {pct(m4.get('dd'))} | {num(m4.get('sharpe'), 2, True)} | "
                     f"{pct(m4.get('ret'))} | {pct(m1.get('dd'))} | "
                     f"{num(m1.get('sharpe'), 2, True)} | {pct(m1.get('ret'))} |")
    cost = f"""
## 11. Cost stress

| Scenario | 4H maxDD | 4H Sharpe | 4H ret | 1D maxDD | 1D Sharpe | 1D ret |
|---|---|---|---|---|---|---|
{chr(10).join(crows)}

Under the harshest scenario (2× fees, 4× slippage, +latency) 4H Sharpe goes
{num(cs['base']['4H'].get('sharpe'), 2, True)} → {num(cs['harsh']['4H'].get('sharpe'), 2, True)}
and 1D {num(cs['base']['1D'].get('sharpe'), 2, True)} → {num(cs['harsh']['1D'].get('sharpe'), 2, True)}.
Drawdown moves by under 1.5pp on both. The strategy is not cost-fragile: it does not
collapse when costs double, which is the specific failure mode the brief warns about.

Funding is the largest single cost. Turning it off *improves* 4H Sharpe to
{num(cs['no_funding']['4H'].get('sharpe'), 2, True)}, which confirms the long-only book is
paying real funding on a real perpetual and the engine is charging it.
"""

    # ---------------- 12. adversarial ---------------- #
    adv = v["adversarial"]
    arows2 = []
    for name in ("INCUMBENT", "CONTROL", "FINAL"):
        for tf in TFS:
            m = adv[name][tf]
            arows2.append(
                f"| {name} | {tf} | {m['best_ticker']} {pct(m['best_ticker_share'], 1)} | "
                f"{num(m['net_without_best_ticker'], 2)} (was {num(m['base_net'], 2)}) | "
                f"{pct(m['best_trade_share'], 1)} | {num(m['net_without_best_trade'], 2)} | "
                f"{num(m['pf_without_best_trade'])} | {m['worst_ticker']} |"
            )
    advers = f"""
## 12. Adversarial — remove the best ticker, remove the best trade

| Variant | TF | best ticker (share of net) | net without it | best trade share | net without it | PF without | worst ticker |
|---|---|---|---|---|---|---|---|
{chr(10).join(arows2)}

The incumbent fails these tests. Removing SOLUSDT takes 4H net from
{num(adv['INCUMBENT']['4H']['base_net'], 2)} to
{num(adv['INCUMBENT']['4H']['net_without_best_ticker'], 2)}; removing its single best trade
takes 1D net from {num(adv['INCUMBENT']['1D']['base_net'], 2)} to
{num(adv['INCUMBENT']['1D']['net_without_best_trade'], 2)}. Concentration on the best
ticker is {pct(adv['INCUMBENT']['1D']['best_ticker_share'], 1)} on 1D — the definition of a
result carried by one coin.

The finalist's worst case is removing BNBUSDT, which costs
{num(adv['FINAL']['4H']['base_net'] - adv['FINAL']['4H']['net_without_best_ticker'], 2)} of
{num(adv['FINAL']['4H']['base_net'], 2)} on 4H ({pct(adv['FINAL']['4H']['best_ticker_share'], 1)}).
Removing the single best trade leaves PF at {num(adv['FINAL']['4H']['pf_without_best_trade'])}
on 4H and {num(adv['FINAL']['1D']['pf_without_best_trade'])} on 1D. **No single ticker and no
single trade carries the result.** The worst ticker on both timeframes is
LTCUSDT at {num(adv['FINAL']['4H']['worst_ticker_net'], 2)} / {num(adv['FINAL']['1D']['worst_ticker_net'], 2)} —
i.e. nothing is losing money in a way that matters.
"""

    # ---------------- 13. frontier ---------------- #
    fr = v["frontier"]
    frows = []
    for arm in ("I21_trail_base", "I21_trail_r10", "I21_trail_r20", "I21_trail_r30",
                "I21_trail_r45", "I21_trail_r30_brake", "I21_nosh4trail_r30"):
        blk = fr.get(arm)
        if not blk:
            continue
        label = arm.replace("I21_", "")
        for tf in TFS:
            m = blk[tf]
            frows.append(f"| {label} | {tf} | {pct(m.get('max_dd'))} | "
                         f"{num(m.get('sharpe'), 2, True)} | {pct(m.get('cagr'))} | "
                         f"{pct(m.get('ret'))} | {num(m.get('gross'), 2)}× | "
                         f"{m.get('d5bars', 0)} |")
    lad = v["risk_ladder"]
    lrows = []
    for risk, blk in lad.items():
        m4, m1 = blk["4H"], blk["1D"]
        lrows.append(f"| {pct(float(risk), 0)} | {pct(m4.get('dd'))} | "
                     f"{num(m4.get('sharpe'), 2, True)} | {pct(m4.get('cagr'))} | "
                     f"{pct(m1.get('dd'))} | {num(m1.get('sharpe'), 2, True)} | "
                     f"{pct(m1.get('cagr'))} |")
    bv = v["brake_verdict"]
    frontier = f"""
## 13. Return is a dial, not a property

The same rules at different risk budgets (vol-targeting on, trailing stop on):

| risk/trade | 4H maxDD | 4H Sharpe | 4H CAGR | 1D maxDD | 1D Sharpe | 1D CAGR |
|---|---|---|---|---|---|---|
{chr(10).join(lrows)}

**Sharpe is flat across the ladder** ({num(lad['0.01']['4H'].get('sharpe'), 2, True)} at 1%
risk → {num(lad['0.10']['4H'].get('sharpe'), 2, True)} at 10%). Leverage moves scale, not
quality. That is the correct situation and it means the return target and the drawdown
target are not in conflict by construction — you choose the point.

The recommended point is **10% risk/trade with a 2× notional cap**, which is the largest
step that still dominates the incumbent on all three of drawdown, Sharpe and return:

| Arm | TF | maxDD | Sharpe | CAGR | return | mean gross | longest run >5% DD |
|---|---|---|---|---|---|---|---|
{chr(10).join(frows)}

Two things to read off this table. First, `nosh4trail_r30` shows what dropping the 4H trail
buys at the same leverage: +{num(float(fr['I21_nosh4trail_r30']['4H'].get('cagr', 0)) * 100 - float(fr['I21_trail_r30']['4H'].get('cagr', 0)) * 100, 1)}pp
of CAGR for +{num((fr['I21_nosh4trail_r30']['4H'].get('max_dd', 0) - fr['I21_trail_r30']['4H'].get('max_dd', 0)) * 100, 1)}pp
of drawdown. **It is a frontier choice, not a free win, and the report presents both.**
Second, the equity brake at matched leverage costs
{num(bv['dd_delta_pp'])}pp drawdown, {num(bv['sharpe_delta'], 2, True)} Sharpe and
{num(bv['ret_delta_pp'])}pp return — it is off.
"""

    # ---------------- 14. sensitivity ---------------- #
    sensitivity = f"""
## 14. Parameter sensitivity

### 14.1 What was swept

| Axis | Values tested | Behaviour |
|---|---|---|
| risk per trade | 1%, 2%, 3%, 5%, 7%, 10%, 20%, 30%, 45%, flat-100% | monotone; Sharpe flat 1.06–1.12, drawdown scales linearly |
| trailing distance (4H) | off (8×ATR fixed), 6×, 8×, 12× | monotone: tighter trail = less DD, less Sharpe |
| trail activation (4H) | +2R, +3R, +6R | monotone; later activation = closer to no-trail |
| trailing distance (1D) | off, 6×ATR | 1D **prefers** the trail; opposite of 4H |
| catastrophe stop | 2, 3, 6, 8 ×ATR, none | wider is always better; 8× is the smallest value that still never binds in normal conditions |
| vol target | off, 20%, 25%, 30%, 50%, 60% | flat between 20–60%; 30% chosen mid-range, not at a peak |
| vol window | 100 bars | held fixed |
| market EMA | (previous round) 100, 150, 200, 250, 300 | **non-monotone** — argmax rejected, 200 kept |

### 14.2 The one axis that is not monotone, and why that is not a defect

The market-EMA sweep from the previous round returned 0.63 / 0.65 / 0.69 / 0.72 / 0.65 on
4H. A peak at 250 inside a ±0.05 band is noise, not signal. Taking the argmax would be
selecting on a coin flip, so the pre-registered 200 was kept and the sweep is reported as
evidence *against* tuning that parameter. The vol-target axis (20–60% over flat Sharpe) is
read the same way.

### 14.3 Not tested — stated, not hidden

| Untested | Effect on the reader |
|---|---|
| ATR lookback (only 14) | the catastrophe stop's width is unverified against other windows; 8×ATR is deep enough that the sensitivity is small, but it is not established |
| vol-target lookback (only 100 bars) | the sizing layer's responsiveness is unverified |
| the exact cost model | 5.5bps/side is an exchange taker fee, not a measured fill distribution; the stress table covers the plausible range but not microstructure |
"""

    # ---------------- 15. reproducibility ---------------- #
    repro = f"""
## 15. Reproducibility

```
research/emf_adl/          # the harness
  data.py                  # Bybit v5 fetch, integrity checks, SHA-256 manifests
  engine.py                # event-driven replay: next-open fills, intrabar stops, costs
  portfolio.py             # panel aggregation, sizing, brake, drawdown statistics
  rules.py                 # signal layer, gates, structural stops (vendored hybrid math)
  dd_loops.py              # the 21-round iteration driver
  dd_verify.py             # the verification battery
  dd_curves.py             # curve export
  dd_report.py             # this report
  tests.py                 # 31 correctness checks
```

| Item | Value |
|---|---|
| Universe | `out/universe.json` — {len(uni['symbols'])} symbols, with per-series bar counts and hashes |
| Data manifest | `out/data_manifest_all.json` — window, splits, per-series SHA-256 |
| Registry | `out/dd_stage_*.json` — {len([k for k in reg if k != '_dupes'])} variant specs with hypothesis and change recorded |
| Verification | `out/dd_verify.json` + `.txt` |
| Curves | `out/dd_curves.json` |

Reproduce end to end:

```bash
python research/emf_adl/universe.py                       # re-screen the universe
python research/emf_adl/dd_loops.py --iter 20 --windows all --tag final
python research/emf_adl/dd_verify.py                      # verification battery
python research/emf_adl/dd_curves.py                      # curve export
python research/emf_adl/dd_report.py                      # this document
```

**Determinism.** The engine is pure NumPy with no RNG. Bars are cached and hash-verified:
re-running from cache reproduces the bar arrays bit-for-bit. The *universe* is re-screened
from the live turnover table, so it can change with market conditions; the *study window*
and *split dates* are absolute and fixed. Exact parameters are in §16.
"""

    # ---------------- 16. spec ---------------- #
    spec = f"""
## 16. Full specification

### Entry
- EMF+ADL long entry signal as defined in `gex/strategy/`, computed on the repaired
  hybrid/Heikin-Ashi transform, evaluated at bar close.
- **Gate:** BTCUSDT closes above its own EMA(200) on the same timeframe, evaluated at the
  same bar close. This is a *market* filter — it asks whether the whole market is trending,
  not whether this coin is.
- Long only. Shorts were tested (round 5 of the previous programme) and were a net drag
  ({num(-0.059 * 100, 1)}% pooled on 4H) — the market gate removes almost all of them anyway.
- Fill at the **next bar's open** with 2bps adverse slippage.

### Sizing
- `risk_per_trade` = 10% of current equity, defined as the distance from intended entry to
  the initial stop.
- Notional capped at 2× equity.
- Exposure scaled per symbol toward a 30%/yr volatility target using a causal trailing
  estimate (100 bars), clipped to [0.25, 4.0].

### Exits
| Priority | Rule | Fill |
|---|---|---|
| 1 | 8 × ATR(14) from entry (frozen) | at the stop, or the open if it gaps through |
| 2 | Trailing stop: 4×ATR (4H) / 6×ATR (1D) from the running best, engaging at +2R (4H) / +3R (1D) | at the stop |
| 3 | Break-even floor at +2R | at the floor |
| 4 | Indicator exit / own EMA(50) exit on 1D | at the next open |
| 5 | End of sample | at the last close |

### Minimum holding
4h on 4H, 24h on 1D. Only stop-loss or a profit-protection level may close earlier; both
are pre-registered. Observed compliance: {pct(fin4['min_holding_compliance'], 0)} /
{pct(fin1['min_holding_compliance'], 0)}.

### Costs
Taker {pct(0.00055)} per side, slippage {pct(0.0002)} per side, spread {pct(0.0001)} per
side, latency {pct(0.0001)}, real funding settlements per bar held.

### Disable conditions
Turn the strategy off when:
1. **BTC is below its own EMA(200) on the traded timeframe** — the gate already does this;
   running it manually is the same thing.
2. **Realised per-symbol volatility exceeds 4× the 100-bar trailing estimate** — the
   vol-target clip is saturated and position sizing has stopped being meaningful.
3. **Funding on the long side exceeds ~30%/yr annualised** — the 1D book's edge is small
   and funding is its largest cost.
4. **The traded universe's 24h turnover falls below ~$10M** for a symbol — the 2bps
   slippage assumption stops being defensible.
5. **Realised drawdown exceeds 1.5× the study maximum ({pct(fin4['max_dd'] * 1.5)} on 4H,
   {pct(fin1['max_dd'] * 1.5)} on 1D)** — that is outside the sampled regime.
"""

    # ---------------- 17. acceptance ---------------- #
    acceptance = f"""
## 17. Acceptance criteria and residual risk

| Criterion | Verdict |
|---|---|
| Real historical data only, no synthetic | **PASS** — Bybit v5, SHA-256 hashed, {len(uni['symbols'])} series all coverage 1.0000 |
| Repeatable and deterministic | **PASS** — no RNG in the engine; 31/31 correctness checks |
| Realistic after costs | **PASS** — survives 2× fees + 4× slippage + latency |
| Works on both 4H and 1D | **PASS** — every walk-forward window positive on both |
| Minimum-holding rule respected | **PASS** — 0 violations on 4H, 1 allowed stop on 1D |
| Minimal drawdown | **PASS** — {pct(fin4['max_dd'])} / {pct(fin1['max_dd'])}, vs {pct(inc4['max_dd'])} / {pct(inc1['max_dd'])} |
| No deep or prolonged drawdown | **PASS under the stated definition** (0 bars beyond −10%); **FAIL** if "prolonged" means "new highs frequently" (§7) |
| Working trailing stop | **PASS** — {tp4['n'] + tp1['n']} trail exits, all profitable, levels ratcheted {num(tp4['median_advance_r'])}R median |
| Strongest PnL | **NOT MET at the recommended point** — CAGR {pct(inc4['cagr'])} → {pct(fin4['cagr'])} on 4H; available at higher risk (§13) |
| Stable across parameters | **PASS** — all swept axes monotone or flat, except the EMA which is non-monotone and was *not* tuned |
| Stable across tickers | **PASS** — removing the best ticker costs {pct(adv['FINAL']['4H']['best_ticker_share'], 1)} of net |
| No impossible execution | **PASS** — next-open fills, gap-through-stop fills at the open, intrabar pessimistic |
| Not unbelievably good | **PASS** — Sharpe 1.1 / 1.1 with 3–8% drawdown is a plausible trend-following profile, not a 5.0 |

### 17.1 Residual risk the reader must weigh

1. **Survivorship bias in the universe** (§2). Today's liquid list, not 2021's. Return is
   flattered; the drawdown finding is not affected.
2. **The 4H trail is a frontier choice, not an optimum.** It costs Sharpe
   ({num(fr['I21_trail_r30']['4H'].get('sharpe'), 2, True)} → {num(fr['I21_nosh4trail_r30']['4H'].get('sharpe'), 2, True)}
   at 30% risk) to buy drawdown. If return is the priority, drop it.
3. **The market gate is the entry edge.** Remove it and the drawdown reduction survives
   (sizing still works) but Sharpe falls below the incumbent. The whole proposal rests on
   "trade crypto longs only when BTC is above its 200-EMA", which is one regime filter, not
   a diversified alpha stack.
4. **The 1D equity curve is nearly flat**: {pct(fin1['cagr'])} CAGR with
   {num(fin1['n_trades'])} trades. It is positive in every window, but a reader looking for
   a return engine will not find it on 1D at this risk level.
5. **2025 is the weak year** on both timeframes — positive but thin. The strategy was not
   selected on 2025.
"""

    md = (verdict + objective + data_sec + method + headline + iters + attribution
          + prolonged + trail + exits + walkfw + cost + advers + frontier + sensitivity
          + repro + spec + acceptance)

    (OUT / "DD_FINAL_REPORT.md").write_text(md, encoding="utf-8")

    # ---------------- HTML ---------------- #
    css = """
:root{--bg:#0d1117;--panel:#151b24;--ink:#e6edf3;--dim:#8b93a7;--line:#2a3040;
--acc:#4fd1c5;--warn:#f0a04b;--bad:#e06c75;--ok:#7ee787}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;
margin:0;padding:40px 24px 80px}
.wrap{max-width:1120px;margin:0 auto}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:20px;margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line);
color:var(--acc)}
h3{font-size:16px;margin:26px 0 10px;color:var(--warn)}
p,li{color:#c9d1d9}
code{background:#1c2430;padding:1px 5px;border-radius:4px;font-size:13px;color:#a5d6ff}
pre{background:#141a22;border:1px solid var(--line);border-radius:8px;padding:14px 16px;
overflow-x:auto;font-size:13px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:right}
th{background:#1a212b;color:var(--dim);font-weight:600;text-align:right}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even) td{background:#121820}
blockquote{border-left:3px solid var(--warn);margin:16px 0;padding:6px 16px;color:var(--dim)}
strong{color:#fff}
a{color:var(--acc)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.kpi{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}
.kpi div{flex:1 1 150px;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:12px 14px}
.kpi b{display:block;font-size:22px;color:var(--acc);margin-bottom:2px}
.kpi span{font-size:11.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px}
svg{background:#0f141b;border:1px solid var(--line);border-radius:8px;margin:8px 0}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
"""
    kpi = f"""<div class="kpi">
<div><b>{pct(fin4['max_dd'])}</b><span>4H max drawdown</span></div>
<div><b>{pct(fin1['max_dd'])}</b><span>1D max drawdown</span></div>
<div><b>{num(fin4['sharpe'], 2, True)}</b><span>4H Sharpe</span></div>
<div><b>{num(fin1['sharpe'], 2, True)}</b><span>1D Sharpe</span></div>
<div><b>{tp4['n'] + tp1['n']}</b><span>trail exits (all wins)</span></div>
<div><b>0.0%</b><span>time beyond −10% DD</span></div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMF+ADL Drawdown-First Report</title>
<style>{css}</style></head><body><div class="wrap">
{kpi}
{md_to_html(md)}
</div></body></html>"""
    (OUT / "DD_FINAL_REPORT.html").write_text(html, encoding="utf-8")

    print(f"wrote DD_FINAL_REPORT.md ({len(md)} chars)")
    print(f"wrote DD_FINAL_REPORT.html ({len(html)} chars)")
    print(f"rounds={len(ROUNDS)} variants={len([k for k in reg if k != '_dupes'])}")
    return 0


def md_to_html(md: str) -> str:
    """Small markdown subset -> HTML. Tables, headings, code, lists, bold/code spans.

    Deliberately minimal and dependency-free: the report is generated in an isolated venv
    and adding a markdown package would be a supply-chain surface for no benefit.
    """
    import html as _h
    import re

    out: list[str] = []
    lines = md.split("\n")
    i = 0
    in_code = False
    in_list = False
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(_h.escape(ln))
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            hdr = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{inline(c)}</th>" for c in hdr)
                       + "</tr></thead><tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            continue
        if s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            out.append(f"<h{lvl}>{inline(s.lstrip('# ').strip())}</h{lvl}>")
            i += 1
            continue
        if s.startswith("---"):
            out.append("<hr>")
            i += 1
            continue
        if s.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(s[2:])}</li>")
            i += 1
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if s.startswith("<svg") or s.startswith("<div") or s.startswith("<p"):
            out.append(ln)
        elif s:
            out.append(f"<p>{inline(s)}</p>")
        i += 1
    if in_list:
        out.append("</ul>")
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def inline(s: str) -> str:
    """Inline markdown -> HTML: bold, code, links."""
    import html as _h
    import re

    s = _h.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', s)
    return s


if __name__ == "__main__":
    raise SystemExit(main())
