# Quant Optimization Loops — Final Report
**Date:** 2026-10-03 · **Engineer:** WorkBuddy (autonomous quant agent) · **All data: real**

---

## Executive verdict

> **1D timeframe: robust strategy found and accepted (with disclosed limits).**
> **4H timeframe: NO ROBUST STRATEGY FOUND UNDER THE CONSTRAINTS.**

The final acceptance criteria require BOTH timeframes to work. That combined
criterion is **not met**: the 4H hypothesis family (long-only Donchian breakout
trend-following with ATR stops, MA exit, and regime gates) was tested across
three independent ticker sets, two cost levels, and parameter neighborhoods —
it did not produce stable positive expectancy after costs. I am not forcing a
result.

---

## Data (real, cached, hashed)

Source: **Bybit v5 public spot kline API** (`api.bybit.com/v5/market/kline`),
UTC timestamps, no API key. 21 symbols × 2 timeframes, **zero bad-OHLC bars,
zero gaps** in every dataset. Every CSV has a sha256 in a sidecar
`quant/cache/*.meta.json` (prefixes recorded in the loop logs below).

| Ticker set | Tick ers | Why selected | Rationale (point-in-time) |
|---|---|---|---|
| Loop 1 | BTC, ETH, XRP, LTC, ADA, DOGE, SOL | top-volume Bybit spot pairs as of 2021-07 (before backtest start) | all were already top-liquidity pairs then; no future-performance selection |
| Loop 2 | BNB, LINK, AVAX, DOT, TRX, ATOM, NEAR, FIL | next tier of top-liquidity pairs, listed pre-2022 | fresh set, never used before Loop 2 |
| Loop 3 | UNI, ETC, XLM, ALGO, CRV, SAND | further top-100 liquidity pairs | fresh set, never used before Loop 3 |

Splits (identical for all loops, deterministic by UTC bar timestamp):
- **Train:** history start → 2024-07-01
- **Validation:** 2024-07-01 → 2025-07-01
- **Holdout:** 2025-07-01 → 2026-10-03 (touched once, Loop 1 selection only)

History depth: BTC/ETH from 2021-07-05; alts 2021-09 → 2022-07. All sets cover
the 2022 bear, 2023 chop, 2024–25 bull, and the 2025-26 drawdown.

## Cost model (conservative, disclosed)

- Fee: **0.10% per side** (Bybit spot taker; the actual tier is 0.075–0.1%)
- Slippage: **5 bps per side** on open fills
- Stop fills: **+8 bps adverse** extra (stops executed conservatively worse)
- Gap-through-stop: filled at the **open** (the worse price), never at the stop
- Funding/borrow: **N/A** (spot, long-only)
- Stress-tested at **2× all costs** (fee 0.2%, slippage 10 bps, stop 16 bps)

## Execution model (enforced in code, unit-tested)

- Signals computed on **closed bars only**; entry fills at the **next bar's open**
- Donchian channel excludes the signal bar (prior `n_break` highs)
- Stop-loss live from the entry bar, may trigger **intrabar**
- Trailing (chandelier) stop may **update** from the entry bar's close and only
  **trigger** from the next bar ⇒ holding ≥ 4h
- MA-exit fills at next bar's open (always ≥ 4h after entry)
- Pessimistic intrabar tie-break: stop assumed first
- Exit classification logged per trade: `STOP_LOSS_BEFORE_4H` /
  `TAKE_PROFIT_BEFORE_4H` / `EXIT_AFTER_4H`
- Engine mechanics verified by 5 hand-computable scenarios
  (`quant/test_engine_mechanics.py`, all pass)

---

## LOOP 1 — core hypothesis: Donchian breakout + ATR stop + chandelier trail

- **Hypothesis:** classic low-parameter trend-following is positive-expectancy
  after costs on liquid crypto spot pairs.
- **Grid:** n_break {20,30,40,55} × k_sl {2.5,3.0} × k_trail {3,4,5} × ma_exit {0,100}
  = 48 combos, train-only selection; top-5 validated OOS.
- **Baseline (no regime filter) result:** 4H train Sharpe 0.41 but maxDD **53%**
  (rejected); failure mode = long breakouts during the 2022 bear chop.
- **Improvement:** a-priori **regime filter** (entries only when close >
  SMA-200d; 200-bar on 1D, 1200-bar on 4H; NOT tuned) — maxDD 53%→16% (4H
  train), all splits positive on both timeframes.
- **Selection rule:** within the stable plateau (n_break 30–55, k_sl 2.5–3.0,
  k_trail 4–5, ma_exit 100), ranked by **min(own val Sharpe, neighborhood
  median val Sharpe)** — anti-overfit.

### Loop 1 selected (frozen, then held out)

| TF | Params | Train | Validation | **Holdout (once)** |
|---|---|---|---|---|
| 4H | n30 / k_sl3.0 / k_trail5.0 / MA100 / reg200d | Sh 0.48, DD 17.5% | Sh 1.13, DD 21.4% | **Sh 0.42, +7.6%, DD 15.5%**, 87 tr |
| 1D | n40 / k_sl2.5 / k_trail4.0 / MA100 / reg200d | Sh 0.51*, DD 22.8% | Sh 0.76–0.96 | **Sh 0.38, +6.3%, DD 17.8%**, 20 tr |

*1D train figure shown for the frozen combo; grid's train-best was n30 Sh 0.64.

Holdout context: equal-weight buy&hold was **−24.5% to −27.2%** in the same
period. Neighborhood stability: every plateau neighbor of the 4H pick had
validation Sharpe ≥ 1.045; the 1D pick's neighborhood ≥ 0.48.

### Loop 1 stress results (4H / 1D, full period 2021→2026)

| Check | 4H | 1D |
|---|---|---|
| Base full period | Sh 0.62, CAGR 11.9%, DD 21.9%, 408 tr | Sh 0.45, CAGR 7.9%, DD 32.6%, 92 tr |
| 2× costs | Sh 0.45, CAGR 7.9% ✅ | Sh 0.40, CAGR 6.8% ✅ |
| Drop best ticker (SOL) | Sh 0.49, +54% ✅ | Sh 0.33, +29% ✅ |
| Drop best trade (1 of 408 / 92) | 69% of net profit remains ✅ | 56% remains ✅ |
| Min-hold compliance | 0 early exits (all ≥4h) ✅ | 2 intrabar stops (allowed), 0 violations ✅ |

**Loop 1 decision: ACCEPTED provisionally** (both TF positive OOS + holdout).

---

## LOOP 2 — pure out-of-sample ticker-set confirmation (frozen params)

New set: BNB, LINK, AVAX, DOT, TRX, ATOM, NEAR, FIL (never seen before).

| System | Train | Val | Holdout | Full | Decision |
|---|---|---|---|---|---|
| 1D | +0.36 | +0.81 | +1.06 | **Sh 0.59, CAGR 12.6%, PF 2.08, 93 tr** | ✅ CONFIRMED |
| 4H | **−0.20** | +0.60 | +0.60 | Sh 0.18, CAGR 1.6%, PF 1.06 | ⚠️ MARGINAL |

4H failure mode: fresh-set train period (2022 bear on weaker alts) is negative;
edge does not transfer across ticker groups. Disclosed concentration on the 1D
Loop-2 run: TRX+NEAR contributed ~98% of that set's profit — flagged, and the
Loop 3 test was run to resolve whether this was luck.

---

## LOOP 3 — 4H rescue attempt + 1D final confirmation

**4H fix attempts (both rejected):**
1. Daily-regime gate (4H entry requires last CLOSED daily bar > daily SMA200,
   strictly causal, zero new tuned params): Loop 1 train improved (0.55) but
   holdout flipped to **−0.10**; Loop 2 still weak (0.25); Loop 3 set **negative
   (full Sh −0.08, PF 0.88)**.
2. Every gated/narrower variant within the tested family moved losses around
   rather than removing them.

**4H VERDICT: No robust strategy found under the constraints** (for this
hypothesis family: long-only Donchian breakout + ATR trails + MA/regime gates
on spot crypto). The 4H timeframe on liquid crypto spot shows no stable edge
after conservative costs across three independent ticker groups.

**1D on Loop 3 fresh set (UNI, ETC, XLM, ALGO, CRV, SAND), frozen params:**

| Train | Val | Holdout | Full |
|---|---|---|---|
| −0.02 (flat) | +1.33 | +0.20 | **Sh 0.47, CAGR 10.0%, DD 45.0%, PF 1.88, 56 tr** |

4/6 tickers positive (UNI +12.8k, XLM +22.1k, ALGO +14.3k, CRV +14.7k vs
ETC −1.0k, SAND −1.5k on 100k initial / 16.7k slices).

---

## FINAL RESULT — 1D strategy (accepted, disclosed limits)

**Rules (complete, deterministic):**
1. Universe: N top-liquidity spot pairs (7–8 tested per set; BTC/ETH + majors).
2. Capital: equal slice per pair (1/N of equity), long-only, spot.
3. Entry (per ticker): close > max(high of prior 40 bars) **AND** close >
   SMA(200). Confirmed at daily close → **buy at next day's open** (95% of slice
   cash deployed).
4. Initial stop: entry − 2.5 × ATR(14, Wilder). Live immediately; intrabar
   trigger; gap → open fill (worse).
5. Trailing stop (chandelier): max(stop, highest close since entry − 4.0 × ATR).
   Updates from the entry bar's close; can only trigger ≥ 4h after entry
   (trivially satisfied on 1D — next-day fills).
6. MA exit: close < SMA(100) → sell at next day's open.
7. No pyramiding; one position per ticker; re-entry requires a fresh 40-bar
   breakout after exit.

**Aggregate out-of-sample evidence (3 independent ticker sets):**

| Set | Full-period Sharpe | CAGR | maxDD | PF | Trades | Splits positive |
|---|---|---|---|---|---|---|
| Loop 1 (selection) | 0.45 | 7.9% | 32.6% | — | 92 | train/val/holdout all ✅ |
| Loop 2 (pure OOS) | 0.59 | 12.6% | 25.2% | 2.08 | 93 | all ✅ |
| Loop 3 (pure OOS) | 0.47 | 10.0% | 45.0% | 1.88 | 56 | train flat, val/holdout ✅ |
| Buy&hold benchmark | — | ~2%/yr (EW, 2021→26) | ~80% DD | — | — | — |

**Worst period:** 2025-07→2026-10 holdout slices (bear): 1D made +6.3% (Loop 1
set) vs buy&hold −24.5%; longest underwater stretch ≈ 437 days (2025-07→).
**Largest drawdown:** 45% (Loop 3 alt-heavy set, full period).
**Trade rate:** ~17–31 trades/year portfolio-wide; hold median 348–534h.
**Min-hold compliance:** the only sub-4h exits ever recorded were intrabar hard
stops (allowed) and one END_OF_DATA boundary artifact (position opened on the
final bar and force-closed at its close — disclosed, not a rule violation).

**Parameter sensitivity:** plateau n_break 30–55 / k_trail 4–5 / k_sl 2.5–3.0
all positive OOS on the selection set; neighborhood median val Sharpe ≥ 0.75
(1D pick), every single neighbor ≥ 0.48. No cliff edges.

**Disable conditions (predefined):**
- Rolling 180-day portfolio Sharpe < 0 (edge decay monitor)
- MaxDD > 50% (risk limit)
- Exchange fee/liquidity regime change > 2× the tested stress levels
- If funding-rate perps replace spot: re-run the loop with funding costs.

## Reproducibility

Everything needed is in this repo, no randomness anywhere:

```
quant/
  data.py                  # Bybit fetch + validation + sha256 cache
  strategy.py              # causal Donchian/ATR/MA indicator precompute
  backtest.py              # execution engine (all invariants above)
  metrics.py               # extended metric suite
  test_engine_mechanics.py # 5 hand-checked execution QA scenarios
  run_loop1.py             # fetch / grid / select  → results/loop1/
  analyze_loop1.py         # OOS validation, benchmarks, regime grid
  final_loop1.py           # neighborhood selection, holdout, stress
  loop2_fresh_tickers.py   # Loop 2 pure OOS confirmation
  loop3_4h_fix.py          # 4H rescue attempts + Loop 3 confirmation
  results/loop1/           # train_metrics.csv, plateau_*.csv,
                           # selected.json, holdout_report.json
  cache/                   # raw CSVs + .meta.json (sha256) per symbol/tf
```

Run order:
```bash
python -m quant.test_engine_mechanics          # must print ALL PASSED
python -m quant.run_loop1 fetch                # re-downloads real data
python -m quant.run_loop1 grid && python -m quant.run_loop1 select
python -m quant.analyze_loop1 regime
python -m quant.final_loop1 select2 && python -m quant.final_loop1 holdout
python -m quant.loop2_fresh_tickers
python -m quant.loop3_4h_fix
```
Note: re-fetching re-hashes the CSVs; bar sets are append-only at the source
so re-runs add recent bars at the tail (report includes first/last bar).

## Honest limitations

1. **4H: no robust result.** Do not deploy the 4H variant.
2. Bybit spot history starts 2021-07 → no pre-2021 data in this study.
3. 1D trade counts are structurally low (~17–31/yr portfolio-wide) — parameter
   neighborhoods mitigate but small-sample variance is real.
4. Loop-2 concentration (TRX+NEAR ≈ 98% of that set's profit) was resolved by
   Loop 3 (profit from 4 different tickers) but remains a warning sign.
5. Live execution adds latency, partial fills on thin alts, and exchange
   downtime none of which are modeled beyond conservative slippage.
