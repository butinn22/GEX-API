# Alligator Confluence Trend-Following Strategy — Research Report

**Package:** `quant/alligator/` · **Data:** Bybit spot OHLCV (cached, integrity-checked, `quant/data.py`) · **Engine:** `quant/alligator/engine.py` (event-driven, same no-lookahead contract as `quant/backtest.py`) · **Date of run:** 2026-10-04

> **Honesty statement up front:** this is a *robust, conservative* trend-follower, not a money printer. It preserves capital extremely well (max DD < 3% at 0.5% risk/trade in every test) and keeps positive expectancy out-of-sample, but absolute returns are modest because exposure is low (9–14%) and the 2025-2026 holdout was mostly range-bound — the worst regime for any trend strategy. All weaknesses are disclosed in §11–§12.

---

## 1. Full strategy rules

**Name:** Alligator Confluence Breakout (long-only spot, 4H).

**Core idea:** only buy strength when four independent trend confirmations agree (confluence), enter on a momentum breakout, risk a fixed fraction of equity, let the trailing stop run winners, exit when the Alligator trend is spent.

**Confluence stack — all four must hold at the signal bar close:**

| # | Component | Bull rule |
|---|-----------|-----------|
| 1 | **Bill Williams Alligator** | `lips > teeth > jaw` (SMMA5/8/13 of median price, forward-shifted 3/5/8) **and** jaw rising (trend developing, not sleeping) |
| 2 | **Smart-money structure (HH/HL)** | last two *confirmed* swing highs ascending (HH) **and** last two *confirmed* swing lows ascending (HL). Pivots are fractals (3 bars each side) confirmed only 3 bars after they form — strictly causal |
| 3 | **Chaikin ADL** | `ADL > EMA20(ADL)` — accumulation, not distribution |
| 4 | **Ease of Movement (EMF 14)** | `EMF > 0` — price advancing without volume churn against |

## 2. Entry conditions

Signal confirmed at **close of bar t**, filled at **open of bar t+1** (next-bar execution — no lookahead):

- `close[t] > max(high[t-30 .. t-1])` — 30-bar Donchian breakout of *prior* bars (channel excludes t itself), and
- the full confluence stack holds at t, and
- flat (position check before add — the engine can never stack signals while in position) and the 3-bar post-exit cooldown has elapsed.

Rejected on train: `pullback` entries (buy-the-dip to the lips) and mixing both modes — both lowered Sharpe (§12).

## 3. Exit conditions

Priority order each bar:

1. **Intrabar stop** (live from the entry bar; gap → fill at the worse open; pessimistic tie-break).
2. **Trailing stop** — ratchet only upward, armed after the min hold (1 bar):
   `max(highest_close − 5·ATR14,  last_confirmed_HL − 0.5·ATR14)`  → 51% of holdout exits. Winners run.
3. **Signal exit** — close below the **jaw** (trend spent) or a **structure break** (close below the last confirmed HL after a bull structure) → fill at next open → 30% of exits.
4. **Time stop** — if held ≥ 45 bars (4H) and never reached +1R, exit at next open → throttles dead trades.

## 4. Stop-loss logic

- **Initial stop** = tighter of `last_confirmed_HL − 0.5·ATR` and `entry − 2.5·ATR`, bounded to `[1·ATR, 4·ATR]` distance.
- **Trailing stop** as in §3, never loosens.

## 5. Position sizing

`qty = (equity_slice × 0.5%) / stop_distance`, capped at 95% of slice cash (spot, no leverage). Risk per trade is constant in % terms; volatility (ATR + structure) sets the distance, hence the size. The risk ladder (§9) shows 0.25% / 0.5% / 1.0% — Sharpe is invariant, CAGR and DD scale ~linearly.

## 6. Recommended timeframe and instruments

- **Timeframe: 4H.** The 1D variant was tested on the same grid and is **rejected**: negative Sharpe on train across all 144 combinations (confluence too rare on daily bars, ~65 trades in 3 years). This is reported, not hidden.
- **Instruments:** liquid top-tier Bybit spot pairs, portfolio of 5–15 symbols, equal capital slices. Validated on 7 in-sample (BTC, ETH, XRP, LTC, ADA, DOGE, SOL) and 14 fresh out-of-sample pairs (§9).
- Works on any liquid 24/7 crypto spot market; FX/index CFDs plausible but untested here.

## 7. Indicator parameters

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Alligator jaw/teeth/lips | SMMA 13·8 / 8·5 / 5·3 of median | ATR | Wilder 14 |
| Swing pivots | 3 left / 3 right (confirmed +3) | ADL EMA | 20 |
| EMF | SMA 14 | Donchian `n_break` | 30 |
| `k_sl_atr` | 2.5 | `k_trail` | 5.0 |
| struct buffer | 0.5·ATR | exit level | jaw |
| risk/trade | 0.5% | cooldown | 3 bars |
| time stop | 45 bars (4H) | warmup | 250 bars |

## 8. Backtest assumptions

- **Costs:** taker fee 0.1%/side, slippage 5 bps/side on open fills, +8 bps adverse on intrabar stop fills (Bybit spot, conservative).
- **Execution:** signal at close t → market fill at open t+1; stops intrabar; force-close at last bar (flagged END_OF_DATA).
- **Portfolio:** $100k, 7 equal slices (per-symbol engine runs, equity summed — same construction as `quant/run_loop1`).
- **Splits (UTC):** TRAIN 2021-07→2024-07 · VAL 2024-07→2025-07 (stability check only, never used for selection) · HOLDOUT 2025-07→2026-10 (touched once).
- **Selection:** composite train score = sortino + 0.5·sharpe + 2·calmar + win_rate + 0.25·min(PF,4), hard gates n≥60 trades and maxDD≤45%. 144-combo grid.
- Self-checks (`python -m quant.alligator.selfcheck`): pivot confirmation lag, prefix-causality (truncation test), hand-computed engine trade, notional cap, flat-before-entry — **all pass**.

## 9. Performance metrics (final frozen params, 0.5% risk/trade)

| Metric | TRAIN | VAL | HOLDOUT |
|---|---|---|---|
| Net profit | +5.8% | +1.9% | +1.2% |
| CAGR | 1.90% | 1.91% | 0.94% |
| Sharpe | 0.91 | 0.97 | 0.56 |
| Sortino | 0.55 | 0.61 | 0.33 |
| Calmar | 0.68 | 0.96 | 0.36 |
| **Max drawdown** | **2.81%** | **1.99%** | **2.66%** |
| Profit factor | 1.75 | 1.55 | 1.38 |
| Win rate | 28.2% | 33.3% | 31.5% |
| Payoff (W/L) | 4.45 | 3.09 | 3.00 |
| Expectancy / trade | +0.122% slice | +0.106% slice | +0.105% slice |
| Trades | 195 | 99 | 92 |
| Avg / median hold | 86.9h / 60h | 81.9h / 68h | 75.8h / 60h |
| Exposure | 14.0% | 13.7% | 9.4% |

Low win rate + high payoff (≈3–4.5) = classic trend-following profile: many small controlled losses, few large winners. Exit mix (holdout): TRAIL 51% / SIGNAL 30% / STOP 18%.

**Risk ladder (portfolio-level choice, not a signal parameter):**

| risk/trade | trainval CAGR | trainval maxDD | Sharpe | holdout CAGR | holdout maxDD |
|---|---|---|---|---|---|
| 0.25% | 0.96% | 1.43% | 0.92 | 0.48% | 1.34% |
| **0.5% (base)** | **1.91%** | **2.81%** | **0.92** | **0.94%** | **2.66%** |
| 1.0% | 3.76% | 5.49% | 0.93 | 1.80% | 5.24% |

**Fresh-ticker OOS (14 symbols never used in selection), full period 2022→2026-10:** CAGR +0.30%, Sharpe 0.19, PF 1.10, maxDD 2.4% — thin but positive through the 2022-2024 alt bleed; **holdout segment: CAGR +0.89%, Sharpe 0.57, PF 1.37, maxDD 1.5%** — statistically indistinguishable from the in-sample universe's holdout. No universe-specific overfit detected.

**Walk-forward (7 anchored 6-month windows, 2023→2026):** 5/7 positive, every re-optimization pick drawn from the same family `{breakout, n_break 30–55, k_sl 2.5–3.0, k_trail 4–5}` — the parameter *family* is stable even where the exact pick flips. Worst window −1.6 Sharpe on a 1.2% DD (chop regime).

## 10. Equity curve description

See `results/equity_curve.svg` (also attached in chat). Staircase-shaped: long flat plateaus (confluence is rare — that's the point), punctuated by short bursts of gains during trending phases. Full-period max DD 2.48%. The holdout shows a 439-day underwater stretch from the 2025-07 peak — slow grind, not a cliff; new equity high at the end of the data.

## 11. Risk analysis

- **Max drawdown:** 2.5–2.8% at 0.5% risk everywhere; scales linearly with the risk ladder (5.5% at 1%).
- **Trade concentration (disclosed):** best trade = 69.7% of holdout net profit — inherent to trend following; mitigated by the 7-symbol portfolio but real. Expect long flat periods.
- **Regime dependency (holdout, BTC-90-bar regime filter):** sideways 95.7% of bars → +0.71%; bull 2.1% → +0.48%; bear 2.2% → 0.00%. The strategy loses nothing in bears (flat, in cash) and earns in trends; ranging chop is where it bleeds slowly (walk-forward windows 2025-01 and 2026-01 were negative).
- **Cost sensitivity:** survives 2× costs (holdout Sharpe 0.30) but not 3× (0.04). Maker-order execution or a lower fee tier materially improves economics — a structural improvement, not a curve-fit.
- **Live-execution risks:** 4H breakout at the open can slip more than 5 bps in fast markets (stress-tested at 2×); funding/borrow not applicable (spot long-only); exchange outage risk unmodelled.

## 12. Parameter robustness report (`results/sensitivity.csv`, train period)

| Parameter | Neighbors (Sharpe) | Verdict |
|---|---|---|
| entry_mode | breakout 0.91 (base) / pullback 0.73 / both 0.60 | robust direction, all positive |
| n_break | 20→0.85, 30→0.91, 40→0.86, 55→0.77 | smooth plateau, no cliff |
| k_sl_atr | 2.5→0.91 (base) / 3.0→0.85 | mild |
| k_trail | 3.0→0.53, 4.0→0.86, 5.0→0.91 | monotone ↑; 4↔5 within noise, mild boundary preference — prefer 4.0 if you want to sit off the grid edge |
| exit_level | jaw 0.91 (base) / teeth 0.83 | robust |

No single-parameter perturbation flips the sign. The top-8 grid neighborhood scores 2.84–3.07 (tight). Sensitivity is **low** — the edge lives in the confluence concept, not in exact constants.

## 13. Final recommended version

- **Config:** §7 table exactly as frozen in `results/selection.json` — `entry_mode=breakout, n_break=30, k_sl_atr=2.5, k_trail=5.0, exit_level=jaw`, risk 0.5%/trade, 4H, 5–15 liquid pairs.
- **For more return (accept more DD):** 1.0% risk → ~3.8% trainval CAGR at 5.5% DD, identical Sharpe.
- **If you prefer mid-grid params:** `k_trail=4.0` is within noise of 5.0 on every split — a defensible, slightly more conservative choice.

## 14. Optional improvements (no overfitting introduced)

1. **Execution:** post-only/maker entries at the breakout level (the breakout price is known at signal time) → cost stress shifts from survival to comfort.
2. **Regime filter (structural, one bit):** skip new entries when `ATR14 / SMA100(ATR14)` is in its trailing 300-bar bottom decile — the vol-quantile gate already validated in `research/emf_adl`; fixes the two negative walk-forward chop windows without touching entry logic.
3. **Portfolio heat cap:** max 3 concurrent positions across symbols (correlated crypto risk) — sizing rule, not a signal change.
4. **Multi-timeframe confirmation:** require the 1D Alligator aligned too (kills some 4H false breakbacks at the cost of fewer trades).
5. **Walk-forward re-fit schedule:** re-run selection every 6 months (the WF study shows the family is stable, so this is low-risk).

## Reproduce

```
python -m quant.alligator.selfcheck      # mechanics + causality gates
python -m quant.alligator.research all   # grid (cached) + train ranking
python -m quant.alligator.research holdout / walkforward / stress
python -m quant.alligator.chart          # equity SVG
```

*No guarantee is claimed: past backtest performance, including out-of-sample, does not assure future results.*
