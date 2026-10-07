# Alpha diagnostic run — real crypto panel (quantitative-alpha-architect pipeline)

**Date:** 2026-10-07 · **Dataset:** repo research cache `quant/cache/*_4h.csv` (real, no synthetic).
**Panel:** 10 liquid perps (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, AVAX, LTC), 4h closes,
inner-joined → **10,010 bars, 2022-03-10 → 2026-10-03**. Single-asset OHLCV = BTCUSDT 4h (11,497 bars).
**Toolkit:** `quantitative-alpha-architect` — `self_check.py` **49/49 passed** before the run.
**Artifacts:** `alpha_report.json`, `alpha_features.csv` (this folder).

> Scope: this is a **structure diagnostic**, not a strategy verdict and not a price forecast.
> It measures latent structure in the panel; the frozen-preset re-validation (Round-4 condition
> #1) remains a separate walk-forward exercise.

## Measured stages

| Stage | Quantity | Value |
| --- | --- | --- |
| 1 resample | chronological → volume bars | 11,497 → **1,493** bars (bucket 12,437; compression 7.70) |
| 2 transform | fractional `d` / DF after | d=0.2, `tau=-3.62` → **p<0.01 (stationary)**; DFA α price 1.508 → fractional 1.331 → log-ret 0.557 |
| 3 denoise (RMT) | MP bound / real factors | upper **1.0642**; **1 eigenvalue above** (7.221) → **1 real factor**, 90% noise |
| 4 extract | VPIN p95 / Kyle λ | VPIN p95 **0.6047** (≤0.70); λ mean 0.201, trend ratio 1.46 |
| 4 extract | transfer entropy (A2→A0 vs A0→A2) | **0.741 vs 0.701**, asym **0.027** → error correction, not lead-lag |
| 5 model | HMM 2-state | high-vol share **82.7% (first 60%) → 47.8% (last 40%)**; separation 1.458; transition 0.995/0.992 |
| 5 model | OU spread | θ=0.0002 → **half-life 3,468 bars ≫ 1,345 usable**; unit root **not** rejected |
| 5 model | tail dependence | λ_U=**0.642**, λ_L=0.633 (independent ref 0.05) |
| 6 validation | purged K-fold | 5 folds, **train/test overlap = 0**; a naive split would leak **1,370 rows** |

## Diagnostics (decision rules)

1. **No order-flow toxicity** — VPIN p95 0.60 ≤ 0.70 → balanced flow, no informed accumulation.
2. **No exploitable lead-lag** — TE asymmetry 0.027 → do **not** build a directional lead-lag rule.
3. **Symmetric coupling = cointegration channel** — trade the spread's deviation, not information direction.
4. **Correlation matrix noise-dominated** — only 1 factor above MP → naive PCA keeps spurious structure.
5. **No exploitable mean reversion** — OU half-life beyond sample → not tradeable as stat-arb here.
6. **Hidden tail dependence** — λ_U 0.64 vs 0.05 → assets co-crash far beyond linear correlation; size joint risk off the copula.

## Degenerate fits / caveats (reported, not hidden)

- **Kalman (stage 3) is degenerate on this series**: `r=0.0`, `q=3.03e6`, residual std 0.0,
  `grid_inner_solution=false` — the MLE is pinned to a grid edge (tiny observation noise relative
  to the search grid at this price scale). **Treat the Kalman denoising output as unreliable here.**
- **OU spread fit** sits at/behind the sample length (half-life 3,468 bars) and the unit root is not
  rejected — the OU parameters are not trustworthy; they are reported only as a non-finding.
- The panel uses a fixed inner-join window (2022-03 on), so earlier 2021 listing behaviour is excluded.

## What this establishes for the programme

The crypto panel the frozen `confluence_breakout` family trades is **one-factor, noise-dominated and
tail-dependent**, with **no measured lead-lag or mean-reversion edge** across these assets. That is
independent corroboration that the Round-4 conclusion holds: the shipped preset numbers must be
**re-validated by walk-forward** (not trusted on the in-sample re-run), and any edge must come from
per-asset timing under regime shifts — which the HMM shows have been frequent (high-vol share halved
from the first 60% to the last 40% of the sample).

## Reproduce

```bash
.venv/Scripts/python.exe <skill>/scripts/self_check.py                       # 49/49
.venv/Scripts/python.exe D:/TEMPORARYFILES/opencode/build_alpha_inputs.py     # builds inputs
.venv/Scripts/python.exe <skill>/scripts/alpha_pipeline.py \
    --input deliverables/software-company/audit/alpha-run/ohlcv.csv \
    --panel deliverables/software-company/audit/alpha-run/panel.csv \
    --out-dir deliverables/software-company/audit/alpha-run --target-bars 1500
```
