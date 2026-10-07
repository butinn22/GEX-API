# Walk-forward re-validation — frozen `confluence_breakout` presets (engine v2)

**Engine:** `2.0.0` · **generated:** 2026-10-07T12:06:48.840640+00:00 · **data:** real `quant/cache` (no synthetic).

> The presets are **frozen** research artefacts, so this is a calendar out-of-sample evaluation (no parameter fitting → no train/validation leakage). Every figure below is read from `results.json`.

## 0. Verdict
- **`alligator_4h` (4h) — NOT ROBUST.** Positive OOS windows **3/5**, mean breadth **48.0%** (share of tickers profitable per window), harsh-cost Sharpe **0.22**, no-lookahead **pass**, best-trade removal structural check **pass**. Full-sample return 3.53%, Sharpe 0.60, median ticker Sharpe 0.22.
- **`donchian_1d` (1d) — NOT ROBUST.** Positive OOS windows **3/5**, mean breadth **33.3%** (share of tickers profitable per window), harsh-cost Sharpe **0.45**, no-lookahead **pass**, best-trade removal structural check **pass**. Full-sample return 31.89%, Sharpe 0.49, median ticker Sharpe 0.19.

The robustness bar (≥4/5 positive windows **and** mean breadth ≥50% **and** harsh-cost Sharpe > 0 **and** both correctness checks) is **not met by either preset**. Breadth is the binding failure: returns are carried by a minority of tickers.

## 1. Constraints restated
- Real bars only (repo cache); no synthetic candles, no fabricated fills.
- Point-in-time: the project engine fills at bar `i+1` open off a bar-`i` close signal; verified by truncation (below).
- Frozen presets: **no** parameter search on any window, so the OOS windows are clean.
- Equal-weight, live-ticker-aware panel (tickers are averaged only where they have data).
- No funding modelled (spot-style cache; perps funding not available in the cache) — an untested axis.

## `alligator_4h` (4h)

**Universe:** 21 USDT perps. Manifest (n bars, first→last, integrity, sha256 prefix):

| sym | bars | first | last | badOHLC | zeroVol | nonMono | sha256 |
|---|---|---|---|---|---|---|---|
| ADA | 10851 | 2021-10-21 | 2026-10-03 | 0 | 0 | 0 | 0aa75ea6fb6b |
| ALGO | 10718 | 2021-11-12 | 2026-10-03 | 0 | 0 | 0 | 53fb2032b105 |
| ATOM | 10718 | 2021-11-12 | 2026-10-03 | 0 | 0 | 0 | 0a380e0acfe5 |
| AVAX | 10718 | 2021-11-12 | 2026-10-03 | 0 | 0 | 0 | b674c9dfe511 |
| BNB | 10010 | 2022-03-10 | 2026-10-03 | 0 | 0 | 0 | 1a2c8163cd51 |
| BTC | 11497 | 2021-07-05 | 2026-10-03 | 0 | 0 | 0 | 4eecf3b7da8d |
| CRV | 10882 | 2021-10-16 | 2026-10-03 | 0 | 0 | 0 | 752ad1cb8d11 |
| DOGE | 11157 | 2021-08-31 | 2026-10-03 | 0 | 0 | 0 | 5d48a5d78391 |
| DOT | 11157 | 2021-08-31 | 2026-10-03 | 0 | 0 | 0 | 5ade178d16d1 |
| ETC | 9170 | 2022-07-28 | 2026-10-03 | 0 | 0 | 0 | 94fd2316a927 |
| ETH | 11497 | 2021-07-05 | 2026-10-03 | 0 | 0 | 0 | 8fe801293837 |
| FIL | 10851 | 2021-10-21 | 2026-10-03 | 0 | 0 | 0 | 0eb2789236cc |
| LINK | 11023 | 2021-09-22 | 2026-10-03 | 0 | 0 | 0 | 2be635412634 |
| LTC | 11157 | 2021-08-31 | 2026-10-03 | 0 | 0 | 0 | 865858077a04 |
| NEAR | 10004 | 2022-03-11 | 2026-10-03 | 0 | 0 | 0 | 5f9f892d90bb |
| SAND | 10779 | 2021-11-02 | 2026-10-03 | 0 | 0 | 0 | 418c7d51f5cf |
| SOL | 10851 | 2021-10-21 | 2026-10-03 | 0 | 0 | 0 | e17ee5660293 |
| TRX | 10112 | 2022-02-21 | 2026-10-03 | 0 | 0 | 0 | 49a3552c9b50 |
| UNI | 11023 | 2021-09-22 | 2026-10-03 | 0 | 0 | 0 | 9bbd1424b587 |
| XLM | 11122 | 2021-09-06 | 2026-10-03 | 0 | 0 | 0 | 7ad4ef2c2f03 |
| XRP | 11410 | 2021-07-20 | 2026-10-03 | 0 | 0 | 0 | 9296aca084c7 |

**Correctness (no-lookahead by truncation):** prefix signals 58 == full signals before the cut 58 → **identical**.

**Walk-forward windows (calendar-aligned):**

| # | start | end | return | maxDD | trades | live | breadth |
|---|---|---|---|---|---|---|---|
| 1 | 2021-07-05 | 2022-07-23 | -0.77% | 0.88% | 127 | 20 | 35% |
| 2 | 2022-07-23 | 2023-08-10 | 0.36% | 1.30% | 234 | 21 | 48% |
| 3 | 2023-08-10 | 2024-08-28 | 1.72% | 1.08% | 282 | 21 | 71% |
| 4 | 2024-08-28 | 2025-09-15 | 2.25% | 1.19% | 295 | 21 | 48% |
| 5 | 2025-09-15 | 2026-10-03 | -0.02% | 0.70% | 209 | 21 | 38% |

- **Worst window:** #1 2021-07-05→2022-07-23 return -0.77%, maxDD 0.88%.
- **Holdout (last window, read once):** return -0.02%, breadth 38%.

**Cost robustness (Sharpe):**

| scenario | sharpe | total_return |
|---|---|---|
| base | 0.599 | 3.53% |
| slippage_2x | 0.523 | 3.07% |
| slippage_4x | 0.371 | 2.16% |
| fees_2x | 0.447 | 2.61% |
| harsh | 0.220 | 1.26% |

**Adversarial:**
- Remove best ticker (`TRX`): return becomes 2.79%, Sharpe 0.47.
- Remove best trade — ledger (exact): PF 1.398 → 1.332 (1147→1146 trades).
- Remove best trade — engine: suppress entry over 2024-11-12→2024-11-26; specific trade present before=1 after=0 (**removed**); raw count 1147→1147 (a re-entering strategy can form a replacement trade, so the count need not drop by one).
- Concentration: best trade = 16.55% of net P&L, top-5 = 64.48% (currency); scale-free best-trade return-share 50.04%.

**Per-year panel return:** 2021:-0.62%, 2022:-0.15%, 2023:1.64%, 2024:2.59%, 2025:-0.21%, 2026:0.26%

## `donchian_1d` (1d)

**Universe:** 21 USDT perps. Manifest (n bars, first→last, integrity, sha256 prefix):

| sym | bars | first | last | badOHLC | zeroVol | nonMono | sha256 |
|---|---|---|---|---|---|---|---|
| ADA | 1809 | 2021-10-21 | 2026-10-03 | 0 | 0 | 0 | b8c486c53868 |
| ALGO | 1787 | 2021-11-12 | 2026-10-03 | 0 | 0 | 0 | afad53468906 |
| ATOM | 1787 | 2021-11-12 | 2026-10-03 | 0 | 0 | 0 | cdb3e1148187 |
| AVAX | 1787 | 2021-11-12 | 2026-10-03 | 0 | 0 | 0 | 8ac44f79d722 |
| BNB | 1669 | 2022-03-10 | 2026-10-03 | 0 | 0 | 0 | 4cbfd56f990b |
| BTC | 1917 | 2021-07-05 | 2026-10-03 | 0 | 0 | 0 | 6ed62e7f42e6 |
| CRV | 1814 | 2021-10-16 | 2026-10-03 | 0 | 0 | 0 | 654c8efcb086 |
| DOGE | 1860 | 2021-08-31 | 2026-10-03 | 0 | 0 | 0 | 423993dce86d |
| DOT | 1860 | 2021-08-31 | 2026-10-03 | 0 | 0 | 0 | b8bb8d597bbd |
| ETC | 1529 | 2022-07-28 | 2026-10-03 | 0 | 0 | 0 | e5403b327b86 |
| ETH | 1917 | 2021-07-05 | 2026-10-03 | 0 | 0 | 0 | 88d76c93562f |
| FIL | 1809 | 2021-10-21 | 2026-10-03 | 0 | 0 | 0 | 285931757c6b |
| LINK | 1838 | 2021-09-22 | 2026-10-03 | 0 | 0 | 0 | 17c15d1535fe |
| LTC | 1860 | 2021-08-31 | 2026-10-03 | 0 | 0 | 0 | cdd736e1c4b8 |
| NEAR | 1668 | 2022-03-11 | 2026-10-03 | 0 | 0 | 0 | e05fd65bd0e3 |
| SAND | 1797 | 2021-11-02 | 2026-10-03 | 0 | 0 | 0 | 8721ed19fd5a |
| SOL | 1809 | 2021-10-21 | 2026-10-03 | 0 | 0 | 0 | dec882d6a5c8 |
| TRX | 1686 | 2022-02-21 | 2026-10-03 | 0 | 0 | 0 | a2aa07b7037f |
| UNI | 1838 | 2021-09-22 | 2026-10-03 | 0 | 0 | 0 | 9c0f25052d28 |
| XLM | 1854 | 2021-09-06 | 2026-10-03 | 0 | 0 | 0 | 4361c84805e4 |
| XRP | 1902 | 2021-07-20 | 2026-10-03 | 0 | 0 | 0 | 19eaf32d5aa7 |

**Correctness (no-lookahead by truncation):** prefix signals 8 == full signals before the cut 8 → **identical**.

**Walk-forward windows (calendar-aligned):**

| # | start | end | return | maxDD | trades | live | breadth |
|---|---|---|---|---|---|---|---|
| 1 | 2021-07-05 | 2022-07-23 | 0.00% | 0.00% | 0 | 20 | 0% |
| 2 | 2022-07-23 | 2023-08-10 | -10.68% | 11.04% | 27 | 21 | 5% |
| 3 | 2023-08-10 | 2024-08-27 | 22.50% | 12.12% | 49 | 21 | 71% |
| 4 | 2024-08-27 | 2025-09-14 | 15.85% | 16.88% | 38 | 21 | 57% |
| 5 | 2025-09-14 | 2026-10-03 | 4.17% | 2.66% | 12 | 21 | 33% |

- **Worst window:** #2 2022-07-23→2023-08-10 return -10.68%, maxDD 11.04%.
- **Holdout (last window, read once):** return 4.17%, breadth 33%.

**Cost robustness (Sharpe):**

| scenario | sharpe | total_return |
|---|---|---|
| base | 0.491 | 31.89% |
| slippage_2x | 0.482 | 31.14% |
| slippage_4x | 0.464 | 29.66% |
| fees_2x | 0.473 | 30.40% |
| harsh | 0.447 | 28.18% |

**Adversarial:**
- Remove best ticker (`NEAR`): return becomes 22.62%, Sharpe 0.39.
- Remove best trade — ledger (exact): PF 1.655 → 1.404 (126→125 trades).
- Remove best trade — engine: suppress entry over 2024-10-29→2024-12-11; specific trade present before=1 after=0 (**removed**); raw count 126→126 (a re-entering strategy can form a replacement trade, so the count need not drop by one).
- Concentration: best trade = 38.42% of net P&L, top-5 = 103.52% (currency); scale-free best-trade return-share 27.11%.

**Per-year panel return:** 2021:0.00%, 2022:-3.80%, 2023:8.05%, 2024:18.73%, 2025:-1.48%, 2026:5.74%

## 2. Untested axes (what this study does NOT establish)
- No funding/borrow cost (perps funding absent from the cache).
- No maker/taker fee tiering beyond the 1×/2× sweep; no latency model.
- Only the two shipped presets were evaluated (frozen) — the strategy's parameter neighbours were **not** swept; this is a re-validation, not a re-optimisation.
- Panel is USDT-perp crypto only; equities/FX were excluded.
- The engine's `Trade.entry_time` stamps the exit bar (a reporting quirk); the event ledger was used for holding windows, so this does not affect the results.

## 3. Conditions where these presets should be disabled
- Panel breadth per window falls below 50% (currently already at/below it).
- Harsh-cost Sharpe (4× slippage + 2× fees) turns negative.
- Two consecutive OOS windows negative.
- Best-ticker removal turns the full-sample return negative.
- Rolling 6-month panel drawdown exceeds the worst window in this study.

## 4. Reproducibility
- `harness.py` reads `quant/cache` (hashes in the manifest above) and writes `results.json`.
- Command: `PYTHONPATH=. .venv/Scripts/python.exe deliverables/software-company/audit/walkforward/harness.py`
- Engine version and all inputs are recorded in `results.json`.
