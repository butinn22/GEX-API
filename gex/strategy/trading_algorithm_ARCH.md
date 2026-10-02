"""Architecture analysis for Pine Script → Python translation.

## Phase 0: Analysis

1. **Script type:** `strategy` — multi-strategy with OR union.
2. **Stateful variables:** `ph, pl, phL, plL` (pivot tracker), `vwap` (polyline/points array,
   `p, vol` cumulative), `rising, falling` (two-pole filter counters), `lastAddBar`,
   `lastTPCloseBarLong/Short`, `totalGreenWeight/RedWeight` and `greenBars/redBars`
   (small trend table, reset every `LengthBarsAmount` bars).
3. **Multi-timeframe:** `request.security` fetches std and HA tickers at `timeframe.period`
   (current TF). No lower/higher TF calls — no lookahead risk in Python.
4. **Repainting risk:** `ticker.heikinashi` can repaint on real-time bars (HA needs next bar's
   close). In Python we process fully-closed historical bars → no repainting.

## Architecture Plan

┌─────────────────────────────────────────────────────────────┐
│                    CombinedEMFMFStrategy                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Preprocess: HA candles (vectorized), hybrid candles       │
│ 2. novelsrc = avg(hlcc4, AvgCandle, sourceformas)           │
│ 3. Common EMAs (novelsrc-based + close-based)               │
│ 4. Bar trend analysis (std + hybrid, 15-bar weighted)       │
│ 5. Flat zone filter                                          │
│ 6. Adaptive VWAP (pivot-based, exponential decay)            │
│ 7. ADL chain: diff → cumsum → EMA2 → ADL                   │
│ 8. ADL-RSI (ubb/lbb/lm)                                     │
│ 9. ADL-BB/linreg (basisbb, upperbb, lowerbb)                │
│ 10. ADL EMAs/SMAs (adl50/100/200/1000)                      │
│ 11. MACD/Signal/TL on ADL                                   │
│ 12. Two-pole filter (numba jit) on ADL + novelsrc           │
│ 13. Strategy A entries/exits/adds                            │
│ 14. Strategy B entries/exits/adds                            │
│ 15. Combined OR: longEntry = longEntryA or longEntryB       │
│ 16. TP with cooldown / trailing / exit execution            │
│ 17. Small trend table (reset counters)                      │
│ 18. TEMA/DEMA decorative plots                               │
├─────────────────────────────────────────────────────────────┤
│ Output: DataFrame with signal columns + metadata columns     │
└─────────────────────────────────────────────────────────────┘

## Performance-critical sections
- **Adaptive VWAP**: previously used nested O(n²) loop. Keep single-pass O(n).
- **Two-pole filter**: recursive by definition → `numba.jit`.
- **Bar trend analysis**: vectorized with rolling window (already O(n) via shift())
- **Everything else**: fully vectorized pandas/numpy.

## Lookahead bias safeguards
- All `shift()` calls use positive shift (past data only)
- Signals generated on bar `i` reference only data up to bar `i`
- No `request.security` with future data
- HA candles use shift(1) for open (Pine semantics: ha_open = (ha_open[1] + ha_close[1])/2)
"""
