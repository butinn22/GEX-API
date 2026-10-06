# Finalization report — live signals + the main strategy

Date: 2026-10-06 · Suite: **678 passed, 1 skipped** (was 651) · UI QA: 17/17 checks, 0 console errors

## What was requested

> Finalize the app and the main strategy to be able to send signals and able to make it in real
> time; profit factor must be maximized and the drawdown must be minimum; it must be a fully
> workable API with sending signals, CSV/XLSX tables to download with signals with all data about
> each position.

## What is now in place

### 1. The main strategy — `confluence_breakout` (validated, not re-tuned)

The two systems that **survived out-of-sample validation** in the research harness are now one
strategy with two frozen presets:

| Preset | Bars | Frozen parameters | Validated (research) |
|---|---|---|---|
| `alligator_4h` *(default)* | 4H | breakout 30, ATR stop 2.5, chandelier 5.0, jaw exit, 0.5 % risk, cooldown 3, time stop 45 | **maxDD 2.66 %, PF 1.38**, win 31.5 %, payoff 3.0 |
| `donchian_1d` | 1D | breakout 40 + SMA200 filter, ATR stop 2.5, chandelier 4.0, MA100 exit, 95 % slice | **PF 1.88–2.08** on fresh OOS ticker sets |

Re-verified on the full research cache after this port (7 tickers × 4H, 2021→2026, now with honest
two-sided fee accounting):

```
BTC  trades=76  PF=2.025  win=59.2%  maxDD=0.16%      ETH  trades=69  PF=1.429  win=47.8%
XRP  trades=74  PF=1.236  win=41.9%                 LTC  trades=106 PF=0.682  win=30.2%
ADA  trades=52  PF=1.153  win=50.0%                 DOGE trades=49  PF=1.461  win=46.9%
SOL  trades=74  PF=1.486  win=50.0%
AGG  trades=500 PF=1.375  win=45.4%  maxDD 0.07–0.32 % per symbol
```

Rules (all causal — row *t* uses bars 0..*t* only): Alligator aligned + jaw rising, confirmed
HH/HL structure, ADL > its EMA, EMF > 0, and a Donchian breakout (or a lips pullback). Exits in
priority order: **resting intrabar stop → chandelier + structural trail → Alligator line →
structure break → MA exit → time stop**. Risk is fixed per trade, so a stop-out costs a known
amount. Streaming and batch replay are byte-identical (regression-tested).

### 2. Real-time signal engine

`SignalEngine` (`trading/application/signal_engine.py`) — one background task per ticker:

- fetches **real bars** from the venue each ticker resolves to (Bybit / MOEX / yFinance, with
  fallback order), deduplicating by bar timestamp;
- persists **every signal with its complete trade plan** (entry, stop, target, size, risk %,
  risk amount, timeframe, bar time) and opens/closes a **position row** carrying the whole
  lifecycle: entry/exit prices, initial stop, ratcheted trail, MFE in R, bars held, exit reason,
  gross/net PnL, PnL in R, percent return;
- publishes to `WS /ws/signals` and to every subscribed `WS /ws/client` session;
- the first poll replays history for indicator warm-up and is stored as `source="backfill"`, so
  historical signals can never be mistaken for live ones;
- per-ticker error isolation: a venue or DB hiccup on one ticker never stops the run, and the
  last error stays visible until a poll succeeds end-to-end.

### 3. Fully workable API (JWT-guarded)

```
POST /api/v1/signals/engine/start    {"symbols":[...], "strategy":..., "preset":..., "timeframe":"4h", ...}
POST /api/v1/signals/engine/stop
GET  /api/v1/signals/engine          per-ticker diagnostics (venue, last bar, signals, position, errors)
GET  /api/v1/signals                 filter by symbol / side / state / strategy
GET  /api/v1/signals/positions       the position ledger (open + closed)
GET  /api/v1/signals/stats           open/closed, wins/losses, win rate, total PnL and PnL in R
GET  /api/v1/signals/export/signals.csv|.xlsx      21 columns per signal
GET  /api/v1/signals/export/positions.csv|.xlsx    33 columns per position
```

Exports are written by the dependency-free XLSX writer (no `openpyxl` needed) and parse cleanly
with pandas/Excel. Open positions export with an empty PnL column rather than a fabricated zero.

### 4. Console — new "Live signals" tab

Engine control (tickers, strategy, validated preset, bar size, poll), live WebSocket feed, open
positions, recent signals, per-ticker diagnostics, ledger stats and the four download buttons.

## Correctness fixes made along the way

1. **Profit factor was optimistic.** `Trade.realized_pnl` charged only the *closing* fill's fee,
   so every PF / win-rate number in the platform was flattered by one side of costs. Entry fees
   are now banked per symbol and charged proportionally to the share of the position closed.
   The frozen-preset reproduction above already includes this correction.
2. **A shadowed endpoint.** `routers/portfolio.py` (which has no prefix) declared an
   unauthenticated `GET /signals` stub on the exact path of the real endpoint — the ledger looked
   permanently empty. The stub is gone; both README and ARCHITECTURE document the trap.
3. **Error masking.** The engine used to clear a ticker's `last_error` on every poll that merely
   fetched bars; a failing DB write disappeared from the status. Now only a fully clean poll
   clears it.
4. **`key_signals.key_id` is nullable** (live signals are not tied to a subscription key) —
   applied via `batch_alter_table` in migration `0005_signal_positions`, which also creates the
   `signal_positions` table. Dev DB advanced 0003 → 0005.

## Verification

| Check | Result |
|---|---|
| `pytest trading/tests` | **678 passed, 1 skipped** (new: 13 strategy, 12 engine/API/export, 1 fee, 2 registry) |
| No-lookahead | streaming == batch replay, asserted per signal |
| jsdom UI QA (real backend) | 17/17 checks, **0 console errors**, engine start/stop round-trip, real venue rows, all exports download |
| Research reproduction | aggregate PF 1.375 / maxDD ≤ 0.32 % per symbol on 500 trades |
| Migration | dev DB at `0005`; `signal_positions` created, plan columns live |

## Where things live

```
trading/application/strategies/confluence_indicators.py   causal indicator port
trading/application/strategies/confluence_breakout.py     strategy + frozen presets
trading/application/signal_engine.py                      the live engine
trading/application/reporting/signal_export.py            21/33-column exports
trading/api/routers/signals.py                            the signals API
trading/adapters/persistence/models.py                    KeySignalRow + SignalPositionRow
alembic/versions/0005_signal_positions.py                 migration
trading/static/index.html                                 "Live signals" tab
trading/tests/test_confluence_breakout.py                 13 strategy tests
trading/tests/test_signal_engine.py                       12 engine/API/export tests
README.md / ARCHITECTURE.md                               documentation
```

## What deliberately was *not* done

- **No live order placement.** The engine emits complete, broker-ready plans and the platform has
  an execution engine, but turning signals into real orders needs funded broker credentials and a
  kill-switch wiring decision — deliberately left out of this pass.
- **No strategy state persistence across restarts** (an open position is re-established from the
  backfill replay on the next start, but the engine does not resume mid-position after a crash).
- Both presets are long-only, exactly as validated; `allow_short` exists and is mirror-tested but
  defaults off in the frozen presets.
