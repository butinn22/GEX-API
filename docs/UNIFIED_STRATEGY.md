# Unified Strategy — Trend Confluence × EMF+ADL × Momentum

This document specifies the **unified strategy**, the **per-ticker parameter
presets**, the **redesigned optimizer**, the **signal API keys**, and the
**`/API_KEY/{key}` live dashboard**. It records what was built, how it works,
and every assumption made where the requirements left room for interpretation.

---

## 1. The unified strategy: `trend_confluence_unified`

**Trend Confluence stays the core.** The unified strategy
(`trading/application/strategies/trend_confluence_unified.py`,
class `UnifiedTrendStrategy`, version `1.0.0`) **subclasses**
`TrendConfluenceStrategy`, so every existing behaviour — regime detection,
trendline engine, confluence zones, options walls, gamma-flip filter,
invalidation exits, ATR trailing, RANGE handling, warm-up gate, batch
`prepare()` fast path with streaming fallback — is inherited, not reimplemented.

On top of that core, two restored components are merged in:

### 1.1 EMF + ADL (`gex_emf` / `EMAFilterTrendStrategy`) — integrated

* The **full `StrategySettings` parameter set** (all ~50 EMF+ADL fields:
  `tp_percent`, `trailing_percent`, `length_adl`, `damping`, `length_bb`,
  `mult_bb`, `rsi_length`, `atr_length`, `verification_threshold`,
  `use_atr_stops`, `atr_tp_mult`, `atr_sl_mult`, regime/flat detector knobs, …)
  is available in the unified schema as a nested **`emf`** block.
* **Entries**: `emf_mode` selects how the EMF+ADL combined entry signal
  (`combined_long_entry` / `combined_short_entry`, i.e. strategy A ∨ B) interacts
  with a Trend-Confluence entry:
  * `"bonus"` (default) — the TC entry always fires; EMF+ADL agreement adds
    `emf_bonus` (default 0.25) to the signal strength. Chosen as the default
    because the unified strategy must be an *enhanced* Trend Confluence: it can
    never produce fewer entries than pure TC, so it can never silently go quiet
    when the two entry logics disagree.
  * `"require"` — a TC entry only fires when EMF+ADL agrees at the same bar
    (strict intersection: higher precision, fewer trades — best selected per
    ticker by the optimizer).
  * `"off"` — EMF entries are ignored (pure TC behaviour).
* **Exits** (all additional, never replacing TC exits):
  * `use_emf_exits=True` — the EMF combined indicator exit
    (`combined_long_exit` / `combined_short_exit`) closes the position.
  * `use_risk_exits=True` — the EMF take-profit / trailing-stop overlay
    (`take_profit_price` / `trailing_stop_price` from `_RiskMixin`,
    ATR-based or percentage-based per the `emf` block) closes the position.
    The ATR is frozen at entry, exactly like the standalone `gex_emf` adapter.
* If the EMF frame is unavailable (e.g. history shorter than 60 bars), a
  `"require"` gate **passes through** and the miss is counted in
  `emf_skip_count` — a missing confirmation must not silently zero the strategy.

### 1.2 Momentum — integrated

* Fields: `use_momentum`, `momentum_period` (default 10, the standalone
  strategy's `period`), `momentum_mode`, `momentum_bonus`.
* `momentum_mode`:
  * `"gate"` — longs need ROC(`momentum_period`) > 0, shorts < 0.
  * `"bonus"` (default) — ROC agreement adds `momentum_bonus` (0.15) to strength.
  * `"off"` — ignored.

Default configuration (`emf_mode="bonus"`, `momentum_mode="bonus"`) is a
genuine **merge**, not a config union: every TC entry fires, is sized up by
EMF+ADL and momentum agreement, and is managed by TC exits **plus** the EMF
indicator exit and the EMF take-profit / trailing-stop overlay — a
trend-following strategy that demonstrably uses all three parameter sets.

### 1.3 The unified parameter schema

One flat/nested JSON object per ticker:

```
{
  "strategy": "trend_confluence_unified",
  "strategy_version": "1.0.0",
  "symbol": "NVDA",
  "timeframe": "1d",
  "source": "auto",
  "params": {
    ... all 27 TrendConfluence fields (ema_fast..atr_trail_mult) ...,
    "use_emf": true, "emf_mode": "require", "emf_bonus": 0.25,
    "use_emf_exits": true, "use_risk_exits": true,
    "use_momentum": true, "momentum_period": 10,
    "momentum_mode": "bonus", "momentum_bonus": 0.15,
    "emf": { ... full EMF+ADL StrategySettings block ... },
    "options": { ... options walls snapshot ... }
  }
}
```

`UnifiedTrendParams` **inherits** `TrendConfluenceParams` (so TC fields can
never drift out of sync) and adds the integration knobs above.

### Assumptions documented

* The requirement calls the strategy "EMF + ADL"; in this codebase it is
  `gex_emf` / `EMAFilterTrendStrategy` ("EMD + ADL" in some docs). Same thing.
* "Momentum profile" — no saved Momentum profile existed anywhere; the
  standalone strategy's current default (`period=10`, the value used by the
  dashboard demo portfolio) is adopted as `momentum_period` default.
* Name collision: both TC and EMF+ADL have a `use_trailing` knob. The unified
  schema keeps TC's flat `use_trailing` (ATR trailing) and puts EMF's
  `use_trailing` / `use_take_profit` / `tp_percent` / … inside the nested
  `emf` block — no field is lost, none is ambiguous.
* The existing `trend_confluence` and `gex_emf` strategies remain registered
  and untouched — the unified strategy is a new registry entry, so no existing
  backtest/result changes.

---

## 2. Per-ticker parameter presets (`strategy_presets`)

New table + repository + service (`trading/application/presets.py`):

| Column | Purpose |
|---|---|
| `id` | PK |
| `symbol` | ticker (indexed, with strategy) |
| `strategy` | strategy name (default `trend_confluence_unified`) |
| `strategy_version` | version the params were validated against |
| `params_json` | the full unified parameter object |
| `source` | `backtest` \| `manual` \| `optimizer` |
| `optimizer_run_id` | traceability to the optimizing run |
| `is_default` | one default preset per (symbol, strategy) |
| `created_at` / `updated_at` | metadata |

* **Backtest is the source of truth**: `POST /api/v1/presets/from-backtest/{symbol}`
  runs the backtest module with the current/default params over real data for
  that ticker and saves the result as the default preset (source=`backtest`).
* **Presets store the *complete* configuration**: both automated paths
  (`from-backtest` and the optimizer's `save_preset`) expand their (partial)
  overrides into the strategy's full resolved parameter set — every TC field,
  the integration knobs and the complete resolved `emf` block
  (`UnifiedTrendStrategy.resolved_params()`, ~35 flat fields + the 52-field
  EMF block) — so a saved preset is self-contained per ticker and an API-key
  snapshot taken from it carries **all** strategy settings. Manual
  `POST /api/v1/presets` / `PATCH` store exactly what the client sends
  (partial overrides are legal; the strategy fills its defaults at build time).
* Presets can be edited (`PATCH`), re-optimized (the optimizer stores its best
  params as source=`optimizer` presets), listed, deleted.
* Alembic migration `0004_presets_signal_keys.py` adds the table.

## 3. Optimizer redesign

`trading/application/backtest/optimize.py`:

* **Objective is selectable** — `sharpe` (default, unchanged), `sortino`,
  `calmar`, `total_return`, `profit_factor`, `win_rate`, `max_drawdown`
  (maximised as `-max_drawdown`). The overfit/thin-sample discounting logic is
  preserved and applied to the chosen metric.
* **`UNIFIED_GRID`** — a default sweep over the unified schema
  (`zone_atr`, `min_confluence`, `atr_trail_mult`, `emf_mode`,
  `momentum_mode`, nested `emf.*` keys like `emf.atr_tp_mult`). Grid keys use
  dotted paths for nested fields (`"emf.atr_tp_mult"` → sets
  `params["emf"]["atr_tp_mult"]`).
* Deterministic: grid enumeration is `itertools.product` (order-stable);
  no RNG anywhere in the sweep.
* **Global optimization** — `optimize_all_tickers(...)` iterates the ticker
  universe (or an explicit list), runs `optimize_strategy` per ticker,
  **isolates failures** (one bad ticker is logged and skipped, never aborts the
  run), and reports per-ticker outcomes. Exposed as
  `POST /api/v1/backtest/optimize/global` + `GET …/optimize/global/{run_id}`
  progress polling (current ticker, completed/failed counts, ETA, results).
* **Individual optimization** — the existing `POST /api/v1/backtest/optimize`
  gains `objective` + `save_preset` (persists best params as the ticker's
  default preset, source=`optimizer`).

## 4. Signal API keys (`signal_keys`)

Distinct from the existing broker-credential `api_keys` table (which stays
untouched): a signal key **references** the complete strategy configuration.

* Key format: `sk_` + 43 chars `secrets.token_urlsafe(32)` — unique, securely
  generated, stored in plain text (it is a configuration reference, not a
  secret credential; the platform DB is local/encrypted-at-rest by policy).
* `config_json` stores: strategy name+version, exchange (broker), tickers with
  per-ticker params (or preset references), timeframe, source, limit, costs.
* Lifecycle: create → list → activate/deactivate → revoke (soft, `revoked_at`).
* `last_used_at` is stamped on every signal generation.

### Signal generation (`POST /api/v1/signal-keys/{key}/generate`)

For each enabled ticker: load the latest bars, build the unified strategy from
the key's config (falling back to the ticker's default preset), and run the
**same portfolio backtest engine** the backtest module uses. Derived rows are
persisted: `key_signals` (timestamp, ticker, broker, side, signal type, reason,
strength, price, strategy+version, preset id, source) and `key_trades` (full
trade ledger with entry/exit times, prices, quantity, fees, gross/net PnL,
return %, holding duration, exit reason). Generation is **idempotent**: rows
for the key are recomputed over the whole lookback window on each refresh
(delete + reinsert), so the dashboard is always consistent with the latest
data and there is no incremental-state drift.

### Trade-event ledger: flips split into exit + entry

`key_trades` rows are reconstructed from the engine's per-fill event ledger
(`trade_log.py`). A **flip** (an opposite-side order larger than the current
position) is recorded as *two* events — the exit of the old side (with its
realized PnL) and the entry of the new side — via `events_from_fill()`.
Without the split, positions opened by a flip had no entry event and
closed-trade reconstruction could not pair them (a 44-trade backtest
produced 1 exportable trade). Both the backtest engine and the order-row
export replay use the same splitting, so their ledgers stay identical.

## 5. The `/API_KEY/{key}` dashboard

* `GET /API_KEY/{key}` — HTML dashboard. **The key in the URL is the
  credential** (documented: this is a local/personal deployment; the route is
  unguessable and revocable). Unknown key → 404 page; revoked → 410 page.
* **Charts equivalent to the backtest charts**: the dashboard reuses the exact
  server-side SVG chart functions (`charts.py`: `line_chart`, `drawdown_chart`,
  `histogram_chart`) on the key's live equity/drawdown/PnL distribution —
  same visuals as the backtest HTML report.
* **Metrics summary**: total trades, wins/losses, win rate, total P/L, avg
  P/L, max drawdown, profit factor, Sharpe, plus all other backtest metrics.
* **Exports**: `GET /API_KEY/{key}/trades.csv` and `…/trades.xlsx` with the
  complete column set (trade id, key, broker, ticker, strategy, version, preset,
  entry/exit timestamps, direction, prices, quantity, fees, gross/net PnL,
  return %, duration, exit reason, source).
* **Near-real-time**: the page auto-polls `/API_KEY/{key}/data` every 60 s,
  has a manual **Refresh** button, and always shows the last-update time.

## 6. Traceability & error handling

* Every signal/trade row carries: key id, exchange, strategy name + version,
  preset id, source (`live` | `replay`), timestamp.
* Optimizer runs get an `optimizer_run_id` recorded into presets.
* Missing data, failed tickers, revoked keys, empty histories and export
  failures surface as visible errors in API responses / dashboard, and are
  logged (structlog) with correlation ids.

## 7. UI workflow

New **"Optimize & Deploy"** tab in `trading/static/index.html`:

1. pick ticker(s) or a universe,
2. `Optimize Selected Ticker` / `Optimize All Tickers` (progress + results),
3. review/edit the saved preset,
4. `Choose API / Broker` (bingx | tbank) → `Create API Key`,
5. `Open Live Dashboard` (`/API_KEY/{key}`), download CSV/Excel, monitor.

## 8. Known limitations

* The dashboard equity curve is computed per trade-event granularity over the
  configured lookback window (the same engine/grid as a backtest run), refreshed
  on demand — not a websocket tick stream.
* Broker "ticker support" validation is category-based (crypto → bingx,
  ru → tbank, us/fx/sectors → signal-only, no direct order routing); orders are
  not auto-placed — keys generate **signals**.
* Global optimization is sequential per ticker (deterministic, cancellable);
  large universes take minutes per ticker.

---

## 9. API reference (new endpoints)

| Method & path | Purpose |
|---|---|
| `POST /api/v1/presets` | Save a preset (manual/backtest/optimizer source). |
| `GET /api/v1/presets?symbol=&strategy=` | List presets. |
| `GET /api/v1/presets/default/{symbol}` | The ticker's default preset (or `null`). |
| `POST /api/v1/presets/from-backtest/{symbol}` | **Backtest = source of truth**: validates params on real data, saves default preset with backtest metrics in the notes. |
| `PATCH /api/v1/presets/{id}` | Edit params / promote to default. |
| `DELETE /api/v1/presets/{id}` | Delete a preset. |
| `POST /api/v1/backtest/optimize` | Individual optimization — `objective` selectable, `save_preset` persists the winner. |
| `POST /api/v1/backtest/optimize/global` | Start all-tickers optimization → `run_id`. |
| `GET /api/v1/backtest/optimize/global/{run_id}` | Progress: current ticker, completed/failed, ETA, per-ticker results + errors. |
| `POST /api/v1/backtest/optimize/global/{run_id}/cancel` | Stop at the next ticker boundary. |
| `POST /api/v1/signal-keys` | Create a key for the chosen broker (requires tickers; params default to presets). |
| `GET /api/v1/signal-keys` | List keys (full config visible to the admin). |
| `DELETE /api/v1/signal-keys/{id}` | Revoke (soft). |
| `PATCH /api/v1/signal-keys/{id}?active=` | Enable/disable. |
| `POST /api/v1/signal-keys/{key}/generate` | Regenerate signals + paper trades (idempotent). |
| `GET /API_KEY/{key}` | Live dashboard (HTML — the key is the credential). |
| `GET /API_KEY/{key}/data` | Metrics, totals, latest signals/trades, last update. |
| `GET /API_KEY/{key}/charts` | Server-side SVG equity / drawdown / PnL histogram. |
| `GET /API_KEY/{key}/trades.csv` / `.xlsx` | Full-column trade exports. |
| `POST /API_KEY/{key}/refresh` | Regenerate now (used by the page's auto-refresh). |

Errors: unknown key → 404; revoked/disabled → 410; bad broker/params → 400;
data unavailable → 502. Nothing fails silently — generation errors are
returned in the report and shown in the dashboard.

## 10. What changed (files)

**New**

| File | Contents |
|---|---|
| `trading/application/strategies/trend_confluence_unified.py` | `UnifiedTrendParams` + `UnifiedTrendStrategy` (the merge). |
| `trading/application/presets.py` | `PresetService` (CRUD + `from_backtest` + `save_optimization`). |
| `trading/application/global_optimize.py` | `GlobalOptimizeRunner` (progress, failure isolation, ETA). |
| `trading/application/signal_keys.py` | `SignalKeyService` (create/revoke/generate/summary). |
| `trading/adapters/persistence/preset_repository.py` | Preset rows. |
| `trading/api/routers/presets.py`, `signal_keys.py`, `dashboard.py` | The three new routers. |
| `alembic/versions/0004_presets_signal_keys.py` | 4 new tables. |
| `trading/tests/test_trend_confluence_unified.py`, `test_presets.py`, `test_optimize_unified.py`, `test_signal_keys.py` | 34 new tests. |

**Modified** — `strategy_registry.py` / `strategy_factory.py` (register +
build the unified strategy; nothing existing changed behaviour), `backtest/
optimize.py` (objectives, nested grid keys, `UNIFIED_GRID`, deterministic
reconstruction), `backtest/trade_log.py` + `backtest/engine.py` (flip fills
split into exit+entry ledger events — `events_from_fill`), `reporting/
trade_export.py` (generic `table_to_csv/xlsx` + flip-split replay),
`api/schemas.py`, `api/routers/backtest.py`, `main.py` (mount routers),
`static/index.html` (Optimize & Deploy tab), `README.md`.

**Breaking changes** — none. The strategy-registry test that pinned the
exact strategy-name set was extended (documented). Every pre-existing
strategy, endpoint and result is untouched; the unified strategy is a new
registry entry and all new tables are additive (migration `0004`). The one
*behavioural* change in shared code: the backtest event ledger now emits
two events for a flip fill (exit + entry) instead of one (exit only) —
`BacktestResult.events` grows accordingly; consumers that assumed one event
per fill were updated (`trade_export.py`). The `Trade` ledger, metrics and
equity curves are unchanged.

## 11. Tests

`trading/tests` — the new suites cover: unified param schema/validation,
factory integration, EMF require/bonus/off equivalence vs pure
trend-confluence, momentum gate, prepare≡streaming equivalence, backtest
sanity; preset CRUD + single-default invariant + JSON round-trip +
complete-config snapshots from `from_backtest`/optimizer saves; optimizer
objectives + nested `emf.*` grid keys + determinism + global runner
(completion, preset persistence, failure isolation); signal-key lifecycle,
generation idempotency, preset resolution, broker-support warnings, the
`/API_KEY/{key}` page/data/charts/CSV/XLSX/refresh endpoints and 404/410
error paths; trade-ledger flip splitting (unit + engine level, regression
for the missing-entry-events bug).

An end-to-end smoke script against a live server ships as
`scripts/smoke_unified_workflow.py` (login → preset-from-backtest →
optimize+save → global optimize → key create → generate → dashboard →
exports → refresh → revoke → 410). Full suite: **833 passed, 1 skipped** (834 collected)
after the final fixes.
