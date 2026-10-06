# PRD — Strategy Hub: Versioned Strategy Store for Lab, Backtest & Live API

**Status**: Draft v1.0 · **Owner**: Product (Alice) · **Repo**: `E:\gex api`

---

## 1. Product Goals

Upgrade the existing **Strategy Lab** (the `pine` console tab) from a per-symbol
parameter editor into the **central hub for backtesting and strategy
management**: one versioned, per-ticker strategy store that the Lab, the
Backtest module, and the live-signal API all read from and write to.

**What "central hub" means concretely:**

1. **Single source of truth** — every saved per-ticker configuration lives in
   one store (`strategy_presets`, extended) with identity, versioning, metrics
   snapshot, and deployment status. No configuration exists only in the UI or
   only in a request body.
2. **Test = deploy** — anything the Lab saves was validated by a backtest run;
   anything the API/live engine consumes is byte-identical to what was tested.
   No manual parameter re-entry anywhere in the loop.
3. **Auditable lifecycle** — every version records where it came from
   (optimization run ID), how it performed (metrics snapshot), and whether it
   is cleared for live use. Promotion and rollback are first-class actions.

**Success criteria (measurable):**

- After an optimize run, saving the best settings produces a **new version**
  of that ticker's strategy with source run ID and metrics — the previous
  version remains intact and restorable.
- A saved strategy can be loaded into a Backtest run, reopened in the Lab, and
  served by the API **without any parameter re-entry** (round-trip identity:
  the API serves exactly the params JSON that was backtested).
- A strategy with missing/inconsistent params or without a linked successful
  backtest **cannot** be promoted to live-enabled; the gate returns a
  structured reason.
- Version history with metrics and status is visible in the console; promote
  and rollback each take ≤ 2 clicks.
- The optimizer's default objective explicitly rewards profit and win rate
  and penalizes drawdown (multi-objective composite).

---

## 2. User Stories

**Persona**: quant analyst working per-ticker on the `trend_confluence_pine`
strategy via the console.

| # | Story |
|---|-------|
| US-1 | As an **analyst**, after an optimization run in the Lab, I want to save the best settings for a ticker as a **named, versioned strategy** (with the run ID and headline metrics) so that I never lose a good configuration to the next sweep. |
| US-2 | As an **analyst**, I want to **reopen** a saved strategy in the Lab and re-optimize it, with results stored as a **new version**, so that iteration is additive and I can compare versions. |
| US-3 | As an **analyst**, I want to **promote** a validated version to live-enabled (or **roll back** to a prior version) so that the API and live signal engine start/stop using exactly that configuration. |
| US-4 | As a **backtester**, I want to load any Lab strategy as the input configuration for a backtest run so that what I test is what was saved — not a hand-typed copy. |
| US-5 | As the **API / live signal engine**, I want to resolve each ticker's current live-enabled strategy version from the same store the Lab writes to, so that live signals are generated with parameters identical to those tested. |
| US-6 | As an **analyst**, I want a list of all saved per-ticker strategies with version history, metrics, and deployment status, so that I can see at a glance what is live, what is experimental, and what each version achieved. |
| US-7 | As an **analyst**, I want the optimizer to maximize profit and win rate while mitigating drawdown, so that "best" means robust, not just curve-fit return. |

---

## 3. Requirement Pool

### Requirement mapping (user's 6 bullets → features)

| User bullet | Features |
|---|---|
| 1. Save best run as named versioned strategy | F1, F2 |
| 2. Conversion to deployable strategy object | F3 |
| 3. Data flow Lab ↔ Backtest ↔ API | F4, F5 |
| 4. Go-live validation gate | F6 |
| 5. Management list + promote/rollback | F7 |
| 6. Multi-objective optimizer | F8 |

### P0 — Must have (first release)

| ID | Feature | Detail |
|----|---------|--------|
| F1 | **Versioned strategy entity** | Extend `StrategyPresetRow` (alembic migration, `batch_alter_table`) with: `strategy_name` (user-facing name), `version` (int, per symbol+strategy), `timeframe`, `metrics_json` (headline snapshot: total_return, sharpe, max_drawdown, win_rate, n_trades), `status` ∈ `backtest_only` \| `live_enabled`, `backtest_ref` (link to the validating backtest/result row). Existing columns (`source`, `optimizer_run_id`, `is_default`) are reused: `is_default` marks the **active version**. One live-enabled version per (symbol, strategy) enforced in the service layer. |
| F2 | **Save-from-optimization** | Lab's save action after `/api/v1/backtest/optimize` calls an extended `PresetService.save_optimization` that: expands winner params to the full resolved set (existing `full_params` behavior), writes metrics snapshot + `optimizer_run_id`, and increments `version` instead of overwriting. |
| F3 | **Conversion to deployable strategy object** | One serialization path (`PresetService` → strategy config consumed by `strategy_factory.build_strategy` and `SignalEngine` ticker configs): the stored params JSON *is* the deployable object. Signal-key `config_json` references the strategy version (by preset id) instead of embedding a params copy, so live and backtest cannot drift. |
| F4 | **Backtest loads Lab strategies** | Backtest tab / `/api/v1/backtest/*` accepts a `preset_id` (or symbol → active version) and runs with those params verbatim. |
| F5 | **Lab reopens saved strategies** | Lab tab lists saved strategies for the current symbol; selecting one loads its params into the editor and into the optimizer's starting point; a new optimize+save produces a new version (F2). |
| F6 | **Go-live validation gate** | Promote-to-live (`POST /api/v1/presets/{id}/promote`) enforces: (a) params complete — pass through `full_params`/`resolved_params` with no missing required keys; (b) consistent — strategy builds via `build_strategy` without `StrategyError`; (c) linked to a successful backtest — `backtest_ref`/metrics snapshot exists (from the optimization run or an explicit validation run). Structured error reasons on failure. |
| F7 | **Management: list + version history + promote/rollback** | Console list of saved per-ticker strategies: name, symbol, version, metrics, status badge, timestamps. Version-history drawer per strategy. Buttons: **Promote** (make this version active + live_enabled after gate), **Rollback** (re-activate a prior version; if that version was live-enabled before, re-run the gate). |
| F8 | **Multi-objective optimizer score** | Extend the `profit_win` composite in `optimize.py` with an explicit drawdown penalty (e.g. validation `max_drawdown` discount on top of the existing profit gate + win-rate multiplier + overfitting/thin-sample discounts). Expose the score breakdown in the optimize response so the Lab can show *why* a candidate won. |

### P1 — Should have

| ID | Feature | Detail |
|----|---------|--------|
| F9 | **Demote / disable live** | Explicit action to flip a live-enabled strategy back to `backtest_only` (stops the API serving it) without deleting history. |
| F10 | **Version diff view** | Side-by-side param comparison between two versions in the console (plain JSON key/value diff — no charting needed). |
| F11 | **Rollback safety confirmation** | Console confirms rollback when the current live version differs, showing both versions' metrics. |
| F12 | **Deploy-tab integration** | Signal-key creation in the deploy tab offers "use saved strategy version" as source, wired to F3. |

### P2 — Nice to have

| ID | Feature | Detail |
|----|---------|--------|
| F13 | **Walk-forward / robustness badge on versions** | Persist walk-forward and/or Monte-Carlo results per version as an extra quality signal for the go-live decision. |
| F14 | **Tags & search** | Tag strategies (experimental / production / retired) and filter the management list. |
| F15 | **Auto-save candidate runner-ups** | Store top-N optimizer candidates per run for later comparison instead of only the winner. |
| F16 | **Notes / changelog per version** | Free-text analyst notes on each version (extend existing `notes`). |

**Explicitly deferred**: multi-user permissions, cross-strategy portfolios,
one-click live order execution (signals only — the platform's current scope),
and any non-SQLite persistence.

---

## 4. UI Design Notes

All changes stay inside the single-file console (`trading/static/index.html`),
reusing existing tab/card/badge/button conventions and vanilla-JS fetch
patterns. The **Strategy Lab (pine) tab** grows a right-hand (or bottom)
management panel; no new tab is needed.

**Lab tab layout after upgrade:**

```
┌─ Strategy Lab (pine) ─────────────────────────────────────────────────┐
│ Ticker: [BTCUSDT ▾]   Strategy: trend_confluence_pine   TF: [1h ▾]    │
│                                                                       │
│ ┌─ Saved strategies (BTCUSDT) ──────────────────────────────────────┐ │
│ │ ▸ BTC momentum v3      v3  ● live_enabled   ret 12.4% wr 58% dd 6%│ │
│ │     [Promote] [Rollback] [Open in Lab] [History ▾]                │ │
│ │ ▸ BTC momentum v2      v2  ○ backtest_only ret 9.1%  wr 55% dd 8% │ │
│ │ ▸ BTC momentum         v1  ○ backtest_only ret 4.2%  wr 51% dd 11%│ │
│ └────────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ ┌─ Parameter editor ──────────────┐  ┌─ Optimizer ──────────────────┐ │
│ │ zone_atr      [0.5]             │  │ grid (PINE_GRID)             │ │
│ │ min_confluence [3]              │  │ [Run optimization]           │ │
│ │ tp_percent    [1.5]             │  │                              │ │
│ │ trailing_pct  [0.8]             │  │ ▸ result: score 0.71         │ │
│ │ [Save as new version]           │  │   profit ✓  win ✓  dd ✓      │ │
│ │ [Validate & backtest]           │  │   [Save best as v4]          │ │
│ └─────────────────────────────────┘  └──────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

Component notes:

- **Saved strategies list**: one row per *latest version* of each named
  strategy for the selected ticker; expandable to full version history
  (version, created, source badge `optimizer/backtest/manual`, metrics,
  status). Status badges reuse existing badge styling:
  `● live_enabled` (accent/green tone) vs `○ backtest_only` (neutral).
- **Buttons per row**: `Promote` (runs validation gate; on failure shows a
  toast with the structured reason), `Rollback` (P1: with confirm), `Open in
  Lab` (loads params into the editor — F5).
- **Save flow**: after a successful optimize run, the result panel shows the
  score breakdown (profit / win-rate / drawdown components from F8) and a
  single **"Save best as new version"** button — the name defaults to the
  strategy's existing name, version auto-increments.
- **Backtest tab**: a small "load from saved strategy" selector (symbol →
  active version, or pick a specific version) feeding the existing backtest
  run form — no layout rework.
- **Status changes are server-driven**: the list re-fetches after every
  promote/rollback/save so the badges always reflect the store.

---

## 5. Open Questions

| # | Question | Recommended default |
|---|----------|---------------------|
| Q1 | When a strategy is promoted to live, should the API serve it **immediately** (hot-swap running `SignalEngine` tasks) or only for **newly created** signal keys? | Immediate hot-swap per-ticker: the engine already rebuilds per-ticker tasks; rollback must be just as fast, and "promote" that only affects future keys defeats the rollback story. If hot-swap proves risky, fall back to new-keys-only in P1. |
| Q2 | Should `rollback` automatically re-run the go-live validation gate on the old version, or trust its historical metrics snapshot? | Re-run the gate: params were valid when saved, but the store may have migrated since. Cost is one backtest-sized validation, and it keeps the invariant "everything live passed the gate". |
| Q3 | One named strategy per (symbol, strategy-class), or can multiple named strategies coexist for the same ticker (e.g. "BTC momentum" and "BTC mean-revert")? | Allow multiple named strategies per ticker; exactly one may be live_enabled per (symbol, strategy-class). Matches the user's "list saved per-ticker strategies" wording and avoids forcing renames. |
| Q4 | Should the multi-objective score's weights (profit vs win-rate vs drawdown) be user-tunable in the Lab, or fixed defaults? | Fixed defaults in P0 (extend `profit_win` with the drawdown penalty); expose weights as advanced inputs in P1. Keeps the first release explainable. |
| Q5 | Versioning granularity: new version on **every save** (including manual param edits) or only on optimize/backtest saves? | Every save creates a new version — manual edits get `source=manual` versions too. Simpler mental model ("history never lies"), and manual edits before promotion should be traceable anyway. |

---

*End of PRD — ~340 lines budget respected.*
