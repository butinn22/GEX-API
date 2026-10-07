# Round 5 — Configuration & Strategy Management Controls (Delete / Update / Refresh)

**Date:** 2026-10-07 · **Root:** `E:\gex api`
**Mandate:** make every configuration and strategy safely manageable — complete delete,
update, refresh/validate controls, backend ⇄ frontend parity, permissions, safe deletes,
state updates, guidance, tests and docs.
**Decisions carried forward:** risk-first; fail-closed in production; **non-destructive
deletes** (soft/revoke where history matters).

---

## 1. Inventory of affected entities

| Entity (domain language) | Kind | Backend surface |
| --- | --- | --- |
| Broker API key (`api_keys`) | Configuration / credential | `/keys` |
| Signal key (`signal_keys`) | Configuration (deployable strategy+broker reference) | `/signal-keys` |
| Strategy preset / version (`strategy_presets`) | Strategy configuration (versioned) | `/presets` |
| Live strategy runner | Strategy runtime | `/strategies/{name}/start|stop` |
| Stored backtest run (`backtest_results`) | Record produced by a strategy run | `/backtest/results` |
| Order (`orders`) | Record | `/orders` |
| Signal-position ledger (`signal_positions`) | Record | `/signals/positions` |
| Signal engine | Runtime | `/signals/engine/start|stop` |

## 2. Backend ⇄ frontend parity (after this round)

| Object | Create | Update | Refresh / validate | Delete |
| --- | --- | --- | --- | --- |
| Broker API key | ✅ Accounts tab | ✅ `PATCH /keys/{id}/settings` | — | ✅ `DELETE /keys/{id}` (confirm) |
| Signal key | ✅ Deploy tab | ✅ **Enable/Disable** (`PATCH /signal-keys/{id}?active=`) | ✅ **Regenerate** (`POST …/{key}/generate`), ✅ **Purge summary cache** (`POST …/cache/purge`) | ✅ Revoke (`DELETE …/{id}`, confirm), ✅ **Purge signals** (`DELETE …/{id}/signals`, confirm) |
| Strategy preset/version | ✅ Strategy Lab / Deploy | ✅ `PATCH /presets/{id}`; promote/rollback/demote | ✅ **Validate** (`GET /presets/{id}/validate`) | ✅ version delete · ✅ group delete (confirm) |
| Live strategy runner | — | — | ✅ **Start / Stop** (`POST /strategies/{name}/start|stop`) | ✅ (Stop) |
| Stored backtest run | ✅ `POST /backtest` | — | ✅ **Refresh** (`GET /backtest/results`) | ✅ per-row Delete · ✅ **Clear all** (confirm) |
| Order record | ✅ `POST /orders` | — | ✅ **Refresh** (`GET /orders`) | ✅ per-row Delete · ✅ **Purge all** (confirm) |
| Signal positions | live engine | — | ✅ `GET /signals/positions` | ✅ `DELETE /signals/positions` (confirm) |

## 3. Added this round

### 3.1 Backend
- **`GET /api/v1/backtest/results`** (`trading/api/routers/backtest.py`) — list stored runs
  newest-first (`limit` 1–500), complementing the existing `DELETE /results` and
  `DELETE /results/{id}` so the console is no longer download-only. Returns
  `[{id, strategy, symbol, created_at, metrics}]`.
- **Bulk endpoints (R5-4):** `POST /orders/bulk-delete`, `POST /backtest/results/bulk-delete`,
  and `POST /signal-keys/bulk` (`{"ids": [...], "action": enable|disable|revoke}`). Each is
  transactional and idempotent, returning `{deleted|updated, missing}` so a batch never
  fails on an unknown id.
- **`POST /keys/{id}/validate`** and **`POST /orders/{id}/cancel`** (see R5-2/R5-3).

### 3.2 Console (`trading/static/index.html`)
- **Signal-key actions** (per row): Enable/Disable, Regenerate, Purge signals — alongside the
  existing Dashboard/Revoke. Regenerate and Purge show a loading/disabled state; Purge and
  Revoke confirm first.
- **Purge summary cache** button in the Signal-keys header.
- **Stored backtest runs** panel: Refresh, per-row Delete (confirm), Clear all (confirm).
- **Live strategy runners** panel: strategy selector (from `GET /strategies`) + Start/Stop,
  with status line.
- **Order history** panel: Refresh, per-row Delete, Purge all (confirm), with a helper note
  that deleting a record does **not** cancel a live broker order.
- **Validate** button on each saved strategy in the Strategy Lab hub (runs the go-live gate
  without promoting).
- Empty/error/retry states for each new list; all destructive actions confirm.

### 3.3 Tests
- `trading/tests/test_backend_correctness.py::test_backtest_result_delete_and_purge` extended
  to assert `GET /backtest/results` lists the seeded row (delete/purge already covered).
- Existing management/authz suites re-run: `test_backend_correctness`, `qa_round3_management`,
  `test_authz_regression`, `test_signal_keys`, `test_preset_delete` → **84 passed**.
- Console JS syntax-checked (2 script blocks, 0 errors) with Node 24.

### 3.4 Docs
- `README.md` gains a **“Managing configurations & strategies”** matrix documenting every
  control, its endpoint, and the deletion semantics.

## 4. Permissions & safety

- Every new console action calls an endpoint already mounted under `require_auth`
  (`trading/main.py` `_GUARD`); nothing new is anonymous. `tests/test_authz_regression.py`
  sweeps the guarded routes.
- Deletes follow the agreed semantics: signal keys are **soft-revoked**; presets are
  **versioned** (a row delete removes one version; the group delete is atomic and 409 while
  live); stored runs/orders are **hard deletes of local records only**; signal purges
  hard-delete generated rows that the next generate rebuilds.
- All destructive console actions use `confirm()` naming the target and its consequences.

## 5. Verification evidence

| Check | Result |
| --- | --- |
| `pytest` focused management/authz/preset suites | **84 passed** |
| `ruff check trading` | All checks passed |
| Console JS syntax (`node` + `vm.Script`, 2 blocks) | 0 errors |
| Console flow tests (`npm test` in `trading/tests/console`) | **2/2 harnesses pass** (management controls: signal keys / runs / orders / **bulk** / runners / accounts / **a11y**; preset: delete 5 scenarios + duplicate) |
| Migration gate (SQLite) | `alembic upgrade head → downgrade base → upgrade head → alembic check` **clean** (new `0008`) |
| FK contract | `test_backend_correctness::test_signal_derived_tables_have_non_destructive_fks` |
| Full suite (`pytest trading/tests -q`) | **874 passed, 1 skipped** (860 before this pass; +14 new tests incl. bulk + FK contract) |
| New endpoint behaviour | `GET /backtest/results`, `POST /keys/{id}/validate`, `POST /orders/{id}/cancel`, `POST /orders/bulk-delete`, `POST /backtest/results/bulk-delete`, `POST /signal-keys/bulk`, BingX cancel/get_order_status asserted in tests |

## 6. Before → after (key workflows)

- **Signal key lifecycle.** Before: create + revoke only. After: create, **disable** (stop
  generating without losing the key), **re-enable**, **regenerate** on demand, **purge**
  generated signals, purge the summary cache, revoke — each with confirm/loading/feedback.
- **Stored runs.** Before: the console held only the last run and could download it. After:
  a managed list with refresh, per-run delete and clear-all.
- **Strategy validation.** Before: promotion was the only way to find out a preset fails the
  go-live gate. After: a **Validate** action reports the gate result without side effects.
- **Strategy runtime.** Before: no console control over `/strategies/{name}/start|stop`.
  After: a runners panel to start/stop with status.
- **Order history.** Before: no console surface at all though the API existed. After:
  refresh/list/delete/purge.

## 7. Remaining / deferred (honesty)

| # | Item | Severity | Blocker reason | Exact next step |
| --- | --- | --- | --- | --- |
| R5-1 | ~~New console interactions are not covered by automated tests~~ **DONE** — added `trading/tests/console/` (`package.json` + `run_all.js` + `management_controls_flow.js`), `npm test` runs on jsdom; new CI `console-tests` job. | — | — | — |
| R5-2 | ~~No “cancel live order”~~ **DONE** — `BingxBroker.cancel_order`/`get_order_status` implemented (symbol-scoped `<symbol>:<orderId>`); `POST /orders/{id}/cancel` + a console **Cancel** action for open orders. | — | — | — |
| R5-3 | ~~No broker “test connection”~~ **DONE** — `trading/application/credential_check.py` (`build_broker`/`check_broker`, timeout, dry-run guard, no secrets) + `POST /keys/{id}/validate` + a console **Test** action. | — | — | — |
| R5-4 | ~~No bulk actions~~ **DONE** — multi-select (select-all + row checkboxes) with bulk **Delete selected** for stored runs and orders, and bulk **Enable/Disable/Revoke selected** for signal keys, backed by `POST /orders/bulk-delete`, `POST /backtest/results/bulk-delete`, `POST /signal-keys/bulk` (idempotent, summarized). | — | — | — |
| R5-5 | ~~No preset duplicate/clone~~ **DONE** — Strategy Lab hub **Duplicate** action (`POST /presets` from an existing version, new named strategy). | — | — | — |
| R5-6 | ~~Accessibility~~ **DONE** — labeled bulk-action groups (`role="group"` + aria-label), `aria-label` on select-all checkboxes and the runner select, `aria-live` on selection counts; jsdom harness asserts every management control has an accessible name. (Native `confirm()` dialogs are inherently accessible.) | — | — | — |
| R5-7 | ~~Reference integrity~~ **DONE** — migration **`0008_referential_integrity`** adds non-destructive `ON DELETE SET NULL` FKs (`key_signals/signal_positions/key_trades → signal_keys`, `*.preset_id → strategy_presets`; `key_trades.key_id` made nullable); it **reports** existing orphans to the log and never deletes them. `alembic upgrade→downgrade→upgrade→check` clean; FK contract covered by a test. | — | — | — |

## 8. Acceptance criteria

| Criterion | Status |
| --- | --- |
| Every config/strategy delete control exists where required | ✅ (signal keys, presets, runners, runs, orders, positions) |
| Every update control exists where required | ✅ (keys settings, signal-key enable/disable, preset edit) |
| Every refresh/validate control exists where required | ✅ (regenerate, cache purge, preset validate, runs/orders/runners refresh) |
| Frontend control ⇄ backend capability parity | ✅ for the entities above (R5-2/R5-3 gaps noted) |
| Destructive actions confirm + explain consequences | ✅ |
| AuthZ enforced server-side | ✅ (`require_auth` mounts; authz sweep) |
| State updates after mutation (list/selectors refresh) | ✅ (each handler reloads its list) |
| Tests + docs | ✅ backend; ⚠️ console automation deferred (R5-1) |
