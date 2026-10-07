# Product Completeness & Usability Audit — GEX-API Trading Console

**Author:** Alice (Product Manager) · **Round:** 1 (read-only discovery)
**Scope:** the product as a user experiences it — `trading/static/index.html` (single-file console),
`trading/main.py`, `trading/api/routers/*`, and the user-facing docs.
**Method:** full read of the 3,961-line console; cross-check of every control against its router;
`file:line` evidence throughout. No product files were modified.

---

## Executive summary

The console is a remarkably complete, dependency-free SPA with six tabs covering the entire
"backtest → optimize → deploy → monitor" loop. Every backend the console *calls* is answered —
there are no dead buttons. The gaps are not in what was built but in **safety and state honesty**:

1. **Destructive actions are unguarded and irreversible.** Deleting a broker account
   (`index.html:2724`) and revoking a live signal key (`index.html:3765`) fire immediately with **no
   confirmation** and **no archive/undo** anywhere in the product. One misclick on a live-trading
   credential is permanent.
2. **Expired sessions are invisible.** The JWT lives in `localStorage` and is used on boot without
   validation (`index.html:3956, 3778`). When it expires (default 8 h), every write fails with a
   transient toast and the operator is never returned to the login screen — the console *looks*
   alive while silently discarding work.
3. **Load failures masquerade as empty data.** `loadKeys`, `btLoadPresets`, `loadSignalKeys`,
   `loadDeployPresets` and `bootData` all swallow fetch errors and render "No … yet" empty states
   (`index.html:2691, 2250, 3720, 3738, 3929`). A transient outage reads as "you have nothing",
   inviting duplicate re-entry.
4. **Several backend capabilities are unreachable from the UI:** signal-key enable/disable
   (`PATCH /signal-keys/{id}`, `signal_keys.py:104`), ledger reset
   (`DELETE /signals/positions`, `signals.py:114`), and the entire live order/position/portfolio
   surface (`portfolio.py:60-155`). A user can be *forced into manual API/DB work* to use features
   the platform already ships.
5. **Docs are internally inconsistent on counts and endpoints** — five different test counts across
   five files (378 / 418 / 629 / 678 / 739) against an actual collected count of **740**.

Overall product verdict: **not yet production-ready** — safe to demo, unsafe to operate
irreversibly. The fix set is small and mostly front-end.

**Missing-control counts by severity:** blocker **1** · critical **2** · high **3** · medium **9** · low **3**.

---

## 1. Tab / panel inventory

Six tabs (`data-tab`), all in one file. Boot: `selectTab()` (`index.html:2962`) lazily initialises
each tab.

### Tab `backtest` — "Backtest" (default, `index.html:487-785`)
- **Purpose:** build a multi-ticker basket, route each ticker to a data venue, run a portfolio
  backtest / Monte-Carlo, and inspect results (equity, drawdown, rolling Sharpe, monthly heatmap,
  distribution, attribution, correlation, MC fan/CI, trade analysis, optimizer leaderboard).
- **Journey:** add tickers (manual / ⚡ Auto-select / + Add) → each row auto-resolves a fetcher chip
  via `GET /data/detect/{symbol}` → optional "Saved strategies" binding → **Run backtest**
  (`POST /backtest/portfolio`) → results render → per-row **Analyze**/**Optimize** → **Transfer
  basket** (validate/deploy to a signal key).
- **Controls:** Timeframe, Bars, Cash, Fee %, Slip %, Position, Universe, How many, Default strategy,
  ⚡ Auto-select, + Add ticker, Clear all, Load-from-saved-strategy + Apply-to-matching / Apply-to-All
  / Apply-to-Selected, per-row symbol/toggle◉/remove✕/strategy/weight/params/override, Monte-Carlo
  toggle + Paths + Resampler, Force-refresh, Synthesize, Copy JSON, Copy cURL, Run, Stop, HTML report,
  Send tickers to API setup, **Validate & preview**, **Deploy to live signals**, Copy payload JSON.

### Tab `accounts` — "Accounts" (`index.html:788-826`)
- **Purpose:** store encrypted broker API credentials and per-account routing/risk; preview order
  routing.
- **Journey:** pick exchange (BingX/TBANK) → enter Label / API key / API secret (TBANK account id) →
  **Add** → card list of wallets → edit instruments/risk/max-pos/leverage/active → **Save**;
  **Delete** per account; routing preview by symbol.
- **Controls:** Exchange, **Add**, Label, API key, API secret, TBANK account id, per-account
  Instruments / Risk / Max pos % / Leverage / active / **Save** / **Delete**, Symbol + **Preview**.

### Tab `tools` — "Tuning & export" (`index.html:829-892`)
- **Purpose:** pre-live auto-tune with risk profiles; download trade ledgers and HTML reports.
- **Journey:** symbol + strategy + risk profile → **Run auto-tune** → volatility + long/short ATR
  targets; export live trades (CSV/Excel) or a stored run by id; download HTML report.
- **Controls:** Symbol, Strategy, Risk profile, **Run auto-tune**; Live-trades **CSV**/**Excel**;
  run id + **CSV**/**Excel**; **Download HTML report**.

### Tab `pine` — "Strategy lab" (Strategy Hub) (`index.html:894-1021`)
- **Purpose:** the versioned strategy store — edit/persist per-ticker presets, optimise with a
  selectable objective, promote/rollback/demote, browse version history.
- **Journey:** Symbol → saved-strategy cards (Promote / Rollback / Demote / Open in Lab / History) →
  edit params → **Save as new version** / **Save edit as version** / **Set default** / **Delete** /
  **Reset to defaults** → tick sweep params → **Run optimization** / Stop sweep / Apply recommended
  → Best configuration → **Apply to settings** / **Create optimizing version**; Ranking table.
- **Controls:** ~50 param fields + sweep grid, Objective, Timeframe, Bars, Run optimization,
  Stop sweep, Apply recommended values, Save as new version, Save edit as version, Set default,
  Delete, Reset to defaults, Use basket symbol, Apply to settings, Create optimizing version, and the
  hub actions above.

### Tab `signals` — "Live signals" (`index.html:1024-1099`)
- **Purpose:** run the real-time signal engine and watch the live feed, open positions and
  per-ticker diagnostics; export signals/positions.
- **Journey:** tickers + strategy + validated preset + bar + poll + history → **Start engine** /
  **Stop** / **Refresh** → live WS feed (`/ws/signals`), open positions, recent signals, engine
  tickers; 4 export buttons; 8 s auto-poll while the tab is visible.
- **Controls:** Tickers, Strategy, Validated preset, Bar, Poll (s), History bars, Start engine,
  Stop, Refresh, Signals CSV/Excel, Positions CSV/Excel.

### Tab `deploy` — "Optimize & deploy" (`index.html:1101-1171`)
- **Purpose:** optimize the unified strategy per ticker / all tickers, create signal keys, and manage
  them (open dashboard / revoke).
- **Journey:** tickers + objective → Optimize Selected / All / Stop → per-ticker results + saved
  presets → choose broker → **Create API Key** → signal-key table → **Open Live Dashboard** /
  **Revoke**.
- **Controls:** Tickers, Use backtest tickers, Objective, Optimize Selected Ticker, Optimize All
  Tickers, Stop, Saved-presets table, Broker, **Create API Key**, signal-keys table
  (Open Live Dashboard, Revoke).

### App-level chrome
- API-pill (live/demo/down), stale-params pill, theme toggle, Sign out; fixed action bar
  (Run backtest / Stop / HTML report / Send tickers to API setup); toasts; a **demo mode** that
  silently activates on any network failure (`index.html:1302-1303`).

---

## 2. Control matrix

Legend: **P** present · **~** partial · **✗** missing · **–** N/A.
Columns: add · edit · del · arch (archive/restore) · dup · exp (export) · imp (import) · ref (refresh)
· retry · cancel · clr (clear/reset) · del-conf (confirm-on-destructive).

| Entity (surface) | add | edit | del | arch | dup | exp | imp | ref | retry | cancel | clr | del-conf |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Broker API keys / accounts** (Accounts) | P | ~¹ | P | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | – | ✗ |
| **Strategy presets / versions** (Strategy Hub) | P | ~² | P | ~³ | ✗ | ✗ | ✗ | ✗ | – | P⁴ | ~⁵ | ~⁶ |
| **Signal keys** (Deploy) | P | ✗ | P⁷ | ~⁷ | ✗ | ✗ | ✗ | ✗ | – | – | – | ✗ |
| **Baskets** (Backtest) | P | P | ~⁸ | ✗ | ✗ | ~⁹ | ✗ | – | ✗ | – | P | ✗ |
| **Stored backtest results** | P | – | ✗ | ✗ | ✗ | ~¹⁰ | ✗ | ✗ | ~ | P | ✗ | – |
| **Orders (live broker)** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | – |
| **Positions (live broker)** | – | ✗ | ✗ | ✗ | – | ✗ | ✗ | ✗ | – | – | ✗ | – |
| **Live signal engine** (Signals) | P | – | – | – | – | P | ✗ | P | ✗ | ~¹¹ | ✗¹² | ✗ |
| **Local clients** (`/ws/client`) | – | – | – | – | – | – | – | ✗ | – | – | – | – |

Footnotes:
1. Settings only (instruments/risk/max-pos/leverage/enabled). **Label, API key, API secret, account id
   are not editable** after creation (`index.html:2698-2720`; backend `keys.py:79`).
2. Params + notes editable (`PATCH /presets/{id}`), but **no rename** of a saved strategy.
3. `Demote` flips live→`backtest_only`; `Rollback` re-activates a prior version — no true "archive".
4. Reset = `pine-reset` resets the **editor form**, not stored history (`index.html:3343`).
5. Same as 4 — a form reset, not a data reset.
6. Delete + rollback confirm ✓ (`3387, 3321`); **demote does not** (`3325`).
7. `DELETE` is a *soft* revoke (`signal_keys.py:97`) so an "archive" exists server-side, but the UI
   offers **no un-revoke/restore** and no enable/disable even though `PATCH` supports it.
8. Remove single row / Clear all — no confirm (`3834, 3845`); no way to restore.
9. "Validate & preview" produces a copyable JSON payload only — **no file download** (`2414`).
10. Export requires **typing a numeric run id** by hand (`878, 3894`); there is no run list.
11. Stop engine only; a single in-flight poll cannot be cancelled.
12. `DELETE /signals/positions` exists (`signals.py:114`) but is not surfaced — cannot clear the ledger.

---

## 3. Missing controls (severity · evidence · user impact)

| # | Sev | Missing control | Evidence (file:line) | User impact |
|---|-----|-----------------|----------------------|-------------|
| 1 | **blocker** | Confirm + undo on **Delete broker account** | `index.html:2724-2727` | One click permanently destroys an encrypted broker credential and its routing config; no confirm, no archive, recovery only via DB/broker. The single highest-risk trap. |
| 2 | **critical** | Confirm on **Revoke signal key** | `index.html:3759-3768` | Immediate soft-delete of a key that may be driving a live deployment; no confirm, and no UI path to re-enable. |
| 3 | **critical** | **Expired-session / 401 handling** | `index.html:3956` (boot trusts stale token), `1309-1313` (401→throw), `3798` (login only on manual logout) | After JWT expiry all writes fail as transient toasts; the operator isn't returned to login and in-progress basket state is lost on the eventual reload. |
| 4 | **high** | **Load failures rendered as empty state** | `index.html:2691, 2250, 3720, 3738, 3929` | A backend hiccup shows "No accounts / no saved strategies / no keys yet" — misleads the user into re-entering existing data. |
| 5 | **high** | **Basket persistence: save / load / name / import** | `index.html:2374-2444` (transfer only) | A built basket exists only in memory; leaving the tab / reload loses it. Re-entry is manual; no file import. |
| 6 | **high** | **Archive / restore / undo** for accounts, keys, presets, baskets | none | Every delete is final; no recycle bin, no undo — forces manual DB edits to recover. |
| 7 | medium | **Stored-backtest-run list** (export by id today) | `index.html:878, 3894-3895` | User must already know the numeric result id; no discoverable list to export from. |
| 8 | medium | **Preset / strategy export·import** (file) | Strategy Hub has no download/upload | Can't back up or share a tuned strategy as a file; only copy-as-text of params. |
| 9 | medium | **Account export / import** | Accounts tab | Can't bulk-migrate or back up wallets. |
| 10 | medium | **Signal-key enable/disable** (API exists) | `signal_keys.py:104-117`; UI `3721-3733` only revoke | To pause a key without revoking it, the operator must call the API/DB by hand. |
| 11 | medium | **Clear the signal/position ledger** (API exists) | `signals.py:114-117`; no UI caller | Test/demo rows accumulate with no in-console reset. |
| 12 | medium | **Confirm on Clear-all basket & Remove ticker** | `index.html:3845, 3834` | "Clear all" wipes the whole basket in one click. |
| 13 | medium | **Confirm on Demote live version** | `index.html:3325-3328` | Demoting stops the API serving a live strategy with no confirmation (unlike delete/rollback). |
| 14 | medium | **Test-connection / verify-credentials** for a broker account | Accounts tab | Wrong keys are only discovered when live signals/routing fail. |
| 15 | medium | **Live orders / positions / portfolio views + order ticket** | `portfolio.py:60-155` (API only) | The platform ships a trading read/write surface the console never shows. Partly by design (signals-only, `FINALIZATION_REPORT.md:122`), but the read-only position/portfolio views would still be expected in production. |
| 16 | low | **Duplicate** a strategy version / account / basket | none | No "clone" convenience; users re-create by hand. |
| 17 | low | **Explicit Refresh** on Accounts / Deploy / Strategy Hub | `loadKeys/loadDeployPresets/pineLoadHub` only on tab-open/symbol-change | Stale lists with no manual refresh; only the Signals tab has one. |
| 18 | low | **Retry** for a failed optimize candidate / failed ticker row | `index.html:2664` (error row is terminal) | Must re-run the whole operation manually. |

---

## 4. State-handling gaps

| State | Where handled | Gap |
|-------|---------------|-----|
| **Empty** | Backtest `empty-card`; Accounts "No accounts yet"; Signals "No open positions/No signals yet/Engine stopped"; Deploy "no signal keys/presets yet"; Strategy Hub "No saved strategies…"; `bt-preset` "no saved strategies". | Routing-preview output starts blank with no placeholder (`2751`). More importantly, the **empty state is shown on error** too (see below). |
| **Loading** | Run/Stop status line with spinner (`2312`); optimizer "sweeping N combinations… Ns elapsed" (`3424`); engine badge; dashboard "loading…". | **No loading state** for `loadKeys`, `loadSignalKeys`, `loadDeployPresets`, `pineLoadHub`, `routingPreview`, and all downloads — buttons don't disable, panels go blank then pop in. |
| **Error** | Inline `.note.err` boxes on forms; toasts for most failures; gate reasons via `pineGateToast`. | Several loads **swallow** errors and fall through to empty state (`2691, 2250, 3720, 3738, 3929`) — no persistent, retryable error banner. A user cannot distinguish "no data" from "load failed". |
| **Permission / expired session** | **Absent.** Token trusted from `localStorage` at boot without a validity probe (`3956`); 401 throws per-call (`1309-1313`); the only logout is the manual button (`3798`). | No 401 interceptor, no redirect to login, no "session expired" state. Login screen also prints the default credentials (`index.html:472`). |
| **Not-found** | Export 404 → toast ✓. | The live dashboard page (`dashboard.py:69/72/75`) raises `HTTPException` → raw JSON `{"detail": "unknown API key"}` for a wrong/revoked/disabled key instead of a friendly page; the friendly 404/410 handling in its inline JS (`dashboard.py:383`) is never reached because the initial document load already failed. |
| **Offline** | `pingApi()` at boot; on network failure the console **silently enters demo mode** (`1302-1303, 3948-3949`). | A mid-session network drop flips subsequent calls to synthetic data with only a pill + transient toast; a user can mistake demo numbers for live results. |

---

## 5. Docs accuracy findings

| # | Claim | Where | Reality | Severity |
|---|-------|-------|---------|----------|
| D1 | Test counts: **378** | `ARCHITECTURE.md:159` | `pytest --collect-only` = **740 collected**; latest log = 739 passed, 1 skipped (`pytest_t08.log:18`) | high (trust) |
| D2 | Test counts: **418** | `README.md:41, 294` | same as D1 | high |
| D3 | Test counts: **629 passed, 1 skipped** | `UNIFIED_STRATEGY.md:334` | same as D1 | high |
| D4 | Test counts: **678 passed, 1 skipped** | `FINALIZATION_REPORT.md:3` | same as D1 | high |
| D5 | Dashboard "auto-polls … every **15 s**" | `UNIFIED_STRATEGY.md:220` | Code auto-refreshes every **60 s** (`dashboard.py:424`); page text says 60 s (`dashboard.py:346`); ARCHITECTURE/README also say 60 s | medium |
| D6 | Auto-tune endpoint `POST /api/v1/autotune` | `SPEC-local-signal-sprint.md:106` | Actual path is `POST /api/v1/backtest/autotune` (`backtest.py`, console `2781`) | medium |
| D7 | Redis-backed **distributed** rate limiting "reused, not rebuilt" for the trading API | `ARCHITECTURE.md:27-29`; `README.md:12, 323`; `.env.example` (Redis "also used for … rate limiter") | The API middleware uses an **in-memory per-IP `TokenBucket`** only (`trading/api/middleware.py:20-38`). `RedisTokenBucket` exists (`trading/adapters/ratelimit/token_bucket.py:98`) but is **not wired** into the middleware. | high (misleading claim) |
| D8 | "Gap analysis": `BaseFetcher / Monte-Carlo / TBANK / BINGX / WebSocket` marked **❌ missing** | `ARCHITECTURE.md:92-100` | All are implemented and shipped (routers, adapters, WS in `websockets.py`/`local_client_ws.py`); §5 of the same doc says "Steps 1–14 … now implemented". The gap table contradicts the current repo. | medium |
| D9 | `api/` described as "… + **WebSocket stub**" | `README.md:12, 14` | Real WS endpoints exist (`/ws/signals`, `/ws/orders`, `/ws/positions`, `/ws/client`) | low |
| D10 | README "Backtest & Monte-Carlo API" table omits live trading surface | `README.md:50-62` | `/portfolio`, `/positions`, `/orders` (incl. `POST`), `/keys/routing`, `/data/ohlcv`, `/data/detect`, `/strategies/{name}/start|stop`, `/local-clients` are undocumented in the README table | medium |
| D11 | "No live order placement" | `FINALIZATION_REPORT.md:122` | A `POST /api/v1/orders` endpoint **does** place broker orders (`portfolio.py:117-155`) when credentials exist — the statement is true only for signal→order automation, not for the API | low (ambiguous) |
| D12 | `trading/static/` described as "multi-ticker backtest console (API-key manager + backtest UI)" | `README.md:14` | The console now has six tabs incl. Strategy Hub, Live signals, Optimize & Deploy — description is understated | low |

Verified-accurate doc references: `run.bat` → `scripts/serve.py` ✓; `scripts/verify_backtest.py`,
`scripts/make_shortcut.py`, `scripts/smoke_unified_workflow.py` all exist ✓;
`.env.example` variable names all match `trading/config.py` ✓.

---

## 6. Draft acceptance criteria — "ready for production" (product view)

**P0 — must pass before any live/real-money use**
1. Every destructive action (delete account, revoke key, clear basket, demote, delete version)
   requires an explicit confirmation showing what will be lost, **or** is reversible from an
   in-product archive/undo. (Closes #1, #2, #12, #13, #6.)
2. An expired or invalid session is detected on the next request and the user is returned to a login
   screen with a clear "session expired" message; no silent write failures, and unsaved basket state
   is preserved. (Closes #3.)
3. No data-loading failure is ever rendered as an empty state: every list panel has distinct
   **loading / empty / error(+retry)** states, and a failed load shows an error with a Retry action.
   (Closes #4, #18, §4.)
4. The console never silently substitutes demo data mid-session: demo/fallback mode is explicitly
   user-acknowledged (banner), not just a pill. (Closes §4 Offline.)

**P1 — should pass before general production rollout**
5. Core entities are recoverable and portable: accounts, presets and signal keys can be archived/
   restored, and presets/baskets can be exported and imported as files. (Closes #6, #8, #9.)
6. A basket can be saved by name, reloaded, and imported; the saved basket survives a reload.
   (Closes #5.)
7. Every mutating backend capability that a user needs is reachable in the console — signal-key
   enable/disable, ledger clear — or is explicitly documented as API-only. (Closes #10, #11, #15.)
8. Stored backtest runs are listed and selectable for export (no hand-typed numeric id). (Closes #7.)
9. The `/API_KEY/{key}` page renders friendly 404 / revoked / disabled **pages** (not raw JSON).
   (Closes §4 Not-found.)

**P2 — quality**
10. Consistent loading affordances (disable + spinner) on all fetch buttons; optimistic lists refresh
    with an explicit control. (Closes #17.)
11. One canonical test count is published and matches CI; endpoint tables in README/ARCHITECTURE
    match the routers; the "distributed rate limiting" and "15 s refresh" claims are corrected.
    (Closes D1–D12.)

---

*End of report.*
