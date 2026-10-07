# Round 2 — Frontend UX Spec & Docs-Correction Punch List

**Author:** Alice (Product Manager) · **Input:** `audit/product-audit.md` (Round 1)
**Status:** implementation-ready spec · **Read-only:** no product files were modified to produce this.
**Audience:** software-engineer (build) + team-lead (verification).

All element ids / function names / line anchors below refer to the current tree
(`trading/static/index.html` unless stated). Line numbers are the **current** positions and are
expected to drift once edits are applied — match on the quoted code, not the number.

**⚠ Canonical test count — the number MOVED during this round; re-verified live:**

- **Round-1 baseline:** `739 passed, 1 skipped` (740 collected) — `_qa_fullsuite.log` (11:59) / `pytest_t08.log`.
- **Current tree (Round-2 remediation in progress):** `768 passed, 1 skipped` (**769 collected**) —
  `_qa_fullsuite_round2.log:18` (11:07), re-confirmed by `pytest trading/tests --collect-only -q` → 769.
- Cause: the engineer added an untracked test file `trading/tests/test_authz_regression.py` mid-round
  (+29 tests). The suite is therefore a **moving target** while remediation is in flight.
- **Rule (see §2.1):** docs must state the count of the **frozen** suite at release. Re-pin once, AFTER
  the last Round-2 test lands, from a single final `pytest trading/tests -q` run.

---

## 1. FRONTEND UX SPEC

### (a) Confirmation before destructive actions

**Pattern to reuse** (already in the file): `if (!confirm("…")) return;`
— see the existing guarded deletes at `index.html:3321` (rollback) and `index.html:3390`
(pine delete). Use the same native `confirm()` for P0; an in-app modal is a P2 refinement.

#### A1 — Delete broker account

- **Where:** `loadKeys()` delete handler, `index.html:2724-2727` (currently:
  `$$("button[data-del]", host).forEach((b) => b.onclick = async () => { try { await json("/keys/" + b.dataset.del, { method: "DELETE" }); loadKeys(); } …`.
- **Change:** insert a confirm guard as the first statement of the handler, resolve a human label
  from the card, and abort on cancel.
- **Exact behavior:**
  1. `const card = b.closest(".route");`
  2. `const label = (card.querySelector("b")?.textContent || ("account #" + b.dataset.del)).trim();`
  3. `if (!confirm('Delete broker account "' + label + '"?\n\nThis permanently removes its encrypted API credentials and its routing/risk settings. This cannot be undone.')) return;`
  4. proceed with the existing `DELETE` + `loadKeys()`.
- **Undo / archive:** **not offered at P0.** The backend `DELETE /keys/{id}` is a *hard* delete
  (`trading/api/routers/keys.py:69-76`, 204, no soft flag). An undo/archive requires backend work
  (new soft-delete column or `?archive=true`) → track as a P1 dependency, **not** in this frontend spec.

#### A2 — Revoke signal key

- **Where:** `$("dp-keys").addEventListener("click", …)` → the `rev` branch, `index.html:3762-3768`
  (currently calls `DELETE /signal-keys/{id}` with no guard).
- **Change:** confirm guard before the `json("/signal-keys/" + rev.dataset.revokeKey, { method: "DELETE" })` call.
- **Exact behavior:**
  1. `const key = rev.dataset.revokeKey;`
  2. `if (!confirm('Revoke signal key ' + key.slice(0, 12) + '…?\n\nLive signal generation for this key stops immediately. The key is soft-deleted and cannot be re-enabled from this console.')) return;`
  3. proceed with the existing `DELETE` → `toast("Key revoked", "ok")` → `loadSignalKeys()`.
- **Undo / archive:** the backend revoke is **soft** (sets `revoked_at`; `signal_keys.py:97-101`), but
  there is **no UI un-revoke path**, so the message must say it is not reversible *from the console*.
  Wiring the existing `PATCH /signal-keys/{id}?active=` for a restore/disable is item **#10** (P1).

> **Consistency note:** also guard the two *basket* actions flagged in Round 1 (#12) with a light confirm,
> since the lead approved P1: `btn-clear` (`index.html:3845`) →
> `if (!confirm("Remove all tickers from the basket?")) return;`; and per-row remove
> (`index.html:3834`) can stay unguarded (single row, cheap to re-add).

---

### (b) Expired-session / 401 handling

**Goal:** a stale/expired JWT must never leave the console looking functional. On any 401 the app
clears the token and returns to the login screen with a visible message.

**New helper** (add near `showLogin`/`showApp`, `index.html:3774-3782`):

```js
let _authHandled = false;
function handleAuthFailure() {
  if (S.demo) return;                 // demo has no session
  if ($("view-app").hidden) return;   // already on the login screen (e.g. wrong password)
  if (_authHandled) return;           // de-dupe: many in-flight calls can 401 at once
  _authHandled = true;
  try { localStorage.removeItem("gex.token"); } catch (e) {}
  S.token = null;
  $("password").value = "";
  showLogin();
  $("login-error").hidden = false;
  $("login-error").textContent = "Your session expired — please sign in again.";
  setApiPill("down", "signed out");
}
```

Reset `_authHandled = false` on a successful login (in the `login-form` submit handler after
`showApp()`), so a later expiry is handled again.

**Hook A — central, on any 401 response.** `json()` is the single funnel for all JSON calls
(`index.html:1276-1316`). In the `if (!res.ok) { … }` block (currently `1309-1313`), before building
the error:

```js
if (res.status === 401) handleAuthFailure();
```

Apply the same one-liner to the two other funnels so downloads/reports can't bypass it:
`blobGet()` (`index.html:1317-1322`, after `if (!res.ok) throw …`) and `textPost()`
(`index.html:1323-1332`, in its `if (!res.ok)`).

> The `/auth/token` call itself goes through `json()`; because `view-app` is hidden while the login
> screen is shown, a 401 from a wrong password is correctly ignored by the `if ($("view-app").hidden) return;` guard.

**Hook B — boot-time probe** (currently trusts the stored token blindly at `index.html:3956`:
`if (S.token) { showApp(); await bootData(); selectTab("backtest"); }`).

Replace with a probe against a protected endpoint before revealing the app:

```js
if (S.token) {
  try {
    await json("/keys", { timeoutMs: 4000 });   // protected → 401 if the JWT is stale/expired
    _authHandled = false;
    showApp(); await bootData(); selectTab("backtest");
  } catch (e) {
    if (e && e.status === 401) {                 // stale token from a previous session
      localStorage.removeItem("gex.token"); S.token = null;
      showLogin();
      $("login-error").hidden = false;
      $("login-error").textContent = "Your session expired — please sign in again.";
    } else {
      // network/other failure → keep today's demo fallback
      S.demo = true; enterDemo("Backend unreachable — showing demo data");
      showApp(); await bootData(); selectTab("backtest");
    }
  }
} else showLogin();
```

**Post-login:** in the `login-form` submit success path (`index.html:3790-3792`) set
`_authHandled = false;`.

**Affected element ids:** `view-login`, `view-app`, `login-error`, `password`,
`api-pill` / `api-pill-text`. **State vars:** `S.token`, `localStorage["gex.token"]`, `S.demo`.

---

### (c) Empty-vs-error state for the five loaders

**Rule:** a failed fetch must render an **error** treatment (message + Retry), never the
"No … yet" empty copy. Add one reusable helper next to `toast()` (`index.html:1240`):

```js
function panelError(host, msg, retry) {
  host.innerHTML =
    '<div class="note err" style="margin-top:6px">' +
    esc(msg) + ' <button class="btn btn-ghost btn-sm" data-retry>Retry</button></div>';
  const b = host.querySelector("[data-retry]");
  if (b && retry) b.onclick = () => { host.innerHTML = ""; retry(); };
}
```

Per loader (match the quoted current code; anchors are current lines):

| # | Loader | Current (failure → empty) | Required error state | Retry action |
|---|--------|---------------------------|----------------------|--------------|
| C1 | `loadKeys()` `index.html:2689-2693` | `catch (e) { keys = []; }` → `"No accounts yet — connect one on the left."` | Set a failure flag; render into `#accounts-list`: `panelError($("accounts-list"), "Couldn't load accounts — the server didn't respond.", loadKeys)` | re-call `loadKeys()` |
| C2 | `btLoadPresets()` `index.html:2246-2258` | `catch (err) { btPresetState.rows = []; }` → `<option>— no saved strategies yet —</option>` | Keep the empty option text for the *real* empty case; on failure add a note element `#bt-preset-error` (new, hidden by default, placed after `#bt-preset` at `index.html:551`) rendered with `panelError($("bt-preset-error"), "Couldn't load saved strategies.", btLoadPresets)` and select text `⚠ strategies unavailable` | re-call `btLoadPresets()` |
| C3 | `loadSignalKeys()` `index.html:3718-3734` | `catch (err) { return; }` (leaves stale table) | Render into `#dp-keys`: `<tr><td colspan="6" class="neg">Couldn't load signal keys. <button class="btn btn-ghost btn-sm" data-retry-keys>Retry</button></td></tr>` | bind `data-retry-keys` → `loadSignalKeys()` |
| C4 | `loadDeployPresets()` `index.html:3736-3745` | `catch (err) { return; }` (leaves stale table) | Render into `#dp-presets`: `<tr><td colspan="5" class="neg">Couldn't load presets. <button … data-retry-presets>Retry</button></td></tr>` | bind → `loadDeployPresets()` |
| C5 | `bootData()` `index.html:3928-3939` | `try { S.instruments = (await json("/data/instruments")) || []; } catch (e) { S.instruments = []; }` (silent) | Do **not** silently continue: set `S.instrumentsError = true`, `toast("Couldn't load the instrument list — type symbols manually.", "err")`, `setApiPill("down", "degraded")`, and render a note under the routing card (`#routes`, `index.html:574`) with `panelError(..., "Instrument list unavailable.", bootData)`; still seed default tickers | re-call `bootData()` |

**Real empty states are preserved:** the four `"No … yet"` messages must still render when the request
**succeeds with an empty list**. The fix is only to branch on *failure* vs *empty* — track a boolean
(e.g. `S.loadError.keys = true`) or `try/catch` separately, and never read `catch` as empty.

---

## 2. DOCS-CORRECTION PUNCH LIST

**Canonical test count — pin to the FROZEN suite, not a point-in-time value.**
The count grew during this round (740 baseline → **769 now**). The engineer MUST take the number from a
single final `pytest trading/tests -q` run (after the last Round-2 test is added) and paste those exact
words into every location. **As of this writing the run is `768 passed, 1 skipped` (769 collected)**, so
the "Corrected text" column below uses that value; substitute the final frozen value if it differs.

### 2.1 Test counts (5 conflicting values → one)

| File:line | Current text | Corrected text |
|-----------|--------------|----------------|
| `README.md:41` | `.venv/Scripts/python.exe -m pytest trading/tests -q       # 418 tests` | `… -q       # 769 tests (768 passed, 1 skipped)` |
| `README.md:294` | `**Done & tested (418 tests):**` | `**Done & tested (769 tests — 768 passed, 1 skipped):**` |
| `ARCHITECTURE.md:159` | `Steps 1–14 above are now implemented and covered by the test suite (378 tests).` | `… covered by the test suite (769 tests).` |
| `docs/UNIFIED_STRATEGY.md:334` | `exports → refresh → revoke → 410). Full suite: **629 passed, 1 skipped**` | `… Full suite: **768 passed, 1 skipped**` |
| `docs/FINALIZATION_REPORT.md:3` | `Date: 2026-10-06 · Suite: **678 passed, 1 skipped** (was 651) · UI QA: 17/17 checks, 0 console errors` | `Date: 2026-10-06 · Suite (at report time): **678 passed, 1 skipped** · current suite: **769 (768 passed, 1 skipped)** · UI QA: 17/17 checks, 0 console errors` |
| `docs/FINALIZATION_REPORT.md:98` | `| `pytest trading/tests` | **678 passed, 1 skipped** (new: 13 strategy, …) |` | `| `pytest trading/tests` | **768 passed, 1 skipped** (769 collected) |` |

> **Historical-doc caveat:** `FINALIZATION_REPORT.md` and `SPEC-local-signal-sprint.md` are dated
> point-in-time records. Update the *living* docs (README, ARCHITECTURE, UNIFIED) to the canonical
> number; for the dated reports prefer the "at report time … current suite …" wording above rather than
> silently rewriting history.
> `pytest_t08.log` is a stale scratch log (739 passed, 1 skipped) — leave as-is; it is not a doc.

### 2.2 Dashboard refresh interval

| File:line | Current text | Corrected text |
|-----------|--------------|----------------|
| `docs/UNIFIED_STRATEGY.md:220` | `* **Near-real-time**: the page auto-polls `/API_KEY/{key}/data` every 15 s,` | `… every 60 s,` (matches `dashboard.py:424` `setInterval(…, 60000)` and the page's own "(auto-refresh every 60 s)" at `dashboard.py:346`) |

`dashboard.py` code is correct — **no code change**; only the doc is wrong. (README/ARCHITECTURE already
say 60 s.)

### 2.3 Wrong autotune path

| File:line | Current text | Corrected text |
|-----------|--------------|----------------|
| `docs/SPEC-local-signal-sprint.md:106` | `live trading; exposed as `POST /api/v1/autotune`.` | `live trading; exposed as `POST /api/v1/backtest/autotune`.` |

Verified: the handler is declared in `trading/api/routers/backtest.py` (`@router.post("/autotune")`)
under the `/backtest` prefix → `/api/v1/backtest/autotune`; the console calls
`/backtest/autotune` (`index.html:2781`).

### 2.4 "Redis-backed distributed rate limiting" claim

Reality: `gex/` has a Redis+Lua bucket, and `trading/adapters/ratelimit/token_bucket.py:98`
(`RedisTokenBucket`) **exists**, but the **API middleware** (`trading/api/middleware.py:15-38`) uses an
**in-memory per-IP `TokenBucket` only** — Redis is never wired in.

| File:line | Current text | Corrected text |
|-----------|--------------|----------------|
| `README.md:12` | `- `adapters/` — rate limiter (token bucket, Redis+Lua), BingX …` | `- `adapters/` — rate limiter (token bucket; in-memory `TokenBucket` + an **unwired** `RedisTokenBucket`), BingX …` |
| `README.md:324` | `rate limiter (in-memory + Redis/Lua) + **API rate-limit middleware**, CORS,` | `rate limiter (in-memory `TokenBucket` + optional `RedisTokenBucket`) + **in-memory, per-IP API rate-limit middleware**, CORS,` |
| `ARCHITECTURE.md:27` | `- Rate limiting is already **distributed (Redis + Lua)** with a `Decision`` | `- Rate limiting in `gex` is **distributed (Redis + Lua)** with a `Decision` (the `trading` API middleware is currently in-memory per-IP — see §3 row 1)` |
| `ARCHITECTURE.md:93` | `| 1 | Redis-backed strict rate limiting | ✅ exists | `gex/adapters/ratelimit` (Redis+Lua token bucket) reused; add per-exchange rule config |` | `| 1 | Redis-backed strict rate limiting | ⚠️ partial | `gex` has Redis+Lua; the `trading` API middleware uses an in-memory per-IP bucket (`trading/api/middleware.py`) — `RedisTokenBucket` present but not wired |` |
| `.env.example:37-38` | `# Redis broker (Celery) + result backend. Also used for the run-cancellation`<br>`# registry and the token-bucket rate limiter.` | `# Redis broker (Celery) + result backend. Also used for the run-cancellation`<br>`# registry. (The API rate-limit middleware is in-memory per-IP and does not use Redis.)` |

### 2.5 Stale ARCHITECTURE §3 gap table

The table at `ARCHITECTURE.md:88-104` is an **initial-state snapshot** that now contradicts the same
document's §5 ("Steps 1–14 … now implemented") and the repo. Update the status column to reality
(all artefacts verified present on this tree):

| Line | Requirement | Current status | Corrected status |
|------|-------------|----------------|------------------|
| `92` | Unified async `BaseFetcher` | ❌ missing | ✅ exists — `trading/ports/fetcher.py` + `trading/adapters/fetchers/*` |
| `93` | Redis-backed strict rate limiting | ✅ exists | ⚠️ partial — see §2.4 |
| `94` | Pluggable `Strategy` ABC + `Signal`/`OrderIntent` | ❌ missing | ✅ exists — `trading/ports/strategy.py`, `trading/domain/` |
| `95` | TA-Lib / pandas-ta indicators | ⚠️ partial | ⚠️ partial (unchanged — TA-Lib C backend only in Docker) |
| `96` | Monte-Carlo backtest engine | ❌ missing | ✅ exists — `trading/application/backtest/monte_carlo.py` |
| `97` | TBANK adapter | ❌ missing | ✅ exists — `trading/adapters/brokers/tbank.py` (+ `tbank_stream.py`) |
| `98` | BINGX adapter | ❌ missing | ✅ exists — `trading/adapters/brokers/bingx.py` (+ `bingx_ws.py`) |
| `99` | REST /strategies /backtest /signals /orders /positions /portfolio /data | ⚠️ partial | ✅ exists — all routers mounted in `trading/main.py:83-96` |
| `100` | WebSocket live signals/streams | ❌ missing | ✅ exists — `trading/api/websockets.py`, `local_client_ws.py` |
| `101` | JWT auth, rate limiting, Swagger | ⚠️ partial | ✅ exists — `trading/api/auth.py`, `middleware.py`, `/docs` |
| `102` | Celery + Redis tasks | ⚠️ partial | ✅ exists — `trading/tasks.py` (`celery_app`) |
| `103` | Postgres + TimescaleDB | ❌ missing | ⚠️ partial — DDL/`trading/adapters/persistence/timescale.py`, async sessions; needs a Docker/Postgres host |
| `104` | Docker, Alembic, Pytest>85%, Prometheus/Grafana, structured logging | ⚠️ partial | ✅ exists — `Dockerfile`, `docker-compose.yml`, `alembic/versions/0001-0006`, `trading/observability.py`, `.github/workflows/ci.yml` |

Also add a one-line banner directly under `## 3. Gap analysis (spec vs repo)`:
`> Status column reflects the current tree (2026-10). It was captured as the initial 2026-10-02 spec-vs-repo snapshot.`

---

## 3. ACCEPTANCE CRITERIA (one pass/fail per control)

Each criterion is a single testable statement. `PASS`/`FAIL` is binary; "see §" points to the spec above.

### 3.1 Frontend controls

| ID | Criterion (testable) |
|----|----------------------|
| AC-1 | Deleting a broker account (Accounts tab → **Delete**) shows a `confirm()` naming the account; dismissing it leaves the account and issues **no** `DELETE /keys/{id}`; accepting issues the delete. |
| AC-2 | Revoking a signal key (Deploy tab → **Revoke**) shows a `confirm()` naming the key prefix; dismissing it issues **no** `DELETE /signal-keys/{id}`; accepting revokes and refreshes the table. |
| AC-3 | Booting with a stale/expired token in `localStorage["gex.token"]` never renders the app: the app probes a protected endpoint, and on 401 clears the token and shows the login screen with the text "Your session expired — please sign in again." |
| AC-4 | Any API call returning **401** while the app view is visible clears the token and returns to the login screen (with the same message) exactly once per expiry — verified by forcing a 401 on `GET /keys` and on a download. |
| AC-5 | A 401 from `POST /auth/token` (wrong password) does **not** navigate away — the inline "Sign-in failed: …" message still shows. |
| AC-6 | For each of `loadKeys`, `btLoadPresets`, `loadSignalKeys`, `loadDeployPresets`, `bootData`: when the endpoint returns a **non-2xx/network error**, the panel shows an error message **plus a working Retry** and does **not** show the "No … yet" empty copy. |
| AC-7 | For the same five loaders: when the endpoint returns **200 with an empty list**, the original "No … yet" empty copy is shown and **no** error/Retry is shown. |
| AC-8 | Clicking Retry in any of the five error states re-invokes the loader and, on success, replaces the error with the correct content. |

**How to test (QA):** intercept `fetch` in jsdom (or use a proxy) to return `401` for a protected path
and `500`/abort for each loader path; assert on DOM (`#accounts-list`, `#bt-preset-error`,
`#dp-keys`, `#dp-presets`, `#routes`) for the presence of `[data-retry]` and the absence of
`No … yet`; assert `localStorage.getItem("gex.token") === null` and `#view-login` visible after a 401.

### 3.2 Docs corrections

| ID | Criterion (testable) |
|----|----------------------|
| AC-9 | `grep -rnE "418 tests|378 tests|629 passed|678 passed" README.md ARCHITECTURE.md docs/` returns **no** live-doc hit for the old counts (README/ARCHITECTURE/UNIFIED updated to the frozen value below). |
| AC-10 | Every occurrence of the test count in README/ARCHITECTURE/UNIFIED states the **same single value**, and that value matches a fresh `pytest trading/tests -q` run at freeze (currently `768 passed, 1 skipped` / 769 collected; must be re-pinned after the last Round-2 test lands). |
| AC-11 | `docs/UNIFIED_STRATEGY.md` contains **no** "every 15 s" for the dashboard; the stated interval equals 60 s and matches `dashboard.py:424`. |
| AC-12 | `docs/SPEC-local-signal-sprint.md` references `POST /api/v1/backtest/autotune` (not `/api/v1/autotune`), and that path exists in `trading/api/routers/backtest.py`. |
| AC-13 | No doc/`.env.example` claims the trading API rate limiter is Redis-backed/distributed: `grep -rn "Redis-backed strict rate limiting\|token-bucket rate limiter"` yields only the corrected per-IP-in-memory wording; `ARCHITECTURE.md:93` is `⚠️ partial`. |
| AC-14 | `ARCHITECTURE.md` §3 gap table no longer marks `BaseFetcher`, `Strategy ABC/Signal`, `Monte-Carlo`, `TBANK`, `BINGX`, REST routes, WebSocket, JWT, Celery as `❌ missing`; each matches the verified artefact cited in §2.5, and the snapshot banner is present. |

---

*End of Round-2 spec.*
