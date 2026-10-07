# QA Verification Report — GEX-API Production-Readiness Audit

**Author:** 严过关 (QA Engineer) · **Date:** 2026-10-07 · **Root:** `E:\gex api`
**Scope:** baseline test suite, coverage & CI gate, live dynamic probing, auth-enforcement sweep, offline verifier + unified smoke, added regression tests.

All commands below were executed on Windows (Git Bash) with `venv/Scripts/python.exe` (Python 3.14, pytest 9.1.1).
Raw logs / harnesses are in this folder: `_qa_fullsuite.log`, `_qa_fullsuite_round2.log`, `_qa_coverage.log`,
`_qa_probe.py` + `_qa_probe_results.json`, `_qa_server.log`, `_qa_verify_backtest.out`, `_qa_smoke.out`.

---

## 1. Baseline test result

```
$ venv/Scripts/python.exe -m pytest trading/tests -q
739 passed, 1 skipped, 1 warning in 197.85s (0:03:17)   # EXIT 0
```

Round-2 regression (after the added tests, see §6):

```
$ venv/Scripts/python.exe -m pytest trading/tests -q
768 passed, 1 skipped, 1 warning in 174.18s (0:02:54)   # EXIT 0
```

Verdict: **suite green, no failures/errors, reproducible.** (Matches the documented baseline of 739 passed / 1 skipped.)

---

## 2. Coverage result & CI-gate verdict

The exact CI gate (`.github/workflows/*.yml`) is:

```
ruff check trading
pytest trading/tests -q --cov=trading --cov-report=term-missing --cov-fail-under=85
```

`pytest-cov` and `ruff` were **not installed** in the venv (the `[dev]` extra was not installed); I installed them to evaluate the gate:

```
$ venv/Scripts/python.exe -m pip install pytest-cov ruff
Successfully installed coverage-7.16.2 pytest-cov-7.1.0 ruff-0.16.10
```

### 2a. Coverage gate — **PASS**

```
$ venv/Scripts/python.exe -m pytest trading/tests -q --cov=trading --cov-report=term-missing --cov-fail-under=85
TOTAL                                                        15898   1044    93%
Required test coverage of 85% reached. Total coverage: 93.43%
739 passed, 1 skipped, 1 warning in 201.35s (0:03:21)   # EXIT 0
```

`--cov-fail-under=85` **passes** (93.43% ≥ 85%).

### 2b. Lint gate — **FAIL**

```
$ venv/Scripts/python.exe -m ruff check trading
Found 416 errors.
[*] 295 fixable with the `--fix` option   # EXIT 1
```

**CI verdict: the pipeline would go RED on the lint step** (`ruff`), despite the test + coverage steps passing. This is a release blocker for a green build.

### 2c. Per-module coverage (critical surface)

| Module | Cover | Module | Cover |
|---|---|---|---|
| `api/routers/export.py` | 100% | `api/routers/backtest.py` | 77% |
| `api/middleware.py` | 100% | `api/routers/presets.py` | 74% |
| `security.py` | 100% | `api/routers/signal_keys.py` | 72% |
| `application/signal_engine.py` | 94% | `api/routers/keys.py` | 69% |
| `application/reporting/trade_export.py` | 100% | `api/routers/portfolio.py` | 56% |
| `application/signal_keys.py` | 93% | `api/routers/strategies.py` | 53% |
| `main.py` | 93% | `api/routers/data.py` | 76% |

Low-coverage *critical* modules: `routers/portfolio.py` (live broker paths 56%), `routers/strategies.py` (53% — the start/stop handlers are barely exercised), `routers/keys.py` (69%).

---

## 3. Dynamic-probe table (live server on :8199)

Booted with a dedicated QA DB so the dev DB was untouched:
```
$ TRADING_DATABASE_URL="sqlite+aiosqlite:///./qa_audit.db" \
  venv/Scripts/python.exe -m uvicorn trading.main:app --port 8199
INFO: Application startup complete.  Uvicorn running on http://127.0.0.1:8199
```
Harness: `venv/Scripts/python.exe deliverables/software-company/audit/_qa_probe.py` (raw results in `_qa_probe_results.json`).

| Endpoint | Method | Expected | Actual | Body excerpt | Verdict |
|---|---|---|---|---|---|
| `/health` | GET | 200 | 200 | `{"status":"ok"}` | PASS |
| `/metrics` | GET | 200 | 200 | `# HELP python_gc_objects_collected_total …` | PASS |
| `/docs` | GET | 200 | 200 | Swagger UI HTML | PASS |
| `/` | GET | 200 | 200 | `<!doctype html>…` | PASS |
| `/favicon.ico` | GET | 200 | 200 | ICO bytes | PASS |
| `/static/index.html` | GET | 200 | 200 | HTML | PASS |
| `/api/v1/auth/token` | POST | 200+token | 200 | `{"access_token":"eyJ…"}` | PASS |
| `/api/v1/auth/token` (wrong pass) | POST | 401 | 401 | `{"detail":"Invalid credentials"}` | PASS |
| `/api/v1/auth/token` (bad user) | POST | 401 | 401 | `{"detail":"Invalid credentials"}` | PASS |
| `/api/v1/keys` | GET | 401/403 | **401** | `{"detail":"Not authenticated"}` | PASS |
| `/api/v1/keys` (+token) | GET | 200 | 200 | `[]` | PASS |
| `/api/v1/keys` (tampered token) | GET | 401 | 401 | `{"detail":"Invalid or expired token"}` | PASS |
| `/api/v1/signals` | GET | 401 | 401 | `{"detail":"Not authenticated"}` | PASS |
| `/api/v1/signals` (+token) | GET | 200 | 200 | `[]` | PASS |
| `/api/v1/backtest` (real, synthetic bars) | POST | 200 | 200 | `…"metrics":{"total_return":-0.0078…}` `result_id=1` | PASS |
| `/api/v1/signals/export/signals.csv` | GET | 200 non-trivial | 200 | 195 bytes, `attachment; filename="signals.csv"` | PASS |
| `/api/v1/signals/export/signals.xlsx` | GET | 200 non-trivial | 200 | 1768 bytes, xlsx MIME | PASS |
| `/api/v1/signals/export/positions.csv` | GET | 200 non-trivial | 200 | 337 bytes | PASS |
| `/api/v1/signals/export/positions.xlsx` | GET | 200 non-trivial | 200 | 1864 bytes | PASS |
| `/API_KEY/does-not-exist-123` | GET | 404 | 404 | `{"detail":"unknown API key"}` | PASS |
| `/API_KEY/does-not-exist-123/data` | GET | 404 | 404 | `{"detail":"unknown API key"}` | PASS |
| `/health` × 420 (burst) | GET | 429 + Retry-After | 429 | `429_count=141 first_at=278 Retry-After=1` | PASS |

Rate limiting confirmed: 429 with a `Retry-After` header appears once the configured 300-req/min bucket is drained (first rejection at request #278, i.e. after the initial probes consumed tokens). Behaviour matches `RateLimitMiddleware`.

**All functional probes pass.** No 5xx, no crashes, exports carry non-trivial bytes, auth round-trips correctly.

---

## 4. Auth-enforcement sweep (every router, no `Authorization` header)

Method: read every router in `trading/api/routers/`; hit each route anonymously; a response of **401/403** ⇒ protected, anything else (200/422/404/409) ⇒ the request reached the handler ⇒ **not auth-gated**.

### 4a. PROTECTED (correct) — JWT-guarded routers

| Router | Guard | Result |
|---|---|---|
| `keys.py` (POST/GET/DELETE/PATCH) | `router = APIRouter(..., dependencies=[Depends(require_auth)])` | 401 ✅ |
| `signal_keys.py` (POST/GET/DELETE/PATCH/{key}/generate) | router-level `require_auth` | 401 ✅ |
| `baskets.py` (POST /export, /deploy) | router-level `require_auth` | 401 ✅ |
| `signals.py` (all 11 routes incl. engine start/stop, DELETE /positions, exports) | router-level `require_auth` | 401 ✅ |
| `export.py` (all 4 exports) | router-level `require_auth` | 401 ✅ |
| `portfolio.py` → `POST /orders` only | route-level `dependencies=[Depends(require_auth)]` | 401 ✅ |

### 4b. NOT AUTH-GATED — **unauthenticated mutations (release blocker)**

| Route | Method | Anonymous status | Router file |
|---|---|---|---|
| `/api/v1/presets` | POST | 422 (passed auth, failed body) | `presets.py` (no `require_auth`) |
| `/api/v1/presets/from-backtest/{symbol}` | POST | **200 — created preset id=1** | `presets.py` |
| `/api/v1/presets/{id}/promote` | POST | 422 (gate) | `presets.py` |
| `/api/v1/presets/{id}/rollback` | POST | 409 | `presets.py` |
| `/api/v1/presets/{id}/demote` | POST | 409 | `presets.py` |
| `/api/v1/presets/{id}` | **PATCH** | **200 — mutated preset** | `presets.py` |
| `/api/v1/presets/{id}` | **DELETE** | 404 (no auth gate) | `presets.py` |
| `/api/v1/strategies/{name}/start` | POST | **200 — started a live runner** | `strategies.py` |
| `/api/v1/strategies/{name}/stop` | POST | 200 | `strategies.py` |
| `/api/v1/backtest` | POST | **200 — ran + persisted a backtest** | `backtest.py` |
| `/api/v1/backtest/monte-carlo` | POST | 200 | `backtest.py` |
| `/api/v1/backtest/portfolio` | POST | 422 | `backtest.py` |
| `/api/v1/backtest/portfolio/monte-carlo` | POST | 422 | `backtest.py` |
| `/api/v1/backtest/portfolio/report` | POST | 422 | `backtest.py` |
| `/api/v1/backtest/analyze` | POST | 200 | `backtest.py` |
| `/api/v1/backtest/optimize` | POST | 200 | `backtest.py` |
| `/api/v1/backtest/optimize/global` | POST | **200 — started a 206-ticker optimization run** | `backtest.py` |
| `/api/v1/backtest/optimize/global/{run_id}/cancel` | POST | 404 (no gate) | `backtest.py` |
| `/api/v1/backtest/autotune` | POST | 200 | `backtest.py` |
| `/api/v1/backtest/cancel/{token}` | POST | **200 — cancelled an arbitrary run token** | `backtest.py` |

**→ 20 unauthenticated mutating routes** (11 in `backtest.py`, 7 in `presets.py`, 2 in `strategies.py`).

### 4c. NOT AUTH-GATED — unauthenticated reads (information disclosure)

| Route | Method | Status | Leak |
|---|---|---|---|
| `/api/v1/orders` | GET | 200 `[]` | order history |
| `/api/v1/positions` | GET | 200 `[]` | live positions |
| `/api/v1/portfolio` | GET | 200 `{"cash":0.0,…}` | account view |
| `/api/v1/backtest/cancel` | GET | 200 `["d61d8…"]` | **active run tokens** |
| `/api/v1/strategies` | GET | 200 | catalogue |
| `/api/v1/data/instruments` | GET | 200 | symbol universe |

### 4d. Key-gated (by design, not JWT)

`GET /API_KEY/{key}` and `POST /API_KEY/{key}/refresh` — the URL key *is* the credential (documented in `dashboard.py`). Bogus key ⇒ 404 ✅. This is a deliberate design choice; flagged for the architect, not counted as a defect.

---

## 5. Bugs reproduced (with repro commands)

Severity buckets: **High = 3, Medium = 2, Low = 1.**

**BUG-1 [HIGH] — Entire preset CRUD surface is unauthenticated.**
An anonymous caller can create, version, edit, promote/rollback/demote and delete per-ticker strategy presets (the deploy-time control plane).
```bash
curl -s -X POST http://127.0.0.1:8199/api/v1/presets/from-backtest/SYNTH   # → 200, created preset id=1
curl -s -X PATCH http://127.0.0.1:8199/api/v1/presets/1                   # → 200, mutated
curl -s -X DELETE http://127.0.0.1:8199/api/v1/presets/999999            # → 404 (no auth gate reached)
```
Source: `trading/api/routers/presets.py:43` (`APIRouter(prefix="/presets", tags=["presets"])` — no `dependencies=[Depends(require_auth)]`).

**BUG-2 [HIGH] — Strategy lifecycle is unauthenticated.**
```bash
curl -s -X POST http://127.0.0.1:8199/api/v1/strategies/sma_crossover/start   # → {"status":"running"}
```
Source: `trading/api/routers/strategies.py:10` — no `require_auth`.

**BUG-3 [HIGH] — Backtest / optimizer / autotune / cancel are unauthenticated (resource-exhaustion + token tampering).**
```bash
curl -s -X POST http://127.0.0.1:8199/api/v1/backtest -H 'content-type: application/json' -d '{}'          # → 200 (real run)
curl -s -X POST http://127.0.0.1:8199/api/v1/backtest/optimize/global                                      # → 200 run_id=… total=206
curl -s -X POST http://127.0.0.1:8199/api/v1/backtest/cancel/abc                                           # → 200 {"cancelled":true}
```
Source: `trading/api/routers/backtest.py:86` — no `require_auth`. Amplifier: CORS is `allow_origins=["*"]` (`main.py:73`), so any website the operator visits can silently drive these endpoints.

**BUG-4 [MEDIUM] — Unauthenticated read of trading data + run-token leak.**
`GET /api/v1/orders|positions|portfolio` and `GET /api/v1/backtest/cancel` return 200 anonymously; the latter discloses active optimization run tokens.

**BUG-5 [MEDIUM] — CI lint gate is red.**
`ruff check trading` → **416 errors** (exit 1). The pipeline cannot pass as configured; the `[dev]` extra (ruff, pytest-cov) is also not installed in the venv.

**BUG-6 [LOW] — Default admin credentials shipped.**
`trading/config.py:23-24` defaults `admin_username="admin"`, `admin_password="admin"` (and `secret_key="dev-secret-change-me"`). Safe only if every deployment overrides env vars (see architect report).

> Note on provenance: probes that returned 200 executed **real work** (created a preset, started a live runner, launched a 206-ticker global optimization). This is direct evidence that the routes are reachable, not merely that they lack a decorator.

---

## 6. Tests added

**One new file, `trading/tests/test_authz_regression.py`** — 29 tests, all passing (`29 passed in 2.43s`), deterministic & offline (ASGITransport; no network). It targets gaps the sweep found **uncovered** — no existing test was renamed, weakened, or deleted.

| # | Test | Gap closed |
|---|---|---|
| 1 | `test_tampered_token_is_rejected` (3 params) | only *missing* token was tested; now garbage / non-JWT / foreign-signed JWT → 401 |
| 2 | `test_missing_token_is_rejected` | explicit 401 + exact detail |
| 3 | `test_auth_enforced_before_body_validation` | anonymous malformed body → 401 (never 422 ⇒ validator shape doesn't leak) |
| 4 | `test_authenticated_malformed_body_is_422` | valid token + bad `exchange` → 422 |
| 5 | `test_guarded_routes_reject_anonymous` (19 params) | **systematic** sweep of every JWT-guarded route — fails loudly if a future route is added without `require_auth` |
| 6 | `test_delete_missing_api_key_404` / `test_delete_missing_signal_key_404` / `test_patch_missing_signal_key_404` | deleting/patching a non-existent resource ⇒ clean 404 |
| 7 | `test_rate_limit_429_has_retry_after_header` | middleware 429 must advertise `Retry-After` (previously only the 429 status was asserted) |

Coverage-gap analysis (journeys still without a test, recommended for follow-up): anonymous **mutation** guard (the sweep documents the gap; a *positive* test would currently fail by design), duplicate signal-key creation, malformed CSV/blob exports, WebSocket `notify_local_clients=false` path, `routers/portfolio.py` live-broker branches (56% cover), `routers/strategies.py` start/stop (53%).

---

## 7. Offline verifier & unified smoke

**`scripts/verify_backtest.py` — PASS (offline, exit 0):**
```
$ PYTHONPATH=. venv/Scripts/python.exe scripts/verify_backtest.py
=== buy_and_hold (499 bars) === total_return -0.1512  sharpe -0.118  max_drawdown +0.3939 …
=== sma_crossover (499 bars) === total_return -0.3031  sharpe -1.134  profit_factor 0.064  n_trades 7 …
=== Monte-Carlo GBM (10,000 paths, 252 steps) === mean -0.0820  90% interval [-0.4850, +0.4771]
EXIT=0
```

**`scripts/smoke_unified_workflow.py` — PASS (23/23 steps, exit 0), run against my :8199 server:**
```
$ SMOKE_BASE=http://127.0.0.1:8199 venv/Scripts/python.exe scripts/smoke_unified_workflow.py
[PASS] health · login · preset from backtest · optimize + save preset · presets list ·
       default preset · global optimize start/done · create API key · generate signals ·
       dashboard HTML/data/charts · CSV export · XLSX export · manual refresh · revoke ·
       revoked key -> 410 · audit trail
SMOKE TEST PASSED: all steps green   EXIT=0
```
(Note: this script is **not** fully offline — it performs real data fetches/optimization. It nonetheless completed green end-to-end.)

---

## 8. Remaining unverified / limitations

- **Live broker integration** (`BingxBroker`, `TbankBroker` real order placement) — cannot be exercised without live credentials; `POST /orders` was verified only to the "no credentials → 400 / no token → 401" boundary. `routers/portfolio.py` is 56% covered.
- **WebSocket** streams (`/ws/signals`, `/ws/client`) were not long-poll tested live under load (unit coverage exists).
- **Rate-limit semantics under concurrency** verified sequentially (429 + Retry-After confirmed); distributed/multi-worker buckets not tested (single-process token bucket, in-memory).
- **Coverage represents `trading/` only** (as the CI gate specifies). `gex/`, `quant/`, `research/` are outside the gate.
- **`ruff` errors were not triaged individually** (416 found); the finding is the *gate result*, not a per-rule breakdown.
- The dynamic sweep used `json={}` bodies; a 422/404 there proves "no auth gate", but does not enumerate every downstream validation branch.

---

## Appendix — exact commands

```bash
# baseline
venv/Scripts/python.exe -m pytest trading/tests -q
# coverage + CI gate (after: pip install pytest-cov ruff)
venv/Scripts/python.exe -m pytest trading/tests -q --cov=trading --cov-report=term-missing --cov-fail-under=85
venv/Scripts/python.exe -m ruff check trading
# dynamic probes
TRADING_DATABASE_URL="sqlite+aiosqlite:///./qa_audit.db" venv/Scripts/python.exe -m uvicorn trading.main:app --port 8199 &
venv/Scripts/python.exe deliverables/software-company/audit/_qa_probe.py
# verifier + smoke
PYTHONPATH=. venv/Scripts/python.exe scripts/verify_backtest.py
SMOKE_BASE=http://127.0.0.1:8199 venv/Scripts/python.exe scripts/smoke_unified_workflow.py
# new tests
venv/Scripts/python.exe -m pytest trading/tests/test_authz_regression.py -q
```
