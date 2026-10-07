# GEX-API — Round 2: Exact Implementation Design (READ-ONLY)

- **Author:** 高见远 (Architect)
- **Inputs:** `architecture-audit.md` (A1–A22), `engineer-code-audit.md` (ENG-01..23), QA round-1.
- **Purpose:** give the Engineer an implement-without-re-discovery spec for the five highest-risk, cross-cutting areas. **No product code was modified to produce this.**
- **Verifies/extends:** A1 (auth), A3+A13 (schema/CI), A4 (rate limit), A6 (metrics), A5 (timezone).

Legend for tiers used below: **public** = no credential · **valid-jwt** = `require_auth` (any valid bearer) · **admin-amplified** = still `require_auth`, but semantically high-blast-radius (go-live / engine / mutate) — see §1.7 · **key-gated** = the `{key}` path segment is the credential (no JWT) · **ws-jwt** = WebSocket JWT handshake.

---

## 0. Decisions at a glance

| # | Area | Decision |
|---|---|---|
| 1 | Auth perimeter | Enforce at the **`include_router` mount** in `main.py` (one file) for all business routers; keep `auth` public; keep `/API_KEY/*` key-gated; gate `/metrics`; disable `/docs` in prod. |
| 2 | WebSocket auth | `/ws/{signals,orders,positions}`: **JWT via `Sec-WebSocket-Protocol`** (fallback `?token=`). `/ws/client`: **token inside the existing `handshake` frame**. |
| 3 | Schema drift | Fix **both** classes: add migration **0007** to set `created_at/updated_at NOT NULL` (batch), and set `compare_type=False` in `alembic/env.py` for the FLOAT≡Double spelling diff. |
| 4 | Startup schema | Replace `await init_db()` in the lifespan with a new `run_migrations()` (alembic `upgrade head`); drop the request-time `init_db()` calls in `backtest.py`. Keep `init_db()` (create_all) **test-only**. |
| 5 | Rate limit | **Fix in-memory correctly AND correct the docs** (keep Redis as an opt-in backend). Add a dedicated login bucket, bound `_buckets`, add a WS connection cap. |
| 6 | Metrics/alerting | **Wire up** (cheap) + fix `prometheus.yml`/compose; instrument the five idle counters; schedule `broker_health`. |
| 7 | Timezone | Add a `UTCDateTime` `TypeDecorator` (ORM-level) that normalises naive→aware-UTC on read; no DDL change. |

---

## 1. AUTH PERIMETER  (A1 · ENG-01/02/03 · QA)

### 1.1 Complete route → required-tier table

All `/api/v1/*` routes are shown without the prefix except where noted. `file:line` is where the route (or router) is declared.

| # | Method | Path | Router file:line | Current | Required tier |
|---|---|---|---|---|---|
| 1 | POST | `/api/v1/auth/token` | `auth.py:16` | public | **public** (login) |
| 2 | GET | `/health` | `main.py:99` | public | **public** (liveness) |
| 3 | GET | `/` · `/favicon.ico` · `/static/*` | `main.py:109,112,117` | public | **public** (SPA shell) |
| 4 | GET | `/metrics` | `main.py:104` | public | **gated** (env token) — §1.5 |
| 5 | GET | `/docs` · `/redoc` · `/openapi.json` | FastAPI defaults | public | **disabled in prod** — §1.5 |
| 6 | POST | `/api/v1/keys` | `keys.py:39` | valid-jwt | valid-jwt |
| 7 | GET | `/api/v1/keys` | `keys.py:60` | valid-jwt | valid-jwt |
| 8 | DELETE | `/api/v1/keys/{key_id}` | `keys.py:69` | valid-jwt | valid-jwt |
| 9 | PATCH | `/api/v1/keys/{key_id}/settings` | `keys.py:79` | valid-jwt | valid-jwt |
| 10 | GET | `/api/v1/keys/routing` | `keys.py:99` | valid-jwt | valid-jwt |
| 11 | POST | `/api/v1/backtest` | `backtest.py:462` | **public** | valid-jwt |
| 12 | POST | `/api/v1/backtest/monte-carlo` | `backtest.py:544` | **public** | valid-jwt |
| 13 | POST | `/api/v1/backtest/portfolio` | `backtest.py:581` | **public** | valid-jwt |
| 14 | POST | `/api/v1/backtest/portfolio/monte-carlo` | `backtest.py:611` | **public** | valid-jwt |
| 15 | POST | `/api/v1/backtest/portfolio/report` | `backtest.py:640` | **public** | valid-jwt |
| 16 | POST | `/api/v1/backtest/analyze` | `backtest.py:674` | **public** | valid-jwt |
| 17 | POST | `/api/v1/backtest/optimize` | `backtest.py:718` | **public** | valid-jwt (admin-amplified: `save_preset`) |
| 18 | POST | `/api/v1/backtest/optimize/global` | `backtest.py:841` | **public** | valid-jwt (admin-amplified: writes presets) |
| 19 | GET | `/api/v1/backtest/optimize/global/{run_id}` | `backtest.py:896` | **public** | valid-jwt |
| 20 | POST | `/api/v1/backtest/optimize/global/{run_id}/cancel` | `backtest.py:907` | **public** | valid-jwt |
| 21 | POST | `/api/v1/backtest/autotune` | `backtest.py:917` | **public** | valid-jwt |
| 22 | POST | `/api/v1/backtest/cancel/{token}` | `backtest.py:960` | **public** | **public (capability)** — see §1.6 |
| 23 | GET | `/api/v1/backtest/cancel` | `backtest.py:975` | **public** | valid-jwt (leaks live run tokens) |
| 24 | GET | `/api/v1/strategies` | `strategies.py:13` | **public** | valid-jwt |
| 25 | GET | `/api/v1/strategies/{name}/schema` | `strategies.py:21` | **public** | valid-jwt |
| 26 | POST | `/api/v1/strategies/{name}/start` | `strategies.py:45` | **public** | valid-jwt (admin-amplified: starts live runner) |
| 27 | POST | `/api/v1/strategies/{name}/stop` | `strategies.py:66` | **public** | valid-jwt (admin-amplified) |
| 28 | GET | `/api/v1/portfolio` | `portfolio.py:60` | **public** | valid-jwt |
| 29 | GET | `/api/v1/positions` | `portfolio.py:85` | **public** | valid-jwt |
| 30 | GET | `/api/v1/orders` | `portfolio.py:104` | **public** | valid-jwt |
| 31 | POST | `/api/v1/orders` | `portfolio.py:117` | valid-jwt | valid-jwt |
| 32 | GET | `/api/v1/data/instruments` | `data.py:22` | **public** | valid-jwt |
| 33 | GET | `/api/v1/data/ohlcv/{symbol}` | `data.py:28` | **public** | valid-jwt |
| 34 | GET | `/api/v1/data/sources` | `data.py:55` | **public** | valid-jwt |
| 35 | GET | `/api/v1/data/detect/{symbol}` | `data.py:60` | **public** | valid-jwt |
| 36 | GET | `/api/v1/data/categories` | `data.py:66` | **public** | valid-jwt |
| 37 | GET | `/api/v1/data/universe` | `data.py:72` | **public** | valid-jwt |
| 38 | GET | `/api/v1/export/backtest/{result_id}/trades.csv` | `export.py:58` | valid-jwt | valid-jwt |
| 39 | GET | `/api/v1/export/backtest/{result_id}/trades.xlsx` | `export.py:66` | valid-jwt | valid-jwt |
| 40 | GET | `/api/v1/export/live-trades.csv` | `export.py:74` | valid-jwt | valid-jwt |
| 41 | GET | `/api/v1/export/live-trades.xlsx` | `export.py:79` | valid-jwt | valid-jwt |
| 42 | POST | `/api/v1/presets` | `presets.py:114` | **public** | valid-jwt |
| 43 | GET | `/api/v1/presets` | `presets.py:137` | **public** | valid-jwt |
| 44 | GET | `/api/v1/presets/latest` | `presets.py:149` | **public** | valid-jwt |
| 45 | GET | `/api/v1/presets/versions` | `presets.py:161` | **public** | valid-jwt |
| 46 | GET | `/api/v1/presets/default/{symbol}` | `presets.py:178` | **public** | valid-jwt |
| 47 | POST | `/api/v1/presets/from-backtest/{symbol}` | `presets.py:190` | **public** | valid-jwt |
| 48 | GET | `/api/v1/presets/{preset_id}/validate` | `presets.py:215` | **public** | valid-jwt |
| 49 | POST | `/api/v1/presets/{preset_id}/promote` | `presets.py:224` | **public** | valid-jwt (**admin-amplified: GO-LIVE**) |
| 50 | POST | `/api/v1/presets/{preset_id}/rollback` | `presets.py:241` | **public** | valid-jwt (admin-amplified) |
| 51 | POST | `/api/v1/presets/{preset_id}/demote` | `presets.py:263` | **public** | valid-jwt (admin-amplified) |
| 52 | PATCH | `/api/v1/presets/{preset_id}` | `presets.py:277` | **public** | valid-jwt |
| 53 | DELETE | `/api/v1/presets/{preset_id}` | `presets.py:299` | **public** | valid-jwt |
| 54 | POST | `/api/v1/signal-keys` | `signal_keys.py:59` | valid-jwt | valid-jwt |
| 55 | GET | `/api/v1/signal-keys` | `signal_keys.py:92` | valid-jwt | valid-jwt |
| 56 | DELETE | `/api/v1/signal-keys/{key_id}` | `signal_keys.py:97` | valid-jwt | valid-jwt |
| 57 | PATCH | `/api/v1/signal-keys/{key_id}` | `signal_keys.py:104` | valid-jwt | valid-jwt |
| 58 | POST | `/api/v1/signal-keys/{key}/generate` | `signal_keys.py:120` | valid-jwt | valid-jwt |
| 59 | POST | `/api/v1/baskets/export` | `baskets.py:44` | valid-jwt | valid-jwt |
| 60 | POST | `/api/v1/baskets/deploy` | `baskets.py:62` | valid-jwt | valid-jwt |
| 61 | POST | `/api/v1/signals/engine/start` | `signals.py:50` | valid-jwt | valid-jwt (admin-amplified) |
| 62 | POST | `/api/v1/signals/engine/stop` | `signals.py:69` | valid-jwt | valid-jwt (admin-amplified) |
| 63 | GET | `/api/v1/signals/engine` | `signals.py:74` | valid-jwt | valid-jwt |
| 64 | GET | `/api/v1/signals` | `signals.py:79` | valid-jwt | valid-jwt |
| 65 | GET | `/api/v1/signals/positions` | `signals.py:95` | valid-jwt | valid-jwt |
| 66 | GET | `/api/v1/signals/stats` | `signals.py:109` | valid-jwt | valid-jwt |
| 67 | DELETE | `/api/v1/signals/positions` | `signals.py:114` | valid-jwt | valid-jwt |
| 68-71 | GET | `/api/v1/signals/export/{signals,positions}.{csv,xlsx}` | `signals.py:121,133,145,156` | valid-jwt | valid-jwt |
| 72 | GET | `/api/v1/local-clients` | `local_client_ws.py:148` | valid-jwt | valid-jwt |
| 73 | GET | `/API_KEY/{key}` | `dashboard.py:129` | key-gated | **key-gated** (path key) |
| 74 | GET | `/API_KEY/{key}/data` | `dashboard.py:144` | key-gated | key-gated |
| 75 | GET | `/API_KEY/{key}/charts` | `dashboard.py:200` | key-gated | key-gated |
| 76 | GET | `/API_KEY/{key}/trades.csv` | `dashboard.py:252` | key-gated | key-gated |
| 77 | GET | `/API_KEY/{key}/trades.xlsx` | `dashboard.py:262` | key-gated | key-gated |
| 78 | POST | `/API_KEY/{key}/refresh` | `dashboard.py:277` | key-gated | **key-gated + throttled** (§1.5) |
| 79 | WS | `/ws/signals` | `websockets.py:49` | public | **ws-jwt** |
| 80 | WS | `/ws/orders` | `websockets.py:54` | public | **ws-jwt** |
| 81 | WS | `/ws/positions` | `websockets.py:59` | public | **ws-jwt** |
| 82 | WS | `/ws/client` | `local_client_ws.py:118` | public | **handshake-frame token** |

### 1.2 How `require_auth` gets applied — **at the mount, not per-route**

Chosen mechanism: **`dependencies=[Depends(require_auth)]` on each `include_router` call in `trading/main.py`** (lines 83–95), so a future router cannot be forgotten (the single source of truth is one file). This is a **FastAPI dependency, not middleware**, because:
- the SPA already knows how to send `Authorization: Bearer` (§1.4) and every endpoint that should 401 does so uniformly with the exact `{"detail": …}` bodies `test_authz_regression.py:65-96` already asserts;
- middleware auth (e.g. path-prefix allow-listing) is more error-prone and breaks `/API_KEY/*` and WS which must stay off the JWT path.

Concrete edits (`trading/main.py`):
```python
from fastapi import Depends
from trading.api.deps import require_auth

_GUARD = [Depends(require_auth)]
app.include_router(auth.router,        prefix=_API)              # PUBLIC (login)
app.include_router(keys.router,        prefix=_API, dependencies=_GUARD)
app.include_router(backtest.router,    prefix=_API, dependencies=_GUARD)   # was public
app.include_router(strategies.router,  prefix=_API, dependencies=_GUARD)   # was public
app.include_router(portfolio.router,   prefix=_API, dependencies=_GUARD)   # was public
app.include_router(data.router,        prefix=_API, dependencies=_GUARD)   # was public
app.include_router(export.router,      prefix=_API, dependencies=_GUARD)
app.include_router(presets.router,     prefix=_API, dependencies=_GUARD)   # was public
app.include_router(signal_keys.router, prefix=_API, dependencies=_GUARD)
app.include_router(baskets.router,     prefix=_API, dependencies=_GUARD)
app.include_router(signals.router,     prefix=_API, dependencies=_GUARD)
app.include_router(ws_router)                       # WS handlers auth themselves (§1.6)
app.include_router(local_client_router)             # /ws/client auths itself; GET already guarded in-file
app.include_router(dashboard.router)                # /API_KEY/*  key-gated (NO JWT)
```
Notes for the Engineer:
- **Router-level deps already in-file** (`keys.py:22`, `signal_keys.py:33`, `signals.py:37`, `baskets.py:33`, `export.py:24`) can stay — the dependency is cached, so a double-declaration is a no-op. Keeping them makes the `test_authz_regression` sweep self-documenting.
- **`/backtest/cancel/{token}` must be exempted** from the mount guard (§1.6) — either split it into a tiny unguarded router, or move the guard from the mount to per-route for `backtest`. Recommendation: keep the mount guard on `backtest` and add the single public cancel route to a **separate 2-line router** included without `_GUARD`.
- The engineer’s suggested alternative (a global `Depends(require_auth)` on the whole `/api/v1` mount) is **rejected**: it would also guard `/api/v1/auth/token` and `/api/v1/backtest/cancel/{token}` incorrectly.

### 1.3 Failure semantics (must match existing tests)

`require_auth` (`deps.py:16-25`) already returns **401** with `{"detail":"Not authenticated"}` (no bearer) or `{"detail":"Invalid or expired token"}` (bad/expired) — keep this exactly; `test_authz_regression.py:65-96` asserts both. Because the mount guard runs as a dependency, auth is evaluated **before** body validation, so an anonymous *malformed* request is 401, never 422 (locked in by `test_authz_regression.py:78-81`). No `403` path exists today; do not introduce one unless `require_admin` (§1.7) is added — then use 403 for “valid token, insufficient role” and keep 401 for “no/invalid token”.

### 1.4 Frontend token attachment — enabling auth **must not** break the SPA

The SPA (`trading/static/index.html`, 3961 lines) **already implements bearer auth**, so no SPA change is required for the JWT-guarded routes:
- token stored: `S.token = localStorage.getItem("gex.token")` → `index.html:1250`;
- injected on every API call: `headers()` → `if (S.token) h.Authorization = "Bearer " + S.token` → `index.html:1265-1268`;
- **all** API calls funnel through `json()` (`:1276`), `blobGet()` (`:1317`), `textPost()` (`:1323`), each of which calls `headers()`. Confirmed: the only raw `fetch()` calls are inside these helpers plus `fetch("/health")` (`:3924`, public) — nothing hits a business route without the helper;
- login already POSTs `/auth/token` → `index.html:3789`; logout clears the token → `:3798`.

Three concrete cross-cutting traps the Engineer MUST handle:

1. **`navigator.sendBeacon(API + "/backtest/cancel/" + token)`** (`index.html:3915`) **cannot set an `Authorization` header.** → `/backtest/cancel/{token}` must remain reachable without a bearer (capability token). See §1.6.
2. **Demo mode** sets `S.token = "demo"` (`index.html:3797`) and short-circuits `json()` to `mock()` (`:1277`) — so demo never reaches the server. No change needed, **but** the Engineer must NOT add a startup `GET` probe that would 401 the demo token (there is none today; `bootData()` only runs when `S.token` truthy and non-demo).
3. **Unauthenticated load path**: `index.html:3956` runs `showApp()+bootData()` only `if (S.token)`; otherwise the login view shows. Behaviour is already correct for a guarded backend.

Optional hardening (recommended, small): in `json()`/`blobGet()`/`textPost()`, on `status === 401` clear `S.token`+`localStorage` and route to login instead of throwing a raw error — improves UX when a token expires (TTL 480 min today).

### 1.5 Routes intentionally public + the `/metrics` & `/docs` decision

**Stay public (justified):**
- `POST /api/v1/auth/token` (`auth.py:16`) — the login endpoint itself.
- `GET /health` (`main.py:99`) — orchestrator liveness; payload is a constant `{"status":"ok"}` with no internals.
- `GET /`, `/favicon.ico`, `/static/*` (`main.py:109-121`) — the SPA shell must load before a token exists.
- `POST /api/v1/backtest/cancel/{token}` (`backtest.py:960`) — capability-token route, required by `sendBeacon` (§1.6).

**Gate them:**
- `GET /metrics` (`main.py:104`): **gate it.** It exposes internal operational data (queue/latency/broker state) and prometheus scrapes it server-to-server, so it does **not** need to be browser-reachable. Recommended: keep the route but require a static token via env `TRADING_METRICS_TOKEN` (sent as `Authorization: Bearer <token>` or `?token=` for the scrape job); update `docker/prometheus.yml` to send it (`authorization:` / `params`). Network-level restriction (bind to internal network only) is equally acceptable; pick one and apply it.
- `GET /docs`, `/redoc`, `/openapi.json`: **disable in production.** Do **not** try to JWT-guard Swagger UI (breaks the UI's own fetch). Recommended: construct `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` unless `TRADING_ENABLE_DOCS=1` (`main.py:66-71`). Keep them on in dev/CI.

### 1.6 WebSocket auth design

**Browser streams** `/ws/signals`, `/ws/orders`, `/ws/positions` (`websockets.py:49-61`):
- **Mechanism: JWT via the `Sec-WebSocket-Protocol` subprotocol** — the client opens `new WebSocket(url, ["gex.jwt", "<JWT>"])`; the server reads `ws.headers["sec-websocket-protocol"]`, validates the JWT **before** `ws.accept()`, and on success echoes the chosen subprotocol back in `accept(subprotocol="gex.jwt")`. **Fallback:** accept `?token=<JWT>` as a query param.
- **Justification:** the browser WebSocket API cannot set an `Authorization` header; of the two workable options the **subprotocol is preferred over a query param** because query strings are written to access logs / proxies / Referer, whereas the subprotocol header is not logged by uvicorn by default. Reuse `decode_access_token` (`security.py:57-63`) so failure semantics match `require_auth`.
- **Frontend change (tiny):** `sigWsUrl()` at `index.html:2821` returns a bare URL and `new WebSocket(...)` at `:2883`; change to `new WebSocket(sigWsUrl(), ["gex.jwt", S.token])`. On close `4401`/`1008`, clear the token and return to login.
- On auth failure: `await ws.close(code=1008)` (policy violation) **without** `accept()`.

**Machine producer** `/ws/client` (`local_client_ws.py:118`):
- **Mechanism: token inside the existing `handshake` frame.** Add a required `token` field to the handshake (`_handle_message`, `local_client_ws.py:60-69`); validate it **before** registering the session / before the socket may emit `signal` frames. Invalid → send an `{"type":"error","code":"UNAUTHORIZED"}` frame and `close(1008)`; do **not** register via `registry.connect()`/`handshake`.
- **Justification:** it is a **non-browser** client — arbitrary bots — so subprotocol/query plumbing is awkward; the protocol already has a first-frame `handshake` (the tests at `test_local_client_ws.py:26-37` prove ordering is enforced). A dedicated static token via env `TRADING_LOCAL_CLIENT_TOKEN` is simplest; accepting a valid JWT as well is a low-cost bonus.
- **Test impact:** `test_local_client_ws.py` and `test_ws.py` must pass the token (add a fixture that mints one).

### 1.7 `require_admin` — define the tier, but do **not** block on it

Today there is exactly one credential (`settings.admin_username`), so **admin-amplified collapses to valid-jwt**. The tier is listed so that the most destructive operations are *labelled and centralised* for a future multi-user split. Recommended now:
- Implement nothing new beyond `require_auth` **unless** you want defence-in-depth: add a `require_admin` dependency (`trading/api/deps.py`) that asserts `decode_access_token(...) == settings.admin_username`, and attach it (route-level) to the admin-amplified set only: `presets/{id}/promote|rollback|demote`, `DELETE /presets/{id}`, `POST /backtest/optimize` (+`/global`), `strategies/{name}/start|stop`, `signals/engine/start|stop`, `keys` create/delete, `signal-keys` create/delete. This keeps reads under plain valid-jwt.

### 1.8 Test updates required by the auth change (blocking for green CI)

- **16 test files** construct the app via `TestClient`/`ASGITransport` (verified). Those that call now-guarded routes anonymously will now get 401. Add a shared fixture in `trading/tests/conftest.py` that logs in and returns `{"Authorization": "Bearer …"}` and apply it as the default header on the client (or extend each test’s client). Known affected: `test_presets.py`, `test_optimize_endpoint.py`, `test_optimize_unified.py`, `test_strategy_hub_api.py`, `test_api_portfolio.py`, `test_api_baskets.py`, `test_qa_acceptance_transfer.py`, `test_signal_keys.py`, `test_signal_engine.py`, `test_autotune.py`, `test_trade_export.py`, `test_ws.py`, `test_local_client_ws.py`.
- **Extend the existing sweep** `test_authz_regression.py:100-130` (`_GUARDED_ANON_REQUESTS`) to include at least: `POST /presets`, `POST /presets/1/promote`, `DELETE /presets/999999`, `GET /presets`, `POST /backtest`, `POST /backtest/optimize/global`, `GET /strategies`, `POST /strategies/sma_crossover/start`, `GET /portfolio`, `GET /positions`, `GET /orders`, `GET /data/ohlcv/BTC`, `GET /data/instruments`, `GET /backtest/cancel`. Each must 401.
- **Explicitly assert the two exceptions stay public:** `POST /backtest/cancel/{token}` (206-safe) and `POST /auth/token` return non-401.

---

## 2. SCHEMA ALIGNMENT  (A3 · A13)

### 2.1 The exact model↔migration drift to fix

Two classes, both surfaced by `alembic check` (run in round 1 → **FAILED**).

**(a) Nullability: `created_at`/`updated_at` are NOT NULL in the models but NULLABLE in the DB.**

| Column | Model (says NOT NULL) | Migration (created NULLABLE) |
|---|---|---|
| `api_keys.created_at` | `models.py:23-25` | `0001_initial.py:28` |
| `orders.created_at` | `models.py:43-45` | `0001_initial.py:46` |
| `backtest_results.created_at` | `models.py:57-59` | `0002_backtest_results.py:26` |
| `strategy_presets.created_at` | `models.py:115-117` | `0004_presets_signal_keys.py:31` |
| `strategy_presets.updated_at` | `models.py:118-120` | `0004_presets_signal_keys.py:32` |
| `signal_keys.created_at` | `models.py:142-144` | `0004_presets_signal_keys.py:45` |
| `key_trades.created_at` | `models.py:271-273` | `0004_presets_signal_keys.py:93` |
| `signal_positions.created_at` | `models.py:238-240` | `0005_signal_positions.py:75` |
| `signal_positions.updated_at` | `models.py:241-243` | `0005_signal_positions.py:76` |

**(b) Type spelling: FLOAT vs Double** — models infer `Double` from `Mapped[float]`; every migration used `sa.Float()`. Affected columns and their migration origin:
- `orders`: `quantity`, `limit_price`, `stop_price`, `filled_quantity` → `0001_initial.py:38,41,42,43` (model `models.py:35,38,39,40`);
- `key_signals`: `strength`, `price` → `0004_presets_signal_keys.py:60,61`; `entry_price`, `stop_loss`, `take_profit`, `position_size`, `risk_pct`, `risk_amount` → `0005_signal_positions.py:36` (loop);
- `key_trades`: `entry_price`, `exit_price`, `quantity`, `fee`, `gross_pnl`, `net_pnl`, `pct_return`, `holding_seconds` → `0004_presets_signal_keys.py:80-87`;
- `signal_positions`: all float columns (`entry_price`, `quantity`, `mfe_r`, `stop_price`, `take_profit`, `trail_price`, `best_price`, `worst_price`, `exit_price`, `risk_amount`, `risk_pct`, `gross_pnl`, `net_pnl`, `pnl_r`, `pct_return`, `unrealised_pnl`, `mark_price`) → `0005_signal_positions.py:54-74`.

### 2.2 Recommended resolution for the drift

- **(a) Make the DB columns NOT NULL via a new migration 0007** (the intent is a timestamped row; `server_default=func.now()` guarantees every existing row has a value, so the tighten is safe). Do **not** loosen the models — that would hide the inconsistency. Use `batch_alter_table` (required on SQLite; on Postgres it is a plain `ALTER`);
  ```
  alembic/versions/0007_created_at_not_null.py  (down_revision="0006")
  upgrade():   for each (table, cols) in [(api_keys, [created_at]), (orders, [created_at]),
                                         (backtest_results, [created_at]),
                                         (strategy_presets, [created_at, updated_at]),
                                         (signal_keys, [created_at]),
                                         (key_trades, [created_at]),
                                         (signal_positions, [created_at, updated_at])]:
                   with op.batch_alter_table(table) as b:
                       for c in cols: b.alter_column(c, existing_type=sa.DateTime(timezone=True),
                                                     nullable=False, existing_server_default=sa.func.now())
  downgrade(): mirror, nullable=True (batch)
  ```
  **SQLite caveat (the skill’s warning):** if the batch fails it leaves a `_alembic_tmp_<table>` behind — the Engineer must drop any `_alembic_tmp_*` before retrying. After the migration, verify with `PRAGMA table_info(<t>)` that `notnull=1`, and re-run `alembic check` (must be clean).
- **(b) Stop comparing type *spelling*** — set, in `alembic/env.py`, `context.configure(..., compare_type=False)` in both `run_migrations_offline` (`env.py:28-33`) and `run_migrations_online` (`env.py:41`), with a comment: *“FLOAT and Double are the same storage on our SQLite/Postgres targets; we compare structure + nullability, not type spelling.”* Also add **`render_as_batch=True`** to both configure calls (fixes A18) so future `--autogenerate` emits batch ops on SQLite. *Alternative* (only if strict type checking is desired): annotate the ~40 columns with explicit `Float` in `models.py` — more edits, same result.

### 2.3 Startup strategy — replace `init_db()`/`create_all` with `alembic upgrade head`

Add a migration runner to `trading/adapters/persistence/database.py` and call it from the lifespan:
```python
# database.py — new
def _sync_url() -> str:                     # mirror alembic/env.py:18-23
    url = settings.database_url
    return (url.replace("sqlite+aiosqlite:///", "sqlite:///")
               .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
               .replace("postgresql://", "postgresql+psycopg2://"))

async def run_migrations() -> None:         # off-loaded; alembic is sync
    def _upgrade() -> None:
        from alembic import command
        from alembic.config import Config
        cfg = Config("alembic.ini")         # script_location=alembic, prepend_sys_path=.
        cfg.set_main_option("sqlalchemy.url", _sync_url())
        command.upgrade(cfg, "head")
    await asyncio.to_thread(_upgrade)
```
- **`trading/main.py:52`**: replace `await init_db()` → `await run_migrations()`.
- **Remove the request-time `init_db()` calls** (each currently runs `create_all` per request): `trading/api/routers/backtest.py:425` (`_persist_result`), `:451` (`_load_preset`), `:812` (`_save_optimized_preset`). The app is already migrated at boot; delete these lines.
- **Keep `init_db()`(create_all) but mark it TEST-ONLY.** ~20 test files call `await db.init_db()` to build a fresh schema fast (e.g. `test_api.py:15`, `test_authz_regression.py:36`); keep the function unchanged for them and add a docstring “**test/bootstrap only — production uses `run_migrations()`**”. Do **not** call it from the app path.
- **Compose (multi-replica safety):** run migrations once before the app starts — change the `app` service `command` to `sh -c "alembic upgrade head && uvicorn trading.main:app --host 0.0.0.0 --port 8000"`, so N replicas don’t race the upgrade (the app-level `run_migrations()` then finds the DB already at head — idempotent).
- **Fresh DB:** `alembic upgrade head` on an empty DB builds the full schema (proven in round 1), removing the previous “app-created DB can never be migrated” trap.

### 2.4 CI migration step (A13)

In `.github/workflows/ci.yml`, add between **Install** (`:16-19`) and **Test** (`:22-23`):
```yaml
      - name: Migrations (forward/backward + drift gate)
        env: { TRADING_DATABASE_URL: "sqlite+aiosqlite:///./ci.db" }
        run: |
          alembic upgrade head
          alembic downgrade base
          alembic upgrade head
          alembic check          # FAILS the build if models drift from migrations
```
Also add a `docker build .` smoke job (A11) and keep `python-version: "3.12"` (parity with the image; 3.14 is dev-only).

---

## 3. RATE LIMITING  (A4)

**Recommendation: fix the in-memory limiter correctly AND correct the docs; keep Redis as an opt-in backend (do not make it mandatory).** Rationale: the app ships single-worker (`docker-compose.yml:48` — one uvicorn, no `--workers`), the `RedisTokenBucket` API is sync (`token_bucket.py:111-118`) and cannot be awaited from the async middleware as-is, and making Redis mandatory would make the API fail closed when Redis is down. Do the in-memory fixes now; wire Redis behind a flag as a follow-up if multi-worker is ever deployed.

Concrete changes in `trading/api/middleware.py` (currently `:15-38`):
1. **Dedicated login bucket.** When `request.url.path == "/api/v1/auth/token"` (POST), use a much tighter bucket, e.g. capacity `TRADING_LOGIN_RATE_LIMIT` (default **10/min**) keyed `f"login:{client}"`, plus a **failed-attempt counter + lockout** (e.g. 5 failures → 15-min lock) — currently login shares the 300/min global bucket (`config.py:39`).
2. **Bound `_buckets` growth (A4 leak).** Cap the dict (e.g. 50k entries) and evict least-recently-used on insert beyond the cap; or run a periodic prune of buckets idle > N minutes. Today `_buckets` (`middleware.py:20`) grows forever.
3. **Trusted-proxy handling.** Only read `X-Forwarded-For` when `TRADING_TRUST_PROXY=1` (new setting); otherwise keep `request.client.host` (`middleware.py:30`). Fixes “all clients behind the proxy share one bucket”.
4. **WebSocket connection cap.** Add a per-IP **concurrent** connection counter in the WS accept path (`websockets.py:_run_stream:28`, `local_client_ws.py:ws_client:118`) — `BaseHTTPMiddleware` never sees the `websocket` scope, so the HTTP limiter cannot cover WS.
5. **Redis opt-in (optional).** Add `TRADING_RATE_LIMIT_BACKEND=local|redis`. For `redis`, add an **async** variant of `RedisTokenBucket` (redis.asyncio `eval`) and select it in `RateLimitMiddleware.__init__`, falling back to local on connect failure.
6. **Docs correction (A21):** change `README.md:12,324` and `ARCHITECTURE.md:27,93` from “Redis+Lua, already distributed” to “in-process token bucket per worker; set `TRADING_RATE_LIMIT_BACKEND=redis` for a Redis+Lua shared bucket”.

Keep the existing `Retry-After` behaviour (`middleware.py:36`) — `test_authz_regression.py:157-167` already guards it.

---

## 4. METRICS / ALERTING  (A6)

**Recommendation: wire up (the instrumentation is cheap) + repair the Prometheus/compose wiring. Do not strip the claim.** Rationale: `broker_health.py` already sets `BROKER_STATUS`; the counters are already declared with sensible labels (`observability.py:20-38`); the only missing pieces are call sites and two compose/prometheus lines. Stripping would remove genuinely useful operational signal.

Concrete changes:
1. **`docker/prometheus.yml`** (currently `:1-9`, no `rule_files`/`alerting`): add
   ```yaml
   rule_files: ["/etc/prometheus/rules/*.yml"]
   alerting:
     alertmanagers:
       - static_configs: [{ targets: ["alertmanager:9093"] }]
   ```
2. **`docker-compose.yml`**: mount the rules file into `prometheus` (`./docker/prometheus-rules.yml:/etc/prometheus/rules/prometheus-rules.yml:ro`) and add an `alertmanager` service (`prom/alerts` image, mount `./docker/alertmanager.yml`, port 9093).
3. **Instrument the five idle counters** (all currently only touched in tests):
   - `ORDERS_TOTAL.inc` after a successful `broker.place_order` (`portfolio.py:142-145`, and in `application/execution.py` if used);
   - `STRATEGY_SIGNALS.inc` in `signal_engine._handle_signal` (`signal_engine.py:459-502`) and in the local-client signal path (`local_client_ws.py:85`);
   - `DATA_FETCH_ERRORS.inc` in `fetchers/registry.py` fallback loop (`:62`) and `http_util.get_json` non-200 (`:22`);
   - `RATE_LIMIT_HITS.inc` in `middleware.py` when rejecting (`:32`);
   - `BACKTEST_DURATION.observe` around the `run_backtest` calls (`backtest.py:525,561`).
4. **`BROKER_STATUS`**: `application/broker_health.py` is never called — add a Celery-beat entry (in `tasks.py:beat_schedule:65-78`) that runs `broker_health.check(...)` per configured exchange, so `TradingBrokerDown` can actually fire.

---

## 5. TIMEZONE  (A5)

**Mechanism: one ORM-level `TypeDecorator` that normalises storage/read to aware UTC.** Round-1 proved SQLite returns **naive** datetimes for `DateTime(timezone=True)`, while Postgres (asyncpg, `timestamptz`) returns **aware** ones → any `aware_now - row.dt` raises on SQLite (`signal_keys.py:138,511,530` are the concrete sites). A single decorator fixes every column at once and is a **no-op on Postgres**.

```python
# trading/adapters/persistence/types.py  (new)
from datetime import datetime, timezone
from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator

class UTCDateTime(TypeDecorator):
    """Aware-UTC on read on every backend; stores UTC; no DDL change."""
    impl = DateTime(timezone=True)
    cache_ok = True
    def process_bind_param(self, value, dialect):
        if value is None: return None
        if value.tzinfo is None: value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    def process_result_value(self, value, dialect):
        if value is None: return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
```
Engineer steps:
1. Replace every `mapped_column(DateTime(timezone=True), ...)` in `models.py` (e.g. `:23,43,57,115,118,142,166,184,212,225,238,241,255,271`) with `mapped_column(UTCDateTime(), ...)` — the `impl` is identical DDL, so **no migration is needed** and `alembic check` stays clean.
2. Add a regression test asserting `row.entry_time.tzinfo is not None` after a fresh read on SQLite (mirror the round-1 proof).
3. Audit the few tz-sensitive computations (`signal_keys.py:138` holding-seconds, any “stale” checks) — they become correct automatically once reads are aware.

Alternative (smaller blast radius, weaker): a free helper `as_utc(dt)` applied at each comparison site — rejected because it is easy to miss a site.

---

## 6. Cross-cutting risks, ordering & verification

**Recommended landing order** (each step independently testable):
1. **A5 timezone decorator** (no DDL, no API surface change — safest first).
2. **Schema: migration 0007 + `env.py` compare/render flags + `run_migrations()`** and remove request-time `init_db()`; land the CI migration step.
3. **A1 auth** at the `include_router` mount + WS auth; update the ~16 test fixtures + extend the `test_authz_regression` sweep; exempt `POST /backtest/cancel/{token}`.
4. **A4 limiter** + doc correction; **A6 metrics/alerts** wiring.

**Highest regression risks to watch:**
- **Auth vs SPA/`sendBeacon`** (§1.4/§1.6) — the one thing that can silently break the console and cancel-on-unload.
- **Migration 0007 on SQLite** — batch + `_alembic_tmp_*` leftovers (skill warning). Verify `PRAGMA table_info` + `alembic check` after.
- **Test suite** — 16 HTTP tests + 2 WS tests must send credentials; the coverage gate (`--cov-fail-under=85`, `ci.yml:23`) will drop if tests are merely disabled instead of updated.

**Verification checklist (hand to QA):**
- `alembic upgrade head` → `alembic check` PASS on empty DB; `downgrade base` → `upgrade head` clean; no `_alembic_tmp_*`.
- Anonymous sweep (extended `test_authz_regression`) → every business route 401; `POST /auth/token` and `POST /backtest/cancel/{token}` unaffected.
- SPA end-to-end in a browser: login → every tab loads with a token; logout → login view; **no** 401 in the network tab except the intended public calls; WS streams authenticate.
- Rate limit: 429 with `Retry-After`; login bucket locks out after the configured failures; WS connection cap enforced.
- Metrics: `/metrics` non-zero for orders/signals/fetch-errors/rate-limit after traffic; `curl :9090/api/v1/rules` shows the three groups; Alertmanager reachable.
- TZ: SQLite round-trip returns aware datetimes; no `TypeError` in holding-seconds paths.

**Unresolved / assumptions:** single-admin identity (§1.7) and single-worker deploy (§3) are assumed; if multi-user or multi-worker is intended, escalate before implementing §1.7 and the Redis backend.
