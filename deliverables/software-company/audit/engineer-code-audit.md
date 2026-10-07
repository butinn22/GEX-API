# Engineer — Code / API-Contract / Validation Audit (Round 1, READ-ONLY)

**Scope:** `E:\gex api\trading` — FastAPI routers (`trading/api/routers/`), schemas, deps, security, persistence models, signal engine/service, fetchers, and the single-file console `trading/static/index.html`.
**Method:** static read of every route, schema, service and the console; JS syntax-checked with `node --check` (Node v22.22.2) on both `<script>` blocks — **both parse clean (exit 0)**, no `True/False/None` literal leak.
**Not covered here (other teammates):** dynamic endpoint behaviour, DB/migration, infra, docs.

---

## 1. Executive summary

The API is well-structured (typed schemas, parameterised SQLAlchemy, structured validation errors on baskets/presets), but it has a **systemic authentication gap**: authentication is applied **per-router**, and six routers were left *off* the guard. As a result, **strategy go-live, preset mutation, global optimization, live-strategy start/stop, all backtests, market-data reads, and both WebSocket surfaces are reachable with no credentials**. Broker account/position/order reads (`/portfolio`, `/positions`, `/orders`) are also anonymous. Combined with a **stored-XSS sink in the anonymous `/API_KEY/{key}` dashboard** and **unauthenticated signal ingestion via `/ws/client`**, this is the dominant risk cluster (4 critical findings).

Secondary issues: **no idempotency on order placement** (real-money duplicate-order risk), **unvalidated `poll_seconds` → event-loop busy-loop DoS**, **frontend destructive actions without confirmation**, and assorted consistency/dead-code items. No Python-literal JS-injection defect was found in the console.

**Severity counts:** Blocker 0 · Critical 4 · High 4 · Medium 8 · Low 6 · Info 2 (**24 findings**).

---

## 2. Findings

### Critical

| ID | Sev | Category | file:line | Evidence | Impact | Recommended fix |
|----|-----|----------|-----------|----------|--------|-----------------|
| ENG-01 | critical | authz | `presets.py:43` (router), `:114,:224,:241,:263,:299`; `backtest.py:86,:718,:841`; `strategies.py:10,:45,:66` | `router = APIRouter(prefix="/presets", tags=["presets"])` — **no `dependencies=[Depends(require_auth)]`**; same for `/backtest` and `/strategies` | Anyone (no token) can `POST /presets/{id}/promote` (flips a strategy **live**), `rollback`/`demote`/`delete` presets, `POST /backtest/optimize?save_preset=true` and `POST /backtest/optimize/global` (writes preset store), and start/stop live strategy runners. Full control of the live-signal configuration. | Add `dependencies=[Depends(require_auth)]` to the `backtest`, `presets`, and `strategies` routers (or a global `Depends(require_auth)` on the `/api/v1` mount), then keep only truly public routes (auth/health) exempt. |
| ENG-02 | critical | authz / PII | `portfolio.py:33` (router), `:60,:85,:104` | `router = APIRouter(tags=["trading"])` with no dependency; `GET /portfolio`→cash, `GET /positions`→live broker positions, `GET /orders`→order history | Anonymous disclosure of trading account state (balances, open positions, orders) for any configured broker. | Guard the router with `require_auth` (the `POST /orders` route already carries it at `:119`). |
| ENG-03 | critical | authz | `local_client_ws.py:118` (`/ws/client`), `websockets.py:49` (`/ws/signals`) | `ws_client` is `await ws.accept()` with **no auth**; a client can `{"type":"signal",...}` which is `signal_hub.publish(...)`ed and answered with `native_payloads` (BingX/TBank order payloads). `/ws/signals` streams live trade plans anonymously. | Unauthenticated actors can inject signals into the platform's stream and read all live signals/plans. | Require a token on the WS handshake (query param or first frame) before `accept()`, or bind the local-client socket to an authenticated session; guard `/ws/signals` similarly. |
| ENG-04 | critical | XSS | `dashboard.py:335-338` (and `:135`) | `_page()` f-string: `tickers: {tickers or "—"}`, `<span class="badge">strategy: {strategy}</span>` where `tickers`/`strategy` come from `config` (symbols are attacker-supplied via basket/backtest, never escaped) | **Stored XSS** on the anonymous dashboard page: a key created with a symbol like `<img src=x onerror=…>` executes in any visitor's browser. | HTML-escape every interpolated value (`html.escape`) in `_page()`, or render via `textContent`/`json.dumps` into a data island. |

### High

| ID | Sev | Category | file:line | Evidence | Impact | Recommended fix |
|----|-----|----------|-----------|----------|--------|-----------------|
| ENG-05 | high | idempotency | `portfolio.py:117-155` | `POST /orders` (`status_code=202`) has no idempotency key; each call → `broker.place_order` + new `OrderRow` | Network retry / double-click places **duplicate live orders**. | Accept an `Idempotency-Key` header (or client order id); dedupe before `place_order`; return the prior order on replay. |
| ENG-06 | high | UX / destructive | `index.html:2724-2727` (account delete), `:3765` (key revoke) | `b.onclick = async () => { await json("/keys/"+id,{method:"DELETE"}); … }` — **no `confirm()`**; preset delete at `:3390` *does* confirm | One mis-click irreversibly deletes encrypted broker credentials / revokes a live key. | Add `confirm()` (and a typed-name check for credential deletion) before destructive DELETE/revoke calls. |
| ENG-07 | high | input-validation / DoS | `signals.py:50-51`; `signal_engine.py:139-140` | `async def engine_start(payload: dict[str, Any])` (untyped body) → `poll_seconds=float(raw.pop("poll_seconds",60.0))`, `bars=int(raw.pop("bars",500))` — **no range/positivity checks** | `poll_seconds: 0` (or negative) yields `asyncio.sleep(0)` → tight async busy-loop; `bars` unbounded → memory. No OpenAPI schema for this body. | Define a `SignalEngineStartRequest` pydantic model (`poll_seconds: float = Field(ge=5, le=86400)`, `bars: int = Field(ge=50, le=100000)`, `timeframe: Literal[...]`, `symbols: list[str]`) and validate `len(symbols) <= MAX_TICKERS`. |
| ENG-08 | high | validation | `keys_service.py:55`; `index.html:2739-2748` | `encrypt(self._secret, api_secret or "")` accepts a blank secret; frontend `addKey()` only checks `if (!payload.api_key)`; `account_id` optional | A BingX account can be saved with an empty secret (non-functional); TBANK saved without `account_id` silently routes to `""` (`portfolio.py:49`). | Require `api_secret` for `bingx` and `account_id` for `tbank` server-side (return 400) and enforce in `addKey()`. |

### Medium

| ID | Sev | Category | file:line | Evidence | Impact | Recommended fix |
|----|-----|----------|-----------|----------|--------|-----------------|
| ENG-09 | medium | API-contract | `baskets.py:37-41`, `presets.py:107-111`, `backtest.py:105-109` vs `keys.py:*`, `signals.py:*` | Error bodies are inconsistent: `{"detail": "<str>"}` (most) vs `{"detail": {"code":"validation_failed","errors":[…]}}` (baskets) / `{"code":…,"reasons":…}` (presets) vs `{"message":…,"run_token":…}` (cancel). | Clients cannot parse errors uniformly (the console does `b.detail || statusText`). | Standardise on one error envelope (`{code, message, details}`) via a shared handler. |
| ENG-10 | medium | destructive scope | `signals.py:114-117`; `signal_engine.py:668-675` | `DELETE /signals/positions` runs `delete(SignalPositionRow)` with only an optional `only_open` filter — clears the **whole global ledger**; no key/symbol scope, no confirmation, no idempotency token. | A single call wipes the entire position-ledger history for all keys. | Scope the delete (by `key_id`/`symbol`), require an explicit `confirm=true`, and audit-log it. |
| ENG-11 | medium | pagination | `portfolio.py:104`, `keys.py:60`, `presets.py:137/:149`, `signal_keys.py:92`; `export.py:74,79`; `signals.py:121-164` | `GET /orders`, `/keys`, `/presets`, `/signal-keys` return the full table; `live-trades` export scans all orders; signal exports allow `limit<=100000`. | Unbounded result sets → latency/memory growth as tables grow. | Add `limit/offset` (with caps) or cursor pagination; cap export windows. |
| ENG-12 | medium | unsafe deserialisation | `export.py:50`; `presets.py:51,55`; `signal_keys.py:43,398` | `json.loads(row.trades_json or "[]")`, `json.loads(row.params_json...)`, `json.loads(row.config_json...)` then consumed/echoed | Corrupted/malicious stored JSON is trusted blindly; `TradeEvent.from_dict` invalid data could raise at request time. | Validate parsed JSON against a schema (pydantic) before use; keep the current try/except fallbacks. |
| ENG-13 | medium | secrets / auth | `config.py:22-24`, `security.py:47-54` | Defaults `secret_key="dev-secret-change-me"`, `admin_password="admin"`, `admin_username="admin"`; JWT TTL 480 min; no login lockout/throttle beyond 300 rpm global. | If env not set in prod, forgeable JWTs, guessable admin creds, long-lived tokens. | Fail-fast in prod when defaults are detected; add login throttling; shorten TTL. |
| ENG-14 | medium | dead code / broken option | `data.py:13-19,:55`; `registry.py:92-101` | `_SOURCES` advertises `"webull"`, but `default_registry()` registers only MOEX/yFinance/Bybit — `webull` is never registered. | `GET /data/ohlcv/{sym}?source=webull` always 502; `/data/sources` advertises a dead source. | Register `WebullFetcher` or drop `webull` from `_SOURCES`. |
| ENG-15 | medium | SSRF / path-injection | `yfinance.py:106`; `data.py:28-44` (unauthenticated) | `get_json(self._client, f"/{symbol}", params=params)` — `symbol` is inserted into the request path verbatim via the anonymous `/data/ohlcv/{symbol}`. | Host is fixed, but arbitrary paths on the yahoo host (and any future base-url change) are reachable; input is unauthenticated. | Validate/normalise `symbol` against a charset allowlist (`^[A-Za-z0-9._-]+$`) before building URLs; guard `/data`. |
| ENG-16 | medium | consistency | `index.html:3349-3361`, `:3709-3712`, `:3554` vs `keys.py:39` | `pine-save`, `dp-create-key`, `pine-save-best` do **not** disable the button during the request → duplicate-submit creates duplicate presets/keys. | Duplicate presets/signal keys on double-click. | Disable the button + re-enable in `finally` (pattern already used by `beginRun`). |
| ENG-17 | medium | concurrency | `signal_keys.py:99-101,546` | `_summary_cache: dict[int, KeySummary]` is a process-global dict, never evicted, inconsistent across workers. | Unbounded memory growth; dashboard shows stale/blank summaries under multi-worker. | Bound/evict (LRU) or move to Redis; document single-worker assumption. |

### Low / Info

| ID | Sev | Category | file:line | Evidence | Impact | Recommended fix |
|----|-----|----------|-----------|----------|--------|-----------------|
| ENG-18 | low | CSV injection | `trade_export.py:50-56,116-123` | Cells written verbatim; no guard for leading `= + - @`. | Exported TRADE_CSV opened in Excel could execute a formula if a symbol/reason is crafted. | Prefix cells starting with `=+-@` with `'` (or quote+prefix). |
| ENG-19 | low | dead code | `dashboard.py:91-97` | `_key_guard(row)` is redundant — `_load_key` already guards, and the comment says so. | Minor confusion. | Remove `_key_guard` and its call sites. |
| ENG-20 | low | API-consistency | `signal_keys.py:104-117` | `PATCH /signal-keys/{key_id}` takes `active` as a **query param** with default `True` — a bare PATCH silently re-enables. | Surprising semantics; inconsistent with body-based PATCH elsewhere. | Move to a body model with required `active`. |
| ENG-21 | low | hardening | `backtest.py:975-978` | `GET /backtest/cancel` (unauthenticated) lists active run tokens. | Leaks run tokens (also enabling cancel). | Fold under `require_auth` (covered by ENG-01). |
| ENG-22 | low | info-leak | `dashboard.py:157` | `dashboard_data` returns `"key": row.key` (the secret) in JSON. | The URL already holds it, but it invites logging/referrer leakage. | Return a masked key in the payload. |
| ENG-23 | info | validation | `signals.py:79-92` | `GET /signals` accepts `state`/`strategy` as free strings (no enum/pattern, unlike `side`/`status`). | Typos silently return empty. | Constrain `state` with a `Literal`/pattern. |
| ENG-24 | info | idempotency | `signal_keys.py:59`, `baskets.py:62` | `POST /signal-keys` and `POST /baskets/deploy` create a new key each call (no dedupe). | Duplicate keys on retry. | Optional idempotency key. |

---

## 3. Route-contract appendix

Legend: **Auth** = `yes` (router/route guarded by `require_auth`), `no`, `key` (URL key is the credential). All routes are under `/api/v1` except the dashboard (`/API_KEY/...`), `/ws/*`, `/health`, `/metrics`.

| Method | Path | Auth | Request | Response | Notes |
|--------|------|------|---------|----------|-------|
| POST | `/auth/token` | no | `LoginRequest` | `TokenResponse` | login |
| GET/POST | `/keys`, `/keys/{id}` (DEL), `/keys/{id}/settings` (PATCH), `/keys/routing` | yes | `ApiKeyCreate`/`ApiKeySettingsUpdate` | `ApiKeyOut` | `DELETE` 204/404 ok; routing `symbol` required |
| POST | `/backtest` | **no** | `BacktestRequest` | `BacktestResponse` | persists `BacktestResultRow` |
| POST | `/backtest/monte-carlo` | **no** | `MonteCarloRequest` | `MonteCarloSummaryOut` | |
| POST | `/backtest/portfolio` | **no** | `PortfolioBacktestRequest` | `PortfolioBacktestResponse` | |
| POST | `/backtest/portfolio/monte-carlo` | **no** | `PortfolioMonteCarloRequest` | `MonteCarloSummaryOut` | |
| POST | `/backtest/portfolio/report` | **no** | `PortfolioBacktestRequest` | HTML | |
| POST | `/backtest/analyze` | **no** | `BacktestRequest` | `AnalyzeResponse` | |
| POST | `/backtest/optimize` | **no** | `OptimizeRequest` | `OptimizeResponse` | `save_preset` writes store |
| POST | `/backtest/optimize/global` | **no** | `GlobalOptimizeRequest` | `GlobalOptimizeStatus` | writes presets |
| GET | `/backtest/optimize/global/{run_id}` | **no** | — | `GlobalOptimizeStatus` | 404 ok |
| POST | `/backtest/optimize/global/{run_id}/cancel` | **no** | — | `CancelOut` | 404 ok |
| POST | `/backtest/autotune` | **no** | `AutoTuneRequest` | dict | |
| POST | `/backtest/cancel/{token}` | **no** | — | `CancelOut` | 200 even if unknown (by design) |
| GET | `/backtest/cancel` | **no** | — | `list[str]` | leaks run tokens |
| GET | `/strategies` | **no** | — | `list[StrategyInfo]` | |
| GET | `/strategies/{name}/schema` | **no** | — | `StrategyParamSchema` | 200 empty if unknown |
| POST | `/strategies/{name}/start` | **no** | — | dict | starts live runner |
| POST | `/strategies/{name}/stop` | **no** | — | dict | |
| GET | `/portfolio` | **no** | — | dict | broker cash/positions |
| GET | `/positions` | **no** | — | `list[dict]` | broker positions |
| GET | `/orders` | **no** | — | `list[OrderOut]` | order history |
| POST | `/orders` | yes | `OrderCreate` | `OrderOut` (202) | no idempotency |
| GET | `/data/instruments`,`/sources`,`/categories` | **no** | — | list | |
| GET | `/data/ohlcv/{symbol}` | **no** | `timeframe,limit,source` | `list[dict]` | 400/502 handled |
| GET | `/data/detect/{symbol}` | **no** | — | dict | |
| GET | `/data/universe` | **no** | `category,n` | dict | 400 handled |
| GET | `/export/backtest/{id}/trades.{csv,xlsx}` | yes | — | file | 404 ok |
| GET | `/export/live-trades.{csv,xlsx}` | yes | — | file | unbounded |
| POST | `/presets` | **no** | `PresetCreate` | `PresetOut` 201 | |
| GET | `/presets`, `/presets/latest`, `/presets/versions`, `/presets/default/{symbol}` | **no** | — | `PresetOut` | |
| POST | `/presets/from-backtest/{symbol}` | **no** | `PresetFromBacktestRequest` | `PresetOut` | |
| GET | `/presets/{id}/validate` | **no** | — | `PresetValidateOut` | 404 ok |
| POST | `/presets/{id}/promote`,`/rollback`,`/demote` | **no** | — | `PresetOut` | 422 gate / 409 / 404; **go-live** |
| PATCH | `/presets/{id}` | **no** | `PresetUpdate` | `PresetOut` | 404 ok |
| DELETE | `/presets/{id}` | **no** | — | 204 | 409 if live / 404 |
| POST | `/signal-keys` | yes | `SignalKeyCreate` | `SignalKeyOut` 201 | |
| GET | `/signal-keys` | yes | — | `list[SignalKeyOut]` | |
| DELETE | `/signal-keys/{id}` | yes | — | 204 | 404 ok (soft revoke) |
| PATCH | `/signal-keys/{id}` | yes | `?active=` | `SignalKeyOut` | 400/404 ok |
| POST | `/signal-keys/{key}/generate` | yes | `?refresh=` | `SignalKeyGenerateReport` | 404/400/502 |
| POST | `/baskets/export`, `/baskets/deploy` | yes | `Basket*Request` | `Basket*Response` | 422 structured |
| POST | `/signals/engine/start`,`/stop` | yes | **dict[Any]** / — | dict | untyped body |
| GET | `/signals/engine`, `/signals`, `/signals/positions`, `/signals/stats` | yes | query filters | dict/list | |
| DELETE | `/signals/positions` | yes | `?only_open=` | dict | global clear |
| GET | `/signals/export/{signals,positions}.{csv,xlsx}` | yes | query | file | |
| GET | `/API_KEY/{key}`, `/data`, `/charts`, `/trades.csv/.xlsx` | **key** | — | HTML/JSON/file | XSS sink; no escaping |
| POST | `/API_KEY/{key}/refresh` | **key** | — | JSON | |
| WS | `/ws/signals`, `/ws/orders`, `/ws/positions` | **no** | — | stream | anonymous |
| WS | `/ws/client` | **no** | — | stream | anonymous signal ingestion |
| GET | `/api/v1/local-clients` | yes | — | dict | |
| GET | `/health`, `/metrics` | no | — | dict/str | |

**Shadowing check:** no active route collisions found. `portfolio.py` (no prefix) exposes `/positions`,`/orders`,`/portfolio` at `/api/v1/*`, while `signals.py` uses `/signals/positions` — distinct paths (the historical `/signals` stub is confirmed removed, `portfolio.py` no longer declares it; see note at `signals.py:167-170`).

## 4. Missing management actions (create → corresponding delete/purge)

| Entity | Created by | Delete / clear / purge | Status |
|--------|-----------|------------------------|--------|
| Backtest result (`backtest_results`) | `POST /backtest` (`_persist_result`) | **none** | ❌ **missing** — rows accumulate with no purge/delete |
| Order (`orders`) | `POST /orders` | **none** (only `GET /orders`) | ❌ **missing** — no cancel/delete per order |
| Broker position / portfolio | broker (read) | **none** | ❌ missing (read-only; may be by design) |
| Position ledger (`signal_positions`) | live engine | `DELETE /signals/positions` (`signal_engine.clear_positions`) | ✅ present (but global/unguarded scope — ENG-10) |
| Per-key signals (`key_signals`) | `POST /signal-keys/{key}/generate` | replaced wholesale on regenerate; **no direct DELETE** | ⚠️ indirect only |
| Preset / version | `POST /presets`, `PATCH /presets/{id}` | `DELETE /presets/{id}` (409 while live) | ✅ present (unauth — ENG-01) |
| Signal key | `POST /signal-keys`, `POST /baskets/deploy` | `DELETE /signal-keys/{id}` (soft revoke) | ✅ present; no hard purge/restore |
| Broker api_key | `POST /keys` | `DELETE /keys/{id}` | ✅ present; no "clear all" |
| Baskets | stateless (`/baskets/export`/`deploy`) | n/a | ✅ |

**Also missing:** no purge for the in-memory `_summary_cache`; no "clear all" for either key type; no delete for `BacktestResultRow` (the console's "Stored backtest run" panel can only download, never remove).

## 5. Frontend findings (`trading/static/index.html`)

- **JS integrity:** both `<script>` blocks pass `node --check` (exit 0). No Python `True/False/None` leak into generated constants — the only `True/False` occurrences (`:3053-3054`) are a `toBool()` helper comparing the **strings** `"True"/"true"`, and `None` at `:1047` is a UI label. ✅ **No literal-injection defect.**
- **Required-field enforcement:** login form OK; `addKey()` (`:2739-2748`) requires only `api_key` — **secret/account_id not required** (ENG-08); `sig-start` requires ≥1 ticker (`:2908`); most numeric inputs have `min/max/step` HTML attributes.
- **Destructive-action confirmation:** account **Delete** (`:2724`) — **no `confirm()`** (ENG-06); signal-key **Revoke** (`:3765`) — no `confirm()`; preset **Delete** (`:3390`) and **rollback** (`:3321`) — confirm ✅.
- **Duplicate-submit protection:** `runBacktest`, signal start/stop, optimization use disable/BeginRun guards ✅; `addKey` (`:2739`), `dpCreateKey` (`:3704`), `pine-save`/`pine-save-best` (`:3345`,`:3554`) — **no guard** (ENG-16).
- **Reset/clear controls:** present — `btn-clear` (clear tickers, `:3845`), `pine-reset` (`:3343`), `ta-close` ✅.
- **Consistency with server rules:** console sends `poll_seconds`, `bars` unvalidated (matches ENG-07); console `sig-preset` offers `"None (strategy defaults)"` → `null` (server handles ✅); `exp-id` download does nothing when empty (silent, minor).

## 6. Security-code findings

- **AuthZ (ENG-01/02/03):** the dominant cluster — router-level auth omissions + anonymous WS.
- **SSRF / path injection (ENG-15):** `symbol` interpolated into the yfinance request path (`yfinance.py:106`); host pinned but path is attacker-controlled on an unauthenticated route. No user-controlled full-URL fetch elsewhere; bybit/moex pass `symbol` as query params (safe). No filesystem path-traversal sinks found (exports are id/int-keyed; dashboard filename uses the validated key).
- **SQL injection:** none found — all queries use SQLAlchemy expressions/parameters (`select(...).where(col==value)`), including `resolve_accounts_for_symbol`.
- **Secrets in logs/errors:** `settings.secret_key`/passwords are never logged; but `dashboard.py:157` echoes the raw key in JSON (ENG-22), and `dashboard.py:287-289` surfaces raw generation exceptions in a 502 body.
- **Unbounded queries / pagination:** ENG-11.
- **Unsafe deserialisation:** ENG-12.
- **Swallowed exceptions:** `presets.py:99-100` (hot-swap, intentionally best-effort, logged ✅), `backtest.py:437-438` (`_persist_result` returns `None` on any error — silent by design but hides DB failures), `backtest.py:824-825` (`_save_optimized_preset` bare `except: pass` — **silent**, hides preset-write failures; consider logging).
- **Rate limiting:** per-IP token bucket 300/min (`middleware.py`), no per-user limit; `/auth/token` not specially throttled (ENG-13).

---

### Round-2 fix priority
1. Guard `backtest`, `presets`, `strategies`, `portfolio`, both WS surfaces (ENG-01/02/03).
2. Escape the dashboard template (ENG-04).
3. Order idempotency + `poll_seconds`/`bars` validation + required broker secret (ENG-05/07/08).
4. Frontend confirmations (ENG-06) and missing delete actions (backtest results, orders).
