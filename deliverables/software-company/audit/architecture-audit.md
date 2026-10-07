# GEX-API — Architecture, Data, Connectors, Infra & Security Audit

- **Round:** 1 (read-only discovery)
- **Auditor:** 高见远 (Architect)
- **Root:** `E:\gex api` (Windows; venv `venv/Scripts/python.exe` = **Python 3.14.7**, `.venv` also present)
- **Toolchain observed:** alembic 1.20.0, SQLAlchemy 2.1.2, SQLite 3.50.4
- **Scope:** DB migrations & access layer, connectors (bingx / tbank / fetchers / redis / celery), rate limiting, security architecture, Docker/compose/CI, docs-vs-code.
- **Product code was NOT modified.** Throwaway SQLite DBs were created under `%LOCALAPPDATA%\Temp\gex_mig` and deleted after the run.

---

## 1. Executive Summary

The `trading/` bounded context is a **well-structured hexagonal design** — frozen-domain dataclasses, explicit ports, adapter isolation, a Redis-backed cooperative-cancellation registry, a correctly bounded signal engine seen-set, and a genuinely secure (256-bit, `secrets.token_urlsafe`) signal-key generator. Several adversarially-written components (BingX HMAC signing, MOEX tz normalisation, the 0005/0006 SQLite batch migrations, the fee-accounting guard) are production-quality.

However, the platform is **not production-ready**. The dominant problems are systemic, not local:

1. **The API has no security perimeter.** Only 5 of ~13 routers require auth. `presets`, `backtest`, `data`, `strategies` and the read half of `portfolio` are **completely anonymous**, and `/metrics`, `/docs`, `/ws/*`, `/ws/client` and the `presets` write surface are open to the internet. An anonymous caller can **promote/rollback/delete the live strategy preset** (which hot-swaps the running live engine) and start/stop the live runner.
2. **Secrets security by default is a footgun.** `dev-secret-change-me` / `admin` / `admin` are the shippable defaults and are the *same* key material used to sign JWTs **and** derive the Fernet key that encrypts every stored broker credential.
3. **Database schema management is split-brained.** The app boots via `Base.metadata.create_all()` while Alembic exists; there is no migration step at startup and no `stamp` for app-created DBs — proven below that `alembic upgrade head` **fails outright** on any DB the app created.
4. **Observability is dead weight.** The alert rules and Alertmanager config are never loaded, and every business metric except `BROKER_STATUS` is never incremented (and `BROKER_STATUS` is never set either) — so **none of the three alerts can ever fire**.
5. **The advertised "distributed Redis+Lua rate limiting" does not exist in the running app.** The middleware is a per-process dict; `RedisTokenBucket` is unreferenced dead code.

**Migration dry-run result: PASS (forward and backward, from empty DB).** `alembic upgrade head` → `0006` cleanly; `downgrade base` → empty; `upgrade head` again → `0006`; the 0005 NULL-row delete + NOT NULL restore and the 0006 version backfill both behave correctly; no `_alembic_tmp_*` leftovers. See §3 for exact commands. Separately, `alembic check` **FAILS** with model↔migration drift (§3.2).

| Severity | Count |
|---|---|
| Blocker | 1 |
| Critical | 2 |
| High | 5 |
| Medium | 6 |
| Low | 8 |
| **Total** | **22** |

---

## 2. Findings Table

Severity legend: **blocker** > **critical** > **high** > **medium** > **low**. Every row cites `file:line`.

| ID | Sev | Area | File:line | Evidence | Impact | Recommended fix | Verification |
|---|---|---|---|---|---|---|---|
| **A1** | **blocker** | Security / AuthZ | `api/routers/presets.py:43`; `backtest.py:86`; `data.py:11`; `strategies.py:10`; `portfolio.py:33,60,85,104`; `dashboard.py:34`; `main.py:94,99,104`; `local_client_ws.py:118` | Routers declared **without** `dependencies=[Depends(require_auth)]`. Only `keys`, `signal_keys`, `signals`, `baskets`, `export` (+1 local-clients route) are guarded. `presets.py` write surface (`POST`, `PATCH`, `DELETE`, `promote/rollback/demote`) and `strategies/{name}/start` are anonymous. | Anonymous internet caller can create/promote/rollback/delete the **live** strategy preset (promote calls `_hot_swap` → `signal_engine.reload_ticker`, flipping the production engine), start/stop live runners, and drive unbounded CPU (backtest/optimize/MC). Full unauthenticated control plane + DoS. | Add `require_auth` to every router (`presets`, `backtest`, `data`, `strategies`, `portfolio`); leave only truly-public liveness. Gate `dashboard.refresh`. | `curl -s -X POST localhost:8000/api/v1/presets/1/promote` returns 200 not 401; audit `grep -rn "APIRouter(" trading/api`. |
| **A2** | **critical** | Security / Secrets | `config.py:22-24`; `docker-compose.yml:34-36`; `security.py:30-33`; `deps.py:29`; `keys_service.py:29` | Defaults `TRADING_SECRET_KEY=dev-secret-change-me`, `TRADING_ADMIN_USERNAME=admin`, `TRADING_ADMIN_PASSWORD=admin`; compose passes `${VAR:-dev-secret-change-me}` etc. `_fernet()` derives the broker-key encryption key from the **same** `secret_key` that signs JWTs. | Two independent crypto purposes share one secret; with defaults, broker API secrets at rest are trivially decryptable and JWTs are forgeable. Compromise of the JWT secret = compromise of all stored broker keys. | Fail-fast at startup if `secret_key` is the dev default / admin password is `admin` in non-dev; split `TRADING_JWT_SECRET` from `TRADING_FERNET_KEY`. | Start app with default secret → must raise a clear config error. |
| **A3** | **critical** | DB / Migrations | `adapters/persistence/database.py:37`; `main.py:52`; `alembic/versions/0001_initial.py:20` | `init_db()` runs `Base.metadata.create_all`; app start never runs `alembic upgrade`. Proven: create schema via `init_db()`, then `alembic upgrade head` → `sqlite3.OperationalError: table api_keys already exists`. | Split-brain schema: a DB the app creates can **never** be migrated (0001 replays DDL); conversely `create_all` never ALTERs existing tables, so any future column added by a migration is missing at runtime on app-created DBs → `no such column` at runtime. | Pick ONE: run `alembic upgrade head` in the lifespan (and drop `create_all`), or `create_all` + `stamp head` on fresh DBs. Add migration step to CI. | On a fresh DB run `alembic current` → should be `0006` after boot. |
| **A4** | **high** | Security / Rate limit | `api/middleware.py:20`; `adapters/ratelimit/token_bucket.py:98`; `README.md:12,324`; `ARCHITECTURE.md:27,93` | Middleware uses an in-process `dict[str, TokenBucket]`; `RedisTokenBucket`/Lua is defined but **never referenced** outside its module. Docs claim "Redis+Lua", "already distributed". | Multi-worker / multi-container deployments enforce the limit **per process** (N× effective limit); limit resets on restart. Login shares the 300/min bucket → ~300 brute-force attempts/min/IP, no lockout. `_buckets` dict grows unbounded (leak). WS not limited (BaseHTTPMiddleware skips `websocket` scope). | Wire `RedisTokenBucket` when Redis is reachable (fallback to local); add a distinct, tighter login bucket + lockout; bound/prune `_buckets`; honour `X-Forwarded-For` behind a proxy. | `grep -rn RedisTokenBucket trading` shows real use; per-IP limit unchanged across 2 workers. |
| **A5** | **high** | Data / Timezone | `adapters/persistence/models.py:23,43,57,115,142,166,184,212,225,238,255`; `application/signal_keys.py:138,511,530` | All timestamps are `DateTime(timezone=True)`, but SQLite returns **naive** datetimes. Proven: insert aware `2026-01-02 03:04:05+00:00`, read back `tzinfo=None`; `datetime.now(timezone.utc) - <read>` → `TypeError: can't subtract offset-naive and offset-aware datetimes`. | Any duration/staleness math over persisted timestamps (holding seconds, “last seen”, TTL) raises on SQLite but works on Postgres (asyncpg `timestamptz` returns aware) → **dev/CI vs prod behavioural divergence**, latent 500s. | Store UTC and normalise on read (SQLAlchemy `TypeDecorator` returning aware UTC), or make all comparisons tz-agnostic. Add a round-trip test. | Test asserts `row.entry_time.tzinfo is not None` on SQLite. |
| **A6** | **high** | Infra / Observability | `docker/prometheus.yml:1-9`; `docker-compose.yml:67-72`; `observability.py:20-38`; `application/broker_health.py`; `docker/alertmanager.yml`; `docker/prometheus-rules.yml` | `prometheus.yml` has **no** `rule_files:` and **no** `alerting:`; Alertmanager is not a compose service. `ORDERS_TOTAL`, `STRATEGY_SIGNALS`, `DATA_FETCH_ERRORS`, `RATE_LIMIT_HITS`, `BACKTEST_DURATION` are defined but never `.inc()`/`.observe()`d outside tests; `broker_health.py` (the only `BROKER_STATUS` writer) is never imported. | Rules `TradingBrokerDown`, `HighDataFetchErrors`, `RateLimitPressure` **can never fire**; orders/signals/signal latency dashboards are empty. Alerting is documented but non-functional. | Mount `prometheus-rules.yml` (`rule_files`), add Alertmanager service + `alerting:` block; instrument the metrics at their call sites (brokers, fetchers, middleware, backtest) and schedule `broker_health`. | `promtool check rules`; `curl localhost:9090/api/v1/rules` shows groups; metrics non-zero after traffic. |
| **A7** | **high** | Security / Exposure | `api/routers/dashboard.py:129,144,200,277`; `main.py:96,104`; `README.md:326` | `/API_KEY/{key}` treats the URL key as the credential; `POST /API_KEY/{key}/refresh` re-runs a full portfolio backtest with no auth and no rate limit; the page auto-refreshes every 60s (`dashboard.py:424`). `/metrics` (internal ops data) and `/docs` are unauthenticated. | Key leaks via Referer/history/proxy logs; unauthenticated refresh is a cheap amplification/CPU-exhaustion vector (each call = full backtest over the window); `/metrics` discloses internals. | Move the key into a header/short-lived signed token; auth+throttle `refresh`; protect `/metrics` (network policy or token); disable `/docs` in prod. | `curl -X POST .../API_KEY/<k>/refresh` (no auth) → 401. |
| **A8** | **medium** | Connectors / Resources | `api/routers/data.py:42`; `tasks.py:178,226,308,328`; `adapters/cache/bar_cache.py:165`; `main.py:56-63` | `data.py` calls `default_registry()` **per request** (new httpx clients, never closed). Celery tasks `asyncio.run(...)` use `loop_registry()` via `load_bars` but never call `aclose_loop_registry()`. The Redis bar-cache client is never disposed on shutdown. | Per-request/per-task file-descriptor & connection leaks; no TLS/connection reuse on `/data/ohlcv`; “Unclosed client” noise; slow leak in long-lived workers. | Use `loop_registry()` in `data.py`; call `aclose_loop_registry()` in a Celery task `finally`; dispose the bar cache in the lifespan. | FD/`httpx` open-connection count stable under load. |
| **A9** | **medium** | DB schema / Integrity | `models.py:157,169,201,252,268`; migrations 0004/0005 | `key_signals.key_id`, `key_trades.key_id`, `signal_positions.key_id`, `*.preset_id` are plain integers with **no `ForeignKey`**, no `ON DELETE`, no unique constraint on “one live_enabled per (symbol,strategy)” (comment admits it is repository-enforced). | Orphan rows when a signal key / preset is deleted; live-deployment invariant can be violated by a racing writer; no DB-level referential safety. | Add FKs (`ondelete="CASCADE"`/`SET NULL` as appropriate) + a partial unique index for the live status. | `PRAGMA foreign_key_list(key_signals)` non-empty; `alembic check` clean. |
| **A10** | **medium** | DB access / API | `adapters/persistence/order_repository.py:38`; `application/signal_keys.py:571`; `key_repository.py:36`; `api/routers/portfolio.py:104`; `dashboard.py:46` | `OrderRepository.list()` (feeds the **unauthenticated** `GET /api/v1/orders`) has no limit/offset; `SignalKeyService.trades()`, `ApiKeyRepository.list()` similarly unbounded; `_summary_cache` / `_generate_locks` grow without eviction. | Unbounded payloads/memory as data grows; DoS via repeated `/orders`; per-key in-memory caches never released. | Add pagination to list endpoints (mirror `signals.py:79-91` which already does limit/offset + bounds). | `GET /orders?limit=…` honoured; memory bounded. |
| **A11** | **medium** | Infra / Docker | `Dockerfile:14-23`; `docker-compose.yml`; `application/instruments.py:19`; `pyproject.toml:41-45` | No `USER` (runs **root**), no `HEALTHCHECK`, no `.dockerignore`. Image copies only `trading/` — the `gex/*.csv` ticker universe is **not copied/packaged**, so `load_instruments()` returns `[]` in the container. `pip install .` ignores the pinned `requirements.txt`. Postgres/Redis ports published with hardcoded creds `gex/gex`; Redis no auth; Grafana default `admin/admin`. | Root container (host escape risk); no orchestration health probe; `/data/*` and `n_tickers` auto-selection broken in the deployed image; unreproducible deps vs the lock file; exposed datastores. | Add non-root `USER`, `HEALTHCHECK`, `.dockerignore`; `COPY gex ./gex` (or package it); install from `requirements.txt`; bind datastore ports to localhost, add Redis auth + Grafana password. | `docker run … id` → non-root; `/data/universe` non-empty in container. |
| **A12** | **medium** | Jobs / Celery | `tasks.py:149,181,233,315,331,339`; `docker-compose.yml:65-78` | `run_backtest_task` has **no** `time_limit` (MC/portfolio do); `fetch_market_data_task` has no retry/time limit; `reconcile_positions_task` and `cleanup_old_backtests_task` are **no-op stubs** but are on the beat schedule and return “success”. | Runaway backtest can pin a worker indefinitely; beat runs two tasks that do nothing → false “reconciliation OK” signal; `TaskResultStore` exists but cleanup is unwired. | Add time limits + retries to every task; implement or unregister the stub beat entries. | `celery -A trading.tasks inspect registered`; beat entries produce real work. |
| **A13** | **medium** | CI / Release gate | `.github/workflows/ci.yml:20-23` | CI runs `ruff check` + `pytest --cov-fail-under=85` only. **No** `alembic upgrade head`, **no** `alembic check`, **no** `docker build`. Python pinned 3.12 vs local 3.14. | The A3/A4 model/migration drift and any broken migration ship green; Docker image never built/validated before merge. | Add `alembic upgrade head` + `alembic check` and a `docker build` job; align the Python version with the lock file. | CI fails on the current drift. |
| **A14** | **low** | Realtime | `application/signal_hub.py:28-30`; `api/websockets.py` | `publish()` uses `q.put_nowait` on an **unbounded** `asyncio.Queue`; WS endpoints are unauthenticated. | A slow/stalled WS subscriber accumulates messages unbounded → memory growth; anonymous subscribers can open many sockets. | Bound the queue (`maxsize`) and drop/close on overflow; authenticate WS. | Load test with a non-draining client → memory flat. |
| **A15** | **low** | Security / CORS | `main.py:73` | `CORSMiddleware(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])`. | Any origin can read the (largely unauthenticated) API responses; widens the A1/A7 blast radius. | Restrict origins to the console host. | Preflight from a foreign origin rejected. |
| **A16** | **low** | Security / Info | `main.py:99-106` | `/health` and `/metrics` are unauthenticated top-level routes. | Internal operational metrics disclosure. | Place `/metrics` behind network policy/token; keep `/health` minimal (already is). | `/metrics` not public. |
| **A17** | **low** | Connectors / Config | `adapters/brokers/tbank.py:57-61`; `api/routers/portfolio.py:46-51` | `TbankBroker` silently enters dry-run when `token` is empty **or** the SDK import fails, even with `sandbox=False`; the router then serves empty/zero views with `configured` flickering (`portfolio.py:67` returns `configured:False`, orders raise 400). | A misconfigured production TBANK key degrades silently to “no trading” instead of a loud error → masked outage. | Log/fail loudly when `sandbox=False` but no token; distinguish “no creds” from “dry-run”. | Startup warns when live mode lacks a token. |
| **A18** | **low** | DB / Portability | `alembic/env.py:38-43`; `0005_signal_positions.py:93-95` | `env.py` sets neither `render_as_batch=True` nor `compare_type`; the 0005 downgrade uses raw `op.drop_column` (SQLite ≥ 3.35 only, works here but not on older hosts). | Future `--autogenerate` on SQLite will emit unusable ALTERs; downgrades fail on old SQLite/Postgres-with-batch hosts. | Set `render_as_batch=True`; document the SQLite floor. | `alembic revision --autogenerate` emits batch ops. |
| **A19** | **low** | Connectors | `adapters/fetchers/http_util.py:21`; `bybit.py`/`moex.py`/`yfinance.py` | No retry/backoff/429 handling in HTTP helpers; only Celery MC/portfolio tasks retry `DataFetchError`. `webull`/`synthetic` fetchers are registered nowhere; `SyntheticFetcher.exchange` is set to `YFINANCE` (`synthetic.py:20`), colliding with the real YF fetcher. | A single transient 5xx/429 loses the data for that request; a synthetic+YF registry would silently shadow YF. | Add bounded retry+backoff and honour `Retry-After` in `get_json`; give the synthetic fetcher its own exchange or document it as test-only. | Fault-injection test: a 429 is retried. |
| **A20** | **low** | Data access | `adapters/persistence/database.py:28`; `main.py:56-63` | Engine created with no `pool_pre_ping`, no pool sizing, and only `dispose()` on shutdown (bar cache redis client left open). | Stale pooled connections after a DB restart → sporadic errors; sizing untuned for prod. | `pool_pre_ping=True`, explicit pool sizes; dispose cache clients on shutdown. | Reconnect test after DB bounce. |
| **A21** | **low** | Docs / Consistency | `README.md:12,324,326`; `ARCHITECTURE.md:27,93,323-327`; `tasks.py:331-342` | README/ARCHITECTURE assert “Redis+Lua” distributed rate limiting and working “alert rules + Grafana”; both are false in the running system. Docs describe WebSocket as a “stub” though real WS + local-client protocol exist. Beat “reconcile” stub returns success. | Operators trust controls that do not exist (rate limiting, alerting) — an availability/security blind spot. | Correct the docs to match reality (or implement the gaps). | Docs diff vs A4/A6/A12. |
| **A22** | **low** | Realtime | `application/signal_engine.py:449-456` | The per-loop fetcher registry is correct, but Celery never closes it (see A8); and `loop_bar_cache()` probes can race the `_next_probe` global under threads. | Minor resource/consistency nits under concurrency. | Centralise loop-scoped resource teardown. | Lint/thread-safety review. |

---

## 3. Migration Dry-Run Evidence

All commands run from `E:\gex api` with `TRADING_DATABASE_URL` pointed at a throwaway DB under `%LOCALAPPDATA%\Temp\gex_mig`. `alembic/env.py:18-23` translates the async URL to a sync driver.

### 3.1 Forward / backward round-trip — **PASS**

```bash
export TRADING_DATABASE_URL="sqlite+aiosqlite:///C:/Users/butin/AppData/Local/Temp/gex_mig/mig_test.db"
venv/Scripts/python.exe -m alembic upgrade head      # empty DB -> 0006
venv/Scripts/python.exe -m alembic current           # -> 0006 (head)

# round-trip
venv/Scripts/python.exe -m alembic downgrade base    # -> all tables dropped, only alembic_version left
venv/Scripts/python.exe -m alembic upgrade head      # -> 0006 again
```

Observed results:
- `upgrade head` on an empty DB creates all 8 tables + `alembic_version`; `downgrade base` leaves **only** `alembic_version`; second `upgrade head` reaches `0006` cleanly.
- **No `_alembic_tmp_*` tables** left behind at any step (checked after each drop/batch).
- Resulting schema columns for all tables verified via `PRAGMA table_info` (counts: `api_keys` 7, `orders` 13, `backtest_results` 6, `strategy_presets` 17, `signal_keys` 9, `key_signals` 22, `key_trades` 20, `signal_positions` 34).

### 3.2 Model ↔ migration drift — **FAIL** (`alembic check`)

```bash
venv/Scripts/python.exe -m alembic check
# FAILED: New upgrade operations detected: [...]
```

Two drift classes (models say one thing, migrations created another):
- **Nullability:** every `created_at`/`updated_at` is `Mapped[datetime]` (→ `NOT NULL`) in `models.py`, but migrations 0001–0006 create them with `server_default=func.now()` and **no** `nullable=False` → **NULLABLE** in the DB. (`alembic check` reports `modify_nullable … True → False` for `api_keys.created_at`, `orders.created_at`, `backtest_results.created_at`, `key_trades.created_at`, `signal_keys.created_at`, `signal_positions.created_at/updated_at`, `strategy_presets.created_at/updated_at`.)
- **Type:** `Mapped[float]` infers `Double` while migrations used `sa.Float()` → `modify_type … FLOAT → Double` on all float columns. Cosmetic on SQLite (both `REAL`) but a real drift flag and a difference the CI gate does not catch.

This is consistent with finding **A3** (two independent schema authorities) and is invisible to CI (**A13**).

### 3.3 SQLite batch-mode specific behaviours — **PASS**

```bash
# 0005 downgrade with a NULL key_id row present
venv/Scripts/python.exe -m alembic downgrade 0004   # 0005 -> 0004
```
- `op.execute("DELETE FROM key_signals WHERE key_id IS NULL")` ran before the batch alter; the NULL row was removed, the surviving row retained, `key_id` correctly restored to `NOT NULL=1`, the added plan columns dropped, `signal_positions` dropped, no `_alembic_tmp` leftover.

```bash
# 0006 backfill from pre-0006 data (rows inserted while at 0005)
venv/Scripts/python.exe -m alembic upgrade 0005
# ... insert 3 AAPL + 1 MSFT strategy_presets rows (no version column yet) ...
venv/Scripts/python.exe -m alembic upgrade head
```
- Backfill produced sequential per-group versions: `AAPL → 1,2,3`; `MSFT → 1`; the `uq_strategy_presets_group_version` unique index held (a duplicate `(symbol,strategy,name,version)` insert is correctly **rejected** — proves the constraint is live); no leftover temp tables.

### 3.4 App-created DB cannot be migrated — **FAIL** (proves A3)

```bash
# create DB the way the app does:
venv/Scripts/python.exe -c "import asyncio; from trading.adapters.persistence.database import init_db; asyncio.run(init_db())"
venv/Scripts/python.exe -m alembic upgrade head
# -> sqlite3.OperationalError: table api_keys already exists
```
`init_db()` (`database.py:37`) creates the **current** model schema; Alembic 0001 then replays `create_table("api_keys")` and aborts. There is no `alembic stamp` path for app-created databases.

### 3.5 Timezone round-trip — **FAIL** (proves A5)

```python
# insert datetime(2026,1,2,3,4,5, tzinfo=utc) into a DateTime(timezone=True) column
read back -> 2026-10-07 07:00:13  tzinfo=None
datetime.now(timezone.utc) - read_value -> TypeError: can't subtract offset-naive and offset-aware datetimes
```

---

## 4. Connector Health Matrix

Legend: ✅ ok · ⚠️ weak/partial · ❌ broken/absent · n/a not applicable.

| Connector | Config validation | Timeouts | Retries/backoff | TLS | Secret handling | Error normalisation | Cleanup | Missing-config behaviour |
|---|---|---|---|---|---|---|---|---|
| **bingx** (`brokers/bingx.py`) | ⚠️ key/secret required by ctor only | ⚠️ httpx default 5s, not explicit | ❌ none; 429 not retried | ✅ https | ⚠️ secret passed to SDK/client; encrypted only at rest via `keys_service` | ✅ `BrokerError` w/ code+msg, `resp.text[:200]` | ✅ `close()`; but router `_broker` never closes per-request clients | ❌ `_broker` returns `None`→400 only at call time; no startup check |
| **bingx_ws** (`brokers/bingx_ws.py`) | ⚠️ listenKey flow only | ⚠️ | ❌ | ✅ wss | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **tbank** (`brokers/tbank.py`) | ⚠️ blank token → silent dry-run even if `sandbox=False` (**A17**) | ❌ SDK blocking calls, no timeout | ❌ | ✅ SDK | ✅ encrypted at rest | ✅ `BrokerError` | ⚠️ session cached, no `close()` | ❌ **silently degrades** (dry-run) instead of erroring |
| **tbank_stream** (`brokers/tbank_stream.py:52-54`) | ✅ `_require_live()` raises clear error in dry-run | n/a | ❌ | ✅ | ⚠️ | ✅ `DataFetchError` | ⚠️ | ✅ clear error |
| **bybit** (`fetchers/bybit.py`) | ✅ | ⚠️ 15s via `get_json` | ❌ | ✅ https | n/a public | ✅ `DataFetchError` on non-200 | ✅ `close()` via registry | ✅ empty → registry fallback |
| **moex** (`fetchers/moex.py`) | ✅ | ⚠️ 15s | ❌ | ✅ https | n/a | ✅ | ✅ | ✅ raises clear error |
| **yfinance** (`fetchers/yfinance.py`) | ✅ window sizing | ⚠️ 15s | ❌ | ✅ | n/a | ✅ | ✅ | ✅ (no book/tape → `DataFetchError`) |
| **webull** (`fetchers/webull.py`) | ⚠️ best-effort | ⚠️ httpx default | ❌ | ✅ | n/a | ✅ | ✅ | ⚠️ not registered anywhere (**A19**) |
| **synthetic** (`fetchers/synthetic.py`) | ✅ | n/a | n/a | n/a | n/a | n/a | n/a | ⚠️ `exchange=YFINANCE` collision (**A19**) |
| **registry / loop_registry** (`fetchers/registry.py`) | ✅ per-loop keying correct | n/a | n/a | n/a | n/a | ✅ aggregated errors | ✅ `aclose_loop_registry` on **app** shutdown; ❌ not called in **Celery** (**A8**) | ✅ `all sources failed` error |
| **redis (ratelimit)** (`adapters/ratelimit/token_bucket.py`) | ⚠️ `RedisTokenBucket` unused by middleware (**A4**) | n/a | n/a | n/a | ⚠️ no auth in URL | ⚠️ | ❌ | ⚠️ |
| **redis (cache)** (`adapters/cache/bar_cache.py`) | ✅ probe w/ cooldown | ✅ 0.5s connect+socket | n/a | n/a | ⚠️ | ✅ degrades to memory | ❌ never disposed (**A8**) | ✅ memory fallback |
| **redis (cancellation)** (`application/cancellation.py`) | ✅ | ✅ 0.15s | n/a | n/a | ⚠️ | ✅ degrades to process-local | ⚠️ | ✅ clear log |
| **celery tasks+beat** (`tasks.py`) | ⚠️ | ⚠️ backtest task has **no** time limit | ⚠️ only MC/portfolio retry | n/a | ⚠️ | ✅ | ❌ loop registry leak | ⚠️ stubs return success (**A12**) |

---

## 5. Security Findings (consolidated)

1. **Anonymous control plane (A1)** — `presets`, `backtest`, `data`, `strategies`, read-`portfolio`, `dashboard`, WS, `/metrics`, `/docs`. Highest-impact issue: promote/rollback/delete of live presets + engine start/stop.
2. **Default secrets & shared key material (A2)** — `dev-secret-change-me`, `admin/admin`; JWT and Fernet share one secret.
3. **Rate limiting ineffective/distributed-claim false (A4)**; login brute-force not specifically throttled; WS unthrottled.
4. **Key-as-credential + unauthenticated heavy regeneration (A7)**.
5. **CORS wildcard (A15)**, **`/metrics` + `/docs` exposure (A16)**.
6. **Positive:** `auth.py:18-19` uses `hmac.compare_digest`; JWTs are HS256 with `exp`/`iat` verified by python-jose (`security.py:47-63`); signal keys use `secrets.token_urlsafe(32)` (`signal_keys.py:62`); cancel tokens are regex-bounded (`cancellation.py:49`); a generic exception handler is **not** registered, so FastAPI's default 500 does not leak tracebacks.
7. **Error handlers**: `TradingError` handler returns `str(exc)`, which for `BrokerError` may include up to 200 chars of upstream `resp.text` (`main.py:77-80`, `bingx.py:110`) — minor info exposure.

---

## 6. Infra Findings (consolidated)

- **Docker:** root user, no `HEALTHCHECK`, no `.dockerignore`, `gex/` CSV universe not shipped, deps not from `requirements.txt`, TA-Lib runtime package name (`libta-lib0`) unverified (**A11**).
- **Compose:** databases published to host with hardcoded creds (`gex/gex`), Redis no auth, Grafana default `admin/admin`, no restart policies, no app healthcheck (**A11**).
- **Monitoring:** Prometheus rules + Alertmanager never wired; metrics mostly un-instrumented (**A6**).
- **CI:** no migration and no Docker validation; Python 3.12 vs local 3.14 (**A13**).
- **Data-plane gaps:** app-boot schema creation vs Alembic (§3.4, **A3**); no FKs (**A9**).

---

## 7. Docs-vs-Code Consistency Findings (A21)

| Claim | Source | Reality |
|---|---|---|
| Rate limiter is “Redis+Lua”, “already distributed” | `README.md:12,324`, `ARCHITECTURE.md:27,93` | In-process `dict` middleware; `RedisTokenBucket` unreferenced. |
| “alert rules + Grafana dashboards” | `README.md:326` | `prometheus-rules.yml`/`alertmanager.yml` not mounted; rules can’t load; metrics never incremented. |
| “Alembic migrations (verified vs SQLite)” | `README.md:327` | `alembic check` **fails**; CI never runs migrations. |
| Reconciliation runs every 5 min | `tasks.py:65-78`, docs | `reconcile_positions_task` is a no-op stub returning success. |
| WebSocket is a “stub” | `README.md:14` | Real `/ws/*` + `/ws/client` protocol + dispatcher exist. |
| “JWT + slowapi” plan | `ARCHITECTURE.md:101,152` | No slowapi; custom limiter. |

---

## 8. Anything Unclear / Assumptions

- **U1 — Two venvs.** Both `venv/` (3.14.7, used) and `.venv/` exist; CI/Docker target 3.12. Which is canonical is unstated; the audit used `venv/`.
- **U2 — Deployed topology.** Rate-limiting, CORS, cancel-registry and the signal hub all assume a **single** uvicorn worker; whether prod runs 1 or N workers changes the severity of A1/A4/A14 materially. `docker-compose.yml` starts one uvicorn with no `--workers`, so single-worker is assumed.
- **U3 — TA-Lib runtime package.** Could not verify offline that Debian ships `libta-lib0`; if not, `docker build`/runtime breaks (A11). Recommend a build check.
- **U4 — `alembic check` failures are partly cosmetic.** On SQLite the FLOAT/Double difference is inert; I flagged it because it indicates the schemas were authored independently and it would trip any CI that runs `alembic check`.
- **U5 — `/data/ohlcv` path parameters.** `symbol` flows into upstream URLs (moex/bybit) with no allow-list; not a classic SSRF (fixed hosts) but an unbounded-input fetch amplifier given it is unauthenticated (A1).
