# GEX-API — Architecture

Auto-trading platform (BingX + TBANK) built on the existing GEX analytics engine.
This document records the target architecture, the key decisions, and the gap
against what already exists in this repo. ADRs are inline; the short version of
each decision is listed at the end.

---

## 1. Current state (what we start from)

`gex/` is a clean/hexagonal analytics engine, ~92k LOC, Russian-documented:

| Ring | Existing | Reuse for trading? |
|---|---|---|
| `gex/domain` | pure entities (numpy/pandas/scipy only, enforced by `ast_guard`) | read-only dependency |
| `gex/ports` | `MarketDataPort` (sync), `RateLimitPort`, `CachePort`, `JobQueuePort`, … | **rate limiter is directly reusable** |
| `gex/adapters` | fetchers (moex/bybit/yf/webull + more), Redis+Lua token bucket, Redis cache, SQLite/Postgres sync SQLAlchemy | fetchers reusable as *data sources*; rate limiter reusable as-is |
| `gex/application` | GEX/signal/scan services | the signal source for strategies |
| `gex/routers` | 24 FastAPI routers | kept separate from the new trading API |
| `gex/strategy` | one Pine→Python strategy (`trading_algorithm`) | to be re-issued behind the new `Strategy` ABC |

Constraints discovered by inspection:

- The engine is **synchronous** and **pandas-first**; exchange/broker I/O is
  inherently async. We do not rewrite the engine.
- Rate limiting is already **distributed (Redis + Lua)** with a `Decision`
  (allowed / retry_after / rule_name) and a local bounded fallback — this is the
  foundation for requirement #1 and is reused, not rebuilt.
- Persistence is sync SQLAlchemy 2.0 (SQLite dev / Postgres prod with graceful
  fallback). No TimescaleDB, no async sessions.

## 2. Target architecture

We add a **new bounded context `trading/`** (sister package to `gex/`), not new
folders inside `gex/`. Trading is a different business context (orders, fills,
risk, backtest) than analytics (signals, GEX, scanners); keeping it separate
protects the 92k LOC engine from churn and lets each context deploy/scale
independently. The only dependencies across the boundary are the existing
**ports** (rate limiter, market data, cache) — never adapters.

```
┌─────────────────────────────── trading/  (new bounded context) ─┐
│  domain/        Signal OrderIntent Order Fill Position Portfolio │
│                 Instrument Bar Tick OrderBook + enums/values/errs│
│  ports/         BaseFetcher(abc,async) BrokerAdapter Strategy    │
│  adapters/      tbank/ bingx/  (broker + websocket)              │
│                 fetcher wrappers over gex market-data adapters   │
│                 ratelimit → delegates to gex RateLimitPort       │
│  application/   ExecutionEngine OrderStateMachine RiskManager    │
│                 BacktestEngine MonteCarlo PathSimulators Metrics │
│  api/           routers + websockets (FastAPI)                   │
│  tasks/         celery tasks + beat schedules                    │
└──────────────────────────────────────────────────────────────────┘
                              │ depends on (ports only)
┌────────────────────────────── gex/  (unchanged analytics engine) ┐
│  ports  MarketDataPort RateLimitPort CachePort JobQueuePort ...   │
│  adapters  fetchers ratelimit(Redis+Lua) cache persistence        │
└──────────────────────────────────────────────────────────────────┘
```

DDD mapping (spec requirement): **Signal**, **Order**, **Position**, **Portfolio**
are domain entities; **Broker** and **Data** are anticorruption-layer
boundaries implemented as ports; **Strategy** is an application service whose
interface (`on_bar`/`on_tick`/`generate_signals`) is itself a port so strategies
are pluggable.

## 3. Gap analysis (spec vs repo)

| # | Requirement | Status | Notes |
|---|---|---|---|
| 1 | Unified async `BaseFetcher` (get_ohlcv/get_orderbook/get_trades) | ❌ missing | existing `MarketDataPort` is sync/DataFrame; new async port + wrappers |
| 1 | Redis-backed strict rate limiting | ✅ exists | `gex/adapters/ratelimit` (Redis+Lua token bucket) reused; add per-exchange rule config |
| 2 | Pluggable `Strategy` ABC + `Signal`/`OrderIntent` dataclasses | ❌ missing | added this turn; existing Pine strategy re-issued behind it later |
| 2 | TA-Lib / pandas-ta indicators | ⚠️ partial | `gex/domain/indicators` exists (pandas); pandas-ta/TA-Lib to be wrapped |
| 3 | Monte-Carlo backtest engine (vectorized + event-driven, GBM/bootstrap/block, CI) | ❌ missing | net-new; numba/joblib for N=10⁴ paths |
| 4 | TBANK (invest-python, unary + streams, sandbox/prod) | ❌ missing | net-new adapter |
| 4 | BINGX (REST V3 + WS, HMAC-SHA256, spot+futures) | ❌ missing | net-new adapter |
| 5 | REST /strategies /backtest /signals /orders /positions /portfolio /data | ⚠️ partial | `gex/routers` cover analytics; trading routes net-new |
| 5 | WebSocket live signals/streams | ❌ missing | net-new |
| 5 | JWT auth, rate limiting, Swagger | ⚠️ partial | `gex/auth` (users/telegram) exists; JWT + slowapi net-new for trading |
| 6 | Celery + Redis tasks | ⚠️ partial | existing queue is Redis **Streams** (`TaskPublisher`), not Celery — see ADR-2 |
| 6 | Postgres + TimescaleDB (OHLCV/trades hypertables) | ❌ missing | sync Postgres exists; Timescale + async migrations net-new |
| 6 | Docker, Alembic, Pytest>85%, Prometheus/Grafana, structured logging | ⚠️ partial | structlog + Prometheus + middleware exist in `gex`; Docker/Alembic/Grafana net-new |

## 4. Key decisions (ADRs)

- **ADR-1 — Sync engine / async trading.** Keep `gex` sync; make the `trading`
  context fully async. Exchange I/O and streams are async-native; forcing them
  sync would waste a worker per connection. The boundary is thin adapters that
  call the sync engine via `asyncio.to_thread` where a strategy needs GEX
  signals. Trade-off: two concurrency models in one repo, but no rewrite of the
  working engine and no thread-per-websocket.
- **ADR-2 — Celery over the existing Redis Streams queue.** Celery is specified
  and adds retries, routing, acks, beat, and result backends that the in-house
  `TaskPublisher` would have to grow organically. We adopt Celery for the
  long-running trading tasks (Monte-Carlo, bulk fetch) and leave the analytics
  pipeline on Redis Streams. Trade-off: a second queue system; accepted because
  the two workloads have different needs (analytics = fire-and-forget fan-out,
  trading = durable, retried, rate-budgeted jobs).
- **ADR-3 — TimescaleDB as a separate storage path from the analytics DB.**
  OHLCV/trades are append-only time series; Timescale hypertables give
  compression + time-partitioned pruning the analytics tables don't need. Keep
  the analytics Postgres untouched, add a dedicated `market` schema. Async
  sessions only in the trading context.
- **ADR-4 — Broker anticorruption layer.** Every broker adapter returns domain
  types (`Order`, `Position`, `Fill`), never raw exchange DTOs. OrderMapper
  (internal ↔ exchange) lives inside the adapter. This is what makes the
  `ExecutionEngine` broker-agnostic and the backtest MatchEngine able to stand
  in for a live broker.
- **ADR-5 — `Signal`/`OrderIntent` split.** A `Signal` is what a strategy
  *thinks* (direction, reason, strength, optional size); an `OrderIntent` is
  what it *wants executed* (side, qty, type, prices, TIF) after risk + sizing.
  Two types instead of one because sizing/risk can veto or transform a signal
  without mutating the signal itself.
- **ADR-6 — Functional position/portfolio (frozen dataclasses).** `Position` and
  `Portfolio` are immutable and `apply_fill()` returns a new instance. This makes
  the event-driven backtest trivially correct (no shared mutable state across
  paths), simplifies testing, and the live `PositionKeeper` just holds the latest
  instance behind a lock.

## 5. Implementation order (maps to the 14-phase roadmap)

1. **This slice — domain + async ports** (`trading/domain`, `trading/ports`).
2. Fetcher adapters + per-exchange rate-limit rules (wraps `gex` fetchers).
3. Storage: Alembic + Timescale hypertables + async repositories.
4. Strategy core: `Strategy` ABC, indicators, 3 example strategies, risk/sizer.
5. Backtest: event-driven engine + match engine + metrics.
6. Monte-Carlo: GBM / residual bootstrap / block bootstrap / historical, CI.
7. Brokers: TBANK, then BINGX (adapters + order mappers + WS reconnect).
8. Execution: `ExecutionEngine`, order state machine, smart orders, kill switch.
9. API + WebSocket + JWT + slowapi.
10. Celery + beat + result store.
11. Observability (metrics/alerts/audit).
12. Testing to >85%, property tests, E2E.
13. Docs (MkDocs).
14. Docker/Helm/deploy hardening.

Steps 1–14 above are now implemented and covered by the test suite (378 tests).
The latest slice added the **portfolio backtest** (multi-ticker, per-ticker
settings), the **Monte-Carlo engine** with confidence intervals and fan bands,
**server-side SVG charts** + self-contained HTML reports, **universe selection**
("choose N tickers"), and **cooperative cancellation** exposed as a UI Stop button
— see ADR 7–10.

---

## ADR short list

1. sync engine / async trading (thin `to_thread` boundary)
2. Celery for trading jobs; Redis Streams stays for analytics
3. TimescaleDB = separate `market` schema, async sessions only in trading
4. broker anticorruption layer → domain types across the boundary
5. `Signal` (intent-to-think) vs `OrderIntent` (intent-to-execute)
6. immutable `Position`/`Portfolio` (`apply_fill` returns new instance)
7. **Portfolio backtesting = N independent single-ticker runs, aligned post-hoc.**
   Each ticker keeps its own strategy, settings and capital slice, and is replayed
   through the *same* `run_backtest` engine; results are then aligned on a common
   time grid and summed. This keeps the per-ticker semantics identical to the
   single-symbol path (no second, subtly-different engine) and makes error
   isolation trivial — one bad ticker degrades to an entry in `errors` and its
   capital stays as cash rather than failing the whole basket.
8. **Monte-Carlo works on a *return series*, not on bars.** The engine takes simple
   returns (derived from any equity curve via `returns_from_equity`) and resamples
   them (GBM on log-returns, residual bootstrap, block bootstrap, historical).
   This decouples the simulator from the data source and lets the same engine run
   on a single strategy or a whole portfolio.
9. **Cancellation is cooperative and Redis-backed, not `asyncio.CancelledError`.**
   A run token lives in Redis so a cancel issued on one worker is observed by
   another; the engine polls it between blocks of paths and raises `RunCancelled`,
   which the API maps to **HTTP 499**. The registry treats an absent flag as
   "run", so a Redis outage degrades to "cannot cancel" rather than "cannot run".
   A cancel that arrives *before* a queued Celery job starts is honoured because
   `cancel()` writes the flag unconditionally — there is no "start wipes the flag"
   window.
10. **Charts are generated server-side as plain SVG strings** (`charts.py`), with
    no JS charting dependency. This keeps the HTML report a single self-contained
    file (works offline, survives email attachments) and makes the visuals
    unit-testable — the tests assert on SVG structure and colors directly.
11. **Data loading, not the replay loop, is the cost of a backtest.** Profiling
    showed the engine replays 1,500 bars in ~3 ms while fetching them took
    seconds, so all optimisation goes into the fetch path: (a) source requests are
    **bounded to the requested bar count** — Yahoo was sent `period1=0` and
    returned ~11,500 daily bars (~1.3 MB) for AAPL to serve 1,500; (b) `load_bars`
    is wrapped in a **TTL cache** keyed by `(symbol, source, timeframe, limit)`,
    because the dominant workflow is re-running the same backtest with one
    parameter changed; (c) fetcher clients are **reused per event loop** (see ADR
    12). A `refresh_data` flag provides an escape hatch from the cache.
12. **The fetcher registry is scoped to the running event loop** (a
    `WeakKeyDictionary` keyed by the loop object), not to the process. httpx
    clients are bound to the loop that first uses them and Celery runs every task
    in a fresh `asyncio.run` loop, so a process-global registry would hand a dead
    client to the next task. Weak keying means the entry dies with its loop. The
    bar *cache* is deliberately process-wide instead — it holds only frozen
    `Bar` values, so it is safe to share across loops and lets a worker reuse data
    between tasks.
13. **A heavy strategy gets one batch hook, not a second engine.** `gex_emf`
    recomputed its whole ~110-column frame inside every `on_bar`, so an n-bar
    replay was O(n²) — minutes for one backtest. Rather than special-case the
    engine, the `Strategy` **port** gains an optional `async def prepare(bars)`
    (default no-op) that the engine awaits once before the replay; `on_bar` then
    does an O(1) row lookup. Two properties make this safe: every indicator in the
    pipeline is **causal** (shift/rolling/ewm, the VWAP pivot recursion, the ADL
    cumsum), so row *i* of the full-history frame equals row *i* of the prefix
    frame — exactly the equality the old recompute relied on; and `prepare`
    records the bar timestamps *and closes* it saw, so a replay of a different
    series (`on_bar` sees an unexpected bar) drops the frame and streams instead
    of indexing signals against the wrong rows. Backwards-compatible: strategies
    that don't override `prepare` behave exactly as before, and the live/streaming
    path is unchanged.
14. **Indicator maths is vectorized where it is a fixed filter.** `_linreg`
    evaluated a least-squares fit per rolling window (~3,000 `lstsq` calls on
    1,500 bars). The endpoint of a least-squares line over a *fixed-length* window
    is a fixed linear filter, so it is computed as one convolution from the closed
    form — 340–1,300× faster and equal to `lstsq` to `1e-13`. The pandas
    `rolling(min_periods=2)` head (windows shorter than `length`) is reproduced
    explicitly so the ramp-in values are bit-identical. The VWAP state machine was
    also rebuilding a length-n pandas Series inside its loop to read one element;
    it now walks numpy arrays.
15. **Risk exits are an explicit opt-in overlay, not a hidden default.** The
    `gex_emf` UI exposed `TP %` / `Trail %`, but the column signals the backtest
    reads carry no take-profit/trailing logic (it lived in `decision.evaluate()`,
    which the engine never calls), so the sliders silently did nothing. The
    adapter now applies the project's own risk maths (percentage **or** ATR) reading
    only bars *before* the current one (no lookahead), gated behind
    `use_risk_exits` (default **off**) so no existing result changes. Chosen over
    making it always-on because silently altering every historical backtest is
    worse than a knob that visibly does nothing.

