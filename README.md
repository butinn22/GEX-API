# GEX-API — Auto-trading platform (BingX + TBANK)

A production-oriented auto-trading platform built on the existing GEX analytics
engine (`gex/`). Architecture and decisions are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Layout

- `gex/` — the unchanged analytics engine (signals, GEX, scanners, fetchers).
- `trading/` — the new trading bounded context (hexagonal):
  - `domain/` — pure dataclasses (Signal, OrderIntent, Order state machine, Position/Portfolio, Bar, value objects).
  - `ports/` — async `BaseFetcher`, `BrokerAdapter`, `Strategy`.
  - `adapters/` — rate limiter (in-memory `TokenBucket` + an **unwired** `RedisTokenBucket`), BingX (HMAC-SHA256), TBANK (dry-run), persistence (encrypted API keys), synthetic fetcher.
  - `application/` — strategy factory, **portfolio backtest** engine, metrics, **Monte-Carlo** engine, server-side **SVG charts** + reporter, cooperative **cancellation**.
  - `api/` — FastAPI REST + JWT + WebSocket stub; `static/` — **multi-ticker backtest console** (API-key manager + backtest UI).

## Run

```bash
git clone https://github.com/butinn22/GEX-API && cd GEX-API
cp .env.example .env                            # then fill in real values
uv venv .venv                                   # once (Python 3.12)
uv pip install --python .venv -e ".[dev]"       # or install listed deps
.venv/Scripts/python.exe -m uvicorn trading.main:app --reload
```

Or just double-click **`run.bat`** (Windows) — it starts the server on the first
free port (8000, 8001, …), waits for `/health`, and opens the browser.
A desktop shortcut can be generated with
`.venv/Scripts/python.exe scripts/make_shortcut.py`.

- Frontend (backtest + API keys): http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- Login uses `TRADING_ADMIN_USERNAME` / `TRADING_ADMIN_PASSWORD` (default `admin`/`admin` — change in prod).
- All configuration lives in `.env` (copy from [`.env.example`](.env.example));
  `trading/config.py` documents each variable.


## Test & verify

```bash
.venv/Scripts/python.exe -m pytest trading/tests -q       # 834 tests (833 passed, 1 skipped)
PYTHONPATH=. .venv/Scripts/python.exe scripts/verify_backtest.py   # metrics + Monte-Carlo CI
```

## Backtest & Monte-Carlo API

The API can backtest **one symbol or a basket of N tickers**, with **individual
settings per ticker** (strategy, params, weight, source, timeframe, limit).

| Method & path | Purpose |
| --- | --- |
| `POST /api/v1/backtest` | Single-symbol event-driven backtest. |
| `POST /api/v1/backtest/monte-carlo` | Backtest one strategy, then simulate its return distribution. |
| `POST /api/v1/backtest/portfolio` | Basket backtest — each ticker runs its **own strategy + settings**; returns combined equity, per-ticker attribution, and a correlation matrix. |
| `POST /api/v1/backtest/portfolio/monte-carlo` | Basket + Monte-Carlo distribution. |
| `POST /api/v1/backtest/portfolio/report` | **Self-contained HTML report** with embedded SVG charts (equity, underwater/drawdown, correlation heatmap, MC fan chart, return histogram). |
| `POST /api/v1/backtest/optimize` | Grid-search one ticker — selectable **objective** (Sharpe/Sortino/Calmar/return/profit factor/win rate/drawdown), optional **preset save**. |
| `POST /api/v1/backtest/optimize/global` + `GET …/optimize/global/{run_id}` | **Optimize every ticker** — progress polling (current ticker, completed/failed, ETA), per-ticker failure isolation, cancel endpoint. |
| `POST /api/v1/backtest/cancel/{token}` | **Stop** an in-flight run — the engine aborts at the next block boundary and the request returns **HTTP 499**. |
| `GET /api/v1/backtest/cancel` | Tokens of runs currently executing in this process. |
| `GET /api/v1/data/universe?category=&n=` | Auto-select **N tickers** from a named universe (`all` round-robins across categories for a diversified basket). |
| `GET /api/v1/data/categories` | Available universes. |

## Unified strategy: optimize → broker → API key → live dashboard

The **Optimize & Deploy** tab (or the API directly) drives the full workflow
for the unified strategy `trend_confluence_unified` — Trend Confluence as the
core, with the **EMF + ADL** parameters and signal logic and the **Momentum**
profile merged into one configuration schema (see
[`docs/UNIFIED_STRATEGY.md`](docs/UNIFIED_STRATEGY.md)):

1. **Presets per ticker** — `POST /api/v1/presets/from-backtest/{symbol}`
   validates a parameter set through a real backtest run and saves it as the
   ticker's default preset (`strategy_presets` table). Presets can be edited
   (`PATCH /api/v1/presets/{id}`), re-optimized and re-defaulted.
2. **Optimize** — `Optimize Selected Ticker` (grid search with a selectable
   objective, winner saved as the preset) or `Optimize All Tickers`
   (`POST /api/v1/backtest/optimize/global` — sequential, deterministic,
   per-ticker failure isolation, progress + ETA polling, cancellable).
   EMF+ADL fields sweep via dotted grid keys (`"emf.atr_tp_mult"`).
3. **Choose API / broker** — bingx or tbank (validated; tickers without an
   order route on that broker are accepted with a visible warning).
4. **Create API key** — `POST /api/v1/signal-keys` creates a unique
   `sk_…` key referencing the complete configuration (strategy, version,
   tickers, optimized params/presets, costs). Keys are revocable/disableable
   and auditable (`last_used_at`, `revoked_at`).
5. **Generate signals** — `POST /api/v1/signal-keys/{key}/generate` runs the
   same portfolio backtest engine on fresh data for every enabled ticker and
   stores the signal + trade ledgers (idempotent — always consistent).
6. **Live dashboard** — open **`/API_KEY/{key}`**: metrics summary, equity /
   drawdown / PnL-distribution charts (the same server-side SVG charts as the
   backtest report), latest signals and trades, **CSV / Excel downloads**
   with the full trade column set, manual Refresh + 60 s auto-refresh with
   last-update time.

```bash
# end-to-end smoke test (needs a running server + admin login):
curl -s -X POST localhost:8000/api/v1/auth/token -H 'content-type: application/json' \
     -d '{"username":"admin","password":"admin"}'   # → token
curl -s -X POST localhost:8000/api/v1/presets/from-backtest/NVDA -H "authorization: Bearer $T" \
     -H 'content-type: application/json' -d '{"strategy":"trend_confluence_unified"}'
curl -s -X POST localhost:8000/api/v1/backtest/optimize -H "authorization: Bearer $T" \
     -H 'content-type: application/json' \
     -d '{"strategy":"trend_confluence_unified","symbol":"NVDA","save_preset":true,"objective":"sharpe"}'
curl -s -X POST localhost:8000/api/v1/signal-keys -H "authorization: Bearer $T" \
     -H 'content-type: application/json' \
     -d '{"exchange":"bingx","tickers":["BTC"],"strategy":"trend_confluence_unified"}'
# → {"key": "sk_…"}  then open  http://localhost:8000/API_KEY/sk_…
```


**Choosing how many tickers:** pass `n_tickers` + `category` (+ `default_*`
settings) instead of an explicit `tickers` list, e.g.

```json
{ "n_tickers": 5, "category": "all", "default_source": "auto",
  "default_strategy": "sma_crossover", "default_limit": 1000 }
```

**Stopping a long run:** every response echoes a `run_token`; send it back to
`POST /api/v1/backtest/cancel/{token}` to stop the run. The UI exposes this as a
**Stop** button. Cancellation is cooperative and Redis-backed, so it also works
for the same run queued on a Celery worker (a cancel that lands *before* the
worker picks up the job is honoured too). Long Monte-Carlo runs are executed in a
worker thread, so the event loop stays free and `/health` keeps answering while a
simulation is running.

### Live signal engine

The platform ships a real-time signal service that turns any strategy into a
live feed of trade plans:

- **Start**: `POST /api/v1/signals/engine/start` with `symbols`, `strategy`,
  `preset`, `timeframe`, `poll_seconds`, `bars`. One background task per ticker
  fetches real bars from the venue the ticker resolves to (Bybit for crypto,
  MOEX for RU names, yFinance otherwise) and feeds only the bars it has not
  seen yet.
- **Every signal is stored** (`key_signals`) with its **complete trade plan** —
  entry, stop, target, size, risk %, risk amount, timeframe, bar time — and
  opens or closes a **position row** (`signal_positions`) that carries the
  whole lifecycle: entry/exit prices, initial stop, ratcheted trail, MFE in R,
  bars held, exit reason, gross/net PnL, PnL in R and percent return.
- **Delivery**: signals are published to `WS /ws/signals` and to every
  subscribed `WS /ws/client` session as a JSON trade plan, so a local client
  (or TradingView-style bridge) can act on them immediately.
- **Reads**: `GET /api/v1/signals` (filter by symbol/side/state/strategy),
  `GET /api/v1/signals/positions`, `GET /api/v1/signals/stats`,
  `GET /api/v1/signals/engine` (per-ticker diagnostics).
- **Exports**: `GET /api/v1/signals/export/signals.csv|.xlsx` and
  `.../positions.csv|.xlsx` — full per-signal and per-position tables, written
  with the dependency-free XLSX writer (no `openpyxl` needed).

The first poll replays the fetch window to warm the indicators up; the signals
it produces are stored with `source="backfill"` so they are clearly
distinguishable from live ones while still establishing the correct position
state.

### Main strategy: `confluence_breakout`

`confluence_breakout` is the port of the two systems that **survived
out-of-sample validation** in the research harness, behind one parameter set:

| Preset | Timeframe | Validated result |
|---|---|---|
| `alligator_4h` (default) | 4H | maxDD 2.7%, profit factor 1.38 OOS, win 31.5%, payoff 3.0 |
| `donchian_1d` | 1D | profit factor 1.88–2.08 on fresh OOS ticker sets |

Both are frozen research artefacts — the documented disable conditions
(rolling Sharpe < 0, drawdown > 50%, cost regime > 2× validated costs) apply.

Entry: Alligator aligned + jaw rising, confirmed HH/HL structure, ADL above its
EMA, EMF > 0, and a Donchian breakout (or a pullback to the lips) — optionally
gated by an SMA trend filter. Exits, in priority order: resting intrabar stop,
chandelier + structural trail, Alligator line, structure break, MA exit, time
stop. Risk is fixed at `risk_pct` of equity per trade (0.5% by default), so a
stop-out costs a known amount.

Reproduce the numbers on the research cache:

```bash
python - <<'PY'
import asyncio, csv, sys
from datetime import datetime, timezone
sys.path.insert(0, ".")
from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategies.confluence_breakout import (
    ConfluenceBreakoutStrategy, preset_params)
from trading.domain import Bar

def load(sym, tf):
    out = []
    with open(f"quant/cache/{sym}USDT_{tf}.csv") as fh:
        for r in csv.DictReader(fh):
            ts = datetime.fromisoformat(r["ts"]).replace(tzinfo=timezone.utc)
            out.append(Bar(timestamp=ts, open=float(r["open"]), high=float(r["high"]),
                           low=float(r["low"]), close=float(r["close"]), volume=float(r["volume"])))
    return out

async def main():
    for sym in ("BTC", "ETH", "XRP", "LTC", "ADA", "DOGE", "SOL"):
        bars = load(sym, "4h")
        p = preset_params("alligator_4h")
        p["timeframe"] = "4h"
        s = ConfluenceBreakoutStrategy(f"{sym}USDT", params=p, preset="alligator_4h")
        res = await run_backtest(s, bars, BacktestConfig(periods_per_year=2190))
        print(f"{sym:6} trades={len(res.trades):4} ret={res.metrics.total_return:7.2f}% "
              f"maxDD={res.metrics.max_drawdown:6.2f}% PF={res.metrics.profit_factor:6.3f} "
              f"win={res.metrics.win_rate:5.1%}")

asyncio.run(main())
PY
```

## Performance

A backtest is dominated by **data loading**, not by the replay loop — the engine
itself replays 1,500 bars in ~3 ms, so three tickers cost single-digit
milliseconds while fetching them cost seconds. Two changes removed that:

1. **Bounded source requests.** The Yahoo fetcher used to send `period1=0`, i.e.
   "give me everything since inception" — AAPL returns ~11,500 daily bars
   (~1.3 MB) of which we kept the last 1,500. The window is now sized from
   `(timeframe, limit)` with holiday/weekend headroom: **786 KB → 87 KB**,
   **0.86 s → 0.18 s** for a single ticker.
2. **An OHLCV cache with a TTL.** `load_bars` caches by
   `(symbol, source, timeframe, limit)` for `TRADING_DATA_CACHE_TTL` seconds
   (default 300; `0` disables). Re-running with a tweaked parameter — the normal
   workflow — re-downloads nothing. Pass `"refresh_data": true` on any backtest
   request to force a fresh download.

Fetcher clients are also reused per event loop, so a run does one TLS handshake
per exchange instead of one per ticker (and no longer leaks a client per ticker).

Measured on 3 real tickers (NVDA/AAPL/BTC-USD, 1,500 bars each, via the API):

| | before | after |
| --- | --- | --- |
| Cold run (downloads) | 5.06 s | **2.38 s** |
| Re-run (same data) | 5.06 s | **0.03 s** |

Cache counters are exposed for debugging: `bar_cache_stats()`.

### EMF + ADL strategy: O(n²) → O(n)

The **EMD + ADL** strategy (`gex_emf`) used to be unusable on real histories —
minutes for a single backtest. It was not the indicator maths; it was *how often*
the maths ran. Its `on_bar` recomputed the entire ~110-column feature frame over
the whole accumulated history **once per bar**, so an n-bar replay cost O(n²)
frames.

Three changes, all numerically exact:

1. **A batch `prepare()` hook on the strategy port.** The event-driven engine now
   calls `strategy.prepare(bars)` once before the replay loop; `on_bar` becomes an
   O(1) row lookup by index (with an O(1) close-price divergence probe, falling
   back to streaming if the bar stream doesn't match what `prepare` saw). The
   default is a no-op, so every other strategy is untouched, and the live/streaming
   path is unchanged.
2. **`_linreg` is a fixed linear filter, not a regression per window.** The
   least-squares endpoint over a fixed-length window has a closed form
   (`c[j] = 1/L + (L-1)(Lk−Σx)/2(LΣx²−(Σx)²)`), so ~3,000 `np.linalg.lstsq` calls
   collapse into one convolution — **340–1,300× faster**, agreeing with `lstsq` to
   `1e-13`.
3. **The VWAP pivot state machine uses numpy arrays.** It rebuilt a length-n pandas
   Series *inside* the loop just to read one element — the per-bar constant now
   dominates nothing.

Result for `gex_emf` on 1,500 bars (through `POST /api/v1/backtest`):

| | before | after |
| --- | --- | --- |
| Backtest | 72.88 s | **0.176 s** (≈**414×**) |
| `run_backtest` equity curve | — | **bit-identical** |
| Trade ledger | — | **identical** |

The project's golden fixture (`tests/test_strategy_golden.py`, 6-dp oracle) and a
causality test (`prepare` must equal streaming signal-for-signal) both pass, so this
is a pure speed-up — no signal changed.

### Risk exits for EMF + ADL (they were inert)

The UI exposed `TP %` / `Trail %` for `gex_emf`, but the column-based signals the
backtest reads had **no take-profit or trailing logic** — that lived in
`decision.evaluate()`, which the engine never calls. So both sliders did nothing.
The adapter now applies the project's own risk maths (percentage **or** ATR-based)
as a per-bar overlay that reads only bars *before* the current one (no lookahead).
It is **off by default** (`use_risk_exits`), so existing results are unchanged;
`tp_percent`, `trailing_percent`, `use_atr_stops` and the `atr_*_mult` knobs now
demonstrably change the outcome.


## Status

**Done & tested (834 tests — 833 passed, 1 skipped):** domain + async ports; event-driven + **vectorized**
backtest engines (no-lookahead fills, fees/slippage, trade ledger) with a batch
**`prepare()` hook** (O(n) replay for heavy strategies); **match engine**
with pluggable commission/slippage models; metrics (Sharpe, Sortino, **Calmar**,
max DD, VaR/CVaR, win rate, profit factor) with bootstrap CIs; Monte-Carlo
simulators (GBM, residual/block bootstrap);
**multi-ticker portfolio backtest** (per-ticker strategy+settings, weighted capital
allocation, time-grid alignment, per-ticker attribution, correlation matrix, error
isolation); **Monte-Carlo engine** over any return series (GBM / residual bootstrap
/ block bootstrap / historical resampling) with **percentile fan bands, confidence
intervals on every metric**, P(profit), VaR/CVaR and a return histogram;
**server-side SVG charts** (equity, underwater, multi-series, fan chart, histogram,
correlation heatmap) embedded in the HTML report; **cooperative cancellation**
(Redis-backed run tokens, HTTP 499, works via API or Celery); **universe selection**
("choose N tickers");
**indicator library** (SMA/EMA/RSI/MACD/
Bollinger/ATR) with an optional TA-Lib backend; **10 strategies** via a **strategy
registry + runner** (dual **long/short SMA crossover with independent per-side
settings**; **EMD + ADL** with optional TP/trailing/ATR exits, vectorized
indicators and an O(n) batch replay); **signal pipeline** (filters → aggregators);
**position sizing**
(fixed / %-risk / Kelly) + RiskManager; **walk-forward** + parameter sensitivity;
**PnL FIFO/LIFO/AVG**; **smart orders** (TWAP/VWAP/iceberg); **reporter** (JSON/HTML/
PDF); **live strategy engine** + **position reconciliation**; **domain event bus**;
parallel **data loader**; **audit log**; real data fetchers (MOEX ISS, Bybit,
yFinance, Webull) with fallback + validator; BingX client (HMAC-SHA256) and TBANK
adapter (real `tinkoff-invest` SDK, lazy session, dry-run fallback); order execution
engine (routing, backoff retry, kill switch) + order persistence; **broker health
monitor**; **WS reconnect manager**; live WebSocket streaming (signals/orders/
positions); encrypted API-key storage + JWT + vanilla-JS key manager; token-bucket
rate limiter (in-memory `TokenBucket` + optional `RedisTokenBucket`) + **in-memory, per-IP API rate-limit middleware**, CORS,
exception handlers; Celery tasks + beat; **TTL cache**; **bulk insert + task-result
store**; Prometheus `/metrics` + **alert rules** + Grafana dashboards + structlog +
correlation IDs; Alembic migrations (verified vs SQLite); **property-based tests**
(hypothesis); `fetch_data` CLI; **BingX WebSocket streams** (market + user-data
parsers/stream); **live signal engine** (per-ticker polling tasks, full trade-plan
signals persisted with a position ledger, `WS /ws/signals` + local-client fan-out,
CSV/XLSX signal and position exports); **TBANK stream bridge** (SDK callbacks → domain); **Timescale
bars/trades storage** + hypertable DDL; **live strategy runner**
(`/strategies/{name}/start|stop` background tasks); Docker image (with TA-Lib) +
compose; one-click launcher (`run.bat` + desktop shortcut) + app icon/favicon;
CI workflow.

**Known remaining (gated only on external keys/infra — everything is wired and offline-tested):**
- Live broker calls + broker WebSocket streams need real BingX/TBANK keys (signing,
  SDK wiring, stream parsers, and bridges are all complete).
- TimescaleDB + Docker compose need a Docker/Postgres host to run (DDL, migrations,
  and all YAML config are written and validated).
- TA-Lib C backend: Docker installs it; locally only the pure-numpy fallback runs
  (no cp312 wheels).
- 24/7 running: the live runner + `/strategies/{name}/start` are wired (synthetic
  feed for demo); production swaps in a realtime feed (broker WS or a data provider).
