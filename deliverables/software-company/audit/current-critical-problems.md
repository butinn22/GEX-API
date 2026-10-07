# Current Critical Problems — Independent Code Review

**Scope:** current tree (`E:\gex api`), reviewed directly (not the Round-1/2 docs).
**Date:** 2026-10-07.
**Method:** read the actual code at each cited `file:line`; every "Critical/High" item below was
confirmed by reading the source, not inferred from the older audits. Items marked *(explore)* were
surfaced by a sub-agent and spot-checked but not exhaustively reproduced.

> Context: the Round-2 remediation (`8d1da1b` … `169aa6c`) did land and its headline fixes are
> **present** on the current tree — router-level auth at the mounts, WS JWT/handshake auth, prod
> default-secret fail-fast, dashboard HTML escaping, `UTCDateTime`, `run_migrations()`, login
> bucket/lockout, `/metrics` gating. The problems below are what **remains** or what those audits
> did **not** cover. The most serious ones are not auth gaps — they are in the **trading and
> backtest logic**, where bugs silently produce wrong P&L and can lose real money.

---

## A. Money-losing / live-execution defects (fix first)

### A1. Stop / trailing-stop orders are silently transmitted as MARKET orders — BingX
`trading/adapters/brokers/bingx.py:227-231`
```python
type_ = {
    "market": "MARKET",
    "limit": "LIMIT",
    "stop_market": "STOP_MARKET",
}.get(intent.order_type.value, "MARKET")
```
`OrderType.STOP` (`"stop"`), `STOP_LIMIT`, and `TRAILING_STOP` are **not in the map**, so `.get(...,
"MARKET")` turns a protective stop into an unconditional market order that fills immediately
instead of resting. A stop intent can therefore open/close at market at the wrong moment. An
unknown order type must raise, never default to MARKET.

### A2. The kill switch is never enforced
`trading/application/execution.py:50-65` defines `KillSwitch.update`, and `:85` stores
`self.kill_switch`, but it is **never called**. Repo-wide grep shows `KillSwitch` only in
`execution.py` and `tests/test_execution.py`. The documented drawdown halt cannot fire; live trading
continues through the configured drawdown.

### A3. `RiskManager.approve` is dead code — risk limits are decorative
`trading/application/risk.py:99-112` implements the only position-notional + drawdown gate.
`approve(` appears only in `risk.py` and `tests/test_risk.py` — no live path (`ExecutionEngine`,
`AccountRouter`, `POST /orders`) calls it. Position caps and drawdown halts are not applied. (Even
if wired, it compares a single intent's notional to `equity * max_position_pct` without existing
exposure, so cumulative over-allocation is allowed.)

### A4. Order retry can duplicate live orders (no idempotency at the broker)
`trading/application/execution.py:88-109` retries any `BrokerError` up to `max_retries` with the same
intent, and `BingxClient.place_order` (`bingx.py:143-153`) sends no `clientOrderID`. If the exchange
accepted the order but the HTTP response failed/timed out, the retry submits a **second** live
order. The REST `Idempotency-Key` cache (`api/routers/portfolio.py:189-201`) does not help: it is
only written **after** `_place_order` succeeds, so a retry following an ambiguous `BrokerError`
misses the cache and places again.

### A5. `OrderIntent` accepts stop orders with no stop price; BingX omits `positionSide`
`trading/domain/orders.py:147-148` validates only `STOP`/`STOP_LIMIT`; `STOP_MARKET` and
`TRAILING_STOP` (both in `enums.py:35,37`) pass with `stop_price=None`. `bingx.py:151-152` then
simply omits `stopPrice`, so a stop-loss can be submitted with no trigger. Separately,
`bingx.py:143-148` never sends `positionSide`, which is required in hedge mode and whose omission
means a BUY that is meant to **close a short** is treated as opening a long.

### A6. TBANK silently "pretends" to place orders
`trading/adapters/brokers/tbank.py:61` sets `self._dry_run = not token or not _sdk_available()`, and
`:96-103` fabricates a `PENDING` Order when dry-run. Callers (`api/routers/portfolio.py:223`,
`ExecutionEngine`) treat it as a real order and persist/publish it. A production deployment missing
the SDK or with an empty token will appear to trade while placing nothing — silent divergence
between local state and the exchange.

### A7. Signals' stop-loss is not sent with the entry
`trading/api/local_client_ws.py:103` calls `native_payloads_for(sig, tbank_lots=...)` **without**
`stop_loss`/`take_profit`, even though the payload builder supports them
(`application/native_payloads.py:151-154`). Positions open with no resting stop; the only protection
is strategy logic that cannot cap loss during a disconnect.

### A8. `Order.apply_fill` bypasses the state machine and corrupts quantity on over-fill
`trading/domain/orders.py:209-224`: it mutates `self.filled_quantity += fill.quantity` at `:216`
**before** the over-fill guard at `:217-218`, so a rejected over-fill leaves the order permanently
over-filled. It also never calls `_transition`, so a PENDING order can jump straight to
FILLED/PARTIAL, transitions the `_TRANSITIONS` table (`:41-49`) forbids. No symbol/side match check
is performed.

### A9. Reconciliation drops a hedge side and double-counts on a side flip
`trading/application/reconcile.py:35` keys broker positions by `symbol`, collapsing a LONG and a
SHORT on the same symbol into one entry (one side is silently dropped). `:39-40` flags a
side-mismatch as `to_insert` but never closes the local opposite side (`to_close` at `:46-48` only
covers symbols entirely absent), so the local store can hold both stale and new exposure.

### A10. BingX `cancel_order` is impossible and is retried anyway
`trading/adapters/brokers/bingx.py:251-255` unconditionally raises
`BrokerError("cancel_order requires symbol...")`. `ExecutionEngine.cancel_order`
(`execution.py:146-147`) retries `BrokerError`, so a cancel can **never** succeed and wastes the
retry budget. The REST cancel route / UI "stop" cannot actually cancel a resting order.

---

## B. Backtest / analytics correctness (wrong reported results)

### B1. Exit orders are sized off current equity, not the held quantity
`trading/application/backtest/engine.py:97-115` (`_size_signal`) always sizes
`qty = equity * position_fraction * strength / price` and **ignores `sig.quantity` / `sig.position_size`**.
Every exit signal (stop/target/trail) therefore gets a fresh equity-sized order rather than the
quantity held. When price moved, the exit over/under-shoots: the position does not fully close and
a small residual — often an opposite-side flip — is left behind while the strategy believes it is
flat. The next "entry" then acts on the residual. This propagates through `run_backtest` and the
portfolio engine (`application/backtest/portfolio.py:398`), so trade ledgers, win rate, profit
factor and drawdown are all computed from a book that does not match the strategy's state.

### B2. `block_bootstrap` removes the drift and never adds it back
`trading/application/backtest/simulators.py:83` computes `residuals = r - r.mean()`, and `:95`
compounds the sampled residuals without re-adding the mean. Every block-bootstrap path is
**driftless** regardless of how profitable the strategy was. Compare `_residual_returns` (`:47-52`)
which correctly adds `r.mean()` back. Any Monte-Carlo using `method="block_bootstrap"` produces a
fan chart, P(profit), VaR/CVaR and mean that are systematically biased toward zero.

### B3. GBM Monte-Carlo double-counts the Itô correction
`trading/application/backtest/monte_carlo.py:106-111` passes `mu = mean(log returns)` — already the
estimated log-drift — into `geometric_brownian_motion`, which subtracts another `0.5σ²`
(`simulators.py:39`: `drift = (mu - 0.5 * sigma**2) * dt`). The simulated drift is
`mean(logret) − 0.5σ²` instead of `mean(logret)`, biasing the default GBM result low.

### B4. Optimizer treats perfectly-good `inf` objectives as the worst possible score
`trading/application/backtest/optimize.py:152`:
```python
return value if math.isfinite(value) else -1e18
```
`profit_factor`, `sortino` and `calmar` return `+inf` for the best possible result (no losses / no
downside / no drawdown; `metrics.py:71-72,91-92,123-124`). Mapping `+inf` to `-1e18` makes the
optimizer **disqualify its best candidates**, so optimizing on those objectives selects a worse
parameter set.

### B5. `sortino` denominator uses only losing observations
`trading/application/backtest/metrics.py:70-73` computes `downside = r[r<0]; dd = sqrt(mean(downside**2))`
— averaging over only the losing count. The semi-deviation must average `min(r,0)²` over **all**
periods. The Monte-Carlo metric (`monte_carlo.py:173-174`) does it correctly, so the two Sortino
values are inconsistent for the same returns.

### B6. Modelled intrabar stops/targets are ignored; fills always happen at the next open
`trading/application/backtest/match_engine.py:56-58` always fills at `bar.open`; the engine never
reads `Signal.stop_loss` / `take_profit`. Strategies detect stops intrabar (e.g.
`strategies/confluence_breakout.py:589-591`) and then the order executes a full bar later at the
next open — so realized stop losses do not respect the modelled risk and the "conservative intrabar
stop" claim in the docstrings is false.

### B7. `walk_forward` does not walk forward
`trading/application/backtest/walk_forward.py:13-36` takes a zero-argument `strategy_factory`, so it
cannot receive a train window; every test window is replayed with fresh cash and no indicator
warm-up, and window 0 is never tested. It cannot produce a leakage-free walk-forward evaluation.

### B8. Portfolio metrics annualize a mixed-cadence union grid with fixed 252
`trading/application/backtest/portfolio.py:422-433` builds the sorted union of every ticker's
timestamps (which may use different `timeframe`s) and passes it to `compute_metrics(...,
periods_per_year=252)`. Consecutive union rows are treated as one period, so Sharpe/CAGR and the
correlation matrix are wrong whenever the legs' bar cadences differ.

### B9. Same-bar trailing-stop lookahead in `confluence_breakout`
`trading/application/strategies/confluence_breakout.py:582-603`: the trail is ratcheted from the
current bar's **close**, then compared to the same bar's **low/high**. At the moment the low printed,
the close did not exist — impossible in real time. Other strategies (`trend_confluence.py:569-573`)
update `_best` after the exit check; this one does not.

### B10. Optimizer's reported "best" metrics are in-sample
`trading/application/backtest/optimize.py:529` re-runs the winner on **all** bars and returns those
metrics, while ranking used the validation split. The headline metrics shown in the UI overstate
expected out-of-sample performance.

---

## C. Security / availability / data-integrity (remaining)

### C1. Production secret validation is bypassable and shares key material
`trading/config.py:30,33,78-95`:
- Only the literal `"dev-secret-change-me"` is rejected. `TRADING_SECRET_KEY=""` (or any weak value)
  passes when admin creds differ, and an empty HS256 key lets anyone forge an admin JWT
  (`security.py:49-63`).
- The admin check requires **both** username and password to be default, so
  `TRADING_ADMIN_PASSWORD=admin` with a changed username is accepted.
- `encryption_secret` falls back to `secret_key` (`:78-80`), so by default the JWT signing key and
  the Fernet key that encrypts stored broker credentials are the **same** — contradicting the
  `security.py` docstring and letting one leak compromise both.

### C2. Unauthenticated dashboard path leaks a DB session on generation errors
`trading/api/routers/dashboard.py:151-201` (and `:207-231`): `_load_key` opens a session directly from
the factory, then `await _ensure_generated(...)` can raise (`SignalKeyError`/`DataFetchError`) with
**no try/finally**, so `await session.close()` at `:200` is skipped. Repeated `/API_KEY/{key}/data`
hits while an upstream source is down exhaust the connection pool. `/API_KEY/{key}` is only
capability-gated, so this is reachable by anyone with a key.

### C3. Unbounded in-memory registries (memory-DoS)
Never evicted: `application/signal_keys.py:102` `_summary_cache` (holds full equity curves),
`api/routers/dashboard.py:48` `_generate_locks`, `application/cancellation.py:110` `_seen`,
`api/middleware.py:96` `_LoginLockout._hits` (keyed by client IP), `api/routers/portfolio.py:50`
`_order_idem_locks` (keyed by an unvalidated, caller-supplied header), `api/ws_limits.py:23`
`_counts`. Several are only bounded when `TRADING_TRUST_PROXY` is off, so a spoofed
`X-Forwarded-For` makes them attacker-unbounded.

### C4. `/ws/client` unauthenticated sessions can be held forever
`trading/api/local_client_ws.py:90-92` handles `ping` via `registry.touch`, and
`application/local_client.py:134-138` shows `touch`/`is_stale` do **not** require an active
handshake. A client that never sends a valid handshake can keep the socket (and registry/outbox
entries) alive indefinitely by pinging. Signal integrity is fine (signals require ACTIVE), but the
connection is a resource leak.

### C5. Preset writes race (two live presets / sporadic 500s)
`trading/application/presets.py:188` does `next_version()` then `create()`; `preset_repository.py:179-191`
does `max(version)+1` non-atomically, and the unique index makes one concurrent insert raise an
uncaught `IntegrityError` → 500. `presets.py:414-416` (`demote_live` then `set_status`) can interleave
two `promote` calls and leave **two** `live_enabled` rows for one `(symbol, strategy)`, breaking the
invariant the live path relies on.

### C6. Unbounded request inputs enable authenticated DoS
`api/schemas.py:700` (`GlobalOptimizeRequest.symbols`, no `max_length`) and `:343`
(`PortfolioBacktestRequest.tickers`, no cap — only `n_tickers` is capped at 50) let one request queue
hours of fetches + grid searches. `global_optimize` also accumulates results unbounded.

### C7. Signal-key capability secret is written to access logs
`dashboard.py:131,284` puts the `sk_` key in the URL path; uvicorn's default access log records the
full path, so every dashboard load/refresh persists the credential to logs. This mirrors the same
leak the WS design deliberately avoided for JWTs (`deps.py:15-18`).

### C8. Orphaned rows / no referential integrity
`application/signal_keys.py:369-383` (`delete_signals`) deletes only `KeySignalRow`, leaving
`KeyTradeRow` rows orphaned; there are still no `ForeignKey`s on `key_signals`/`key_trades`/
`signal_positions` (`adapters/persistence/models.py`). A prior audit flagged this (A9) and it is
still open.

### C9. Order routing ignores per-account scope, `enabled`, and risk profile
`api/routers/portfolio.py:74-90` uses `svc.resolve_credentials(session, exchange)` — the **first**
credential for the exchange — instead of the multi-account router / `resolve_accounts_for_symbol`.
Orders and portfolio reads can hit the wrong account and bypass per-account limits; `enabled=False`
accounts are still used.

### C10. CORS is still a wildcard
`trading/main.py:85` `allow_origins=["*"]`. Bearer tokens (not cookies) blunt classic CSRF, but any
origin can read API/dashboard JSON if it obtains a key. Low-to-medium, but unchanged from the audit.

---

## D. Minor / follow-ups

- `metrics.profit_factor`/`sortino`/`calmar` returning `inf` also flows into API JSON (`Infinity`),
  which is not valid JSON for strict clients.
- BingX `build_query` (`bingx.py:47-50`) uses `str(v)` → `"1e-05"` / `"1.0"` for small/integral
  floats, which can break signatures/parameter parsing for low-priced instruments.
- BingX `place_order` always returns `status=OrderStatus.OPEN` and ignores the exchange response
  (`bingx.py:240-249`); a filled MARKET order is reported OPEN.
- `BingxClient` never uses `server_time()` and sends no `recvWindow` (`bingx.py:74-80`), so host
  clock skew can fail every signed request (including exits).
- `ExecutionEngine.place_multi` (`execution.py:128-142`) catches only `BrokerError`; a `ValueError`
  from sizing aborts the whole fan-out, and raw transport errors (httpx) escape both the retry loop
  and per-account isolation.
- TBANK sandbox flag is inconsistent: `account_router.py:139` uses `extra.get("sandbox", True)`
  while `portfolio.py:88` and `tasks.py:406` use `settings.tbank_sandbox`.
- `routers/portfolio.py:199-201` idempotency is process-local (single-worker only) and only caches on
  success.
- `application/reconcile.py:32` `qty_tol=1e-9` is tight enough that float drift causes persistent
  spurious `to_update` churn.
- Dashboard `_key_guard` (`dashboard.py:93-99`) is redundant (`_load_key` already guards) and is the
  only remaining dead code from the Round-1 audit.
- `BacktestResult.trades` stamps `entry_time == exit_time == bar.timestamp` (`engine.py:187-188`),
  so consumers of the `Trade` rows see a zero holding period.

---

## Suggested fix order

1. **A1, A2, A3, A4, A5, A10** — live-order safety (a wrong order costs real money).
2. **A6, A7, A8, A9** — silent fallbacks / state-machine / reconciliation integrity.
3. **B1, B2, B4** — backtest results that are wrong in a way users will trust and act on.
4. **C1, C2, C3** — auth/DoS hardening (C1 is a genuine prod footgun).
5. **B3, B5–B10, C4–C10, D** — remaining correctness and hygiene.
