# GEX-API — Round 4: Production-Readiness Audit, Remediation & Certification

**Date:** 2026-10-07 · **Root:** `E:\gex api` · **Python:** `.venv` (3.12)
**Scope:** the whole `trading/` bounded context (85 routes, 8 ORM models, 7 migrations,
3,969-line console) plus its `gex/` analytics dependency and research harnesses.
**Method:** direct source read at every cited `file:line`, focused execution of the affected
test suites, `ruff`, offline verifier, and a re-run of the frozen `confluence_breakout`
strategy on the research cache under the corrected engine. Decisions confirmed with the
owner before any irreversible change (see *Decisions*).
**Prior context:** Round-1/2 audits and the WS-1..WS-10 remediation commits are in
`deliverables/software-company/audit/`. This round reviews the **current** tree and fixes
what those rounds left, prioritising risk.

**Working-tree provenance.** On arrival the tree already contained **uncommitted changes
not authored by this round**: the saved-strategy *group delete* feature
(`application/presets.py`, `adapters/persistence/preset_repository.py`,
`api/routers/presets.py`, console `static/index.html`, `tests/test_preset_delete.py`) and
the Round-3 QA harness (`tests/qa_round3_*.py`, `tests/console/`). Those are preserved
untouched; the baseline suite (843 passed) already included them. All edits listed in §3
are this round's; `git diff`/`git status` separates them.

---

## 0. Verdict

**READY WITH CONDITIONS.**

The security perimeter the Round-2 work established is present and correct (router-mount
auth, WS JWT/handshake auth, prod secret validation, dashboard escaping, `/metrics`
gating). This round fixed the **money-risk and correctness defects** the earlier audits
did not reach — the ones that could place a wrong order or report a wrong P&L — added
regression tests, and left the full suite green (**860 passed, 1 skipped**; `ruff` clean).
It is **not** unconditionally production-ready: the conditions in §7 (signature of engine
v2 on the frozen presets, referential-integrity migration, wiring the REST order route
through the risk gate, broker idempotency/cancel/stop-attachment, and the dashboard-key log
leak) must be closed first. No unresolved **blocker** remains in the code that was fixed;
the remaining items are integration/validation work that needs external keys or a product
decision.

---

## 1. Project inventory

| Area | Count / notes |
|---|---|
| API routes (routers + main + WS) | 85 |
| ORM models | 8 (`api_keys`, `orders`, `backtest_results`, `strategy_presets`, `signal_keys`, `key_signals`, `key_trades`, `signal_positions`) |
| Alembic migrations | 7 (`0001`–`0007`) |
| Test files | 168 (`trading/tests` + `tests/`) |
| Frontend | single-file `trading/static/index.html` (3,969 lines) |
| Brokers | BingX (REST V3 + WS, HMAC-SHA256), TBANK (tinkoff-invest SDK + stream) |
| Fetchers | MOEX, yFinance, Bybit, Webull(+synthetic test-only) |
| Background | Celery tasks + beat; Redis cancellation registry; in-memory rate limiter |
| Charts/reporting | server-side SVG (`charts.py`), HTML/PDF/CSV/XLSX exports |

**Architecture:** hexagonal/clean within `trading/` — `domain/` (pure), `ports/`,
`adapters/`, `application/`, `api/`; `gex/` is a read-only analytics dependency reached
only through ports. Layer separation is mostly respected; the notable leak is that
`ExecutionEngine` (application) is bypassed by the REST `/orders` route, which calls broker
adapters directly (finding C9).

---

## 2. Decisions locked with the owner

| # | Decision | Chosen |
|---|---|---|
| 1 | Scope/depth | **Risk-first deep fixes** + documented backlog for UI/a11y/connectors |
| 2 | Backtest correctness | **Fix + version the engine and re-run** (accept changed metrics) |
| 3 | Database integrity | **Non-destructive FK + cleanup** (report orphans, no silent delete) |
| 4 | Production safety | **Fail closed in prod** (enforce risk; require strong, distinct secrets) |

---

## 3. Fixes applied (verified)

### 3.1 Trading correctness — backtest engine v2

Added `ENGINE_VERSION = "2.0.0"` (`trading/application/backtest/engine.py`) and surfaced
it on `BacktestResult.engine_version`, so results produced under different rules are
identifiable.

| ID | Defect | Fix (file) | Test |
|---|---|---|---|
| B1 | Exit orders were re-sized from current equity, ignoring the held quantity → position-managed strategies left a residual (and after a loss, a flipped) position while believing they were flat | `domain/orders.py` `Signal.reduce_only` (opt-in) + `engine.py` `_size_signal`: a `reduce_only` signal on an open position trades the **held** quantity; plain signals keep the documented target-position/flip contract. Set on the exits of `confluence_breakout`, `trend_confluence_unified` and `gex_emf` | `test_backtest_engine.py::test_reduce_only_signal_closes_the_held_quantity`; `test_trade_log.py` (flip contract preserved) |
| B2 | `block_bootstrap` removed the historical mean and never re-added it → every path driftless | `simulators.py` re-adds the mean before compounding | `test_simulators.py::test_block_bootstrap_preserves_historical_drift` |
| B3 | GBM Monte-Carlo subtracted the Itô correction twice, biasing paths low | `monte_carlo.py` inverts the correction (`mu + ½σ²`) | `test_engine_v2_regressions.py::test_gbm_drift_matches_the_historical_log_mean` |
| B4 | Optimizer rated `+inf` (perfect profit-factor/Sortino/Calmar) as the **worst** score, disqualifying its best candidates | `optimize.py` `metric_value`: `+inf→+1e18`, `-inf/nan→-1e18` | `test_engine_v2_regressions.py::test_optimizer_metric_value_treats_infinite_objective_as_best` |
| B5 | Sortino semi-deviation averaged only losing observations | `metrics.py` uses `min(r,0)²` over **all** periods | `test_metrics.py::test_sharpe_and_sortino_match_reference`, `::test_sortino_all_positive_returns_is_infinite` |
| WS0 | Broken suite test (un-awaited `client.delete`) | `test_preset_delete.py:100` `await` added | full suite |

### 3.2 Live-order safety

| ID | Defect | Fix (file) | Test |
|---|---|---|---|
| A1 | Unsupported/stop order types silently became **MARKET** on BingX (a protective stop fills immediately) | `bingx.py` `place_order` maps known types and **raises** on anything else; `STOP→STOP_MARKET`, `STOP_LIMIT→STOP`; `TRAILING_STOP` refused | `test_bingx.py::test_place_order_refuses_to_downgrade_unsupported_type` |
| A2/A3 | `KillSwitch` and `RiskManager.approve` were never called → drawdown/position limits decorative | `execution.py` `ExecutionEngine` now runs a risk gate on `place_order`/`place_multi`; **fail-closed in production** (`enforce_risk` defaults to `settings.is_production`), refusing orders when equity/mark are unknown. New `RiskLimitError` | `test_execution.py::test_risk_gate_*` |
| A5 | `STOP_MARKET`/`TRAILING_STOP` intents accepted with no stop price | `domain/orders.py` `OrderIntent.__post_init__` requires `stop_price` for all stop types | `test_domain.py` (existing) + validation path |
| A8 | `Order.apply_fill` mutated `filled_quantity` before validating (stranding over-fills) and bypassed the state machine | `domain/orders.py`: validate → advance `PENDING→OPEN` → transition to `PARTIAL/FILLED`; symbol/side checks | `test_domain.py::test_order_overfill_does_not_corrupt_filled_quantity`, `::test_order_fill_requires_matching_symbol_and_side`, `::test_order_fill_from_pending_advances_through_open` |
| A6 | TBANK silently dry-ran when a token was set but the SDK was missing | `tbank.py` **raises** in production instead of masking the outage | `test_tbank.py` (dev path unchanged) |

### 3.3 Security / availability

| ID | Defect | Fix (file) | Test |
|---|---|---|---|
| C1 | Prod accepted empty/weak `TRADING_SECRET_KEY`, a default/short admin password, and a missing/shared Fernet key | `config.py` `__post_init__` fails closed on all four | `test_security_hardening.py::test_production_rejects_*` |
| C2 | `/API_KEY/{key}/data|/charts|/trades.*` leaked a pooled DB session on any generation error (connection-pool DoS) | `dashboard.py` wraps each handler in `try/finally: await session.close()` | `test_api.py` / dashboard tests |
| C3 | Unbounded in-memory registries: `_summary_cache`, `_generate_locks`, login-lockout `_hits`, `_order_idem_locks` | bounded LRU / size caps with unlocked-only eviction | `test_api_middleware.py`, `test_signal_keys.py` |
| C4 | An unauthenticated `/ws/client` session could stay alive forever by pinging | `local_client_ws.py` only refreshes liveness for an **ACTIVE** (handshaked) session; otherwise the watchdog closes it | `test_local_client_ws.py` / `test_ws.py` |
| C8 | `delete_signals` orphaned `key_trades` rows | `signal_keys.py` purges child trades in the same transaction (keep the documented signal-count return) | `test_signal_keys.py` |

**Automated checks:** full suite **860 passed, 1 skipped** (`pytest trading/tests -q`); `ruff check trading`
→ *All checks passed*; `scripts/verify_backtest.py` runs clean; engine v2 re-run of
`confluence_breakout` on the research cache succeeds (numbers in §6). Baseline before this
round (same tree, before the fixes) was **1 failed, 843 passed, 1 skipped** — the one failure
was the broken `test_preset_delete` test, now fixed and green.

---

## 4. Reproduced defects (root cause → impact)

The full evidence list (with file:line) is in
`deliverables/software-company/audit/current-critical-problems.md`. Highest-severity
reproductions:

1. **BingX stop → market.** `place_order` used `.get(type, "MARKET")`; `OrderType.STOP`/`TRAILING_STOP`
   are not keys, so a stop order was transmitted as MARKET. Fixed by raising on unknown types.
2. **Kill switch dead.** Grep proved `KillSwitch.update` and `RiskManager.approve` had zero
   non-test call sites; a live engine could keep trading through a 25%+ drawdown. Fixed by the
   `ExecutionEngine` risk gate.
3. **Exit sizing.** `_size_signal` sized exits from equity; with `position_fraction<1` or fees,
   `qty_exit ≠ held`, leaving a residual (and after a loss, a flipped) position. Proven by a
   controlled unit test; fixed via the opt-in `Signal.reduce_only` (the engine's documented
   flip contract for plain signals is preserved — `test_trade_log.py`).
4. **Block-bootstrap drift.** `residuals = r - r.mean()` with no re-add; a profitable strategy's
   Monte-Carlo distribution was centred on zero. Proven by a mean-preservation test.
5. **Optimizer self-sabotage.** `+inf` objective → `-1e18` → scored as "no trades". Fixed.

---

## 5. Database & connectors

**Database**
- Migrations `0001`–`0007` apply forward/backward (Round-1 dry-run PASS); `run_migrations()`
  is called at boot (`main.py:56`) and stamps `create_all`-managed DBs.
- **Still open (condition):** no `FOREIGN KEY`s on `key_signals`/`key_trades`/`signal_positions`;
  a non-destructive migration (`0008`) with `SET NULL`/report-only orphan handling was the chosen
  posture but is **not yet written** (see §7-2). App-level child purge was added (`delete_signals`).
- Preset write races (C5: concurrent `next_version`/`promote`) are unfixed — needs a transaction
  or DB constraint (see §7).

**Connectors**
- BingX signing is HMAC-SHA256 and never logs secrets, but: no `recvWindow`/server-time offset
  (clock skew fails signed calls), `cancel_order` is unimplementable via the adapter and is
  pointlessly retried, `positionSide` is never sent (hedge-mode risk), and there is **no broker
  idempotency key** so a retry after an ambiguous timeout can duplicate a live order.
- TBANK: sandbox flag is inconsistent between paths (`account_router.py` vs `portfolio.py`/
  `tasks.py`); silent dry-run now fails closed in production.
- Fetchers: bounded windows, TTL cache, loop-scoped clients; `webull` still unregistered while
  advertised.

---

## 6. Re-run evidence (engine v2)

`ENGINE_VERSION=2.0.0`, `confluence_breakout` / `alligator_4h` on `quant/cache` (README repro):

```
sym     trades      ret%   maxDD%      PF    win%  sortino
BTC         64      0.10     3.18   1.998    29.7    1.257
ETH         58      0.08     3.39   1.690    36.2    1.141
XRP         47      0.08     4.75   1.844    38.3    0.882
LTC         58     -0.08     8.21   0.338    25.9   -1.317
ADA         48      0.09     2.80   1.887    39.6    1.186
DOGE        45      0.16     5.11   2.634    35.6    1.391
SOL         62      0.06     4.89   1.542    32.3    0.788
```

These differ from the README's frozen "validated" table (maxDD 2.7%, PF 1.38 OOS, win 31.5%).
That is expected: the numbers were produced under the size/statistics rules just corrected,
so **the frozen presets must be re-validated** before they are quoted again (condition §7-1).

---

## 7. Conditions / deferred work (READY **WITH CONDITIONS**)

| # | Condition | Why deferred | Verification |
|---|---|---|---|
| 1 | Re-validate the frozen `confluence_breakout` presets on engine v2 and update README/`docs/UNIFIED_STRATEGY.md` | **DONE** — calendar walk-forward on 21 real USDT perps (v2); **both presets NOT ROBUST** (3/5 positive windows; mean breadth 48%/33%). README corrected. | `deliverables/software-company/audit/walkforward/REPORT.md` + `results.json` (engine `2.0.0`); no-lookahead = identical, best-trade removal = removed |
| 2 | Add migration `0008` for non-destructive FKs + orphan report | SQLite batch-FK migration is high-risk to land without a DB host to dry-run | `alembic upgrade/downgrade` + `alembic check` clean; orphan report |
| 3 | Route REST `POST /orders` through the `ExecutionEngine` risk gate and define a persistent peak-equity source (C9) | Needs a product decision on where live equity comes from; currently only the engine path is gated | Order rejected when equity unknown; integration test |
| 4 | Broker hardening: `clientOrderID` idempotency (A4), BingX `cancel_order` (A10), attach stop-loss to entry (A7), `positionSide` (C6), server-time/`recvWindow` (H4), reconcile hedge support (A9) | Requires the exchange's exact parameter semantics / a hedge-vs-one-way account decision | Mock-transport tests + live smoke with real keys |
| 5 | Treat the dashboard `sk_` key as a header or short-lived signed token; throttle `/API_KEY/{key}/refresh` (C7) | API-contract change affecting the console/README | 401 without token; access logs contain no key |
| 6 | Input caps/pagination: `GlobalOptimizeRequest.symbols`, portfolio `tickers`, `BacktestRequest.bars` (C6); list pagination | Schema/product decision on limits | 422 on oversize; bounded payloads |
| 7 | UI/accessibility breadth sweep and connector-by-connector certification | Explicitly de-scoped this pass (owner chose risk-first) | Backlog produced; per-area audit |

---

## 8. Acceptance-criteria matrix

| Criterion | Status |
|---|---|
| All critical user journeys pass | ⚠️ Backtest/signals/dashboards verified; **live broker journeys unverified** (no keys) |
| Add/edit/delete/clear controls exist | ⚠️ Backend management actions largely present; **frontend sweep deferred** |
| Pages render / states consistent | ⚠️ Not fully re-verified this round |
| Forms validate client + server | ✅ Server-side validated; frontend sweep deferred |
| API endpoints predictable | ✅ Auth/error semantics verified; risk gate added on engine path |
| Connectors authenticate/reconnect/limits | ❌ Live broker flows unverified; several hardening items open (§7-4) |
| Migrations apply cleanly | ✅ `0001`–`0007` verified; `0008` pending |
| FKs/constraints/indexes/transactions | ❌ FKs missing (condition §7-2) |
| Rate limits configured/tested | ✅ Per-IP + login bucket/lockout/WS cap (now bounded) |
| AuthN/AuthZ enforced | ✅ Router-mount guards + WS auth verified |
| No secrets exposed | ⚠️ Prod secrets fail closed; dashboard key still URL-logged (§7-5) |
| No known critical vulns | ⚠️ Remaining high items tracked in §7 |
| Logging/monitoring cover critical paths | ⚠️ Metrics wired (Round-2); live-order instrumentation still thin |
| Backups/recovery | ❌ Out of scope / not verified |
| Tests pass | ✅ Full suite green (860 passed, 1 skipped); `ruff` green |
| No unresolved blocker | ✅ (conditions remain) |

---

## 9. Test evidence

- **Full suite: `pytest trading/tests -q` → 860 passed, 1 skipped** (green) after the fixes.
- Baseline before the fixes (same tree): **1 failed, 843 passed, 1 skipped** — the failure was
  the un-awaited `test_preset_delete` test, now fixed.
- Changed-area suites (backtest engine/metrics/simulators/monte-carlo/optimizer, domain,
  security-hardening, bingx, execution, tbank, middleware, dashboard/api, ws, preset-delete,
  strategy family) all green; the flip contract is pinned by `test_trade_log.py`.
- `ruff check trading` → **All checks passed**.
- `scripts/verify_backtest.py` → exit 0; engine v2 re-run table in §6.

*Note: the shipped test count in README/ARCHITECTURE is stale relative to the tree; the count
is a moving target and should be re-pinned from a single frozen run.*

---

## 10. Remaining risks

- **Unverified live trading.** No real BingX/TBANK keys were available; order placement,
  cancel, stops-attached-to-entry, and hedge-mode behaviour remain unproven end-to-end.
- **Engine v2 invalidates prior published metrics.** Until §7-1 is done, any quoted strategy
  performance is suspect.
- **Single-worker assumptions** (rate limiter, idempotency cache, cancellation registry,
  summary cache) are documented but not enforced; multi-worker deployment changes severity.
- **No DB-level referential integrity** until §7-2.
- **Data-scale DoS** via uncapped symbol/ticker inputs (§7-6) for authenticated callers.

## 11. Recommended next improvements

1. A live-broker integration test suite behind a `TRADING_LIVE_TESTS=1` gate (sandbox accounts).
2. Move process-local state (rate buckets, idempotency, cancellation, summaries) to Redis now
   that the app can assume multiple workers.
3. Structured, correlated audit logging for every order lifecycle transition.
4. A CI job that runs `alembic upgrade/downgrade/check` and a `docker build` (Round-1 A13 is
   still only partially closed).
5. Frontend accessibility pass and the missing-control inventory (explicitly deferred here).
