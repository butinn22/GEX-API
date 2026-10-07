# GEX-API — Living Remediation Plan & Task Checklist

**Project:** `E:\gex api` (GEX Trading API — FastAPI + async SQLAlchemy + Celery; single-file console `trading/static/index.html`)
**Owner of this artifact:** Delivery Director (齐活林) — coordination/tracking only; all engineering/QA output is owned by the respective team member.
**Created:** 2026-10-07 · **Last updated:** 2026-10-07 12:45 (Round 3 — verification in progress)
**Scope approved by user:** ALL tiers (P0 + P1 + P2).

## Source documents (evidence base)

| Doc | Author | Path |
|---|---|---|
| Code/API-contract audit (24 findings) | Engineer | `deliverables/software-company/audit/engineer-code-audit.md` |
| Product audit (controls/docs) | PM | `deliverables/software-company/audit/product-audit.md` |
| Architecture audit (A1–A22) | Architect | `deliverables/software-company/audit/architecture-audit.md` |
| QA verification Round 1 (739p/1s, cov 93.43%, lint red) | QA | `deliverables/software-company/audit/qa-verification.md` |
| Round-2 implementation design | Architect | `deliverables/software-company/audit/round2-design.md` |
| Round-2 product spec (UX spec, AC-1..AC-14) | PM | `deliverables/software-company/audit/round2-product-spec.md` |
| **Round-3 verification (this round)** | QA | `deliverables/software-company/audit/round3/qa-round3-verification.md` |

---

## Status legend
`DONE-VERIFIED` = committed + independently verified with evidence · `COMMITTED-UNVERIFIED` = code committed, awaiting Round-3 verification · `IN-PROGRESS` = working · `PENDING` = not started · `FAIL` = verification failed, fix required · `BLOCKED` = needs external input.

## Progress snapshot (reconstructed 2026-10-07 12:45)
- Round 1 (discovery): **DONE** — 4 reports delivered.
- Round 2 (implementation): **DONE at commit level** — WS-1..WS-10 all committed (`9ebcd17` → `169aa6c`). Tree clean, no further writes since 12:21:33.
- Round 3 (verification): **IN PROGRESS** — QA agent `agent-b7195e3a` running the 23-item acceptance sweep.

> ⚠️ **Round-2 evidence is NOT trustworthy yet.** The authoring run was unstable: a framework failure (`Tool TaskOutput not found`) left the engineer task marked `failed` while a duplicate execution kept writing the same files, so two writers interleaved in one working tree. Commit messages describe intent, not verified behaviour. **Every WS below stays `COMMITTED-UNVERIFIED` until QA produces execution evidence.**

### Incident log
| # | Time | Event | Resolution |
|---|---|---|---|
| INC-1 | 11:33 | Engineer task aborted: `TaskOutput not found in agent software-engineer` (background command polling on an agent type that has no such tool). | Recovery message sent; **worker briefs must forbid `run_in_background`/`TaskOutput`/`BashOutput`** and require foreground commands with large timeouts. |
| INC-2 | 11:38–12:21 | Duplicate concurrent execution of the *same* engineer task wrote the tree (two "hands"), causing file rewrites/clobbers and a false "second session" diagnosis. Old Round-1 engineer died at 11:29 and is **not** implicated. | Confirmed via scheduler log (only 5 sub-agent sessions ever; single writer process). Execution ran to completion; no writes after `169aa6c` (12:21:33); verified quiet at 12:44. |
| INC-3 | 12:02–12:05 | Safety net created while the tree was in flux. | Salvage refs `wip-salvage-120233` (`2916444`) and `wip-salvage-120553` (`91fc556`); backup `C:\Users\butin\gex-wip-backup-120233`. Keep until Round 3 passes. |

---

## Task checklist

### P0 — Blockers / Criticals

| ID | Prio | Sev | Task | Status | Commit | Acceptance criteria / required evidence |
|---|---|---|---|---|---|---|
| REM-001 | P0 | Blocker | `require_auth` on the previously unguarded routers (presets/backtest/strategies/data) + portfolio GETs | COMMITTED-UNVERIFIED | `8d1da1b` | Anonymous request to each mount returns 401; authz sweep covers the 5 newly-guarded routers |
| REM-002 | P0 | Blocker | Authenticate WebSockets `/ws/*` (JWT subprotocol/`?token=`) and `/ws/client` (handshake frame) | COMMITTED-UNVERIFIED | `8d1da1b` | Unauthorized WS connect rejected; authorized connect succeeds |
| REM-003 | P0 | Critical | Remove/guard shipped default secrets; fail-fast in prod | COMMITTED-UNVERIFIED | `58de425` | Booting prod with `dev-secret-change-me`/`admin` fails fast (real invocation) |
| REM-004 | P0 | Critical | Split JWT signing secret from Fernet broker-key secret | COMMITTED-UNVERIFIED | `58de425` | Two independent keys; rotating one does not affect the other |
| REM-005 | P0 | Critical | Schema alignment: migration 0007 (timestamp NOT NULL), `run_migrations()` at boot, drop request-time `init_db()` | COMMITTED-UNVERIFIED | `9ebcd17` | upgrade→downgrade→upgrade passes; `alembic check` clean |
| REM-006 | P0 | Critical | Stored XSS on anonymous `/API_KEY/{key}` dashboard | COMMITTED-UNVERIFIED | `b61eebb` | `<script>`/`"`/`&` in symbol+strategy render escaped |
| REM-007 | P0 | Critical | `confirm()` before destructive UI: Delete broker account; Revoke signal key | COMMITTED-UNVERIFIED | `e26d848` | Both controls gated by a confirmation; cancel aborts the API call |
| REM-008 | P0 | Critical | Expired-session / 401 handling (stale JWT not trusted) | COMMITTED-UNVERIFIED | `e26d848` | 401 forces re-auth; stale token cannot drive a mutation |

### P1 — High

| ID | Prio | Sev | Task | Status | Commit | Acceptance criteria / required evidence |
|---|---|---|---|---|---|---|
| REM-009 | P1 | High | Rate limiter: login bucket (10/min + lockout), bounded buckets, `TRADING_TRUST_PROXY`, WS cap, correct docs | COMMITTED-UNVERIFIED | `3010863` | Burst → 429 + `Retry-After`; lockout after N failures; registry bounded; XFF honoured only with trust flag |
| REM-010 | P1 | High | Metrics/alerting: `rule_files`/`alerting`, alertmanager service, mount YAMLs, increment counters, schedule `broker_health` | COMMITTED-UNVERIFIED | `dc49313` | YAMLs valid + mounted; 5 counter call-sites live; beat scheduled |
| REM-011 | P1 | High | `UTCDateTime` TypeDecorator (aware-UTC on read, no DDL) | COMMITTED-UNVERIFIED | `553928f` | Round-trip on every configured backend returns tz-aware UTC; no schema change |
| REM-012 | P1 | High | Idempotency key on `POST /orders` | COMMITTED-UNVERIFIED | `b61eebb` | Same key twice → exactly one order |
| REM-013 | P1 | High | Validate `/signals/engine/start` body (`poll_seconds>0`) | COMMITTED-UNVERIFIED | `b61eebb` | `poll_seconds<=0` → 422, no busy loop |
| REM-014 | P1 | High | Load failures: error + Retry for the 5 loaders | COMMITTED-UNVERIFIED | `e26d848` | Each loader surfaces an error state with a working Retry |
| REM-015 | P1 | High | CI lint gate green + install `[dev]` extra | COMMITTED-UNVERIFIED | `9bab401`, `24a5114` | `ruff check trading` → 0; review the curated ignore list for silenced defects |
| REM-016 | P1 | High | CI migration gate: upgrade→downgrade→upgrade→`alembic check` | COMMITTED-UNVERIFIED | `24a5114` | Workflow contains the gate and it passes |

### P2 — Medium / Low

| ID | Prio | Sev | Task | Status | Commit | Acceptance criteria / required evidence |
|---|---|---|---|---|---|---|
| REM-017 | P2 | Medium | `/API_KEY/{key}` wrong/revoked → HTML error page | COMMITTED-UNVERIFIED | `b61eebb` | Invalid + revoked key → HTML, not raw JSON |
| REM-018 | P2 | Medium | httpx client leaks (`data.py:42`; Celery never `aclose`) | COMMITTED-UNVERIFIED | `b61eebb` | No per-request client; Celery tasks close clients |
| REM-019 | P2 | Medium | Management actions: backtest_results purge; per-order cancel/delete; `key_signals` DELETE; `_summary_cache` purge | COMMITTED-UNVERIFIED | `b61eebb` | Each endpoint exists, is authorised, and has a test |
| REM-020 | P2 | Medium | Docker: non-root, HEALTHCHECK, include `gex/` CSVs | COMMITTED-UNVERIFIED | `24a5114` | Dockerfile satisfies all three |
| REM-021 | P2 | Medium | Docs corrections (test-count re-pin, 15s→60s, autotune path, Redis claim, ARCHITECTURE §3) | COMMITTED-UNVERIFIED | `c3c5d2c`, `169aa6c` | Every doc claim matches an independently re-derived value |
| REM-022 | P2 | Low | Broker credential validation UX (blank secret, tbank `account_id`) | COMMITTED-UNVERIFIED | `58de425` | Blank/invalid credentials rejected with a clear error |

### Verification

| ID | Prio | Sev | Task | Status | Evidence |
|---|---|---|---|---|---|
| REM-023 | P0 | Blocker | Round-3 QA verification: suite green, ruff clean, migration round-trip + `alembic check`, anonymous-mutation sweep → 401, XSS, controls, AC-1..AC-14 | IN-PROGRESS (`agent-b7195e3a`) | `audit/round3/qa-round3-verification.md` |

---

## Notes / decisions carried forward
- **Auth application point:** at `include_router` mounts in `main.py` (unfootgettable), not middleware/per-route. `auth.router` public; `dashboard.router` key-gated.
- **Deliberate exception:** `POST /backtest/cancel/{token}` stays public (capability token; `navigator.sendBeacon` cannot set headers).
- **Test count is a moving target** — re-pinned to 834 (833 passed, 1 skipped) at `169aa6c`; QA must re-derive it independently.
- **Guarded test files** (must send credentials): 16 HTTP + 2 WS test files; keep `trading/tests/test_authz_regression.py` sweeping the 5 newly-guarded routers.
- **Worker tool constraint (from INC-1):** no `run_in_background` / `TaskOutput` / `BashOutput`; foreground commands with explicit large timeouts.

## Known verified-good (Round-1 baseline — must stay green)
Suite 739p/1s baseline; coverage 93.43% (gate 85%); migrations forward+backward PASS; dynamic probes pass; rate-limit burst → 429 + `Retry-After`.

## Blockers
_None._ Round-3 verification is the critical path; any FAIL routes to the engineer with the QA repro.
