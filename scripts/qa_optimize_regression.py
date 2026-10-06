"""QA independent regression for the Strategy Lab "Run optimization" hang fix.

Covers (without touching the engineer's own tests):
  B. API behaviour: empty grid -> 400 "empty sweep grid"; >512 combos -> 400
     with planned count; small grid -> 200 with candidates/best_params and
     user's trendline_refresh preserved; planned-combos log line emitted.
  C. Cancel flow: slow sweep + POST /backtest/cancel/{token} -> 499 in time.
  E. Edge cases: trendline_refresh 10/11 honored everywhere; grid=None falls
     back to default (<= 64 combos); two concurrent runs isolated by token.

Run:  python scripts/qa_optimize_regression.py
"""
from __future__ import annotations

import asyncio
import io
import logging
import time

import httpx

from trading.main import app

BASE = "http://test"
PASS: list[str] = []
FAIL: list[str] = []


def report(name: str, ok: bool, evidence: str) -> None:
    (PASS if ok else FAIL).append(f"{name}: {evidence}")
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {evidence}")


async def client() -> httpx.AsyncClient:
    c = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE, timeout=300
    )
    r = await c.post("/api/v1/auth/token",
                     json={"username": "admin", "password": "admin"})
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return c


def _optimize_body(**kw) -> dict:
    body = {
        "strategy": "trend_confluence_pine",
        "symbol": "SYNTH",
        "params": {},
        "objective": "profit_win",
        "timeframe": "1d",
        "limit": 1000,
        "save_preset": False,
    }
    body.update(kw)
    return body


# ── B1: empty grid → 400 ───────────────────────────────────────────────

async def b1_empty_grid(c: httpx.AsyncClient) -> None:
    r = await c.post("/api/v1/backtest/optimize",
                     json=_optimize_body(grid={}))
    detail = r.json().get("detail", "")
    ok = r.status_code == 400 and "empty sweep grid" in str(detail)
    report("B1 empty grid -> 400", ok,
           f"status={r.status_code} detail={detail!r}")


# ── B2: oversized grid → 400 with planned count ────────────────────────

async def b2_runaway_grid(c: httpx.AsyncClient) -> None:
    grid = {f"axis_{i}": [1, 2] for i in range(10)}  # 1024 > 512
    r = await c.post("/api/v1/backtest/optimize",
                     json=_optimize_body(grid=grid))
    detail = str(r.json().get("detail", ""))
    ok = r.status_code == 400 and "1024" in detail and "512" in detail
    report("B2 runaway grid -> 400 w/ planned count", ok,
           f"status={r.status_code} detail={detail!r}")


# ── B3: small grid → 200, user refresh preserved, log line ─────────────

async def b3_small_grid(c: httpx.AsyncClient, logbuf: io.StringIO) -> None:
    body = _optimize_body(
        params={"trendline_refresh": 1},
        grid={"zone_atr": [0.5, 0.8], "min_confluence": [2, 3]},
        run_token="qa-small-1",
    )
    t0 = time.perf_counter()
    r = await c.post("/api/v1/backtest/optimize", json=body)
    dt = time.perf_counter() - t0
    data = r.json() if r.status_code == 200 else {}
    ok = (
        r.status_code == 200
        and data.get("n_candidates") == 4
        and data.get("best_params", {}).get("trendline_refresh") == 1
        and "leaderboard" in data and "best" in data
    )
    report("B3 small grid -> 200, refresh=1 preserved", ok,
           f"status={r.status_code} n_candidates={data.get('n_candidates')} "
           f"best_refresh={data.get('best_params', {}).get('trendline_refresh')} "
           f"wall={dt:.1f}s")
    logtext = logbuf.getvalue()
    planned_ok = ("qa-small" in logtext or "starting: 4 planned" in logtext) and \
        "planned combinations from 2 grid axis(es)" in logtext
    finished_ok = ("finished: 4 candidates" in logtext)
    report("B3b planned/finished log lines", planned_ok and finished_ok,
           f"planned_ok={planned_ok} finished_ok={finished_ok} "
           f"log_head={logtext.strip().splitlines()[-1] if logtext.strip() else '<empty>'!r}")


# ── C: cancel flow → 499 within a reasonable time ──────────────────────

async def c1_cancel(c: httpx.AsyncClient) -> None:
    # A 512-combo grid with dense refresh makes the sweep slow enough (~
    # minutes) that cancelling at ~3 s is well before natural completion.
    grid = {"zone_atr": [0.3, 0.5, 0.8] * 1,
            "min_confluence": [2, 3],
            "stop_atr": [1.5, 2.5],
            "tp_r": [0.0, 2.0],
            "atr_trail_mult": [2.0, 3.0],
            "need_rejection": [True, False]}  # 3*2*2*2*2*2 = 96 combos... need slower
    grid = {"zone_atr": [0.3, 0.5, 0.8],
            "min_confluence": [2, 3],
            "stop_atr": [1.5, 2.5],
            "tp_r": [0.0, 2.0],
            "atr_trail_mult": [2.0, 3.0],
            "need_rejection": [True, False]}  # 96
    # bump to >400 combos but under 512: widen one axis
    grid["zone_atr"] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 6*2*2*2*2*2 = 192
    grid["min_confluence"] = [2, 3]
    grid["tp_percent"] = [1.5, 3.0]  # extra axis x2 = 384 combos (pine strategy ok)
    token = "qa-cancel-1"
    body = _optimize_body(grid=grid, run_token=token, limit=2000)

    async def do_cancel():
        await asyncio.sleep(3)
        t0 = time.perf_counter()
        cr = await c.post(f"/api/v1/backtest/cancel/{token}")
        dt = time.perf_counter() - t0
        report("C1 cancel endpoint accepted", cr.status_code == 200 and
               cr.json().get("known") is True and cr.json().get("cancelled") is True,
               f"status={cr.status_code} body={cr.json()} ({dt:.2f}s)")

    cancel_task = asyncio.create_task(do_cancel())
    t0 = time.perf_counter()
    r = await c.post("/api/v1/backtest/optimize", json=body)
    dt = time.perf_counter() - t0
    await cancel_task
    detail = r.json().get("detail")
    ok = (r.status_code == 499 and dt < 15
          and isinstance(detail, dict) and detail.get("run_token") == token)
    report("C1 sweep cancelled -> 499 promptly", ok,
           f"status={r.status_code} wall={dt:.1f}s detail={detail}")


# ── E1: refresh >= SWEEP_MIN_REFRESH honored everywhere ────────────────

async def e1_refresh_at_floor(c: httpx.AsyncClient) -> None:
    for val in (10, 11):
        r = await c.post("/api/v1/backtest/optimize", json=_optimize_body(
            params={"trendline_refresh": val},
            grid={"zone_atr": [0.5, 0.8]},
            run_token=f"qa-refresh-{val}"))
        data = r.json()
        base = data.get("baseline", {}).get("params", {}).get("trendline_refresh")
        best = data.get("best_params", {}).get("trendline_refresh")
        ok = r.status_code == 200 and base == val and best == val
        report(f"E1 refresh={val} honored everywhere", ok,
               f"baseline_refresh={base} best_refresh={best}")


# ── E2: grid=None → default sweep ≤ 64 combos ──────────────────────────

async def e2_default_grid(c: httpx.AsyncClient, logbuf: io.StringIO) -> None:
    r = await c.post("/api/v1/backtest/optimize",
                     json=_optimize_body(run_token="qa-default-grid"))
    data = r.json()
    n = data.get("n_candidates")
    ok = r.status_code == 200 and isinstance(n, int) and 0 < n <= 64
    report("E2 grid=None -> default sweep <= 64", ok,
           f"status={r.status_code} n_candidates={n}")


# ── E3: two concurrent optimizes isolated by run_token ─────────────────

async def e3_concurrent_isolated(c: httpx.AsyncClient) -> None:
    b1 = _optimize_body(params={"trendline_refresh": 7},
                        grid={"zone_atr": [0.5, 0.8]},
                        run_token="qa-conc-A")
    b2 = _optimize_body(params={"trendline_refresh": 9},
                        grid={"min_confluence": [2, 3]},
                        run_token="qa-conc-B")
    r1, r2 = await asyncio.gather(
        c.post("/api/v1/backtest/optimize", json=b1),
        c.post("/api/v1/backtest/optimize", json=b2),
    )
    d1, d2 = r1.json(), r2.json()
    ok = (r1.status_code == 200 and r2.status_code == 200
          and d1.get("run_token") == "qa-conc-A"
          and d2.get("run_token") == "qa-conc-B"
          and d1.get("best_params", {}).get("trendline_refresh") == 7
          and d2.get("best_params", {}).get("trendline_refresh") == 9
          and d1.get("strategy") == d2.get("strategy"))
    report("E3 concurrent runs token-isolated", ok,
           f"r1={r1.status_code} token={d1.get('run_token')} "
           f"refresh={d1.get('best_params', {}).get('trendline_refresh')}; "
           f"r2={r2.status_code} token={d2.get('run_token')} "
           f"refresh={d2.get('best_params', {}).get('trendline_refresh')}")


async def main() -> None:
    logbuf = io.StringIO()
    handler = logging.StreamHandler(logbuf)
    handler.setLevel(logging.INFO)
    logging.getLogger("trading.api.routers.backtest").addHandler(handler)
    logging.getLogger("trading.api.routers.backtest").setLevel(logging.INFO)

    c = await client()
    try:
        await b1_empty_grid(c)
        await b2_runaway_grid(c)
        await b3_small_grid(c, logbuf)
        await e1_refresh_at_floor(c)
        await e2_default_grid(c, logbuf)
        await e3_concurrent_isolated(c)
        await c1_cancel(c)
    finally:
        await c.aclose()

    print("\n==== SUMMARY ====")
    print(f"PASS: {len(PASS)}  FAIL: {len(FAIL)}")
    for f in FAIL:
        print("FAILED:", f)


if __name__ == "__main__":
    asyncio.run(main())
