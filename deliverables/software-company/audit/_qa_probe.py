"""QA dynamic probe harness for the GEX-API audit (offline, deterministic).

Hits the running server on :8199 and records status codes + body excerpts.
Usage:  venv/Scripts/python.exe deliverables/software-company/audit/_qa_probe.py
Writes JSON results next to this file (_qa_probe_results.json).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import httpx

BASE = os.environ.get("QA_BASE", "http://127.0.0.1:8199")


def synth_bars(n: int = 180, seed: float = 0.13) -> list[dict]:
    """Deterministic GBM bars so the backtest never touches the network."""
    bars, price, t0 = [], 100.0, datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        drift = 0.0004 * math.sin(i / 11.0)
        shock = seed * math.sin(i / 3.7) + 0.4 * math.sin(i / 1.3)
        o = price
        c = max(0.5, price * (1 + drift + 0.01 * shock))
        h = max(o, c) * (1 + 0.003 * abs(math.sin(i / 5.0)))
        lo = min(o, c) * (1 - 0.003 * abs(math.sin(i / 7.0)))
        bars.append({"timestamp": (t0 + timedelta(days=i)).isoformat(),
                     "open": round(o, 4), "high": round(h, 4),
                     "low": round(lo, 4), "close": round(c, 4), "volume": 1000.0})
        price = c
    return bars


RESULTS: list[dict] = []


def rec(name, method, path, status, excerpt="", expected=None, verdict=None, note=""):
    RESULTS.append({
        "probe": name, "method": method, "path": path, "expected": expected,
        "actual": status, "body": excerpt[:220], "verdict": verdict, "note": note,
    })
    print(f"[{status}] {method:6} {path:52} {name}  {verdict or ''}")


def main() -> None:
    c = httpx.Client(base_url=BASE, timeout=120.0)
    token = None

    # ── 1. basic endpoints ────────────────────────────────────────────
    for name, method, path, exp_ok, ctype in [
        ("health", "GET", "/health", {200}, "json"),
        ("metrics", "GET", "/metrics", {200}, "text"),
        ("docs", "GET", "/docs", {200}, "html"),
        ("index", "GET", "/", {200}, "html"),
        ("favicon", "GET", "/favicon.ico", {200}, "image"),
        ("static-index", "GET", "/static/index.html", {200}, "html"),
    ]:
        r = c.request(method, path)
        ok = r.status_code in exp_ok
        rec(name, method, path, r.status_code, r.text[:200],
            ",".join(map(str, exp_ok)), "PASS" if ok else "FAIL")

    # ── 2. auth token ─────────────────────────────────────────────────
    r = c.post("/api/v1/auth/token", json={"username": "admin", "password": "admin"})
    ok = r.status_code == 200 and "access_token" in r.json()
    if ok:
        token = r.json()["access_token"]
    rec("login-correct", "POST", "/api/v1/auth/token", r.status_code,
        r.text[:120], "200+token", "PASS" if ok else "FAIL")

    r = c.post("/api/v1/auth/token", json={"username": "admin", "password": "wrong"})
    rec("login-wrong", "POST", "/api/v1/auth/token", r.status_code, r.text[:120],
        "401", "PASS" if r.status_code == 401 else "FAIL")

    r = c.post("/api/v1/auth/token", json={"username": "nobody", "password": "x"})
    rec("login-baduser", "POST", "/api/v1/auth/token", r.status_code, r.text[:120],
        "401", "PASS" if r.status_code == 401 else "FAIL")

    # ── 3. protected endpoint with / without token ────────────────────
    r = c.get("/api/v1/keys")
    rec("keys-no-token", "GET", "/api/v1/keys", r.status_code, r.text[:120],
        "401/403", "PASS" if r.status_code in (401, 403) else "FAIL")

    r = c.get("/api/v1/keys", headers={"Authorization": f"Bearer {token}"})
    rec("keys-with-token", "GET", "/api/v1/keys", r.status_code, r.text[:120],
        "200", "PASS" if r.status_code == 200 else "FAIL")

    r = c.get("/api/v1/keys", headers={"Authorization": "Bearer garbage.token.here"})
    rec("keys-bad-token", "GET", "/api/v1/keys", r.status_code, r.text[:120],
        "401", "PASS" if r.status_code in (401, 403) else "FAIL")

    # ── 4. protected signals endpoints ────────────────────────────────
    auth = {"Authorization": f"Bearer {token}"}
    r = c.get("/api/v1/signals", headers=auth)
    rec("signals-with-token", "GET", "/api/v1/signals", r.status_code, r.text[:120],
        "200", "PASS" if r.status_code == 200 else "FAIL")
    r = c.get("/api/v1/signals")
    rec("signals-no-token", "GET", "/api/v1/signals", r.status_code, r.text[:120],
        "401", "PASS" if r.status_code in (401, 403) else "FAIL")

    # ── 5. real backtest (offline synthetic bars) ─────────────────────
    bars = synth_bars()
    r = c.post("/api/v1/backtest", json={"strategy": "sma_crossover", "symbol": "SYNTH",
                                          "bars": bars}, headers=auth)
    excerpt = r.text[:200]
    ok = r.status_code == 200
    rid = r.json().get("result_id") if ok else None
    rec("backtest-real", "POST", "/api/v1/backtest", r.status_code, excerpt,
        "200", "PASS" if ok else "FAIL",
        note=("result_id=%s" % rid) if ok else "")
    r = c.post("/api/v1/backtest", json={"strategy": "sma_crossover", "symbol": "SYNTH",
                                          "bars": bars})
    rec("backtest-NO-token", "POST", "/api/v1/backtest", r.status_code, r.text[:120],
        "200 (UNAUTH!)", "UNAUTHENTICATED-MUTATION" if r.status_code == 200 else "protected")

    # ── 6. exports ────────────────────────────────────────────────────
    for name, path in [
        ("export-signals-csv", "/api/v1/signals/export/signals.csv"),
        ("export-signals-xlsx", "/api/v1/signals/export/signals.xlsx"),
        ("export-positions-csv", "/api/v1/signals/export/positions.csv"),
        ("export-positions-xlsx", "/api/v1/signals/export/positions.xlsx"),
    ]:
        r = c.get(path, headers=auth)
        nbytes = len(r.content)
        rec(name, "GET", path, r.status_code, f"{nbytes} bytes, ct={r.headers.get('content-type')}",
            "200 non-trivial", "PASS" if r.status_code == 200 and nbytes > 0 else "FAIL",
            note=f"download: {r.headers.get('content-disposition')}")

    # ── 7. dashboard with bogus key ───────────────────────────────────
    r = c.get("/API_KEY/does-not-exist-123")
    rec("dashboard-bogus-key", "GET", "/API_KEY/does-not-exist-123", r.status_code,
        r.text[:120], "404", "PASS" if r.status_code == 404 else "FAIL")
    r = c.get("/API_KEY/does-not-exist-123/data")
    rec("dashboard-bogus-data", "GET", "/API_KEY/does-not-exist-123/data", r.status_code,
        r.text[:120], "404", "PASS" if r.status_code == 404 else "FAIL")

    # ── 8. AUTH ENFORCEMENT SWEEP (no token) ──────────────────────────
    sweep = [
        ("POST", "/api/v1/backtest"),
        ("POST", "/api/v1/backtest/monte-carlo"),
        ("POST", "/api/v1/backtest/portfolio"),
        ("POST", "/api/v1/backtest/portfolio/monte-carlo"),
        ("POST", "/api/v1/backtest/portfolio/report"),
        ("POST", "/api/v1/backtest/analyze"),
        ("POST", "/api/v1/backtest/optimize"),
        ("POST", "/api/v1/backtest/optimize/global"),
        ("POST", "/api/v1/backtest/optimize/global/abc/cancel"),
        ("POST", "/api/v1/backtest/autotune"),
        ("POST", "/api/v1/backtest/cancel/abc"),
        ("GET", "/api/v1/backtest/cancel"),
        ("POST", "/api/v1/presets"),
        ("POST", "/api/v1/presets/from-backtest/SYNTH"),
        ("POST", "/api/v1/presets/1/promote"),
        ("POST", "/api/v1/presets/1/rollback"),
        ("POST", "/api/v1/presets/1/demote"),
        ("PATCH", "/api/v1/presets/1"),
        ("DELETE", "/api/v1/presets/999999"),
        ("POST", "/api/v1/strategies/sma_crossover/start"),
        ("POST", "/api/v1/strategies/sma_crossover/stop"),
        ("GET", "/api/v1/strategies"),
        ("GET", "/api/v1/data/instruments"),
        ("GET", "/api/v1/portfolio"),
        ("GET", "/api/v1/positions"),
        ("GET", "/api/v1/orders"),
        ("POST", "/api/v1/orders"),
        ("POST", "/api/v1/keys"),
        ("DELETE", "/api/v1/keys/999999"),
        ("PATCH", "/api/v1/keys/999999/settings"),
        ("GET", "/api/v1/keys"),
        ("POST", "/api/v1/signal-keys"),
        ("POST", "/api/v1/baskets/export"),
        ("POST", "/api/v1/baskets/deploy"),
        ("POST", "/api/v1/signals/engine/start"),
        ("POST", "/api/v1/signals/engine/stop"),
        ("DELETE", "/api/v1/signals/positions"),
        ("GET", "/api/v1/signals"),
        ("GET", "/api/v1/export/live-trades.csv"),
    ]
    print("\n--- AUTH ENFORCEMENT SWEEP (no Authorization header) ---")
    for method, path in sweep:
        r = c.request(method, path, json={})
        prot = r.status_code in (401, 403)
        # 422/400/404/200/500 => request reached the handler => NOT auth-gated
        mutating = method in ("POST", "PATCH", "DELETE")
        if prot:
            verdict = "PROTECTED"
        else:
            verdict = "UNAUTH-MUTATION" if mutating else "public-read"
        rec(f"sweep {method} {path}", method, path, r.status_code, r.text[:120],
            "401/403" if mutating else "-", verdict)

    # ── 9. rate limiting (burst; configured limit is 300/min) ─────────
    print("\n--- RATE LIMIT BURST (configured=300/min) ---")
    got_429, retry_after, first_429_at = 0, None, None
    for i in range(420):
        rr = c.get("/health")
        if rr.status_code == 429:
            got_429 += 1
            if first_429_at is None:
                first_429_at = i
                retry_after = rr.headers.get("Retry-After")
    rec("rate-limit-burst", "GET", "/health x420", 429 if got_429 else 200,
        f"429_count={got_429} first_at={first_429_at} Retry-After={retry_after}",
        "429 + Retry-After",
        "PASS" if got_429 and retry_after else "FAIL")

    c.close()
    with open(os.path.join(os.path.dirname(__file__), "_qa_probe_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("\nSaved -> _qa_probe_results.json")


if __name__ == "__main__":
    main()
