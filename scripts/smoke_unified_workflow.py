"""End-to-end smoke test for the unified strategy workflow against a live server.

Sequence: login -> preset-from-backtest -> optimize(+save_preset) -> presets list
-> global optimize (single symbol) -> create API key -> generate signals
-> dashboard (HTML/data/charts/CSV/XLSX/refresh) -> revoke -> 410 check.

Usage:
    .venv/Scripts/python.exe scripts/smoke_unified_workflow.py
    SMOKE_BASE=http://127.0.0.1:9000 TRADING_ADMIN_USERNAME=admin ... python ...

Environment: SMOKE_BASE (default http://127.0.0.1:8123),
TRADING_ADMIN_USERNAME / TRADING_ADMIN_PASSWORD (both default ``admin``).
Exits 0 only when every step passes; each step prints PASS/FAIL.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("SMOKE_BASE", "http://127.0.0.1:8123")
USERNAME = os.environ.get("TRADING_ADMIN_USERNAME", "admin")
PASSWORD = os.environ.get("TRADING_ADMIN_PASSWORD", "admin")
FAILURES = []


def call(method, path, token=None, body=None, raw=False):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = resp.read()
            if raw:
                return resp.status, payload, dict(resp.headers)
            return resp.status, (json.loads(payload) if payload else None)
    except urllib.error.HTTPError as e:
        payload = e.read()
        if raw:
            return e.code, payload, dict(e.headers)
        try:
            return e.code, json.loads(payload) if payload else None
        except Exception:
            return e.code, payload.decode(errors="replace")


def step(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"[{status}] {name} {('- ' + str(detail)[:300]) if detail and not cond else ''}")


# 0) health
code, _ = call("GET", "/health")
step("health", code == 200, code)

# 1) login
code, body = call("POST", "/api/v1/auth/token", body={"username": USERNAME, "password": PASSWORD})
step("login", code == 200 and body.get("access_token"), code)
TOKEN = body["access_token"]
AUTH = {"authorization": f"Bearer {TOKEN}"}

# 2) preset from backtest (unified strategy, synthetic source)
code, body = call("POST", "/api/v1/presets/from-backtest/SYNTH",
                  body={"strategy": "trend_confluence_unified", "source": "synthetic",
                        "limit": 600, "notes": "smoke test"},
                  token=TOKEN)
ok = code == 200 and body.get("id") and body.get("is_default") is True
step("preset from backtest", ok, (code, body))
preset1 = body.get("id") if ok else None

# 3) optimize unified with save_preset (small grid: 4 combos)
grid = {"emf_mode": ["bonus", "require"], "momentum_mode": ["bonus", "off"]}
code, body = call("POST", "/api/v1/backtest/optimize",
                  body={"strategy": "trend_confluence_unified", "symbol": "SYNTH",
                        "source": "synthetic", "limit": 600, "objective": "sharpe",
                        "save_preset": True, "grid": grid},
                  token=TOKEN)
ok = code == 200 and body.get("best_params") and body.get("n_candidates", 0) >= 1
step("optimize + save preset", ok, (code, str(body)[:300]))
if ok:
    bp = body["best_params"]
    step("  best params carry integration knobs", bp.get("emf_mode") in ("bonus", "require"), bp.get("emf_mode"))

# 3b) saved preset is the COMPLETE unified config (full schema, not deltas)
code, body = call("GET", "/api/v1/presets/default/SYNTH", token=TOKEN)
ok = code == 200
if ok:
    cfgp = body.get("params", {})
    tc_keys = {"ema_fast", "ema_mid", "ema_slow", "use_trendlines", "use_trailing"}
    ok = (
        tc_keys <= set(cfgp)
        and cfgp.get("emf_mode") in ("bonus", "require", "off")
        and isinstance(cfgp.get("emf"), dict) and len(cfgp["emf"]) >= 20
        and "momentum_mode" in cfgp and "momentum_period" in cfgp
    )
    print(f"       preset param count: {len(cfgp)} | emf block: {len(cfgp.get('emf', {}))} keys")
step("saved preset is complete unified config", ok, (code, str(body)[:200]))

# 4) presets list shows SYNTH presets
code, body = call("GET", "/api/v1/presets", token=TOKEN)
step("presets list", code == 200 and any(p["symbol"] == "SYNTH" for p in body), (code, str(body)[:200]))

# 5) default preset for SYNTH resolves
code, body = call("GET", "/api/v1/presets/default/SYNTH", token=TOKEN)
step("default preset for SYNTH", code == 200 and body.get("is_default") is True, (code, str(body)[:200]))

# 6) global optimize run (single symbol, tiny grid)
code, body = call("POST", "/api/v1/backtest/optimize/global",
                  body={"strategy": "trend_confluence_unified", "symbols": ["SYNTH"],
                        "source": "synthetic", "limit": 600, "objective": "sharpe",
                        "grid": grid},
                  token=TOKEN)
ok = code == 200 and body.get("run_id")
step("global optimize start", ok, (code, str(body)[:300]))
if ok:
    run_id = body["run_id"]
    final = None
    for _ in range(120):
        c, st = call("GET", f"/api/v1/backtest/optimize/global/{run_id}", token=TOKEN)
        if c == 200 and st.get("state") in ("done", "cancelled", "error"):
            final = st
            break
        time.sleep(1)
    ok = final and final.get("state") == "done" and final.get("completed") == 1
    step("global optimize done", ok, final)

# 7) create API key (broker = bingx, ticker SYNTH)
code, body = call("POST", "/api/v1/signal-keys",
                  body={"exchange": "bingx", "label": "smoke", "strategy": "trend_confluence_unified",
                        "tickers": ["SYNTH"], "source": "synthetic", "limit": 600},
                  token=TOKEN)
ok = code in (200, 201) and body.get("key", "").startswith("sk_")
step("create API key", ok, (code, str(body)[:300]))
if not ok:
    print("FATAL: cannot continue without a key")
    sys.exit(1)
KEY = body["key"]
KEY_ID = body.get("id")
cfg = json.loads(body.get("config_json", "{}")) if isinstance(body.get("config_json"), str) else body.get("config", {})
print(f"       key={KEY[:12]}... id={KEY_ID}")

# 8) generate signals for the key
code, body = call("POST", f"/api/v1/signal-keys/{KEY}/generate", token=TOKEN)
ok = code == 200
step("generate signals", ok, (code, str(body)[:400]))
if ok:
    step("  generate report has signals/trades counts",
         "n_signals" in str(body) or "signals" in str(body), body)

# 9) dashboard HTML page
code, payload, _ = call("GET", f"/API_KEY/{KEY}", raw=True)
step("dashboard HTML", code == 200 and b"<html" in payload.lower(), code)

# 10) dashboard data JSON (metrics + per-ticker + signals + trades + totals;
# the equity curve is rendered server-side into /charts)
code, body = call("GET", f"/API_KEY/{KEY}/data")
ok = code == 200 and all(
    k in body for k in ("metrics", "per_ticker", "signals", "trades", "totals")
)
step("dashboard data", ok, (code, str(body)[:300]))
if ok and body.get("trades"):
    t0 = body["trades"][0]
    step("  trade record complete",
         all(k in t0 for k in ("entry_time", "exit_time", "entry_price", "exit_price",
                               "direction", "quantity", "net_pnl", "return_pct",
                               "exit_reason")), t0)
    print(f"       trades in /data: {len(body['trades'])}")

# 11) charts
code, payload, _ = call("GET", f"/API_KEY/{KEY}/charts", raw=True)
step("dashboard charts (SVG)", code == 200 and b"<svg" in payload, code)

# 12) CSV export
code, payload, headers = call("GET", f"/API_KEY/{KEY}/trades.csv", raw=True)
ok = code == 200 and b"," in payload and b"entry" in payload.lower()
step("CSV export", ok, (code, payload[:200]))
if ok:
    hdr = payload.split(b"\n")[0].decode()
    print(f"       csv header: {hdr[:160]}")

# 13) XLSX export (SpreadsheetML zip -> PK magic)
code, payload, headers = call("GET", f"/API_KEY/{KEY}/trades.xlsx", raw=True)
ok = code == 200 and payload[:2] == b"PK"
step("XLSX export", ok, (code, payload[:20]))
print(f"       xlsx content-type: {headers.get('Content-Type')}")

# 14) manual refresh
code, body = call("POST", f"/API_KEY/{KEY}/refresh")
step("manual refresh", code == 200 and body.get("generated_at") and "n_signals" in body, (code, str(body)[:200]))

# 15) revoke key, dashboard must return 410
code, body = call("DELETE", f"/api/v1/signal-keys/{KEY_ID}", token=TOKEN)
step("revoke key", code in (200, 204), (code, str(body)[:200]))
code, payload, _ = call("GET", f"/API_KEY/{KEY}", raw=True)
step("revoked key -> 410 Gone", code == 410, code)

# 16) key list shows revoked
code, body = call("GET", "/api/v1/signal-keys", token=TOKEN)
ok = code == 200 and any(k.get("id") == KEY_ID and k.get("revoked_at") for k in body)
step("key audit trail (revoked_at)", ok, (code, str(body)[:300]))

print()
if FAILURES:
    print(f"SMOKE TEST FAILED: {len(FAILURES)} failure(s): {FAILURES}")
    sys.exit(1)
print("SMOKE TEST PASSED: all steps green")
