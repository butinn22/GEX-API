"""The ``/API_KEY/{key}`` live dashboard.

The key in the URL **is the credential** (unguessable, revocable — this is a
local/personal deployment; the admin console that creates keys is
JWT-protected). The page reuses the backtest module's server-side SVG charts
(``charts.py``) so the live view looks exactly like the backtest report:
equity curve, underwater drawdown and a per-trade PnL histogram, plus the
metrics summary, the recent signals, the full trade table, CSV/Excel exports
and a Refresh button with the last-update time (the page also auto-refreshes).
"""
from __future__ import annotations

import asyncio
import html
import itertools
import json
import math
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from trading.application.backtest.charts import (
    DARK,
    drawdown_chart,
    histogram_chart,
    line_chart,
)
from trading.application.reporting.trade_export import table_to_csv, table_to_xlsx
from trading.application.signal_keys import SignalKeyError, SignalKeyService

__all__ = ["router"]

router = APIRouter(tags=["dashboard"])

#: Full column set of the trade exports (requirement §12).
TRADE_EXPORT_COLUMNS = [
    "trade_id", "api_key", "broker", "ticker", "strategy", "strategy_version",
    "preset_id", "entry_time", "exit_time", "direction",
    "entry_price", "exit_price", "quantity", "fee",
    "gross_pnl", "net_pnl", "return_pct", "holding_hours", "exit_reason",
    "source",
]

#: One generation at a time per key (auto-refresh + manual click can race).
_generate_locks: dict[int, asyncio.Lock] = {}


def _session_factory_value():
    from trading.adapters.persistence import database

    if database._session_factory is None:
        database.configure()
    return database._session_factory


async def _load_key(key: str):
    """Load the key row + service + open session; guard lifecycle up front.

    The session is closed before any guard raises so a 404/410 never leaks a
    pooled connection.
    """
    factory = _session_factory_value()
    session = factory()
    svc = SignalKeyService(session)
    row = await svc.get_by_key(key)
    if row is None:
        await session.close()
        raise HTTPException(404, "unknown API key")
    if row.revoked_at is not None:
        await session.close()
        raise HTTPException(410, "this API key has been revoked")
    if not row.active:
        await session.close()
        raise HTTPException(410, "this API key is disabled")
    return row, svc, session


async def _ensure_generated(key: str, row, svc) -> None:
    """Regenerate when the in-memory summary is cold (e.g. after a restart)."""
    if svc.summary(row.id) is not None:
        return
    lock = _generate_locks.setdefault(row.id, asyncio.Lock())
    if lock.locked():  # another request is already regenerating
        return
    async with lock:
        if svc.summary(row.id) is None:
            await svc.generate(key=row, refresh=False)


def _key_guard(row) -> None:  # kept for symmetry; _load_key already guards
    if row is None:
        raise HTTPException(404, "unknown API key")
    if row.revoked_at is not None:
        raise HTTPException(410, "this API key has been revoked")
    if not row.active:
        raise HTTPException(410, "this API key is disabled")


def _downsample(times: list[str], values: list[float], cap: int = 400) -> tuple[list[str], list[float]]:
    n = len(values)
    if n <= cap:
        return times, values
    step = (n - 1) / (cap - 1)
    idx = sorted({round(i * step) for i in range(cap)})
    return [times[i] for i in idx], [values[i] for i in idx]


def _drawdown(equity: list[float]) -> list[float]:
    arr = np.asarray(equity, dtype=float)
    if len(arr) == 0:
        return []
    peak = np.maximum.accumulate(arr)
    return list(arr / peak - 1.0)


def _pnl_histogram(pnls: list[float], bins: int = 20):
    arr = np.asarray([p for p in pnls if math.isfinite(p)], dtype=float)
    if arr.size == 0:
        return [], []
    counts, edges = np.histogram(arr, bins=bins)
    centers = [(a + b) / 2 for a, b in itertools.pairwise(edges)]
    return [int(c) for c in counts], centers


# ── HTML page ──────────────────────────────────────────────────────────


@router.get("/API_KEY/{key}", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(key: str) -> HTMLResponse:
    # The key path segment is user-controlled (anybody can GET any key), so an
    # unknown/revoked key must render an HTML page, not a raw JSON 4xx body.
    try:
        row, _svc, session = await _load_key(key)
    except HTTPException as exc:
        return HTMLResponse(_error_page(str(exc.detail)), status_code=exc.status_code)
    _key_guard(row)
    await session.close()
    config = json.loads(row.config_json or "{}")
    tickers = ", ".join(t.get("symbol", "?") for t in config.get("tickers", []))
    strategy = config.get("strategy", "")
    return HTMLResponse(_page(key=key, exchange=row.exchange.upper(),
                              tickers=tickers, strategy=strategy))


# ── data (summary + signals + trades + last update) ───────────────────


@router.get("/API_KEY/{key}/data", include_in_schema=False)
async def dashboard_data(key: str) -> JSONResponse:
    row, svc, session = await _load_key(key)
    _key_guard(row)
    await _ensure_generated(key, row, svc)
    summary = svc.summary(row.id)
    trades = await svc.trades(row.id)
    signals = await svc.signals(row.id, limit=50)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gross = sum(t.gross_pnl for t in trades)
    data = {
        "key": row.key,
        "exchange": row.exchange,
        "active": bool(row.active),
        "last_updated": (summary.generated_at.isoformat() if summary else None),
        "metrics": (summary.metrics if summary else {}),
        "per_ticker": (summary.per_ticker if summary else []),
        "generation_errors": (summary.errors if summary else []),
        "totals": {
            "n_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
            "total_pnl": round(sum(t.net_pnl for t in trades), 2),
            "avg_pnl": round(sum(t.net_pnl for t in trades) / len(trades), 2) if trades else 0.0,
            "gross_pnl": round(gross, 2),
            "last_trade_at": trades[-1].exit_time.isoformat() if trades else None,
        },
        "trades": [
            {
                "id": t.id, "symbol": t.symbol, "direction": t.direction,
                "entry_time": t.entry_time.isoformat(), "exit_time": t.exit_time.isoformat(),
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "quantity": t.quantity, "net_pnl": round(t.net_pnl, 2),
                "return_pct": round(t.pct_return, 4),
                "exit_reason": t.exit_reason, "source": t.source,
            }
            for t in trades[-25:]
        ],
        "signals": [
            {
                "timestamp": s.timestamp.isoformat(), "symbol": s.symbol,
                "side": s.side, "state": s.state, "reason": s.reason,
                "price": s.price, "strategy": s.strategy,
            }
            for s in signals
        ],
    }
    await session.close()
    return JSONResponse(data)


# ── charts (same SVG functions as the backtest report) ─────────────────


@router.get("/API_KEY/{key}/charts", include_in_schema=False)
async def dashboard_charts(key: str) -> JSONResponse:
    row, svc, session = await _load_key(key)
    _key_guard(row)
    await _ensure_generated(key, row, svc)
    summary = svc.summary(row.id)
    if summary is None or not summary.equity:
        await session.close()
        return JSONResponse({"equity": "", "drawdown": "", "histogram": ""})
    times, equity = _downsample(summary.times, summary.equity)
    trades = await svc.trades(row.id)
    await session.close()
    _, dd_values = _downsample(summary.times, _drawdown(summary.equity))
    counts, centers = _pnl_histogram([t.net_pnl for t in trades])
    charts = {
        "equity": line_chart(
            times, [("equity", equity)], title="Live equity",
            palette=DARK,
        ),
        "drawdown": drawdown_chart(times, equity, title="Live drawdown", palette=DARK),
        "histogram": histogram_chart(
            counts, centers, title="Trade PnL distribution", palette=DARK,
        ) if counts else "",
    }
    return JSONResponse(charts)


# ── exports ───────────────────────────────────────────────────────────


async def _export_rows(key: str) -> tuple[str, list[list[Any]], dict[str, Any]]:
    row, svc, session = await _load_key(key)
    _key_guard(row)
    config = json.loads(row.config_json or "{}")
    trades = await svc.trades(row.id)
    await session.close()
    rows: list[list[Any]] = []
    for t in trades:
        rows.append([
            t.id, key, row.exchange, t.symbol, t.strategy, t.strategy_version,
            t.preset_id,
            t.entry_time.isoformat() if t.entry_time else "",
            t.exit_time.isoformat() if t.exit_time else "",
            t.direction, t.entry_price, t.exit_price, t.quantity,
            round(t.fee, 6), round(t.gross_pnl, 6), round(t.net_pnl, 6),
            round(t.pct_return, 6),
            round(t.holding_seconds / 3600.0, 3) if t.holding_seconds else 0.0,
            t.exit_reason, t.source,
        ])
    return row.exchange, rows, config


@router.get("/API_KEY/{key}/trades.csv", include_in_schema=False)
async def export_csv(key: str) -> Response:
    _exchange, rows, _config = await _export_rows(key)
    return Response(
        content=table_to_csv(TRADE_EXPORT_COLUMNS, rows),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{key}_trades.csv"'},
    )


@router.get("/API_KEY/{key}/trades.xlsx", include_in_schema=False)
async def export_xlsx(key: str) -> Response:
    _exchange, rows, _config = await _export_rows(key)
    return Response(
        content=table_to_xlsx(TRADE_EXPORT_COLUMNS, rows),
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={"Content-Disposition": f'attachment; filename="{key}_trades.xlsx"'},
    )


# ── refresh (near-real-time regeneration) ─────────────────────────────


@router.post("/API_KEY/{key}/refresh", include_in_schema=False)
async def dashboard_refresh(key: str) -> JSONResponse:
    row, svc, session = await _load_key(key)
    lock = _generate_locks.setdefault(row.id, asyncio.Lock())
    async with lock:
        try:
            report = await svc.generate(key=row, refresh=True)
        except SignalKeyError as exc:
            await session.close()
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # data source down — visible, not silent
            await session.close()
            raise HTTPException(502, f"signal generation failed: {exc}") from exc
    await session.close()
    if isinstance(report.get("generated_at"), datetime):
        report["generated_at"] = report["generated_at"].isoformat()
    return JSONResponse(report)


# ── the page ──────────────────────────────────────────────────────────


def _error_page(message: str) -> str:
    """A minimal, escaped HTML error page (never echo raw JSON to a browser)."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard unavailable</title>
<style>body {{ margin:0; font:14px/1.5 system-ui, sans-serif; background:#0f1420; color:#dbe4f0; }}
main {{ padding:40px 24px; max-width:640px; margin:0 auto; }}
h1 {{ font-size:18px; }} .muted {{ color:#7d8ca3; }}</style></head>
<body><main>
<h1>📡 Live strategy dashboard</h1>
<p>{html.escape(message)}</p>
<p class="muted">Check the key, or ask the operator to re-enable / re-issue it.</p>
</main></body></html>"""


def _page(*, key: str, exchange: str, tickers: str, strategy: str) -> str:
    esc = html.escape
    key_html = esc(key)
    key_js = json.dumps(key)  # a safe JS string literal (handles quotes/backslash)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live dashboard — {esc(key[:10])}…</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #0f1420; color: #dbe4f0; }}
  header {{ padding: 16px 24px; background: #131a2a; border-bottom: 1px solid #223049;
            display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }}
  header h1 {{ font-size: 16px; margin: 0; }}
  .badge {{ background: #223049; border-radius: 4px; padding: 2px 8px; font-size: 12px; }}
  main {{ padding: 20px 24px; max-width: 1200px; margin: 0 auto; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }}
  .card {{ background: #131a2a; border: 1px solid #223049; border-radius: 8px; padding: 12px; }}
  .card .v {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
  .chart {{ background: #131a2a; border: 1px solid #223049; border-radius: 8px; padding: 8px; margin: 12px 0; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  th, td {{ text-align: right; padding: 4px 8px; border-bottom: 1px solid #223049; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .pos {{ color: #35d07f; }} .neg {{ color: #ff6b6b; }}
  .toolbar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 12px 0; }}
  button, a.btn {{ background: #2a4a8f; color: #fff; border: 0; border-radius: 6px; padding: 8px 14px;
        font-size: 13px; cursor: pointer; text-decoration: none; }}
  button:disabled {{ opacity: .5; cursor: wait; }}
  .muted {{ color: #7d8ca3; font-size: 12px; }}
  #err {{ color: #ff6b6b; }}
  h2 {{ font-size: 14px; color: #9fb2cc; margin: 24px 0 8px; }}
</style>
</head>
<body>
<header>
  <h1>📡 Live strategy dashboard</h1>
  <span class="badge">broker: {esc(exchange)}</span>
  <span class="badge">tickers: {esc(tickers or "—")}</span>
  <span class="badge">strategy: {esc(strategy)}</span>
  <span class="badge">key: <code>{key_html}</code></span>
</header>
<main>
  <div class="toolbar">
    <button id="refresh">Refresh</button>
    <a class="btn" href="/API_KEY/{key_html}/trades.csv">Download CSV</a>
    <a class="btn" href="/API_KEY/{key_html}/trades.xlsx">Download Excel</a>
    <span class="muted">last update: <span id="updated">—</span>
      (auto-refresh every 60 s)</span>
    <span id="err"></span>
  </div>

  <div class="cards" id="cards"></div>

  <h2>Equity</h2><div class="chart" id="c-equity"><span class="muted">loading…</span></div>
  <h2>Drawdown</h2><div class="chart" id="c-dd"><span class="muted">loading…</span></div>
  <h2>Trade PnL distribution</h2><div class="chart" id="c-hist"><span class="muted">loading…</span></div>

  <h2>Latest signals</h2>
  <div style="overflow-x:auto"><table id="signals"><thead><tr>
    <th>time</th><th>ticker</th><th>side</th><th>state</th><th>price</th><th>reason</th>
  </tr></thead><tbody></tbody></table></div>

  <h2>Trades (latest 25)</h2>
  <div style="overflow-x:auto"><table id="trades"><thead><tr>
    <th>exit time</th><th>ticker</th><th>dir</th><th>entry</th><th>exit</th>
    <th>qty</th><th>net PnL</th><th>return %</th><th>exit reason</th>
  </tr></thead><tbody></tbody></table></div>
</main>
<script>
const KEY = {key_js};
const $ = (id) => document.getElementById(id);
const fmt = (x, d=2) => (x === null || x === undefined) ? "—" :
    Number(x).toLocaleString(undefined, {{maximumFractionDigits: d}});
function esc(s) {{ const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }}

async function refresh(regenerate) {{
  $("refresh").disabled = true; $("err").textContent = "";
  try {{
    if (regenerate) {{
      const r = await fetch(`/API_KEY/${{KEY}}/refresh`, {{method: "POST"}});
      if (!r.ok) {{ const j = await r.json().catch(() => ({{}}));
        $("err").textContent = "refresh failed: " + (j.detail || r.statusText); }}
    }}
    const res = await fetch(`/API_KEY/${{KEY}}/data`);
    if (res.status === 410 || res.status === 404) {{
      const j = await res.json(); document.body.innerHTML =
        `<main><h2>${{esc(j.detail || "key unavailable")}}</h2></main>`; return;
    }}
    const d = await res.json();
    $("updated").textContent = d.last_updated ? new Date(d.last_updated).toLocaleString() : "—";
    const t = d.totals, m = d.metrics || {{}};
    const cards = [
      ["Total trades", t.n_trades], ["Win rate", (100*t.win_rate).toFixed(1) + " %"],
      ["Total P/L", (t.total_pnl >= 0 ? "+" : "") + fmt(t.total_pnl)],
      ["Avg P/L", fmt(t.avg_pnl)], ["Max drawdown", (100*(m.max_drawdown ?? 0)).toFixed(1) + " %"],
      ["Sharpe", fmt(m.sharpe, 3)], ["Profit factor", fmt(m.profit_factor, 2)],
      ["Return", (100*(m.total_return ?? 0)).toFixed(1) + " %"],
    ];
    $("cards").innerHTML = cards.map(([k, v]) =>
      `<div class="card"><div class="muted">${{k}}</div><div class="v">${{esc(v)}}</div></div>`).join("");
    const tb = $("signals").tBodies[0];
    tb.innerHTML = (d.signals || []).map(s => `<tr>
      <td>${{esc(new Date(s.timestamp).toLocaleString())}}</td><td>${{esc(s.symbol)}}</td>
      <td class="${{s.side === "buy" ? "pos" : "neg"}}">${{esc(s.side)}}</td>
      <td>${{esc(s.state)}}</td><td>${{fmt(s.price, 4)}}</td><td>${{esc(s.reason)}}</td>
    </tr>`).join("") || `<tr><td colspan="6" class="muted">no signals yet — press Refresh</td></tr>`;
    const tt = $("trades").tBodies[0];
    tt.innerHTML = [...(d.trades || [])].reverse().map(x => `<tr>
      <td>${{esc(new Date(x.exit_time).toLocaleString())}}</td><td>${{esc(x.symbol)}}</td>
      <td>${{esc(x.direction)}}</td><td>${{fmt(x.entry_price, 4)}}</td><td>${{fmt(x.exit_price, 4)}}</td>
      <td>${{fmt(x.quantity, 4)}}</td>
      <td class="${{x.net_pnl >= 0 ? "pos" : "neg"}}">${{fmt(x.net_pnl)}}</td>
      <td class="${{x.return_pct >= 0 ? "pos" : "neg"}}">${{(100*x.return_pct).toFixed(2)}} %</td>
      <td>${{esc(x.exit_reason)}}</td>
    </tr>`).join("") || `<tr><td colspan="9" class="muted">no closed trades yet</td></tr>`;

    const ch = await (await fetch(`/API_KEY/${{KEY}}/charts`)).json();
    $("c-equity").innerHTML = ch.equity || '<span class="muted">no data yet</span>';
    $("c-dd").innerHTML = ch.drawdown || '<span class="muted">no data yet</span>';
    $("c-hist").innerHTML = ch.histogram || '<span class="muted">no trades yet</span>';
  }} catch (e) {{ $("err").textContent = "error: " + e.message;
  }} finally {{ $("refresh").disabled = false; }}
}}
$("refresh").addEventListener("click", () => refresh(true));
refresh(false);            // initial load (generates on a cold cache)
setInterval(() => refresh(true), 60000);  // near-real-time auto-refresh
</script>
</body>
</html>"""
