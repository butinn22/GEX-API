/* jsdom regression harness for the configuration/strategy management controls.
 *
 * Loads trading/static/index.html against an in-process fake of the management
 * API and drives the real DOM events:
 *   - signal keys : Enable/Disable, Regenerate, Purge signals, Purge cache
 *   - stored runs : list, Delete, Clear all
 *   - orders      : list, Cancel (live), Delete, Purge all
 *   - runners     : Start/Stop
 *   - accounts    : Test connection
 *
 * Usage: node management_controls_flow.js   (exit 0 = pass, 1 = fail)
 */
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const HTML = path.resolve(__dirname, "../../static/index.html");
const BASE = "http://127.0.0.1:8199/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function jsonRes(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    blob: async () => body,
  };
}

function makeBackend() {
  const calls = [];
  const state = {
    signalKeys: [{
      id: 1, key: "sk_" + "a".repeat(40), exchange: "bingx", label: "demo",
      active: true, revoked_at: null, created_at: "2026-10-01T00:00:00Z",
      last_used_at: null, config: { tickers: [{ symbol: "BTC" }] },
    }],
    runs: [{ id: 5, symbol: "BTC", strategy: "sma_crossover",
             created_at: "2026-10-01T00:00:00Z", metrics: { total_return: 0.1 } }],
    orders: [{ id: "BTC-USDT:1", exchange: "bingx", symbol: "BTC-USDT",
               side: "buy", status: "open", quantity: 1.0, order_type: "market" }],
    strategies: [{ name: "sma_crossover", params: [] }],
    accounts: [{ id: 7, exchange: "bingx", label: "main", api_key_masked: "sk-…",
                 created_at: "2026-10-01T00:00:00Z", settings: { enabled: true } }],
  };

  async function handle(url, opts) {
    const method = (opts.method || "GET").toUpperCase();
    const u = new URL(url, BASE);
    const p = u.pathname.replace(/^\/api\/v1/, "");
    const q = u.searchParams;
    calls.push(method + " " + u.pathname + u.search);

    // boot / shared
    if (u.pathname === "/health") return jsonRes(200, { status: "ok" });
    if (p === "/auth/token") return jsonRes(200, { access_token: "t" });
    if (p === "/data/instruments") return jsonRes(200, []);
    if (p.startsWith("/data/detect/")) return jsonRes(200, { symbol: "X", source: "yfinance" });
    if (p === "/strategies" && method === "GET") return jsonRes(200, state.strategies);
    if (/^\/strategies\/.+\/schema$/.test(p)) return jsonRes(200, { name: "x", groups: [], params: [], defaults: {}, sweep: {} });
    if (p === "/presets/latest") return jsonRes(200, []);
    if (p === "/presets/versions") return jsonRes(200, []);
    if (p === "/presets" && method === "GET") return jsonRes(200, []);
    if (p === "/data/categories" || p === "/backtest/cancel") return jsonRes(200, []);

    // signal keys
    if (p === "/signal-keys" && method === "GET") return jsonRes(200, state.signalKeys);
    if (p === "/signal-keys/cache/purge") return jsonRes(200, { purged: 2 });
    let m = p.match(/^\/signal-keys\/(\d+)\/signals$/);
    if (m && method === "DELETE") return jsonRes(200, { deleted: 3 });
    m = p.match(/^\/signal-keys\/(\d+)$/);
    if (m && method === "PATCH") {
      const row = state.signalKeys.find((k) => k.id === Number(m[1]));
      if (!row) return jsonRes(404, { detail: "signal key not found" });
      row.active = q.get("active") !== "false";
      return jsonRes(200, row);
    }
    m = p.match(/^\/signal-keys\/([^/]+)\/generate$/);
    if (m && method === "POST") return jsonRes(200, { n_signals: 4, n_trades: 2, errors: [] });

    // stored runs
    if (p === "/backtest/results" && method === "GET") return jsonRes(200, state.runs);
    if (p === "/backtest/results" && method === "DELETE") {
      const n = state.runs.length; state.runs = []; return jsonRes(200, { deleted: n });
    }
    m = p.match(/^\/backtest\/results\/(\d+)$/);
    if (m && method === "DELETE") {
      state.runs = state.runs.filter((r) => r.id !== Number(m[1]));
      return jsonRes(204, null);
    }

    // bulk actions
    if (p === "/backtest/results/bulk-delete" && method === "POST") {
      const ids = (JSON.parse(opts.body || "{}").ids) || [];
      const del = state.runs.filter((r) => ids.includes(r.id)).length;
      state.runs = state.runs.filter((r) => !ids.includes(r.id));
      return jsonRes(200, { deleted: del, missing: [] });
    }
    if (p === "/orders/bulk-delete" && method === "POST") {
      const ids = (JSON.parse(opts.body || "{}").ids) || [];
      const del = state.orders.filter((o) => ids.includes(o.id)).length;
      state.orders = state.orders.filter((o) => !ids.includes(o.id));
      return jsonRes(200, { deleted: del, missing: [] });
    }
    if (p === "/signal-keys/bulk" && method === "POST") {
      const b = JSON.parse(opts.body || "{}");
      const ids = b.ids || [];
      let n = 0;
      for (const k of state.signalKeys) {
        if (!ids.includes(k.id)) continue;
        n++;
        if (b.action === "revoke") { k.revoked_at = "now"; k.active = false; }
        else k.active = b.action === "enable";
      }
      return jsonRes(200, { updated: n, missing: [], action: b.action });
    }

    // orders
    if (p === "/orders" && method === "GET") return jsonRes(200, state.orders);
    if (p === "/orders" && method === "DELETE") {
      const n = state.orders.length; state.orders = []; return jsonRes(200, { deleted: n });
    }
    m = p.match(/^\/orders\/(.+)\/cancel$/);
    if (m && method === "POST") {
      const id = decodeURIComponent(m[1]);
      const row = state.orders.find((o) => o.id === id);
      if (!row) return jsonRes(404, { detail: "order not found" });
      row.status = "cancelled";
      return jsonRes(200, row);
    }
    m = p.match(/^\/orders\/(.+)$/);
    if (m && method === "DELETE") {
      const id = decodeURIComponent(m[1]);
      state.orders = state.orders.filter((o) => o.id !== id);
      return jsonRes(204, null);
    }

    // accounts / credentials
    if (p === "/keys" && method === "GET") return jsonRes(200, state.accounts);
    m = p.match(/^\/keys\/(\d+)\/validate$/);
    if (m && method === "POST") return jsonRes(200, { ok: true, message: "connected — USDT balance 1000", mine: false });

    throw new Error("unhandled " + method + " " + p);
  }
  return { handle, state, calls };
}

async function boot() {
  const backend = makeBackend();
  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => errors.push("jsdomError: " + e.message));
  vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));
  const dom = new JSDOM(fs.readFileSync(HTML, "utf8"), {
    url: BASE, runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(window) {
      window.confirm = () => true;
      try { window.localStorage.setItem("gex.token", "t"); } catch (e) {}
      window.fetch = (url, opts) => backend.handle(url, opts || {});
    },
  });
  return { win: dom.window, backend, errors };
}

async function waitFor(fn, ms = 5000, step = 25) {
  const end = Date.now() + ms;
  for (;;) {
    let v; try { v = fn(); } catch (e) { v = false; }
    if (v) return v;
    if (Date.now() > end) return v;
    await sleep(step);
  }
}

function expect(cond, why) { if (!cond) throw new Error(why); }

async function main() {
  const { win, backend, errors } = await boot();
  const doc = win.document;
  const $ = (s) => doc.querySelector(s);
  const $$ = (s) => Array.from(doc.querySelectorAll(s));
  const called = (prefix) => backend.calls.some((c) => c.startsWith(prefix));
  const waitCall = (prefix) => waitFor(() => called(prefix));
  const waitSel = (s) => waitFor(() => $(s));

  await waitFor(() => !$("#view-app").hidden);

  /* ── deploy tab: signal keys ─────────────────────────────────────────── */
  $('button[data-tab="deploy"]').click();
  await waitSel("#dp-keys button[data-toggle-key]");
  expect($("#dp-keys button[data-regen-key]"), "signal key row missing Regenerate");
  expect($("#dp-keys button[data-purge-key]"), "signal key row missing Purge signals");
  expect($("#dp-purge-cache"), "missing Purge summary cache button");

  // Disable (active currently true → sends active=false)
  $("#dp-keys button[data-toggle-key]").click();
  await waitCall("PATCH /api/v1/signal-keys/1?active=false");
  expect(await waitFor(() => !backend.state.signalKeys[0].active), "key was not disabled server-side");

  // Regenerate
  $("#dp-keys button[data-regen-key]").click();
  await waitCall("POST /api/v1/signal-keys/" + encodeURIComponent(backend.state.signalKeys[0].key) + "/generate");

  // Purge signals (confirm returns true)
  $("#dp-keys button[data-purge-key]").click();
  await waitCall("DELETE /api/v1/signal-keys/1/signals");

  // Purge summary cache
  $("#dp-purge-cache").click();
  await waitCall("POST /api/v1/signal-keys/cache/purge");
  console.log("  ok  signal keys — disable / regenerate / purge / purge-cache");

  /* ── stored runs ─────────────────────────────────────────────────────── */
  await waitSel("#dp-runs button[data-del-run]");
  $("#dp-runs button[data-del-run]").click();
  await waitCall("DELETE /api/v1/backtest/results/5");
  expect(await waitFor(() => backend.state.runs.length === 0), "run not deleted server-side");

  backend.state.runs.push({ id: 6, symbol: "ETH", strategy: "sma_crossover",
                            created_at: "2026-10-02T00:00:00Z", metrics: {} });
  $("#dp-runs-refresh").click();
  await waitSel("#dp-runs button[data-del-run]");
  $("#dp-runs-clear").click();
  await waitCall("DELETE /api/v1/backtest/results");
  expect(await waitFor(() => backend.state.runs.length === 0), "clear-all did not empty the runs");
  console.log("  ok  stored runs — delete one / clear all");

  /* ── orders ──────────────────────────────────────────────────────────── */
  await waitSel("#dp-orders button[data-cancel-order]");
  $("#dp-orders button[data-cancel-order]").click();
  await waitCall("POST /api/v1/orders/" + encodeURIComponent("BTC-USDT:1") + "/cancel");
  expect(await waitFor(() => backend.state.orders[0] && backend.state.orders[0].status === "cancelled"),
    "order not cancelled server-side");

  $("#dp-orders button[data-del-order]").click();
  await waitCall("DELETE /api/v1/orders/" + encodeURIComponent("BTC-USDT:1"));
  expect(await waitFor(() => backend.state.orders.length === 0), "order record not deleted");

  backend.state.orders.push({ id: "ETH-USDT:2", exchange: "bingx", symbol: "ETH-USDT",
                              side: "sell", status: "open", quantity: 1.0, order_type: "market" });
  $("#dp-orders-refresh").click();
  await waitSel("#dp-orders button[data-del-order]");
  $("#dp-orders-clear").click();
  await waitCall("DELETE /api/v1/orders");
  expect(await waitFor(() => backend.state.orders.length === 0), "purge-all did not empty orders");
  console.log("  ok  orders — cancel live / delete record / purge all");

  /* ── bulk selection ──────────────────────────────────────────────────── */
  const selectAll = (scope) => {
    const all = doc.querySelector(`[data-sel-all='${scope}']`);
    all.checked = true;
    all.dispatchEvent(new win.Event("change", { bubbles: true }));
  };

  backend.state.runs.push(
    { id: 21, symbol: "BTC", strategy: "sma_crossover", created_at: null, metrics: {} },
    { id: 22, symbol: "ETH", strategy: "sma_crossover", created_at: null, metrics: {} });
  $("#dp-runs-refresh").click();
  await waitFor(() => $$("#dp-runs [data-sel]").length === 2);
  selectAll("runs");
  expect($("#dp-runs-bulk").style.display === "flex", "bulk bar should appear for a non-empty selection");
  $("#dp-runs-del-sel").click();
  await waitCall("POST /api/v1/backtest/results/bulk-delete");
  expect(await waitFor(() => backend.state.runs.length === 0), "bulk run delete did not empty the runs");

  backend.state.orders.push(
    { id: "BTC-USDT:9", exchange: "bingx", symbol: "BTC-USDT", side: "buy", status: "open", quantity: 1, order_type: "market" },
    { id: "ETH-USDT:9", exchange: "bingx", symbol: "ETH-USDT", side: "sell", status: "open", quantity: 1, order_type: "market" });
  $("#dp-orders-refresh").click();
  await waitFor(() => $$("#dp-orders [data-sel]").length === 2);
  selectAll("orders");
  $("#dp-orders-del-sel").click();
  await waitCall("POST /api/v1/orders/bulk-delete");
  expect(await waitFor(() => backend.state.orders.length === 0), "bulk order delete did not empty orders");

  await waitFor(() => $$("#dp-keys [data-sel]").length >= 1);
  selectAll("keys");
  $("#dp-keys-disable-sel").click();
  await waitCall("POST /api/v1/signal-keys/bulk");
  expect(await waitFor(() => backend.state.signalKeys.every((k) => k.active === false)),
    "bulk disable did not disable the selected key");
  console.log("  ok  bulk — runs delete / orders delete / keys disable");

  /* ── live runners ────────────────────────────────────────────────────── */
  await waitFor(() => $$("#dp-runner-name option").length > 0);
  $("#dp-runner-name").value = "sma_crossover";
  $("#dp-runner-start").click();
  await waitCall("POST /api/v1/strategies/sma_crossover/start");
  $("#dp-runner-stop").click();
  await waitCall("POST /api/v1/strategies/sma_crossover/stop");
  console.log("  ok  runners — start / stop");

  /* ── accounts: Test connection ───────────────────────────────────────── */
  $('button[data-tab="accounts"]').click();
  await waitSel("#accounts-list button[data-test]");
  $("#accounts-list button[data-test]").click();
  await waitCall("POST /api/v1/keys/7/validate");
  console.log("  ok  accounts — test connection");

  /* ── accessibility: every management control has an accessible name ──── */
  const accessibleName = (el) => {
    const aria = el.getAttribute("aria-label");
    if (aria && aria.trim()) return aria.trim();
    const by = el.getAttribute("aria-labelledby");
    if (by) {
      const t = by.split(/\s+/).map((id) => (doc.getElementById(id) || {}).textContent || "").join(" ").trim();
      if (t) return t;
    }
    const title = el.getAttribute("title");
    if (title && title.trim()) return title.trim();
    if (el.id) {
      const lab = doc.querySelector(`label[for='${el.id}']`);
      if (lab && lab.textContent.trim()) return lab.textContent.trim();
    }
    const txt = (el.textContent || "").trim();
    if (txt) return txt;
    return (el.getAttribute("placeholder") || "").trim();
  };
  const controls = Array.from(doc.querySelectorAll(
    "#tab-deploy button, #tab-deploy input, #tab-deploy select, #tab-accounts button"));
  const unnamed = controls.filter((el) => !accessibleName(el));
  expect(unnamed.length === 0, "controls without an accessible name: " +
    unnamed.map((e) => e.outerHTML.slice(0, 70)).join(" | "));
  const groups = doc.querySelectorAll("#tab-deploy [role='group']");
  expect(groups.length >= 3, "expected labeled bulk action groups, found " + groups.length);
  console.log("  ok  accessibility — all management controls named, bulk groups labeled");

  expect(errors.length === 0, "js errors: " + errors.slice(0, 3).join(" | "));
  console.log("\nPASS");
}

main().catch((e) => {
  console.log("FAIL: " + (e && e.stack || e));
  process.exitCode = 1;
});
