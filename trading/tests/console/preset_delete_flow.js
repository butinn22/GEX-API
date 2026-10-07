/* jsdom regression harness: console "saved strategies" delete must STICK.
 *
 * Loads trading/static/index.html against a faithful in-process fake of the
 * preset API (same semantics as trading/api/routers/presets.py) and drives the
 * Strategy-lab hub through real DOM events.
 *
 * The bug this guards (Round-3): a deleted saved strategy "reappeared". Two
 * root causes, both asserted here:
 *   1. a failed mutation was swallowed by the demo mock → "Version deleted"
 *      toast while the backend never changed (silent fake success);
 *   2. the hub row is a named-strategy GROUP but Delete only removed the
 *      latest VERSION, so the group came back as an older version.
 *
 * Usage: node preset_delete_flow.js <scenario> [<via>]
 *   scenarios: single-version | multi-version | delete-network-fails | live-enabled
 *   via:       hub (row Delete) | lab (Open in Lab + toolbar Delete)
 * Exit code 0 = assertions passed, 1 = failed.
 */
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const HTML = path.resolve(__dirname, "../../static/index.html");
const BASE = "http://127.0.0.1:8199/";
const STRAT = "trend_confluence_pine";
const SYMBOL = "DELX";

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

function seedRows(scenario) {
  let seq = 100;
  const row = (over) => Object.assign({
    id: ++seq, symbol: SYMBOL, strategy: STRAT, strategy_name: "grp",
    version: 1, is_default: true, status: "backtest_only", source: "manual",
    timeframe: "1d", metrics: {}, params: {}, notes: "",
  }, over);
  if (scenario === "live-enabled") return [row({ status: "live_enabled" })];
  if (scenario === "multi-version")
    return [row({ version: 1, is_default: false }), row({ id: ++seq, version: 2, is_default: true })];
  return [row({})];
}

function makeBackend(scenario) {
  const rows = seedRows(scenario);
  const calls = [];
  let newId = 1000;

  async function handle(url, opts) {
    const method = (opts.method || "GET").toUpperCase();
    const u = new URL(url, BASE);
    const p = u.pathname.replace(/^\/api\/v1/, "");
    const q = u.searchParams;
    calls.push(method + " " + u.pathname + u.search);

    if (u.pathname === "/health") return jsonRes(200, { status: "ok" });
    if (p === "/auth/token") return jsonRes(200, { access_token: "t", token_type: "bearer" });
    if (p === "/keys") return jsonRes(200, []);
    if (p === "/data/instruments") return jsonRes(200, []);
    if (p.startsWith("/data/detect/")) {
      const sym = decodeURIComponent(p.split("/data/detect/")[1]).toUpperCase();
      return jsonRes(200, { symbol: sym, source: "yfinance", fetch_symbol: sym,
        category: "us", synthetic_available: true });
    }
    if (p === "/strategies") return jsonRes(200, []);
    if (/^\/strategies\/.+\/schema$/.test(p))
      return jsonRes(200, { name: STRAT, groups: [], params: [], defaults: {}, sweep: {} });

    const filtered = () => rows.filter((r) =>
      (!q.get("symbol") || r.symbol === q.get("symbol").toUpperCase()) &&
      (!q.get("strategy") || r.strategy === q.get("strategy")) &&
      (!q.get("name") || (r.strategy_name || "") === q.get("name")));

    if (p === "/presets/latest") {
      const best = {};
      for (const r of filtered()) {
        const k = r.symbol + "|" + r.strategy + "|" + (r.strategy_name || "");
        if (!best[k] || (r.version || 1) >= (best[k].version || 1)) best[k] = r;
      }
      return jsonRes(200, Object.values(best));
    }
    if (p === "/presets/versions")
      return jsonRes(200, filtered().slice().sort((a, b) => b.version - a.version));
    if (p === "/presets" && method === "DELETE") {
      if (scenario === "delete-network-fails") throw new TypeError("connection reset");
      const group = rows.filter((r) =>
        r.symbol === (q.get("symbol") || "").toUpperCase() &&
        r.strategy === (q.get("strategy") || "") &&
        (r.strategy_name || "") === (q.get("name") || ""));
      if (!group.length) return jsonRes(404, { detail: "no saved strategy" });
      if (group.some((r) => r.status === "live_enabled"))
        return jsonRes(409, { detail: "saved strategy is live_enabled — demote it before deleting" });
      group.forEach((r) => rows.splice(rows.indexOf(r), 1));
      return jsonRes(204, null);
    }
    if (p === "/presets" && method === "GET") return jsonRes(200, filtered());
    if (p === "/presets" && method === "POST") {
      const b = JSON.parse(opts.body || "{}");
      const r = { id: ++newId, symbol: (b.symbol || "").toUpperCase(), strategy: b.strategy,
        strategy_name: b.strategy_name || "", version: 1, is_default: !!b.is_default,
        status: "backtest_only", source: "manual", timeframe: "", metrics: {}, params: {}, notes: "" };
      rows.push(r);
      return jsonRes(201, r);
    }
    const m = p.match(/^\/presets\/(\d+)$/);
    if (m) {
      const id = Number(m[1]);
      const r = rows.find((x) => x.id === id);
      if (method === "DELETE") {
        if (scenario === "delete-network-fails") throw new TypeError("connection reset");
        if (!r) return jsonRes(404, { detail: `preset ${id} not found` });
        if (r.status === "live_enabled")
          return jsonRes(409, { detail: "preset is live_enabled — demote it before deleting" });
        rows.splice(rows.indexOf(r), 1);
        return jsonRes(204, null);
      }
      if (method === "PATCH") {
        if (!r) return jsonRes(404, { detail: `preset ${id} not found` });
        const copy = Object.assign({}, r, { id: ++newId, version: (r.version || 1) + 1 });
        r.is_default = false;
        rows.push(copy);
        return jsonRes(200, copy);
      }
    }
    throw new Error("unhandled " + method + " " + p);
  }
  return { handle, rows, calls };
}

async function boot(scenario) {
  const backend = makeBackend(scenario);
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

async function waitFor(fn, ms = 4000, step = 25) {
  const end = Date.now() + ms;
  for (;;) {
    let v; try { v = fn(); } catch (e) { v = false; }
    if (v) return v;
    if (Date.now() > end) return v;
    await sleep(step);
  }
}

async function drive(scenario, via) {
  const { win, backend, errors } = await boot(scenario);
  const $ = (s) => win.document.querySelector(s);
  const $$ = (s) => Array.from(win.document.querySelectorAll(s));

  await waitFor(() => !$("#view-app").hidden);
  const tab = $('button[data-tab="pine"]');
  if (tab) tab.click();
  await waitFor(() => win.eval("typeof PINE !== 'undefined' && PINE.ready"));
  await win.eval(
    `(async () => { document.querySelector("#pine-symbol").value = "${SYMBOL}";` +
    ' await pineRefreshStore(); })()');
  await waitFor(() => $$("#pine-hub .route").length > 0);

  const hubBefore = $$("#pine-hub .route").length;
  if (via === "hub") {
    const del = $("#pine-hub button[data-delete]");
    if (!del) return { ok: false, why: "hub row has no Delete button", hubBefore, backend };
    del.click();
  } else {
    const open = $("#pine-hub button[data-open]");
    if (!open) return { ok: false, why: "hub row has no Open-in-Lab button", hubBefore, backend };
    open.click();
    await sleep(60);
    $("#pine-delete").click();
  }
  await sleep(700);

  const errEl = $("#pine-preset-error");
  return {
    ok: true,
    hubBefore,
    hubAfter: $$("#pine-hub .route").length,
    hubNames: $$("#pine-hub .route b").map((b) => b.textContent.trim()),
    backendRows: backend.rows.map((r) => r.id + ":v" + r.version),
    presetError: errEl && !errEl.hidden ? errEl.textContent : null,
    toasts: $$("#toasts .toast").map((t) => t.textContent),
    demoMode: win.eval("typeof S !== 'undefined' ? S.demo : '?'"),
    jsErrors: errors.slice(0, 3),
  };
}

function expect(cond, why) {
  if (!cond) throw new Error(why);
}

async function driveDuplicate() {
  const { win, backend, errors } = await boot("single-version");
  const $ = (s) => win.document.querySelector(s);
  const $$ = (s) => Array.from(win.document.querySelectorAll(s));
  win.prompt = () => "grp copy";
  await waitFor(() => !$("#view-app").hidden);
  const tab = $('button[data-tab="pine"]');
  if (tab) tab.click();
  await waitFor(() => win.eval("typeof PINE !== 'undefined' && PINE.ready"));
  await win.eval(
    `(async () => { document.querySelector("#pine-symbol").value = "${SYMBOL}";` +
    ' await pineRefreshStore(); })()');
  await waitFor(() => $$("#pine-hub .route").length > 0);
  const dup = $("#pine-hub button[data-duplicate]");
  if (!dup) return { ok: false, why: "hub row has no Duplicate button", backend, errors };
  const before = $$("#pine-hub .route").length;
  dup.click();
  await sleep(700);
  return {
    ok: true, before,
    after: $$("#pine-hub .route").length,
    names: $$("#pine-hub .route b").map((b) => b.textContent.trim()),
    postCalled: backend.calls.some((c) => c.startsWith("POST /api/v1/presets")),
    backendRows: backend.rows.length,
    jsErrors: errors.slice(0, 3),
  };
}

async function scenario(name, via, checks) {
  const r = await drive(name, via);
  if (!r.ok) throw new Error(r.why);
  expect(r.jsErrors.length === 0, `js errors: ${r.jsErrors.join(" | ")}`);
  checks(r);
  console.log(`  ok  ${name} (via ${via}) — hub ${r.hubBefore}->${r.hubAfter}, ` +
    `backend ${r.backendRows.length} row(s)`);
  return r;
}

async function main() {
  console.log("console preset-delete regression (jsdom)");

  // 1. a saved strategy with one version disappears from hub AND backend.
  await scenario("single-version", "hub", (r) => {
    expect(r.hubBefore === 1, "expected one hub row to start with");
    expect(r.hubAfter === 0, "hub still shows the deleted strategy");
    expect(r.backendRows.length === 0, "backend still holds the row: " + r.backendRows);
    expect(!r.presetError, "unexpected error surfaced: " + r.presetError);
  });

  // 2. a multi-version group must vanish entirely (no older version resurfacing).
  await scenario("multi-version", "hub", (r) => {
    expect(r.hubBefore === 1, "expected one hub row (a group) to start with");
    expect(r.hubAfter === 0, "group reappeared as an older version: " + r.hubNames);
    expect(r.backendRows.length === 0, "backend still holds versions: " + r.backendRows);
  });

  // 3. a failed delete must NOT fake success (no demo fallback for mutations).
  await scenario("delete-network-fails", "hub", (r) => {
    expect(r.hubAfter === 1, "row vanished although the backend refused/never got it");
    expect(r.backendRows.length === 1, "backend should still hold the row");
    expect(!!r.presetError, "no error surfaced for an unreachable backend");
    expect(r.presetError.indexOf("NOT saved") >= 0,
      "error copy must say the change was not saved, got: " + r.presetError);
    expect(r.demoMode === false, "a mutation must never flip the console into demo mode");
    expect(!r.toasts.some((t) => /deleted/i.test(t)),
      "success toast shown for a failed delete: " + r.toasts.join(" | "));
  });

  // 4. live_enabled is a real refusal, not a fake success.
  await scenario("live-enabled", "hub", (r) => {
    expect(r.hubAfter === 1, "live strategy vanished from the hub");
    expect(r.backendRows.length === 1, "live strategy was deleted");
    expect(!!r.presetError && /live_enabled/.test(r.presetError),
      "live refusal not surfaced, got: " + r.presetError);
  });

  // 5. the lab's per-version Delete still works (version-scoped semantics).
  const lab = await drive("multi-version", "lab");
  expect(lab.ok, "lab flow failed: " + lab.why);
  expect(lab.hubAfter === 1, "version delete should leave the group (older version)");
  expect(lab.backendRows.length === 1, "version delete should remove exactly one version");
  console.log("  ok  multi-version (via lab) — one version removed, group kept");

  // 6. Duplicate creates a new named strategy without touching the original.
  const dup = await driveDuplicate();
  expect(dup.ok, "duplicate flow failed: " + dup.why);
  expect(dup.jsErrors.length === 0, "duplicate js errors: " + dup.jsErrors.join(" | "));
  expect(dup.postCalled, "Duplicate did not POST /presets");
  expect(dup.after === dup.before + 1, "hub did not gain a row: " + dup.names.join(","));
  expect(dup.names.some((n) => n === "grp copy"), "copy not shown: " + dup.names.join(","));
  expect(dup.backendRows === 2, "backend should hold both strategies, got " + dup.backendRows);
  console.log("  ok  duplicate — new named strategy created, original kept");

  console.log("PASS");
}

main().catch((e) => {
  console.log("FAIL: " + (e && e.stack || e));
  process.exitCode = 1;
});
