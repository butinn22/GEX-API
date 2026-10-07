/* Run every *_flow.js console harness in this directory and aggregate results.
 *
 * Each harness is a standalone Node script that exits 0 on pass, non-zero on
 * fail. This runner keeps a single `npm test` entry point for CI.
 */
const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const dir = __dirname;
const flows = fs.readdirSync(dir).filter((f) => f.endsWith("_flow.js")).sort();

let failed = 0;
for (const f of flows) {
  console.log(`\n=== ${f} ===`);
  const r = spawnSync(process.execPath, [path.join(dir, f)], { stdio: "inherit" });
  if (r.status !== 0) {
    failed++;
    console.log(`FAIL ${f} (exit ${r.status})`);
  }
}

console.log(`\n${flows.length - failed}/${flows.length} console flow(s) passed`);
process.exit(failed ? 1 : 0);
