"""CLI: build and persist the study universe. Run once; every stage reads the result."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.emf_adl import run_loops as RL  # noqa: E402

u = RL.select_universe(max_candidates=60, workers=6)
out = Path(__file__).with_name("out") / "universe.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(u, indent=2, default=str), encoding="utf-8")

print("selected %d symbols, rejected %d (short history), excluded %d (non-crypto)"
      % (len(u["symbols"]), len(u["rejected_short_history"]), len(u["excluded_non_crypto"])))
print("\nSELECTED (ranked by real 24h turnover):")
for i, r in enumerate(u["selected"]):
    print("  %2d %-14s turn=%14.0f  4H=%6d 1D=%5d from=%s cov4=%.4f fund=%d"
          % (i, r["symbol"], r["turnover24h"], r["n_bars4H"], r["n_bars1D"],
             (r.get("first_bar4H") or "")[:10], r.get("coverage4H") or 0,
             r["n_funding4H"]))
print("\nREJECTED short history:")
for r in u["rejected_short_history"]:
    print("  %-14s %s" % (r["symbol"], r["reason"]))
