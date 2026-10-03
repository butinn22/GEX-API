"""Export equity and drawdown curves for the drawdown-first finalist trio.

Kept separate from ``dd_verify`` so the report can be regenerated without re-running the
whole adversarial battery, and so the curve data has its own artifact with its own hash.
Every curve is thinned to a fixed point count; the thinning keeps the endpoints, so a
2x-move at the end of the sample is never dropped.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from research.emf_adl import data as D  # noqa: E402
from research.emf_adl import run_loops as RL  # noqa: E402
from research.emf_adl.dd_verify import CONTROL, FINALIST, INCUMBENT, build  # noqa: E402
from research.emf_adl.engine import Costs  # noqa: E402

OUT = RL.OUT
VARIANTS = {"INCUMBENT": INCUMBENT, "CONTROL": CONTROL, "FINAL": FINALIST}


def thin(eq: np.ndarray, ts: np.ndarray, max_points: int = 900) -> dict:
    n = len(eq)
    if n == 0:
        return {"t": [], "v": [], "dd": []}
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    sel = np.arange(n) if n <= max_points else np.unique(
        np.concatenate([np.arange(0, n, int(np.ceil(n / max_points))), [n - 1]])
    )
    return {
        "t": [str(x)[:10] for x in ts[sel]],
        "v": [float(eq[i]) for i in sel],
        "dd": [float(dd[i]) for i in sel],
        "n_bars": int(n),
    }


def main() -> int:
    t0 = time.time()
    uni = RL.load_universe(OUT / "universe.json")
    start, end = D.ms(*RL.STUDY_START), D.ms(*RL.STUDY_END)
    contexts, exclusions = RL.build_contexts(uni["symbols"], start, end)
    print(f"universe={len(uni['symbols'])} contexts={len(contexts)} "
          f"exclusions={len(exclusions)}")

    out: dict = {"variants": {}, "exclusions": exclusions}
    for name, v in VARIANTS.items():
        runs = build(v, contexts, Costs())
        block: dict = {"spec": {k: str(x) for k, x in v.__dict__.items()}, "by_tf": {}}
        for tf, pr in runs.items():
            ts = np.asarray(pr.timestamps, dtype="int64")
            labels = np.array([np.datetime64(int(t), "ms") for t in ts])
            block["by_tf"][tf] = thin(pr.equity, labels)
            print(f"  {name:10s} {tf}: {len(pr.equity)} bars, "
                  f"final {pr.equity[-1]:.4f}x, maxDD {pr.stats['max_dd']:.2%}")
        out["variants"][name] = block

    (OUT / "dd_curves.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote dd_curves.json | elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
