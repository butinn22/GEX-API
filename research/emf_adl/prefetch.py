"""Warm the disk cache for the whole study panel. Run once, then everything is cached."""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.emf_adl import data as D  # noqa: E402

EXCLUDE = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT",
           "USDEUSDT", "USD1USDT", "EURUSDT"}

OUT = Path(__file__).with_name("out")
OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    pool_size = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    t0 = time.time()
    top = D.fetch_top_symbols(80)
    pool = [s for s in top["symbol"].tolist() if s not in EXCLUDE][:pool_size]
    print("POOL", pool, flush=True)
    rep = D.prefetch_panel(
        pool, ("4H", "1D"), D.ms(2021, 7, 1), D.ms(2026, 10, 1), workers=4
    )
    (OUT / "prefetch.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    ok = [r for r in rep if r.get("ok")]
    bad = [r for r in rep if not r.get("ok")]
    print(f"fetched {len(ok)}/{len(rep)} in {time.time() - t0:.0f}s; failures={len(bad)}")
    for r in bad:
        print("  FAIL", r)
