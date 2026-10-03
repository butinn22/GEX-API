#!/usr/bin/env python
"""Deterministic self-check for the hybrid framework and the structure engine.

Two families of checks:

1. **Transform correctness** — the vectorized implementation must equal an independent
   naive loop over all ten derived series, and satisfy the framework's identities.
2. **Structure causality and invariants** — published structure values must be
   reproducible from a truncated history (no look-ahead), and stops/breaks must obey
   their structural rules.

Exits 0 when every check passes, 1 otherwise.

Usage
-----
    python self_check.py
    python self_check.py --bars 800 --seed 7 --json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

try:
    from hybrid_candles import (
        DERIVED,
        add_hybrid_candles,
        closed_form_ha_open,
        compute_arrays,
        reference_transform,
    )
    from market_structure import build_structure, find_pivots
except ImportError:  # pragma: no cover
    from .hybrid_candles import (  # type: ignore
        DERIVED,
        add_hybrid_candles,
        closed_form_ha_open,
        compute_arrays,
        reference_transform,
    )
    from .market_structure import build_structure, find_pivots  # type: ignore

ATOL = 1e-9
RTOL = 1e-12


def synthetic_ohlcv(n: int, seed: int = 11) -> pd.DataFrame:
    """Seeded random walk with intrabar range and a regime change. Deterministic."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate(
        [np.full(n // 3, 0.0006), np.full(n // 3, -0.0008), np.full(n - 2 * (n // 3), 0.0010)]
    )
    vol = 0.012
    rets = drift + rng.normal(0.0, vol, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = close[0] * (1.0 + rng.normal(0, 0.002))
    open_[1:] = close[:-1] * (1.0 + rng.normal(0, 0.0015, n - 1))
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = np.abs(rng.normal(0.0, 0.004, n)) * close
    high = body_hi + wick
    low = body_lo - np.abs(rng.normal(0.0, 0.004, n)) * close
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


class Checker:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.results if not r["ok"]]


# --------------------------------------------------------------------------- #
def check_transform(df: pd.DataFrame, chk: Checker) -> None:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    fast = compute_arrays(o, h, l, c)
    slow = reference_transform(o, h, l, c)

    for name in DERIVED:
        err = float(np.max(np.abs(fast[name] - slow[name])))
        chk.check(
            f"vectorized == naive loop: {name}",
            err <= ATOL,
            f"max abs error {err:.3e}",
        )

    n_cf = min(200, len(df))
    cf = closed_form_ha_open(o[:n_cf], h[:n_cf], l[:n_cf], c[:n_cf])
    err_cf = float(np.max(np.abs(cf - fast["ha_open"][:n_cf])))
    chk.check(
        "closed-form HA open matches recursion (first 200 bars)",
        err_cf <= 1e-8,
        f"max abs error {err_cf:.3e}",
    )

    err_identity = float(
        np.max(np.abs(fast["avg_candle"] - (fast["hybrid_open"] + fast["hybrid_close"]) / 2.0))
    )
    chk.check(
        "identity: avg_candle == (hybrid_open + hybrid_close)/2",
        err_identity <= ATOL,
        f"max abs error {err_identity:.3e}",
    )

    mid_ok = bool(
        np.all(fast["candle_top"] >= fast["avg_candle"] - ATOL)
        and np.all(fast["avg_candle"] >= fast["candle_bottom"] - ATOL)
    )
    chk.check("ordering: candle_top >= avg_candle >= candle_bottom", mid_ok)

    want = np.where(o > c, (o + l) / 2.0, (c + h) / 2.0)
    err_sf = float(np.max(np.abs(fast["sourceformas"] - want)))
    chk.check("sourceformas branch follows open > close", err_sf <= ATOL,
              f"max abs error {err_sf:.3e}")

    explicit_novel = (fast["hlcc4"] + fast["avg_candle"] + fast["sourceformas"]) / 3.0
    err_ns = float(np.max(np.abs(fast["novelsrc"] - explicit_novel)))
    chk.check("novelsrc == mean(hlcc4, avg_candle, sourceformas)", err_ns <= ATOL,
              f"max abs error {err_ns:.3e}")

    nan_cols = [k for k in DERIVED if bool(np.isnan(fast[k]).any())]
    chk.check("no NaN in any derived series", not nan_cols, ", ".join(nan_cols) or "clean")

    # every derived series finite and on a plausible price scale
    scale_ok = all(
        np.all(np.isfinite(fast[k])) and
        np.all(fast[k] > 0.2 * c.min()) and np.all(fast[k] < 5.0 * c.max())
        for k in DERIVED
    )
    chk.check("derived series finite and on price scale", bool(scale_ok))

    frame = add_hybrid_candles(df)
    chk.check(
        "add_hybrid_candles appends exactly the documented columns",
        all(k in frame.columns for k in DERIVED) and len(frame.columns) == len(df.columns) + 10,
        f"{len(frame.columns)} columns",
    )


def check_frame_errors(chk: Checker) -> None:
    import hybrid_candles as hc

    try:
        hc.add_hybrid_candles(pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}))
        ok_min = False
        detail = "short frame was accepted"
    except hc.HybridCandleError as exc:
        ok_min = "at least" in str(exc)
        detail = str(exc)
    chk.check("short history rejected", ok_min, detail)

    try:
        hc.add_hybrid_candles(pd.DataFrame({"open": [1.0] * 10, "high": [1.0] * 10}))
        ok_cols = False
        detail = "missing columns were accepted"
    except hc.HybridCandleError as exc:
        ok_cols = "missing required column" in str(exc)
        detail = str(exc)
    chk.check("missing OHLC columns rejected", ok_cols, detail)

    upper = pd.DataFrame(
        {"Open": [1.0] * 10, "High": [1.1] * 10, "Low": [0.9] * 10, "Close": [1.05] * 10}
    )
    try:
        hc.add_hybrid_candles(upper)
        ok_case = True
        detail = "case-insensitive mapping ok"
    except hc.HybridCandleError as exc:
        ok_case = False
        detail = str(exc)
    chk.check("case-insensitive column mapping", ok_case, detail)


def check_structure(df: pd.DataFrame, chk: Checker) -> None:
    df = add_hybrid_candles(df)
    left = right = 3
    params = dict(pivot_left=left, pivot_right=right, buffer_mult=0.75, trendline_window=50)
    s = build_structure(df, **params)
    n = len(s)

    # --- causality: published structure must not depend on future bars -----------
    k = n - 120
    margin = 3 * (right + 1) + 5
    truncated = build_structure(df.iloc[:k].reset_index(drop=True), **params)
    limit = k - margin
    cols = [
        "last_confirmed_HH",
        "last_confirmed_HL",
        "last_confirmed_LH",
        "last_confirmed_LL",
    ]
    ok_causal = True
    detail = "identical"
    for col in cols:
        a = s[col].to_numpy()[:limit]
        b = truncated[col].to_numpy()[:limit]
        same = np.allclose(a, b, atol=1e-9, rtol=0, equal_nan=True)
        if not same:
            ok_causal = False
            bad = int(np.argmax(~np.isclose(a, b, atol=1e-9, rtol=0, equal_nan=True)))
            detail = f"{col} diverges at bar {bad}"
            break
    chk.check("no look-ahead: structure reproducible from truncated history", ok_causal, detail)

    # --- pivot confirmation lag --------------------------------------------------
    piv = find_pivots(s["hybrid_close"].to_numpy(), left, right)[0]
    idx = np.flatnonzero(piv)
    lag_ok = bool(np.all(s["pivot_high_confirmed_at"].to_numpy()[idx + right] == idx))
    chk.check("pivot confirmation lag == pivot_right bars", lag_ok,
              f"{len(idx)} pivots checked")

    # --- structural stop invariants ---------------------------------------------
    # Monotonicity applies WITHIN a contiguous run: the ratchet legitimately resets
    # (NaN gap) when the trend leg flips.
    sl = s["stop_long"].to_numpy()
    ss = s["stop_short"].to_numpy()

    def _within_runs_monotone(values: np.ndarray, *, increasing: bool) -> bool:
        ok = True
        run: list[float] = []
        for v in values:
            if np.isnan(v):
                ok = ok and _check_run(run, increasing)
                run = []
            else:
                run.append(float(v))
        return ok and _check_run(run, increasing)

    def _check_run(run: list[float], increasing: bool) -> bool:
        if len(run) < 2:
            return True
        d = np.diff(np.asarray(run))
        return bool(np.all(d >= -1e-9) if increasing else np.all(d <= 1e-9))

    chk.check("stop_long never loosens within a trade leg",
              _within_runs_monotone(sl, increasing=True))
    chk.check("stop_short never loosens within a trade leg",
              _within_runs_monotone(ss, increasing=False))

    hl = s["last_confirmed_HL"].to_numpy()
    both = ~np.isnan(sl) & ~np.isnan(hl)
    below_hl = bool(np.all(sl[both] < hl[both])) if both.any() else True
    chk.check("stop_long sits below the last confirmed HL", below_hl)

    lh = s["last_confirmed_LH"].to_numpy()
    both = ~np.isnan(ss) & ~np.isnan(lh)
    above_lh = bool(np.all(ss[both] > lh[both])) if both.any() else True
    chk.check("stop_short sits above the last confirmed LH", above_lh)

    # the ratchet must only tighten relative to the stateless structural level
    raw_l = s["stop_long_raw"].to_numpy()
    both = ~np.isnan(sl) & ~np.isnan(raw_l)
    chk.check("ratcheted stop_long >= raw structural stop",
              bool(np.all(sl[both] >= raw_l[both] - 1e-9)) if both.any() else True)

    raw_s = s["stop_short_raw"].to_numpy()
    both = ~np.isnan(ss) & ~np.isnan(raw_s)
    chk.check("ratcheted stop_short <= raw structural stop",
              bool(np.all(ss[both] <= raw_s[both] + 1e-9)) if both.any() else True)

    # a stop must not be carried across a regime flip: on range/downtrend bars the
    # long ratchet is undefined rather than stale
    tr = s["trend"].to_numpy()
    not_up = tr != "up"
    stale = bool(np.all(np.isnan(sl[not_up])))
    chk.check("long stop ratchet resets outside an uptrend leg", stale)

    # sanity: the live stop stays within a plausible distance of price
    hc_all = s["hybrid_close"].to_numpy()
    dist_l = np.abs(sl - hc_all) / hc_all
    dist_s = np.abs(ss - hc_all) / hc_all
    all_dist = np.concatenate([dist_l, dist_s])
    all_dist = all_dist[~np.isnan(all_dist)]
    med = float(np.median(all_dist))
    worst = float(np.max(all_dist))
    chk.check("median stop distance from price under 25%", med < 0.25, f"median {med:.3f}")
    chk.check("worst stop distance from price under 50%", worst < 0.5, f"worst {worst:.3f}")

    # buffer unit must be a distance on the candle's own scale, not a price level
    buf = s["buffer_distance"].to_numpy()
    buf_pct = buf / hc_all
    chk.check(
        "buffer distance is on candle scale (< 10% of price)",
        bool(np.nanmax(buf_pct) < 0.10),
        f"max {float(np.nanmax(buf_pct)):.4f} of price",
    )

    # --- break invariants --------------------------------------------------------
    bl = s["break_long"].to_numpy()
    hh = s["last_confirmed_HH"].to_numpy()
    hc = s["hybrid_close"].to_numpy()
    idx = np.flatnonzero(bl)
    break_ok = bool(np.all(hc[idx] > hh[idx])) if idx.size else True
    chk.check("every long break closes strictly beyond the HH", break_ok,
              f"{idx.size} breaks")

    bs = s["break_short"].to_numpy()
    ll = s["last_confirmed_LL"].to_numpy()
    idx = np.flatnonzero(bs)
    break_ok = bool(np.all(hc[idx] < ll[idx])) if idx.size else True
    chk.check("every short break closes strictly beyond the LL", break_ok,
              f"{idx.size} breaks")

    labels = s["pivot_label"].astype(str)
    known = set(labels.str.split("+").explode().unique()) - {""}
    chk.check("pivot labels limited to HH/HL/LH/LL", known <= {"HH", "HL", "LH", "LL"},
              ", ".join(sorted(known)))

    # --- trendline sanity --------------------------------------------------------
    tv = s["trendline_value"].to_numpy()
    chk.check("trendline covers most of the series",
              float(np.mean(~np.isnan(tv))) > 0.5,
              f"coverage {float(np.mean(~np.isnan(tv))):.2f}")

    # --- other structure sources -------------------------------------------------
    for src in ("novelsrc", "hybrid_open"):
        try:
            build_structure(df, structure_source=src, **params)
            ok = True
            detail = ""
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail = str(exc)
        chk.check(f"build_structure runs with source={src}", ok, detail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid framework + structure self-check.")
    ap.add_argument("--bars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    df = synthetic_ohlcv(args.bars, args.seed)
    chk = Checker()
    check_transform(df, chk)
    check_frame_errors(chk)
    check_structure(df, chk)

    failed = chk.failed
    if args.json:
        print(json.dumps(
            {"passed": len(chk.results) - len(failed), "failed": len(failed),
             "results": chk.results}, indent=2))
    else:
        for r in chk.results:
            flag = "PASS" if r["ok"] else "FAIL"
            extra = f"  [{r['detail']}]" if r["detail"] else ""
            print(f"{flag}  {r['check']}{extra}")
        print(f"\n{len(chk.results) - len(failed)}/{len(chk.results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
