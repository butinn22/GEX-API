#!/usr/bin/env python
"""Market-structure engine: pivots, HH/HL labelling, breaks, structural stops, trendlines.

All structure is measured on the hybrid series produced by ``hybrid_candles.py``
(never raw OHLC). See ``references/market_structure.md`` for the algorithm rationale
and parameter guidance.

Per-bar outputs (added columns)
------------------------------
pivot_high_raw, pivot_low_raw            geometric pivots (debug only)
pivot_high_confirmed_at, pivot_low_confirmed_at   bar index where the pivot becomes usable
pivot_label                              HH / HL / LH / LL on the confirmation bar
last_confirmed_HH, last_confirmed_HL     running structural levels (uptrend)
last_confirmed_LH, last_confirmed_LL     running structural levels (downtrend)
trend                                    up / down / range
break_long, break_short                  confirmed structural breaks
trendline_value, trendline_slope         dynamic lower/upper bound
stop_long, stop_short                    structural trailing stops (ratcheted)
stop_long_raw, stop_short_raw            stateless structural levels (pivot ± buffer)

Usage
-----
    python market_structure.py --input ohlcv.csv --output structure.csv
    python market_structure.py --input ohlcv.csv --pivot-left 3 --pivot-right 3 --json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

try:  # allow both package-style and flat-script imports
    from hybrid_candles import add_hybrid_candles
except ImportError:  # pragma: no cover
    from .hybrid_candles import add_hybrid_candles  # type: ignore

DEFAULTS = {
    "pivot_left": 3,
    "pivot_right": 3,
    "min_pivot_separation": None,  # -> pivot_right + 1
    "buffer_mult": 1.0,            # multiplier on the smoothed hybrid body range
    "buffer_window": 14,           # smoothing window for the noise unit
    "min_buffer_pct": 0.001,       # buffer floor, as a fraction of price
    "trendline_window": 100,
    "structure_source": "hybrid_close",  # one of hybrid_close / novelsrc / hybrid_open
    "warmup_bars": 30,
}

OUTPUT_COLUMNS = (
    "pivot_label",
    "last_confirmed_HH",
    "last_confirmed_HL",
    "last_confirmed_LH",
    "last_confirmed_LL",
    "trend",
    "break_long",
    "break_short",
    "trendline_value",
    "trendline_slope",
    "stop_long",
    "stop_short",
    "stop_long_raw",
    "stop_short_raw",
)


# --------------------------------------------------------------------------- #
# Pivots
# --------------------------------------------------------------------------- #
def find_pivots(
    values: np.ndarray, left: int, right: int, min_separation: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks of raw pivot highs and lows.

    A pivot at ``i`` is geometric truth but is only *knowable* at ``i + right``.
    Callers must apply that shift before using a pivot as a signal input.

    ``min_separation`` enforces spacing between same-type pivots: if two same-type
    pivots are closer than ``min_separation``, keep the more extreme one.

    Vectorized: the rolling window ``[i-left, i+right]`` maximum at index ``i`` equals
    ``values.rolling(w).max().shift(-right)[i]`` for ``w = left + right + 1``.
    """
    n = len(values)
    w = left + right + 1
    high_mask = np.zeros(n, dtype=bool)
    low_mask = np.zeros(n, dtype=bool)
    if n < w:
        return high_mask, low_mask

    s = pd.Series(np.asarray(values, dtype=np.float64))
    roll_max = s.rolling(w).max().shift(-right).to_numpy()
    roll_min = s.rolling(w).min().shift(-right).to_numpy()
    with np.errstate(invalid="ignore"):
        high_mask = np.asarray(values == roll_max) & ~np.isnan(roll_max)
        low_mask = np.asarray(values == roll_min) & ~np.isnan(roll_min)

    sep = right + 1 if min_separation is None else int(min_separation)
    return _thin(high_mask, values, sep, keep_max=True), _thin(
        low_mask, values, sep, keep_max=False
    )


def _thin(mask: np.ndarray, values: np.ndarray, sep: int, *, keep_max: bool) -> np.ndarray:
    """Drop same-type pivots that are closer than ``sep`` bars, keeping the extreme."""
    if sep <= 1:
        return mask
    out = mask.copy()
    idx = np.flatnonzero(mask)
    keep: list[int] = []
    for i in idx:
        if not keep or i - keep[-1] >= sep:
            keep.append(i)
        else:
            prev = keep[-1]
            better = values[i] > values[prev] if keep_max else values[i] < values[prev]
            if better:
                out[prev] = False
                keep[-1] = i
            else:
                out[i] = False
    return out


# --------------------------------------------------------------------------- #
# Structure series
# --------------------------------------------------------------------------- #
def build_structure(
    frame: pd.DataFrame,
    *,
    pivot_left: int = DEFAULTS["pivot_left"],
    pivot_right: int = DEFAULTS["pivot_right"],
    min_pivot_separation: int | None = DEFAULTS["min_pivot_separation"],
    buffer_mult: float = DEFAULTS["buffer_mult"],
    buffer_window: int = DEFAULTS["buffer_window"],
    min_buffer_pct: float = DEFAULTS["min_buffer_pct"],
    trendline_window: int = DEFAULTS["trendline_window"],
    structure_source: str = DEFAULTS["structure_source"],
) -> pd.DataFrame:
    """Return ``frame`` with the structure columns appended. Requires hybrid columns."""
    required = {
        structure_source,
        "avg_candle",
        "candle_top",
        "candle_bottom",
        "hybrid_open",
        "hybrid_close",
        "novelsrc",
    }
    missing = sorted(c for c in required if c not in frame.columns)
    if missing:
        raise ValueError(
            f"missing hybrid column(s) {missing}; call add_hybrid_candles() first"
        )

    out = frame.copy()
    src = out[structure_source].to_numpy(dtype=np.float64)
    n = len(out)
    high_mask, low_mask = find_pivots(src, pivot_left, pivot_right, min_pivot_separation)

    # --- confirmation shift: a pivot is usable only `pivot_right` bars later -----
    confirm_at_high = np.full(n, -1, dtype=np.int64)
    confirm_at_low = np.full(n, -1, dtype=np.int64)
    for i in np.flatnonzero(high_mask):
        c = i + pivot_right
        if c < n:
            confirm_at_high[c] = i
    for i in np.flatnonzero(low_mask):
        c = i + pivot_right
        if c < n:
            confirm_at_low[c] = i

    # --- HH/HL/LH/LL labelling on confirmation bars -----------------------------
    labels = np.array([""] * n, dtype=object)
    lvl_hh = np.full(n, np.nan)
    lvl_hl = np.full(n, np.nan)
    lvl_lh = np.full(n, np.nan)
    lvl_ll = np.full(n, np.nan)
    trend = np.array(["range"] * n, dtype=object)

    prev_high: float | None = None
    prev_low: float | None = None
    cur_hh = cur_hl = cur_lh = cur_ll = np.nan
    up_leg = False
    down_leg = False

    for i in range(n):
        if confirm_at_high[i] >= 0:
            p = int(confirm_at_high[i])
            level = float(src[p])
            if prev_high is not None:
                if level > prev_high:
                    labels[i] = "HH"
                    cur_hh = level
                else:
                    labels[i] = "LH"
                    cur_lh = level
                up_leg = level > prev_high
            prev_high = level
        if confirm_at_low[i] >= 0:
            p = int(confirm_at_low[i])
            level = float(src[p])
            if prev_low is not None:
                if level > prev_low:
                    labels[i] = (labels[i] + "+HL") if labels[i] else "HL"
                    cur_hl = level
                else:
                    labels[i] = (labels[i] + "+LL") if labels[i] else "LL"
                    cur_ll = level
                down_leg = level < prev_low
            prev_low = level

        lvl_hh[i], lvl_hl[i], lvl_lh[i], lvl_ll[i] = cur_hh, cur_hl, cur_lh, cur_ll
        if up_leg and not down_leg:
            trend[i] = "up"
        elif down_leg and not up_leg:
            trend[i] = "down"
        else:
            trend[i] = "range"

    out["pivot_high_raw"] = high_mask
    out["pivot_low_raw"] = low_mask
    out["pivot_high_confirmed_at"] = confirm_at_high
    out["pivot_low_confirmed_at"] = confirm_at_low
    out["pivot_label"] = labels
    out["last_confirmed_HH"] = lvl_hh
    out["last_confirmed_HL"] = lvl_hl
    out["last_confirmed_LH"] = lvl_lh
    out["last_confirmed_LL"] = lvl_ll
    out["trend"] = trend

    # --- structural breaks: strict hybrid_close beyond the confirmed level -------
    hc = out["hybrid_close"].to_numpy(dtype=np.float64)
    ns = out["novelsrc"].to_numpy(dtype=np.float64)
    ns_rising = np.zeros(n, dtype=bool)
    ns_rising[1:] = ns[1:] > ns[:-1]

    break_long = np.zeros(n, dtype=bool)
    break_short = np.zeros(n, dtype=bool)

    # one level, one signal: the level must be replaced by a fresh, more extreme one
    fired_hh: float | None = None
    fired_ll: float | None = None
    for i in range(n):
        hh = lvl_hh[i]
        if not np.isnan(hh) and (fired_hh is None or hh > fired_hh):
            if hc[i] > hh and ns_rising[i]:  # close-based break + smoothed agreement
                break_long[i] = True
                fired_hh = hh
                continue
        ll = lvl_ll[i]
        if not np.isnan(ll) and (fired_ll is None or ll < fired_ll):
            if hc[i] < ll and not ns_rising[i]:
                break_short[i] = True
                fired_ll = ll

    out["break_long"] = break_long
    out["break_short"] = break_short

    # --- structural trailing stops ----------------------------------------------
    # BUFFER UNIT. `avg_candle` is the hybrid body MID-POINT — a price level, not a
    # distance, so `buffer_mult * avg_candle` is dimensionally invalid (it would put
    # a stop ~75 price units away). The correct reading of "use avg_candle for the
    # buffer" is the candle's own dimension around that midpoint:
    #     candle_range = candle_top - candle_bottom == 2 * (avg_candle - candle_bottom)
    # Smoothed over `buffer_window` bars to survive single-bar dojis, and floored at
    # `min_buffer_pct` of price so a pinched hybrid body cannot produce a zero-width
    # stop. See references/market_structure.md §4.
    candle_range = (
        out["candle_top"].to_numpy(dtype=np.float64)
        - out["candle_bottom"].to_numpy(dtype=np.float64)
    )
    noise_unit = (
        pd.Series(candle_range).rolling(buffer_window, min_periods=1).mean().to_numpy()
    )
    hc_all = out["hybrid_close"].to_numpy(dtype=np.float64)
    noise_unit = np.maximum(noise_unit, min_buffer_pct * hc_all)
    buffer_distance = buffer_mult * noise_unit

    stop_long_raw = np.full(n, np.nan)
    stop_short_raw = np.full(n, np.nan)
    stop_long = np.full(n, np.nan)
    stop_short = np.full(n, np.nan)
    run_long = np.nan
    run_short = np.nan
    for i in range(n):
        hl = lvl_hl[i]
        if not np.isnan(hl):
            stop_long_raw[i] = hl - buffer_distance[i]
        lh = lvl_lh[i]
        if not np.isnan(lh):
            stop_short_raw[i] = lh + buffer_distance[i]

        # A stop only exists while its trend leg is intact. Leaving an uptrend leg
        # clears the ratchet; there is no long stop to carry into a range or downtrend.
        if trend[i] == "up":
            raw = stop_long_raw[i]
            if not np.isnan(raw):
                run_long = raw if np.isnan(run_long) else max(run_long, raw)
            stop_long[i] = run_long
        else:
            run_long = np.nan

        if trend[i] == "down":
            raw = stop_short_raw[i]
            if not np.isnan(raw):
                run_short = raw if np.isnan(run_short) else min(run_short, raw)
            stop_short[i] = run_short
        else:
            run_short = np.nan

    out["buffer_distance"] = buffer_distance
    out["stop_long_raw"] = stop_long_raw
    out["stop_short_raw"] = stop_short_raw
    out["stop_long"] = stop_long
    out["stop_short"] = stop_short

    # --- dynamic trendline (pivot-connected, fallback to rolling regression) -----
    tl_value, tl_slope = dynamic_trendline(
        out[structure_source].to_numpy(dtype=np.float64),
        out["hybrid_open"].to_numpy(dtype=np.float64),
        lvl_hl,
        lvl_lh,
        trend,
        window=trendline_window,
    )
    out["trendline_value"] = tl_value
    out["trendline_slope"] = tl_slope

    return out


def dynamic_trendline(
    src: np.ndarray,
    hybrid_open: np.ndarray,
    lvl_hl: np.ndarray,
    lvl_lh: np.ndarray,
    trend: np.ndarray,
    *,
    window: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor-based structural trendline with a rolling-regression fallback.

    Primary (Option A in the reference): line through the two most recent confirmed
    HL pivots in an uptrend, or LH pivots in a downtrend. Valid only while the slope
    has the correct sign. Falls back to (Option B) a rolling least-squares fit, which
    is fully vectorized and needs no pivot history.
    """
    n = len(src)
    value = np.full(n, np.nan)
    slope_out = np.full(n, np.nan)

    # --- Option B: rolling regression (used as fallback and as slope baseline) ---
    reg_value, reg_slope = rolling_regression(src, window)

    # --- Option A: pivot-connected line -----------------------------------------
    anchors: list[tuple[int, float, str]] = []
    for i in range(n):
        kind = "hl" if not np.isnan(lvl_hl[i]) else ("lh" if not np.isnan(lvl_lh[i]) else None)
        if kind is None:
            continue
        level = lvl_hl[i] if kind == "hl" else lvl_lh[i]
        if anchors and anchors[-1][1] == level and anchors[-1][2] == kind:
            continue  # unchanged level, same type -> not a new anchor
        anchors.append((i, float(level), kind))

    if len(anchors) >= 2:
        a0, a1 = anchors[-2], anchors[-1]
        if a1[0] > a0[0]:
            slope = (a1[1] - a0[1]) / (a1[0] - a0[0])
            xx = np.arange(n)
            line = a0[1] + slope * (xx - a0[0])
            valid_sign = (slope > 0) if a1[2] == "hl" else (slope < 0)
            if valid_sign:
                start = a1[0]
                value[start:] = line[start:]
                slope_out[start:] = slope

    use_reg = np.isnan(value)
    value[use_reg] = reg_value[use_reg]
    slope_out[use_reg] = reg_slope[use_reg]
    return value, slope_out


def rolling_regression(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized least-squares slope/intercept over a trailing window.

    ``x`` is taken as ``0..window-1`` inside each window, so the reported value is the
    fitted level **at the window end** (the current bar). O(n): all window sums come
    from cumulative sums of ``y`` and ``j*y``; no ``rolling.apply`` and no loop.

    Assumes a NaN-free input (forward-fill before calling).
    """
    y = np.asarray(series, dtype=np.float64)
    n = y.shape[0]
    value = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    if n < window or window < 2:
        return value, slope

    x = np.arange(window, dtype=np.float64)
    sx = x.sum()
    sxx = float((x * x).sum())
    denom = window * sxx - sx * sx

    j = np.arange(n, dtype=np.float64)
    cs_y = np.concatenate(([0.0], np.cumsum(y)))
    cs_jy = np.concatenate(([0.0], np.cumsum(j * y)))

    idx = np.arange(n)
    lo = idx - window + 1
    valid = lo >= 0
    iv = idx[valid]
    lv = lo[valid]

    sum_y = cs_y[iv + 1] - cs_y[lv]
    sum_jy = cs_jy[iv + 1] - cs_jy[lv]
    sxy = sum_jy - lv * sum_y  # shift x to 0..window-1 inside each window

    m = (window * sxy - sx * sum_y) / denom
    b_end = (sum_y - m * sx) / window
    value[valid] = b_end
    slope[valid] = m
    return value, slope


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load(path: str) -> pd.DataFrame:
    lowered = path.lower()
    if lowered.endswith(".parquet"):
        return pd.read_parquet(path)
    if lowered.endswith((".json", ".jsonl")):
        return pd.read_json(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hybrid-candle market structure engine.")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o")
    p.add_argument("--pivot-left", type=int, default=DEFAULTS["pivot_left"])
    p.add_argument("--pivot-right", type=int, default=DEFAULTS["pivot_right"])
    p.add_argument("--min-pivot-separation", type=int, default=None)
    p.add_argument("--buffer-mult", type=float, default=DEFAULTS["buffer_mult"],
                   help="multiplier on the smoothed hybrid body range")
    p.add_argument("--buffer-window", type=int, default=DEFAULTS["buffer_window"])
    p.add_argument("--min-buffer-pct", type=float, default=DEFAULTS["min_buffer_pct"],
                   help="buffer floor as a fraction of price (default 0.001)")
    p.add_argument("--trendline-window", type=int, default=DEFAULTS["trendline_window"])
    p.add_argument("--source", default=DEFAULTS["structure_source"],
                   choices=["hybrid_close", "novelsrc", "hybrid_open"])
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        frame = add_hybrid_candles(_load(args.input))
        struct = build_structure(
            frame,
            pivot_left=args.pivot_left,
            pivot_right=args.pivot_right,
            min_pivot_separation=args.min_pivot_separation,
            buffer_mult=args.buffer_mult,
            buffer_window=args.buffer_window,
            min_buffer_pct=args.min_buffer_pct,
            trendline_window=args.trendline_window,
            structure_source=args.source,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    if args.output:
        struct.to_csv(args.output, index=False)

    labels = struct["pivot_label"].astype(str)
    summary = {
        "rows": int(len(struct)),
        "counts": {k: int((labels.str.contains(k)).sum()) for k in ("HH", "HL", "LH", "LL")},
        "break_long": int(struct["break_long"].sum()),
        "break_short": int(struct["break_short"].sum()),
        "trend_last": str(struct["trend"].iloc[-1]),
        "last_bar": {
            "last_confirmed_HH": _f(struct["last_confirmed_HH"].iloc[-1]),
            "last_confirmed_HL": _f(struct["last_confirmed_HL"].iloc[-1]),
            "last_confirmed_LH": _f(struct["last_confirmed_LH"].iloc[-1]),
            "last_confirmed_LL": _f(struct["last_confirmed_LL"].iloc[-1]),
            "stop_long": _f(struct["stop_long"].iloc[-1]),
            "stop_short": _f(struct["stop_short"].iloc[-1]),
            "trendline_value": _f(struct["trendline_value"].iloc[-1]),
            "trendline_slope": _f(struct["trendline_slope"].iloc[-1]),
        },
        "params": {
            "pivot_left": args.pivot_left,
            "pivot_right": args.pivot_right,
            "buffer_mult": args.buffer_mult,
            "buffer_window": args.buffer_window,
            "min_buffer_pct": args.min_buffer_pct,
            "trendline_window": args.trendline_window,
            "structure_source": args.source,
        },
        "buffer_pct_of_price_median": _frac_of_price(struct, "buffer_distance"),
        "stop_distance_pct_median": {
            "long": _frac_of_price(struct, "stop_long"),
            "short": _frac_of_price(struct, "stop_short"),
        },
        "confirmation_lag_bars": args.pivot_right,
        "output": args.output,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        c = summary["counts"]
        print(f"rows={summary['rows']} HH={c['HH']} HL={c['HL']} LH={c['LH']} LL={c['LL']}")
        print(f"breaks long={summary['break_long']} short={summary['break_short']} "
              f"trend(last)={summary['trend_last']}")
        print(f"confirmation lag = {summary['confirmation_lag_bars']} bars "
              f"(signals may only use pivots confirmed on/before the current bar)")
        if args.output:
            print(f"wrote {args.output}")
    return 0


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _frac_of_price(struct: pd.DataFrame, column: str) -> float | None:
    """Median of ``column`` expressed as a fraction of price.

    Use for *distance* columns (``buffer_distance``). For *level* columns
    (``stop_long``) the distance from price is what matters — see
    :func:`_pct_of_price`.
    """
    if column not in struct.columns:
        return None
    price = struct["hybrid_close"].to_numpy(dtype=np.float64)
    vals = struct[column].to_numpy(dtype=np.float64)
    if column.startswith("stop"):
        vals = np.abs(vals - price)
    frac = vals / price
    frac = frac[~np.isnan(frac)]
    return float(np.median(frac)) if frac.size else None


if __name__ == "__main__":
    sys.exit(main())
