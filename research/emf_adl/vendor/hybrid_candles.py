#!/usr/bin/env python
"""Hybrid Candle transform (proprietary Standard/Heikin-Ashi framework).

Deterministic, self-contained implementation of the framework documented in
``references/hybrid_candle_math.md``. Import it; do not reimplement it inside
strategy code.

Added columns
-------------
ha_open, ha_close, hybrid_open, hybrid_close, candle_top, candle_bottom,
avg_candle, sourceformas, hlcc4, novelsrc

Usage
-----
    python hybrid_candles.py --input ohlcv.csv --output hybrid.csv
    python hybrid_candles.py --input ohlcv.csv --json

Import
------
    from hybrid_candles import add_hybrid_candles
    df = add_hybrid_candles(df)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

REQUIRED = ("open", "high", "low", "close")
DERIVED = (
    "ha_open",
    "ha_close",
    "hybrid_open",
    "hybrid_close",
    "candle_top",
    "candle_bottom",
    "avg_candle",
    "sourceformas",
    "hlcc4",
    "novelsrc",
)
# Bars dropped from evaluation by default: the HA recursion seed and the earliest
# derivative values are not trustworthy before this horizon.
DEFAULT_WARMUP = 30
MIN_BARS = 5


class HybridCandleError(ValueError):
    """Raised when the input frame cannot support the transform."""


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #
def reference_transform(open_, high, low, close) -> dict:
    """Naive loop-based reference transform. Test-only ground truth.

    Deliberately written with Python scalars in explicit per-bar loops so that it is
    an independent implementation of :func:`compute_arrays`, not a copy of it.
    """
    n = len(close)
    o = [float(x) for x in open_]
    h = [float(x) for x in high]
    l = [float(x) for x in low]
    c = [float(x) for x in close]

    ha_close = [(o[i] + h[i] + l[i] + c[i]) / 4.0 for i in range(n)]
    ha_open = [0.0] * n
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    hybrid_open = [(o[i] + ha_open[i]) / 2.0 for i in range(n)]
    hybrid_close = [(c[i] + ha_close[i]) / 2.0 for i in range(n)]
    candle_top = [max(hybrid_open[i], hybrid_close[i]) for i in range(n)]
    candle_bottom = [min(hybrid_open[i], hybrid_close[i]) for i in range(n)]
    avg_candle = [(candle_top[i] + candle_bottom[i]) / 2.0 for i in range(n)]
    sourceformas = [
        (o[i] + l[i]) / 2.0 if o[i] > c[i] else (c[i] + h[i]) / 2.0 for i in range(n)
    ]
    hlcc4 = [(h[i] + l[i] + c[i] + c[i]) / 4.0 for i in range(n)]
    novelsrc = [
        (hlcc4[i] + avg_candle[i] + sourceformas[i]) / 3.0 for i in range(n)
    ]
    return {
        "ha_open": np.asarray(ha_open),
        "ha_close": np.asarray(ha_close),
        "hybrid_open": np.asarray(hybrid_open),
        "hybrid_close": np.asarray(hybrid_close),
        "candle_top": np.asarray(candle_top),
        "candle_bottom": np.asarray(candle_bottom),
        "avg_candle": np.asarray(avg_candle),
        "sourceformas": np.asarray(sourceformas),
        "hlcc4": np.asarray(hlcc4),
        "novelsrc": np.asarray(novelsrc),
    }


def closed_form_ha_open(open_, high, low, close) -> np.ndarray:
    """Analytic form of the HA open recursion. NOT numerically stable.

    ``ha_open[i] = ha_open[0]/2**i + sum_{k=1..i} ha_close[i-k]/2**k``

    Exact for short series; term magnitudes underflow to 0 beyond roughly 1000 bars.
    Provided only to demonstrate that the sequential form in :func:`compute_arrays` is
    correct. Never use in production.
    """
    n = len(close)
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    ha_close = (o + h + l + c) / 4.0
    out = np.empty(n, dtype=np.float64)
    seed = (o[0] + c[0]) / 2.0
    for i in range(n):
        acc = seed / (2.0 ** i)
        for k in range(1, i + 1):
            acc += ha_close[i - k] / (2.0 ** k)
        out[i] = acc
    return out


def compute_arrays(open_, high, low, close, *, use_numba: bool = False) -> dict:
    """Vectorized transform over numpy arrays.

    Only the ``ha_open`` recursion is sequential. It runs over pre-extracted float64
    buffers with no pandas indexing — the correct answer to "vectorize the Heikin-Ashi
    open". Every later step is fully vectorized with ``np.where`` / ``np.maximum`` /
    ``np.minimum``. ``use_numba`` opts into a JIT-compiled recursion when ``numba`` is
    installed; it is never required.
    """
    o = np.ascontiguousarray(open_, dtype=np.float64)
    h = np.ascontiguousarray(high, dtype=np.float64)
    l = np.ascontiguousarray(low, dtype=np.float64)
    c = np.ascontiguousarray(close, dtype=np.float64)
    if not (o.shape == h.shape == l.shape == c.shape):
        raise HybridCandleError("open/high/low/close must have identical length")
    n = c.shape[0]
    if n == 0:
        raise HybridCandleError("empty series")

    # Step 1 — Heikin-Ashi base
    ha_close = (o + h + l + c) / 4.0
    ha_open = _ha_open_recursion(o, c, ha_close, use_numba=use_numba)

    # Step 2 — hybrid blending
    hybrid_open = (o + ha_open) / 2.0
    hybrid_close = (c + ha_close) / 2.0

    # Step 3 — candle extremes
    candle_top = np.maximum(hybrid_open, hybrid_close)
    candle_bottom = np.minimum(hybrid_open, hybrid_close)

    # Step 4 — derived sources
    avg_candle = (candle_top + candle_bottom) / 2.0  # == (hybrid_open + hybrid_close)/2
    sourceformas = np.where(o > c, (o + l) / 2.0, (c + h) / 2.0)
    hlcc4 = (h + l + c + c) / 4.0

    # Step 5 — master source
    novelsrc = (hlcc4 + avg_candle + sourceformas) / 3.0

    return {
        "ha_open": ha_open,
        "ha_close": ha_close,
        "hybrid_open": hybrid_open,
        "hybrid_close": hybrid_close,
        "candle_top": candle_top,
        "candle_bottom": candle_bottom,
        "avg_candle": avg_candle,
        "sourceformas": sourceformas,
        "hlcc4": hlcc4,
        "novelsrc": novelsrc,
    }


def _ha_open_recursion(
    o: np.ndarray, c: np.ndarray, ha_close: np.ndarray, *, use_numba: bool = False
) -> np.ndarray:
    """Sequential HA open. Seed = ``(open[0] + close[0]) / 2`` (PineScript ``nz`` fallback)."""
    n = ha_close.shape[0]
    seed = (o[0] + c[0]) / 2.0

    if use_numba:  # pragma: no cover - optional accelerator
        try:
            from numba import njit  # type: ignore

            @njit(cache=True)
            def _loop(seed_value, ha_c):  # noqa: ANN001
                out = np.empty(ha_c.shape[0], dtype=np.float64)
                out[0] = seed_value
                for i in range(1, ha_c.shape[0]):
                    out[i] = 0.5 * out[i - 1] + 0.5 * ha_c[i - 1]
                return out

            return _loop(seed, ha_close)
        except Exception:  # noqa: BLE001 - fall through to the numpy loop
            pass

    out = np.empty(n, dtype=np.float64)
    out[0] = seed
    for i in range(1, n):
        out[i] = 0.5 * out[i - 1] + 0.5 * ha_close[i - 1]
    return out


# --------------------------------------------------------------------------- #
# Frame-level API
# --------------------------------------------------------------------------- #
def _resolve_columns(df: pd.DataFrame) -> dict:
    """Map required columns case-insensitively. Returns {canonical: actual_name}."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in lower]
    if missing:
        raise HybridCandleError(
            f"missing required column(s): {', '.join(missing)}; "
            f"found: {', '.join(map(str, df.columns))}"
        )
    return {c: lower[c] for c in REQUIRED}


def add_hybrid_candles(
    df: pd.DataFrame,
    *,
    forward_fill: bool = True,
    min_bars: int = MIN_BARS,
    drop_warmup: int = 0,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the ten hybrid columns appended.

    Parameters
    ----------
    forward_fill : forward-fill the required OHLC columns before transforming. A NaN
        entering ``ha_open`` poisons every subsequent bar, so this is on by default.
    min_bars : refuse to transform shorter histories.
    drop_warmup : also drop the first ``drop_warmup`` rows from the returned frame.
        Default 0 — dropping is a backtest concern, applied by the caller.
    """
    if not isinstance(df, pd.DataFrame):
        raise HybridCandleError("input must be a pandas DataFrame")
    if len(df) < min_bars:
        raise HybridCandleError(
            f"need at least {min_bars} bars to define the hybrid transform, got {len(df)}"
        )

    cols = _resolve_columns(df)
    out = df.copy()
    work = out[[cols[c] for c in REQUIRED]].astype("float64")
    work.columns = list(REQUIRED)

    if forward_fill:
        work = work.ffill()
    bad = work.isna().sum()
    if int(bad.sum()) > 0:
        detail = ", ".join(f"{k}={int(v)}" for k, v in bad.items() if v)
        raise HybridCandleError(f"NaN values remain after forward-fill: {detail}")

    arrays = compute_arrays(
        work["open"].to_numpy(),
        work["high"].to_numpy(),
        work["low"].to_numpy(),
        work["close"].to_numpy(),
    )
    for name, values in arrays.items():
        out[name] = values

    if drop_warmup:
        out = out.iloc[int(drop_warmup):].reset_index(drop=True)
    return out


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
    parser = argparse.ArgumentParser(description="Compute the Hybrid Candle framework.")
    parser.add_argument("--input", "-i", required=True, help="CSV/Parquet/JSON with OHLC columns")
    parser.add_argument("--output", "-o", help="write the augmented frame here (CSV)")
    parser.add_argument("--no-ffill", action="store_true", help="do not forward-fill inputs")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"bars to drop before evaluation (default {DEFAULT_WARMUP})")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args(argv)

    try:
        raw = _load(args.input)
        frame = add_hybrid_candles(raw, forward_fill=not args.no_ffill)
    except (HybridCandleError, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    if args.output:
        frame.to_csv(args.output, index=False)

    warmup = int(min(max(args.warmup, 0), max(0, len(frame) - 1)))
    summary = {
        "rows": int(len(frame)),
        "warmup_bars_to_drop": warmup,
        "columns_added": list(DERIVED),
        "last_bar": {k: float(frame[k].iloc[-1]) for k in DERIVED},
        "identity_avg_candle_max_abs_error": float(
            np.max(np.abs(frame["avg_candle"] - (frame["hybrid_open"] + frame["hybrid_close"]) / 2.0))
        ),
        "output": args.output,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"rows={summary['rows']} -> added {len(DERIVED)} hybrid columns")
        print(
            f"last bar: novelsrc={summary['last_bar']['novelsrc']:.6f} "
            f"hybrid_close={summary['last_bar']['hybrid_close']:.6f} "
            f"avg_candle={summary['last_bar']['avg_candle']:.6f}"
        )
        print(f"drop >= {warmup} warm-up bars before evaluating")
        if args.output:
            print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
