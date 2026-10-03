"""Signal construction: the real EMF+ADL columns, plus the improvement layer.

Everything in this module is *causal*: row ``i`` of every array depends only on bars
``0..i``. The engine is responsible for the one-bar execution delay.

Two transforms are wired up
---------------------------
``repair_hybrid=False`` — the strategy exactly as it exists in ``gex/strategy`` today.
``repair_hybrid=True``  — the same strategy with a **repaired** Heikin-Ashi open.

The repair exists because ``_FeaturesMixin._heikin_ashi`` builds ``ha_open`` with a
single numpy-free pandas expression::

    ha_open.iloc[1:] = (ha_open.shift(1).iloc[1:] + ha_close.shift(1).iloc[1:]) / 2

``ha_open.shift(1)`` is evaluated **once, from the initial all-NaN series**, so the
recursion never propagates: index 0 and 1 are correct and index 2..n-1 stay NaN. On
real BTCUSDT 4H data that is 911 of 913 bars NaN.

The NaN then disappears silently downstream, because ``DataFrame.max(axis=1)`` skips
NaN. The consequences, measured on the real frame:

* ``candle_top == candle_bottom == hybrid_close``  → the hybrid body has **zero range**.
* ``avg_candle == hybrid_close`` instead of ``(hybrid_open + hybrid_close) / 2``.
* ``median_top`` collapses to ``max(open, close)``.
* ``novelsrc`` — the single source behind every EMA, VWAP context, entry and exit in
  this strategy — is computed from the degenerate body.

The vendored reference transform (``vendor/hybrid_candles.py``, 39/39 self-checks)
computes the recursion correctly and is used as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from gex.strategy.features import _FeaturesMixin
from gex.strategy.settings import StrategySettings
from gex.strategy.trading_algorithm import EMAFilterTrendStrategy

from .engine import SignalSet

#: Bars consumed before any signal may fire. Driven by ``ema200(novelsrc)`` and
#: ``adl200``; 400 bars leaves <2% residual weight in a span-200 EWM. Disclosed, and
#: applied identically to every variant so the comparison stays fair.
WARMUP = 400

_ORIGINAL_HEIKIN_ASHI = _FeaturesMixin._heikin_ashi


def project_hybrid_is_repaired() -> bool:
    """Is the project's own ``_heikin_ashi`` already correct?

    Probed, not assumed. ``gex/strategy/features.py`` was repaired on 2026-10-03, so in a
    patched tree ``repair_hybrid=True`` is a **no-op** and the recorded ``V0_shipped`` /
    ``V1_repaired`` distinction no longer exists. Any report that claims a difference
    between them in a patched tree is mislabelled, so every entry point asks this first.
    """
    probe = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "high": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "low": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        }
    )
    got = np.asarray(_ORIGINAL_HEIKIN_ASHI(probe.copy(), pd)["ha_open"], dtype=float)
    ref = np.asarray(_heikin_ashi_fixed(probe.copy(), pd)["ha_open"], dtype=float)
    return bool(np.allclose(got, ref, rtol=0.0, atol=1e-9))


def _heikin_ashi_fixed(data: pd.DataFrame, pd_module) -> pd.DataFrame:
    """Correct HA recursion: each ``ha_open`` depends on the *updated* previous one."""
    o = data["open"].to_numpy(dtype=float)
    h = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    c = data["close"].to_numpy(dtype=float)
    n = len(c)
    ha_close = (o + h + low + c) / 4.0
    ha_open = np.empty(n, dtype=float)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return pd_module.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": np.maximum.reduce([h, ha_open, ha_close]),
            "ha_low": np.minimum.reduce([low, ha_open, ha_close]),
            "ha_close": ha_close,
        },
        index=data.index,
    )


class hybrid_repair:
    """Context manager: swap in the corrected HA transform for the duration.

    Scoped and reversible, so the same process can measure both the shipped and the
    repaired strategy without cross-contamination.
    """

    def __enter__(self):
        _FeaturesMixin._heikin_ashi = staticmethod(_heikin_ashi_fixed)
        return self

    def __exit__(self, *exc):
        _FeaturesMixin._heikin_ashi = staticmethod(_ORIGINAL_HEIKIN_ASHI)
        return False


#: True when the project's own HA recursion has already been fixed upstream. Evaluated
#: after ``_heikin_ashi_fixed`` exists.
PROJECT_ALREADY_REPAIRED = project_hybrid_is_repaired()


def to_ohlcv(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": bars["open"].to_numpy(dtype=float),
            "high": bars["high"].to_numpy(dtype=float),
            "low": bars["low"].to_numpy(dtype=float),
            "close": bars["close"].to_numpy(dtype=float),
            "volume": bars["volume"].to_numpy(dtype=float),
        }
    )


def build_frame(
    bars: pd.DataFrame,
    settings: StrategySettings | None = None,
    *,
    repair_hybrid: bool = False,
) -> pd.DataFrame:
    """Run the real strategy pipeline and return the full feature frame."""
    strat = EMAFilterTrendStrategy(settings=settings or StrategySettings())
    ohlc = to_ohlcv(bars)
    if repair_hybrid:
        with hybrid_repair():
            return strat.calculate(ohlc, include_decorative=False)
    return strat.calculate(ohlc, include_decorative=False)


def base_signals(
    bars: pd.DataFrame,
    settings: StrategySettings | None = None,
    *,
    repair_hybrid: bool = False,
    warmup: int = WARMUP,
    long_only: bool = False,
    short_only: bool = False,
) -> tuple[SignalSet, pd.DataFrame]:
    """The EMF+ADL entry/exit columns as causal boolean arrays."""
    f = build_frame(bars, settings, repair_hybrid=repair_hybrid)

    def col(name: str) -> np.ndarray:
        v = f[name].to_numpy()
        return np.nan_to_num(v.astype(bool), nan=0).astype(bool)

    n = len(f)
    allowed = np.zeros(n, dtype=bool)
    allowed[warmup:] = True

    entry_long = col("long_entry_signal") & allowed
    entry_short = col("short_entry_signal") & allowed
    exit_long = col("long_exit_signal")
    exit_short = col("short_exit_signal")
    if long_only:
        entry_short = np.zeros(n, dtype=bool)
    if short_only:
        entry_long = np.zeros(n, dtype=bool)

    atr = pd.to_numeric(f["atr"], errors="coerce").to_numpy(dtype=float)
    ss = SignalSet(
        entry_long=entry_long,
        entry_short=entry_short,
        exit_long=exit_long,
        exit_short=exit_short,
        atr=np.nan_to_num(atr, nan=0.0),
        warmup=warmup,
        label="emf_adl" + ("_repaired" if repair_hybrid else ""),
    )
    return ss, f


# --------------------------------------------------------------------------- #
# Improvement layer: gates and structural features
# --------------------------------------------------------------------------- #
@dataclass
class GateSpec:
    """A causal entry filter. ``None`` means 'no gate'."""

    kind: str = "none"
    #: volatility band, as a ratio of ATR to its own long-run average
    vol_low: float = 0.5
    vol_high: float = 2.5
    vol_window: int = 100
    #: for ``vol_quantile``: the quantile cut-points and the trailing window they use
    vol_quantile_window: int = 300
    #: structural break window (hybrid candles)
    struct_pivot_left: int = 3
    struct_pivot_right: int = 3
    #: trend filter: require |close/ema| alignment
    trend_ema: int = 200
    allow_shorts: bool = True


def atr_ratio(f: pd.DataFrame, window: int = 100) -> np.ndarray:
    """ATR divided by its own trailing average — a scale-free vol regime measure."""
    atr = pd.to_numeric(f["atr"], errors="coerce").to_numpy(dtype=float)
    s = pd.Series(atr)
    base = s.rolling(window, min_periods=max(5, window // 4)).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((base > 0) & np.isfinite(base), atr / base, np.nan)
    return r


def structural_arrays(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    """Hybrid candles + confirmed market structure (vendored reference transform).

    Pivots become usable only ``pivot_right`` bars after they form; the vendor's
    ``build_structure`` already applies that confirmation shift, so nothing here can
    see the future.
    """
    from .vendor.hybrid_candles import add_hybrid_candles
    from .vendor.market_structure import build_structure

    ohlc = to_ohlcv(bars)
    hyb = add_hybrid_candles(ohlc)
    st = build_structure(hyb)
    out: dict[str, np.ndarray] = {}
    for name in (
        "trend", "pivot_label", "pivot_high_confirmed_at", "pivot_low_confirmed_at",
        "last_confirmed_HH", "last_confirmed_HL", "last_confirmed_LH", "last_confirmed_LL",
        "break_long", "break_short", "buffer_distance",
        "stop_long_raw", "stop_short_raw", "stop_long", "stop_short",
        "trendline_value", "trendline_slope",
    ):
        if name in st.columns:
            out[name] = st[name].to_numpy()
    out["_frame"] = st
    return out


def build_gate(
    bars: pd.DataFrame, f: pd.DataFrame, spec: GateSpec, side: str
) -> np.ndarray | None:
    """Return a per-bar boolean entry gate, or ``None`` when no gating is requested."""
    n = len(bars)
    if spec.kind == "none":
        return None

    if spec.kind == "vol_band":
        r = atr_ratio(f, spec.vol_window)
        ok = np.isfinite(r) & (r >= spec.vol_low) & (r <= spec.vol_high)
        return ok

    if spec.kind == "vol_quantile":
        # Self-calibrating regime filter. An absolute band on atr/mean(atr) is
        # non-binding in practice — the ratio lives in [0.68, 1.93] across a decade of
        # crypto — so the band is expressed as quantiles of its own trailing window,
        # which binds at a fixed rate on every asset and every timeframe.
        r = pd.Series(atr_ratio(f, spec.vol_window))
        w = spec.vol_quantile_window
        lo = r.rolling(w, min_periods=max(20, w // 4)).quantile(spec.vol_low).to_numpy()
        hi = r.rolling(w, min_periods=max(20, w // 4)).quantile(spec.vol_high).to_numpy()
        rv = r.to_numpy()
        return np.isfinite(rv) & np.isfinite(lo) & np.isfinite(hi) & (rv >= lo) & (rv <= hi)

    if spec.kind == "trend":
        close = bars["close"].to_numpy(dtype=float)
        ema = pd.Series(close).ewm(span=spec.trend_ema, adjust=False).mean().to_numpy()
        return (close > ema) if side == "long" else (close < ema)

    if spec.kind == "struct_break":
        st = structural_arrays(bars)
        trend = st.get("trend")
        if trend is None:
            return None
        if side == "long":
            return np.array([t == "up" for t in trend], dtype=bool)
        return np.array([t == "down" for t in trend], dtype=bool)

    if spec.kind == "struct_break_event":
        st = structural_arrays(bars)
        key = "break_long" if side == "long" else "break_short"
        arr = st.get(key)
        if arr is None:
            return None
        return np.nan_to_num(arr.astype(bool), nan=0).astype(bool)

    raise ValueError(f"unknown gate kind {spec.kind!r}")


def combine_gates(*gates: np.ndarray | None) -> np.ndarray | None:
    real = [g for g in gates if g is not None]
    if not real:
        return None
    out = real[0].copy()
    for g in real[1:]:
        out &= g
    return out


def structural_raw_arrays(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The un-buffered structural pivot levels, used to scale the stop buffer."""
    st = structural_arrays(bars)
    lo, sh = st.get("stop_long_raw"), st.get("stop_short_raw")
    n = len(bars)
    return (
        np.asarray(lo, dtype=float) if lo is not None else np.full(n, np.nan),
        np.asarray(sh, dtype=float) if sh is not None else np.full(n, np.nan),
    )


def structural_stop_arrays(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Structural trailing-stop levels from the vendored market-structure builder.

    ``stop_long``/``stop_short`` are already ratcheted by the vendor (self-check:
    "stop_long never loosens within a trade leg") and sit beyond the last confirmed
    pivot by a hybrid-body-range buffer. The buffer unit is ``candle_top -
    candle_bottom`` — never ``avg_candle``, which is a price level, not a distance.
    """
    st = structural_arrays(bars)
    long_sl = st.get("stop_long")
    short_sl = st.get("stop_short")
    if long_sl is None:
        nan = np.full(len(bars), np.nan)
        return nan, nan
    return (
        np.asarray(long_sl, dtype=float),
        np.asarray(short_sl, dtype=float),
    )
