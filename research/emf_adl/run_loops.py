"""Driver: run the optimisation loops over real data and log everything.

Usage
-----
    python -m research.emf_adl.run_loops --stage ab
    python -m research.emf_adl.run_loops --stage loops
    python -m research.emf_adl.run_loops --stage holdout
    python -m research.emf_adl.run_loops --stage robustness

Structure
---------
A ``SeriesContext`` is built once per (symbol, timeframe) and holds the priced bars, the
real funding series, the EMF+ADL signal columns for both the shipped and the repaired
hybrid transform, and the cached structural arrays. Variants then differ only by gate
and stop rules, so a whole battery costs seconds rather than minutes — and none of that
caching touches the signal maths, which is recomputed from the real pipeline for every
repair flag.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import loop as L
from . import rules as R
from .engine import Costs, StopSpec, run
from .metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emf_adl")

OUT = Path(__file__).with_name("out")
OUT.mkdir(exist_ok=True)

STUDY_START = (2021, 7, 1)
STUDY_END = (2026, 10, 1)
#: Cache for BTC's regime series, keyed by (timeframe, start, end).
_BTC_REGIME_CACHE: dict = {}
#: Walk-forward splits are expressed as *absolute UTC dates*. Deriving them from a
#: fraction of each symbol's own history would give every ticker a different calendar
#: window, so a "validation period" would mean five different things in one panel and
#: cross-ticker comparison would be meaningless.
SPLITS = {
    "train": ((2021, 7, 1), (2024, 1, 1)),
    "valid": ((2024, 1, 1), (2025, 6, 1)),
    "holdout": ((2025, 6, 1), (2026, 10, 1)),
}

EXCLUDE = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT", "USDEUSDT",
    "USD1USDT", "EURUSDT",
}

#: Tokenised traditional-finance instruments listed on Bybit's linear venue. The EMF+ADL
#: logic is a crypto trend model; leaving semiconductor ETFs, gold, crude oil and
#: single-stock perps in the panel would silently turn the study into a multi-asset
#: portfolio and make every per-group comparison meaningless. Excluded and disclosed.
TRADFI_SYMBOLS = {
    "SOXLUSDT", "SNDKUSDT", "MSTRUSDT", "MUUSDT", "SPCXUSDT", "SKHYUSDT",
    "KORUUSDT", "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT", "GOOGUSDT",
    "METAUSDT", "AMZNUSDT", "COINUSDT", "HOODUSDT", "QQQUSDT", "SPYUSDT",
    "CRCLUSDT", "XAUUSDT", "XAUTUSDT", "XAGUSDT", "CLUSDT", "BZUSDT",
    "USOILUSDT", "WTIUSDT", "BRENTUSDT", "GBPUSDT", "JPYUSDT", "AUDUSDT",
    "CADUSDT", "CHFUSDT",
}
STABLE_SYMBOLS = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT", "USDEUSDT",
    "USD1USDT", "USDTUSDT", "PYUSDUSDT",
}
TIMEFRAMES = ("4H", "1D")


def select_universe(*, max_candidates: int = 60, workers: int = 6) -> dict:
    """The single source of truth for which tickers are studied.

    Selection rule, stated plainly so its bias can be argued with:

    1. Rank every linear USDT perpetual by **real 24h turnover** (Bybit
       ``/v5/market/tickers``), highest first.
    2. Drop stablecoins and tokenised traditional-finance instruments (disclosed lists).
    3. Keep only series that **cover the full study calendar on both timeframes**, so
       every ticker is present in every walk-forward window.

    Known bias, disclosed rather than hidden: the ranking uses *today's* turnover, so
    coins that were small in 2021 and large now are over-represented. Point-in-time
    turnover rankings are not available from the public endpoint. Every surviving coin
    must have traded since 2021-07, which removes the worst of it (no coin can enter the
    study by being launched and pumped in 2025), but the residual survivorship bias
    makes these results, if anything, optimistic. It does not manufacture the negative
    results, which is the direction that matters for a rejection.
    """
    top = D.fetch_top_symbols(120)
    cands, excluded = [], []
    for _, row in top.iterrows():
        sym = row["symbol"]
        if sym in STABLE_SYMBOLS:
            excluded.append({"symbol": sym, "reason": "stablecoin pair"})
        elif sym in TRADFI_SYMBOLS:
            excluded.append({"symbol": sym, "reason": "tokenised traditional finance"})
        elif len(cands) < max_candidates:
            cands.append(sym)

    start, end = D.ms(*STUDY_START), D.ms(*STUDY_END)
    D.prefetch_panel(cands, TIMEFRAMES, start, end, workers=workers)

    selected, rejected = [], []
    for sym in cands:
        try:
            ctx_ok = True
            row = {"symbol": sym, "turnover24h": float(top.loc[top["symbol"] == sym, "turnover24h"].iloc[0])}
            for tf in TIMEFRAMES:
                s = D.load_series(sym, tf, start, end)
                _calendar_wins(s.bars.reset_index(drop=True),
                               60 if tf == "4H" else 30)
                row["n_bars" + tf] = int(len(s.bars))
                row["coverage" + tf] = s.manifest.integrity.get("coverage")
                row["first_bar" + tf] = s.manifest.integrity.get("first_bar_utc")
                row["n_funding" + tf] = int(len(s.funding))
            if ctx_ok:
                selected.append(row)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"symbol": sym, "reason": str(exc)[:160]})

    return {
        "selection_rule": (
            "top real 24h turnover USDT perpetuals (Bybit /v5/market/tickers), "
            "crypto only, must cover the full study calendar on 4H and 1D"
        ),
        "source": f"{D.BYBIT} (Bybit v5 public market data)",
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "study_window_utc": [str(pd.Timestamp(D.ms(*STUDY_START), unit="ms", tz="UTC")),
                             str(pd.Timestamp(D.ms(*STUDY_END), unit="ms", tz="UTC"))],
        "n_candidates": len(cands),
        "symbols": [r["symbol"] for r in selected],
        "selected": selected,
        "rejected_short_history": rejected,
        "excluded_non_crypto": excluded,
        "survivorship_bias_disclosure": (
            "Ticker ranking uses current 24h turnover, not point-in-time turnover. "
            "Residual survivorship bias inflates results; it cannot create the negative "
            "out-of-sample results reported for the filtered variants."
        ),
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Series context
# --------------------------------------------------------------------------- #
@dataclass
class SeriesContext:
    symbol: str
    timeframe: str
    bars: pd.DataFrame
    funding: pd.DataFrame
    wins: dict[str, slice]
    signals: dict[bool, object]
    frames: dict[bool, pd.DataFrame]
    #: Bar length in hours, carried on the context so the portfolio layer does not have to
    #: re-derive it from the timeframe label.
    tf_hours: float = 0.0
    gates: dict[str, np.ndarray] = field(default_factory=dict)
    struct_stops: tuple[np.ndarray, np.ndarray] | None = None
    struct_raw: tuple[np.ndarray, np.ndarray] | None = None
    #: The absolute epoch-ms bounds this series was loaded for. Needed so the market
    #: regime can be recomputed for a variant-specific EMA span without re-reading disk.
    start_ms: int = 0
    end_ms: int = 0
    #: BTC's own trend state (+1 above its 200-EMA, -1 below), aligned to this
    #: series' bars. A market-level filter: long-only trend systems bleed in broad
    #: bear regimes, so the gate asks whether the whole market is in an uptrend
    #: rather than only whether this one coin is.
    market_regime: np.ndarray | None = None
    integrity: dict = field(default_factory=dict)


def _calendar_wins(bars: pd.DataFrame, min_bars: int) -> dict[str, slice]:
    """Map the absolute study calendar onto bar indices for one series.

    Bars before the indicator warm-up are excluded even if the study window nominally
    starts earlier, and a series is rejected outright if a window has no data in it —
    silently running a "validation" on three bars is worse than dropping the ticker.
    """
    ts = bars["timestamp"].to_numpy(dtype=np.int64)
    wins: dict[str, slice] = {}
    for name, (s, e) in SPLITS.items():
        a = max(int(np.searchsorted(ts, D.ms(*s), side="left")), R.WARMUP)
        b = int(np.searchsorted(ts, D.ms(*e), side="left"))
        if b - a < min_bars:
            raise D.DataError(
                f"{name} window has {b - a} bars (need {min_bars}) "
                f"— series does not cover the study calendar"
            )
        wins[name] = slice(a, b)
    wins["all"] = slice(R.WARMUP, len(bars))
    return wins


def _btc_regime(timeframe: str, start: int, end: int, span: int = 200):
    """BTC trend state per bar, cached per (timeframe, window, EMA span).

    Returns ``(timestamps, regime)`` where regime is +1 when BTC closes above its own
    ``span``-period EMA and -1 when below. Causal: an EMA at bar i uses bars <= i only.
    ``span`` is a variant parameter, so the cache key must include it.
    """
    key = (timeframe, start, end, span)
    hit = _BTC_REGIME_CACHE.get(key)
    if hit is not None:
        return hit
    s = D.load_series("BTCUSDT", timeframe, start, end)
    c = s.bars["close"].to_numpy(float)
    ema = pd.Series(c).ewm(span=span, adjust=False).mean().to_numpy()
    ts = s.bars["timestamp"].to_numpy(np.int64)
    reg = np.where(c > ema, 1, -1).astype(np.int8)
    _BTC_REGIME_CACHE[key] = (ts, reg)
    return ts, reg


def _align_market_regime(bars: pd.DataFrame, timeframe: str,
                         start: int, end: int, span: int = 200) -> np.ndarray:
    """Project BTC's regime onto this series' bars (last BTC bar at or before each)."""
    btc_ts, reg = _btc_regime(timeframe, start, end, span)
    own = bars["timestamp"].to_numpy(np.int64)
    pos = np.searchsorted(btc_ts, own, side="right") - 1
    out = np.zeros(len(own), dtype=np.int8)
    ok = pos >= 0
    out[ok] = reg[pos[ok]]
    return out


def build_context(symbol: str, timeframe: str, start: int, end: int) -> SeriesContext:
    s = D.load_series(symbol, timeframe, start, end, with_funding=True)
    bars = s.bars.reset_index(drop=True)
    n = len(bars)
    usable = n - R.WARMUP
    if usable < 400:
        raise D.DataError(f"{symbol} {timeframe}: only {usable} usable bars after warm-up")
    min_bars = 60 if timeframe == "4H" else 30
    wins = _calendar_wins(bars, min_bars)

    signals: dict[bool, object] = {}
    frames: dict[bool, pd.DataFrame] = {}
    for rep in (False, True):
        ss, f = R.base_signals(bars, repair_hybrid=rep)
        signals[rep] = ss
        frames[rep] = f

    gates: dict[str, np.ndarray] = {}
    # Structural gates depend only on the (cached, parameter-free) structure build, so
    # they are precomputed. Volatility and trend gates depend on variant parameters and
    # are therefore built on demand in ``evaluate`` — caching them here with defaults
    # silently turned V4 and V5 into clones of the ungated variant.
    for kind in ("struct_break", "struct_break_event"):
        gs = R.GateSpec(kind=kind)
        for side in ("long", "short"):
            g = R.build_gate(bars, frames[True], gs, side)
            if g is not None:
                gates[f"{kind}:{side}"] = np.asarray(g, dtype=bool)
    return SeriesContext(
        symbol=symbol, timeframe=timeframe, bars=bars, funding=s.funding, wins=wins,
        signals=signals, frames=frames, tf_hours=D.TIMEFRAMES[timeframe][1], gates=gates,
        struct_stops=R.structural_stop_arrays(bars),
        struct_raw=R.structural_raw_arrays(bars),
        start_ms=start, end_ms=end,
        market_regime=_align_market_regime(bars, timeframe, start, end),
        integrity=s.manifest.integrity,
    )


def _window_mask(ctx: SeriesContext, window: str) -> np.ndarray:
    keep = np.zeros(len(ctx.bars), dtype=bool)
    keep[ctx.wins[window]] = True
    return keep


def _stops_for(ctx: SeriesContext, v: L.Variant) -> StopSpec:
    ext_l = ext_s = None
    if v.stop_mode == "external" and ctx.struct_stops is not None:
        ext_l, ext_s = (a.astype(float).copy() for a in ctx.struct_stops)
        # Widen (or tighten) the structural stop by scaling its buffer distance,
        # measured from the un-buffered pivot level so the multiplier is meaningful.
        if v.ext_buffer_mult != 1.0 and ctx.struct_raw is not None:
            rl, rs = (a.astype(float) for a in ctx.struct_raw)
            ext_l = np.where(np.isfinite(rl) & np.isfinite(ext_l),
                             rl - v.ext_buffer_mult * (rl - ext_l), np.nan)
            ext_s = np.where(np.isfinite(rs) & np.isfinite(ext_s),
                             rs + v.ext_buffer_mult * (ext_s - rs), np.nan)
    return StopSpec(
        mode=v.stop_mode, atr_mult=v.stop_atr_mult, pct=v.stop_pct,
        external_long=ext_l, external_short=ext_s,
        tp_mode=v.tp_mode, tp_atr_mult=v.tp_atr_mult, tp_pct=v.tp_pct,
        breakeven_at_r=v.breakeven_at_r,
    )


def _slice_to_window(res, sl: slice):
    """Restrict a run to a sub-slice of bars.

    Entries are already masked to the window, so the pre-window equity is exactly flat.
    Leaving that flat stretch in the array would divide the window's mean return and its
    volatility by the same factor and quietly deflate Sharpe, Calmar and drawdown. The
    walk-forward numbers must describe the window, not the window plus dead air.
    """
    start = 0 if sl.start is None else sl.start
    stop = len(res.equity) if sl.stop is None else sl.stop
    trades = [t for t in res.trades if start <= t.entry_bar < stop]
    return replace(res, equity=res.equity[start:stop],
                   timestamps=res.timestamps[start:stop], trades=trades)


def _apply_gates(ctx: SeriesContext, v: L.Variant, el: np.ndarray, es: np.ndarray):
    """AND every configured gate into the two entry masks.

    ``gate`` and ``gate2`` are applied independently, so combinations compose without a
    combinatorial branch. Structural gates come from the shared cache (they are
    parameter-free); volatility and trend gates are parameterised by the variant and are
    therefore rebuilt per variant — cheaper than caching, and impossible to get stale.
    """
    for kind in (v.gate, v.gate2):
        if not kind or kind == "none":
            continue
        if kind == "btc_regime":
            # Market-level filter: only take longs while BTC itself is above its EMA.
            # The span is a variant parameter, so this is computed per variant.
            mr = _align_market_regime(ctx.bars, ctx.timeframe, ctx.start_ms, ctx.end_ms,
                                      span=v.trend_ema)
            if mr is None:
                continue
            gl = mr > 0
            gs_ = mr < 0
        elif kind in ("struct_break", "struct_break_event"):
            gl = ctx.gates.get(f"{kind}:long")
            gs_ = ctx.gates.get(f"{kind}:short")
        else:
            spec = R.GateSpec(
                kind=kind, vol_low=v.vol_low, vol_high=v.vol_high,
                vol_window=v.vol_window, vol_quantile_window=v.vol_quantile_window,
                trend_ema=v.trend_ema,
            )
            gl = R.build_gate(ctx.bars, ctx.frames[v.repair_hybrid], spec, "long")
            gs_ = R.build_gate(ctx.bars, ctx.frames[v.repair_hybrid], spec, "short")
        if gl is not None:
            el = el & gl
        if gs_ is not None:
            es = es & gs_
    return el, es


def evaluate(ctx: SeriesContext, v: L.Variant, costs: Costs, *, window: str = "all"):
    """Apply one variant to one series. All gating is entry masking — no output
    filtering, no lookahead, no peeking at the exit."""
    tf_hours = D.TIMEFRAMES[ctx.timeframe][1]
    ss = ctx.signals[v.repair_hybrid]
    keep = _window_mask(ctx, window)
    el = ss.entry_long & keep
    es = ss.entry_short & keep

    if v.long_only:
        es = np.zeros_like(es)
    if v.short_only:
        el = np.zeros_like(el)
    el, es = _apply_gates(ctx, v, el, es)

    ss = replace(ss, entry_long=el, entry_short=es)
    res = run(
        ctx.bars, ss, symbol=ctx.symbol, timeframe=ctx.timeframe, tf_hours=tf_hours,
        costs=costs, stops=_stops_for(ctx, v), funding=ctx.funding,
    )
    if window != "all":
        res = _slice_to_window(res, ctx.wins[window])
    met = compute_metrics(res.equity, res.trades, tf_hours,
                          timestamps=res.timestamps.astype("int64"))
    return res, met


# --------------------------------------------------------------------------- #
# Variant catalogue
# --------------------------------------------------------------------------- #
def catalogue() -> list[L.Variant]:
    """Hypotheses, grouped by what they isolate.

    Round 1 (V0-V12) separates the defect repair, the entry filter and the exit rule.
    Round 2 (V13+) combines only the parts that survived: the structural gate and the
    structural stop were the two components that improved the risk-adjusted profile
    without inflating either the trade count or the parameter count.
    """
    V = L.Variant
    return [
        # --- controls ------------------------------------------------------ #
        V(name="V0_shipped", repair_hybrid=False,
          hypothesis="EMF+ADL exactly as shipped: indicator crossovers, no stop, no gate."),
        V(name="V1_repaired", repair_hybrid=True,
          hypothesis="Repair the HA recursion so the hybrid body is real. Isolates the "
                     "defect fix from every other change."),
        # --- exit-rule family ---------------------------------------------- #
        V(name="V2_rep_atr_stop", repair_hybrid=True, stop_mode="atr_fixed",
          stop_atr_mult=2.0, tp_mode="atr", tp_atr_mult=3.0,
          hypothesis="Fixed ATR stop + ATR target bounds the exit distribution."),
        V(name="V3_rep_atr_trail", repair_hybrid=True, stop_mode="atr_trail",
          stop_atr_mult=3.0,
          hypothesis="Trailing ATR stop: cut losers, let winners run."),
        V(name="V4_rep_volband", repair_hybrid=True, gate="vol_band",
          vol_low=0.6, vol_high=2.4, vol_window=100,
          hypothesis="Gate out dead and chaotic volatility regimes."),
        # --- entry-filter family ------------------------------------------- #
        V(name="V5_rep_trendgate", repair_hybrid=True, gate="trend", trend_ema=200,
          hypothesis="Trade only with the higher-timeframe trend."),
        V(name="V6_rep_struct", repair_hybrid=True, gate="struct_break",
          hypothesis="Hybrid-candle structure: longs only in confirmed up legs, shorts "
                     "only in confirmed down legs."),
        V(name="V7_rep_structstop", repair_hybrid=True, gate="struct_break",
          stop_mode="external",
          hypothesis="Structural trailing stops beyond the last confirmed HL/LH, buffered "
                     "by the hybrid body range."),
        V(name="V8_rep_struct_atr", repair_hybrid=True, gate="struct_break",
          stop_mode="atr_trail", stop_atr_mult=3.0, tp_mode="atr", tp_atr_mult=4.0,
          hypothesis="Structure times the entry; ATR sizes the risk."),
        V(name="V9_rep_struct_vol", repair_hybrid=True, gate="struct_break",
          gate2="vol_band", vol_low=0.6, vol_high=2.4, stop_mode="atr_trail",
          stop_atr_mult=3.0,
          hypothesis="Structure + volatility band + ATR trail."),
        V(name="V10_struct_longonly", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="atr_trail", stop_atr_mult=3.0,
          hypothesis="Perp shorts fight funding and drift; test the long-only subset."),
        V(name="V11_rep_struct_event", repair_hybrid=True, gate="struct_break_event",
          stop_mode="external",
          hypothesis="Enter on the confirmed break bar only, exit on the structural stop. "
                     "The tightest entry set that still has a risk rule."),
        V(name="V12_ungated_stop", repair_hybrid=True, stop_mode="external",
          hypothesis="Structural stop with no entry filter — does the stop carry the "
                     "result, or the gate?"),
        # --- round 2: combinations of what survived ------------------------- #
        V(name="V13_struct_sl_trend", repair_hybrid=True, gate="struct_break",
          gate2="trend", trend_ema=200, stop_mode="external",
          hypothesis="Structure + trend agreement, structural stop. Two independent "
                     "filters must both agree before risk is taken."),
        V(name="V14_struct_sl_be1", repair_hybrid=True, gate="struct_break",
          stop_mode="external", breakeven_at_r=1.0,
          hypothesis="Structural stop plus profit protection at +1R: stops giving back a "
                     "winner that already paid for itself."),
        V(name="V15_long_sl_trend", repair_hybrid=True, gate="struct_break",
          gate2="trend", trend_ema=200, stop_mode="external", long_only=True,
          hypothesis="Long-only + structure + trend. Tests whether the short side is a "
                     "net drag on this family."),
        V(name="V16_long_struct_be1", repair_hybrid=True, gate="struct_break",
          stop_mode="external", breakeven_at_r=1.0, long_only=True,
          hypothesis="The long-only structural core with +1R profit protection."),
        V(name="V17_struct_sl_wide", repair_hybrid=True, gate="struct_break",
          stop_mode="external", breakeven_at_r=2.0,
          hypothesis="Profit protection only at +2R — looser, so fewer winners are "
                     "stopped at break-even."),
        V(name="V18_trend_sl", repair_hybrid=True, gate="trend", trend_ema=200,
          stop_mode="external",
          hypothesis="Trend filter alone with the structural stop; isolates the gate from "
                     "the hybrid-candle structure."),
        # --- round 3: a volatility gate that actually binds ----------------- #
        V(name="V19_struct_sl_volq", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.2, vol_high=0.8, vol_quantile_window=300,
          stop_mode="external",
          hypothesis="Structural stop plus a self-calibrating volatility quantile filter: "
                     "stand aside when realised vol is in the top or bottom quintile of "
                     "its own trailing year."),
        V(name="V20_long_sl_volq", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.2, vol_high=0.8, vol_quantile_window=300,
          stop_mode="external", long_only=True,
          hypothesis="The long-only structural core with the volatility quantile filter."),
        V(name="V21_struct_sl_short", repair_hybrid=True, gate="struct_break",
          stop_mode="external", short_only=True,
          hypothesis="Diagnostic: the short side on its own. If this is negative the "
                     "long-only variants are not overfitting, they are removing a loss."),
        V(name="V22_long_sl_tight", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.25, vol_high=0.75, vol_quantile_window=300,
          stop_mode="external", long_only=True,
          hypothesis="As V20 with a tighter volatility band — measures how much the "
                     "filter's cut-points matter."),
        V(name="V23_long_sl_be1", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.2, vol_high=0.8, vol_quantile_window=300,
          stop_mode="external", long_only=True, breakeven_at_r=1.0,
          hypothesis="V20 plus +1R profit protection."),
        # --- round 4: trade count vs filter strength ------------------------ #
        V(name="V24_long_sl_volq85", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.15, vol_high=0.85, vol_quantile_window=300,
          stop_mode="external", long_only=True,
          hypothesis="The V22 core with a looser volatility band, to see whether the "
                     "edge survives when the filter stops selecting."),
        V(name="V25_long_sl_volq_trend", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.2, vol_high=0.8, vol_quantile_window=300,
          stop_mode="external", long_only=True, trend_ema=200,
          hypothesis="V22 plus the 200-EMA trend filter as a third agreement test."),
        V(name="V26_dual_sl_volq_trend", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.2, vol_high=0.8, vol_quantile_window=300,
          stop_mode="external", trend_ema=200,
          hypothesis="Both sides, structure + volatility quantile, structural stop. The "
                     "diversified version of V20 for investors who want short exposure."),
        V(name="V27_long_sl_volq50", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.5, vol_high=1.0, vol_quantile_window=300,
          stop_mode="external", long_only=True,
          hypothesis="V22 restricted to the calmest half of the volatility "
                     "distribution — the limit of the vol-filter direction."),
        # --- sensitivity axes -------------------------------------------------- #
        # These exist only to answer 'does a small parameter change destroy it?'.
        # They are the same hypothesis on a grid, not new ideas.
        V(name="V28_lo_volq10_90", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.10, vol_high=0.90, vol_quantile_window=300,
          stop_mode="external", long_only=True,
          hypothesis="Volatility quantile axis, widest band. The near-unfiltered end."),
        V(name="V29_lo_volq35_65", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.35, vol_high=0.65, vol_quantile_window=300,
          stop_mode="external", long_only=True,
          hypothesis="Volatility quantile axis, tight band."),
        V(name="V30_lo_volq2575_w150", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.25, vol_high=0.75, vol_quantile_window=150,
          stop_mode="external", long_only=True,
          hypothesis="Window axis: a short 150-bar memory for the volatility rank."),
        V(name="V31_lo_volq2575_w600", repair_hybrid=True, gate="struct_break",
          gate2="vol_quantile", vol_low=0.25, vol_high=0.75, vol_quantile_window=600,
          stop_mode="external", long_only=True,
          hypothesis="Window axis: a long 600-bar memory for the volatility rank."),
        V(name="V32_lo_struct_nostop", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="none",
          hypothesis="Exit-rule axis: the long-only structural core with no stop at "
                     "all, to separate the entry filter from the risk rule."),
        V(name="V33_lo_struct_atrtrail", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="atr_trail", stop_atr_mult=3.0,
          hypothesis="Exit-rule axis: long-only structural core with a trailing ATR "
                     "stop instead of the structural one."),
        # --- widening the stop, because tightening it demonstrably hurt --------- #
        V(name="V34_lo_struct_sl2x", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="external", ext_buffer_mult=2.0,
          hypothesis="Two structural bodies of buffer instead of one. The stop exists "
                     "to bound the tail, not to trade the noise."),
        V(name="V35_lo_struct_sl3x", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="external", ext_buffer_mult=3.0,
          hypothesis="Three structural bodies of buffer — approaching a "
                     "catastrophe-only stop."),
        V(name="V36_lo_struct_atr6", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=6.0,
          hypothesis="A fixed 6-ATR stop: far enough out to be a disaster brake rather "
                     "than a trade-management tool."),
        V(name="V37_lo_struct_atr8", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="A fixed 8-ATR stop, the widest fixed-risk rule tested."),
        V(name="V38_lo_struct_pcttrail20", repair_hybrid=True, gate="struct_break",
          long_only=True, stop_mode="pct_trail", stop_pct=0.20,
          hypothesis="A 20% trailing stop, expressed in price rather than volatility."),
        # --- round 5: does a market-level regime filter stop the bear bleed? --- #
        V(name="V39_mkt_long_nostop", repair_hybrid=True, gate="btc_regime",
          long_only=True, stop_mode="none",
          hypothesis="Longs only while BTC itself is above its 200-EMA. The per-ticker "
                     "structural gate says 'this coin is trending'; this says 'the "
                     "market is trending'. Bear-regime losses were the largest "
                     "identified weakness."),
        V(name="V40_lo_struct_mkt", repair_hybrid=True, gate="struct_break",
          gate2="btc_regime", long_only=True, stop_mode="none",
          hypothesis="Both filters: the coin must be in a confirmed up leg AND the "
                     "market must be above its 200-EMA."),
        V(name="V41_lo_struct_mkt_atr8", repair_hybrid=True, gate="struct_break",
          gate2="btc_regime", long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="V40 with the catastrophe stop restored — the full candidate."),
        V(name="V42_lo_struct_mkt_sl", repair_hybrid=True, gate="struct_break",
          gate2="btc_regime", long_only=True, stop_mode="external",
          hypothesis="V40 with the structural stop instead of the ATR brake."),
        V(name="V43_mkt_long_atr8", repair_hybrid=True, gate="btc_regime",
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="Market filter alone with the catastrophe stop — isolates the "
                     "market gate from the structural gate."),
        # --- round 6: sweep the one axis left untested in rounds 1-5 ------------ #
        # Sections 12-13 of the report admitted the EMA length was never moved. These
        # close that gap; everything else is identical to V43.
        V(name="V44_mkt_ema100", repair_hybrid=True, gate="btc_regime", trend_ema=100,
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="V43 with a faster market filter (EMA100) — reacts sooner, but "
                     "whipsaws more."),
        V(name="V45_mkt_ema150", repair_hybrid=True, gate="btc_regime", trend_ema=150,
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="V43 with EMA150."),
        V(name="V46_mkt_ema250", repair_hybrid=True, gate="btc_regime", trend_ema=250,
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="V43 with EMA250 — slower filter, longer holds."),
        V(name="V47_mkt_ema300", repair_hybrid=True, gate="btc_regime", trend_ema=300,
          long_only=True, stop_mode="atr_fixed", stop_atr_mult=8.0,
          hypothesis="V43 with EMA300 — the slowest filter tested."),
    ]


# --------------------------------------------------------------------------- #
# Battery
# --------------------------------------------------------------------------- #
def run_battery(
    contexts: dict[tuple[str, str], SeriesContext],
    variants: list[L.Variant],
    costs: Costs,
    *,
    window: str,
    groups: list[list[str]] | None = None,
) -> dict:
    """Evaluate every variant on every series, then pool.

    When ``groups`` is given, each variant is also scored on every ticker group
    separately. A variant that only works on one group of coins is the single most
    common way a backtest lies, and it is invisible in a pooled number — so both are
    reported side by side and the ranking uses the pooled figure.
    """
    results: dict = {}
    for v in variants:
        per_tf: dict = {}
        for tf in TIMEFRAMES:
            res_by_sym = {}
            for (sym, t), ctx in contexts.items():
                if t != tf:
                    continue
                try:
                    r, _ = evaluate(ctx, v, costs, window=window)
                    res_by_sym[sym] = r
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s %s %s failed: %r", v.name, sym, tf, exc)
            pm = L.panel_metrics(res_by_sym, tf)
            pm["flags"] = L.concentration_flags(pm)
            pm["score"] = L.robust_score(pm)
            pm["per_ticker"] = L.per_ticker_table(res_by_sym, tf)
            if groups:
                per_group = {}
                for gi, gs in enumerate(groups):
                    sub = {s: r for s, r in res_by_sym.items() if s in set(gs)}
                    if len(sub) < 3:
                        continue
                    gm = L.panel_metrics(sub, tf)
                    per_group[f"group{gi}"] = {
                        "n": len(sub),
                        "total_trades": gm.get("total_trades", 0),
                        "total_return": gm.get("total_return", 0.0),
                        "sharpe": gm.get("sharpe", 0.0),
                        "median_ticker_sharpe": gm.get("median_ticker_sharpe", 0.0),
                        "max_dd": gm.get("max_dd", 0.0),
                        "share_tickers_positive": gm.get("share_tickers_positive", 0.0),
                    }
                pm["per_group"] = per_group
                if per_group:
                    rets = np.array([g["total_return"] for g in per_group.values()])
                    shps = np.array([g["sharpe"] for g in per_group.values()])
                    pm["group_return_worst"] = float(rets.min())
                    pm["group_return_median"] = float(np.median(rets))
                    pm["group_sharpe_worst"] = float(shps.min())
                    pm["n_groups_positive"] = int((rets > 0).sum())
                    # The selection score shrinks by how much of the cross-section
                    # failed to make money - a variant cannot win on its best group.
                    pm["score"] = pm["score"] * min(1.0, pm["n_groups_positive"] / len(per_group))
            per_tf[tf] = pm
        results[v.name] = {"hypothesis": v.hypothesis, "variant": asdict(v),
                           "by_tf": per_tf}
    return results


def summary_table(results: dict, groups: bool = False) -> str:
    lines = []
    hdr = (f"{'variant':22s} {'TF':3s} {'trk':>3s} {'trd':>5s} {'ret':>9s} {'sharpe':>6s} "
           f"{'medShp':>6s} {'maxDD':>7s} {'PF':>5s} {'pos%':>5s} {'score':>6s}")
    if groups:
        hdr += f" {'gRet':>7s} {'gShpW':>6s} {'+g':>3s}"
    hdr += "  flags"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name, block in results.items():
        for tf in TIMEFRAMES:
            pm = block["by_tf"][tf]
            if not pm.get("n_tickers"):
                lines.append(f"{name:22s} {tf:3s}   (no data)")
                continue
            pf = pm.get("profit_factor", 0.0)
            pf = 99.0 if pf == float("inf") else pf
            row = (
                f"{name:22s} {tf:3s} {pm['n_tickers']:3d} {pm['total_trades']:5d} "
                f"{pm['total_return']:+9.2%} {pm['sharpe']:+6.2f} "
                f"{pm['median_ticker_sharpe']:+6.2f} {pm['max_dd']:7.1%} {pf:5.2f} "
                f"{pm['share_tickers_positive']:5.0%} {pm['score']:6.2f}"
            )
            if groups:
                row += (f" {pm.get('group_return_median', 0.0):+7.2%} "
                        f"{pm.get('group_sharpe_worst', 0.0):+6.2f} "
                        f"{pm.get('n_groups_positive', 0):2d}/{len(pm.get('per_group', {})) or 0:<1d}")
            row += f"  {'; '.join(pm['flags'])}"
            lines.append(row)
    return "\n".join(lines)


def _thin(eq, idx, max_points: int = 800) -> dict:
    """Thin an equity curve for JSON/plotting without changing its shape."""
    n = len(eq)
    if n == 0:
        return {"t": [], "v": []}
    if n <= max_points:
        sel = np.arange(n)
    else:
        step = int(np.ceil(n / max_points))
        sel = np.arange(0, n, step)
        if sel[-1] != n - 1:
            sel = np.append(sel, n - 1)
    return {"t": [str(pd.Timestamp(idx[i])) for i in sel],
            "v": [float(eq[i]) for i in sel]}


def load_universe(path: Path) -> dict:
    if not path.exists():
        u = select_universe()
        path.write_text(json.dumps(_jsonable(u), indent=2), encoding="utf-8")
        return u
    return json.loads(path.read_text(encoding="utf-8"))


def split_groups(symbols: list[str], k: int) -> list[list[str]]:
    """Contiguous groups in turnover-rank order. Rotation is by rank, not by hand."""
    if k <= 1:
        return [list(symbols)]
    per = int(np.ceil(len(symbols) / k))
    return [list(symbols[i * per:(i + 1) * per]) for i in range(k)
            if symbols[i * per:(i + 1) * per]]


def build_contexts(symbols, start, end, *, log_prefix: str = ""):
    contexts: dict[tuple[str, str], SeriesContext] = {}
    exclusions: list[dict] = []
    for sym in symbols:
        for tf in TIMEFRAMES:
            try:
                contexts[(sym, tf)] = build_context(sym, tf, start, end)
            except Exception as exc:  # noqa: BLE001
                exclusions.append({"symbol": sym, "timeframe": tf, "reason": repr(exc)})
                log.warning("%sexcluded %s %s: %r", log_prefix, sym, tf, exc)
    return contexts, exclusions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="loops",
                    choices=["ab", "loops", "holdout", "robustness", "all", "final",
                             "universe", "curves"])
    ap.add_argument("--pool", type=int, default=0,
                    help="limit symbols (0 = whole universe)")
    ap.add_argument("--groups", type=int, default=4,
                    help="how many turnover-ranked ticker groups to split into")
    ap.add_argument("--tag", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--windows", default="train,valid")
    args = ap.parse_args(argv)

    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    if args.stage == "universe":
        u = select_universe(max_candidates=60, workers=6)
        _dump(OUT / "universe.json", u)
        print(f"universe: {len(u['symbols'])} symbols; "
              f"rejected {len(u['rejected_short_history'])}; "
              f"excluded {len(u['excluded_non_crypto'])}")
        for i, r in enumerate(u["selected"]):
            print(f"  {i:2d} {r['symbol']:14s} turn={r['turnover24h']:14.0f} "
                  f"4H={r['n_bars4H']:6d} 1D={r['n_bars1D']:5d}")
        for r in u["rejected_short_history"]:
            print(f"  REJECT {r['symbol']:14s} {r['reason'][:80]}")
        return 0

    uni = load_universe(OUT / "universe.json")
    pool = uni["symbols"][: args.pool] if args.pool else uni["symbols"]
    groups = split_groups(pool, args.groups)
    log.info("universe %d symbols -> %d groups", len(pool), len(groups))
    for gi, g in enumerate(groups):
        log.info("  group %d (%d): %s", gi, len(g), g)

    start, end = D.ms(*STUDY_START), D.ms(*STUDY_END)
    contexts, exclusions = build_contexts(pool, start, end)
    log.info("built %d series contexts in %.0fs (%d exclusions)",
             len(contexts), time.time() - t0, len(exclusions))

    manifest = {
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "source": uni.get("source"),
        "selection_rule": uni.get("selection_rule"),
        "survivorship_bias_disclosure": uni.get("survivorship_bias_disclosure"),
        "study_window_utc": uni.get("study_window_utc"),
        "warmup_bars": R.WARMUP,
        "splits": {k: [str(pd.Timestamp(D.ms(*a), unit="ms", tz="UTC")),
                       str(pd.Timestamp(D.ms(*b), unit="ms", tz="UTC"))]
                   for k, (a, b) in SPLITS.items()},
        "groups": {f"group{i}": g for i, g in enumerate(groups)},
        "series": {
            f"{sym}|{tf}": {
                "n_bars": int(len(c.bars)),
                "first_bar_utc": c.integrity.get("first_bar_utc"),
                "last_bar_utc": c.integrity.get("last_bar_utc"),
                "coverage": c.integrity.get("coverage"),
                "n_gaps": c.integrity.get("n_gaps"),
                "bad_ohlc": c.integrity.get("bad_ohlc"),
                "zero_volume_bars": c.integrity.get("zero_volume_bars"),
                "n_funding": int(len(c.funding)),
                "sha256": D._hash_frame(c.bars),
                "window_bars": {w: int(sl.stop - sl.start) for w, sl in c.wins.items()},
            }
            for (sym, tf), c in sorted(contexts.items())
        },
        "exclusions": exclusions,
    }
    tag = args.tag or "all"
    _dump(OUT / f"data_manifest_{tag}.json", manifest)

    costs = Costs()
    cat = {v.name: v for v in catalogue()}

    if args.stage in ("ab", "loops", "holdout", "all"):
        want = [n for n in args.variants.split(",") if n]
        if args.stage == "ab":
            vs = [v for v in catalogue() if v.name in ("V0_shipped", "V1_repaired")]
        elif want:
            # An explicit shortlist (used for the one-touch holdout) must not
            # silently fall back to the whole catalogue — that would burn the
            # holdout on variants nobody is going to ship.
            vs = [cat[n] for n in want if n in cat]
            missing = [n for n in want if n not in cat]
            if missing:
                raise SystemExit(f"unknown variants: {missing}")
        else:
            vs = catalogue()
        windows = (["all"] if args.stage == "ab"
                   else [w for w in args.windows.split(",") if w])
        for window in windows:
            res = run_battery(contexts, vs, costs, window=window, groups=groups)
            _dump(OUT / f"stage_{args.stage}_{window}_{tag}.json", res)
            print(f"\n=== STAGE {args.stage.upper()} window={window} "
                  f"({len(pool)} symbols, {len(groups)} groups) ===")
            print(summary_table(res, groups=True))
            ranked = sorted(
                ((blk["by_tf"][tf].get("score", -9.9), name, tf)
                 for name, blk in res.items() for tf in TIMEFRAMES),
                reverse=True,
            )
            print("\nTop by pooled robust score:")
            for sc, name, tf in ranked[:12]:
                print(f"  {sc:6.2f}  {name:26s} {tf}")

    if args.stage in ("robustness", "all"):
        names = [n for n in args.variants.split(",") if n] or [
            "V0_shipped", "V1_repaired", "V4_rep_volband", "V12_ungated_stop",
            "V19_struct_sl_volq", "V22_long_sl_tight",
        ]
        base = [cat[n] for n in names if n in cat]
        stress = {
            "base": Costs(),
            "slip_4x": Costs(slippage_bps=4 * 2.0),
            "fee_2x": Costs(fee_rate=0.0011),
            "no_funding": Costs(funding_on=False),
            "harsh": Costs(fee_rate=0.0011, slippage_bps=8.0, latency_bps=3.0),
        }
        out = {}
        for label, c in stress.items():
            out[label] = run_battery(contexts, base, c, window="all", groups=groups)
        _dump(OUT / f"stage_robustness_{tag}.json", out)
        for label, blk in out.items():
            print(f"\n=== ROBUSTNESS [{label}] full sample ===")
            print(summary_table(blk, groups=True))

    if args.stage == "curves":
        names = [n for n in args.variants.split(",") if n] or [
            "V0_shipped", "V1_repaired", "V22_long_sl_tight"]
        out: dict = {}
        for name in names:
            v = cat[name]
            block: dict = {"variant": asdict(v), "hypothesis": v.hypothesis, "by_tf": {}}
            for tf in TIMEFRAMES:
                res_by_sym = {}
                for (sym, t), ctx in contexts.items():
                    if t != tf:
                        continue
                    r, _ = evaluate(ctx, v, costs, window="all")
                    res_by_sym[sym] = r
                eq, idx = L.panel_equity(res_by_sym)
                block["by_tf"][tf] = {
                    "panel": _thin(eq, idx),
                    "per_ticker": {
                        sym: _thin(
                            r.equity,
                            pd.to_datetime(r.timestamps, unit="ms", utc=True).to_numpy(),
                            max_points=300,
                        )
                        for sym, r in sorted(res_by_sym.items())
                    },
                }
            out[name] = block
        _dump(OUT / "curves.json", out)
        print(f"wrote curves for {len(out)} variants on {len(pool)} symbols")
        log.info("curves done in %.1fs", time.time() - t0)
        return 0

    if args.stage == "final":
        names = [n for n in args.variants.split(",") if n] or [
            "V0_shipped", "V1_repaired", "V22_long_sl_tight",
        ]
        base = [cat[n] for n in names if n in cat]
        out = {}
        for window in ("train", "valid", "holdout", "all"):
            out[window] = run_battery(contexts, base, costs, window=window,
                                      groups=groups)
        _dump(OUT / "stage_final.json", out)
        for window, blk in out.items():
            print(f"\n=== FINAL window={window} ===")
            print(summary_table(blk, groups=True))

    log.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
