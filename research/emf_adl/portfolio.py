"""Portfolio construction and the equity-curve drawdown brake.

Why this module exists
----------------------
The per-symbol engine in :mod:`research.emf_adl.engine` cannot control *portfolio*
drawdown, no matter how its stops are tuned, because a stop only ever closes one position.
Portfolio drawdown is a cross-sectional property: it is driven by how much total risk is
live at once. This module supplies the two levers that act on it.

1. **Constant-risk sizing** (:class:`~research.emf_adl.engine.Sizing`). Each trade risks a
   fixed fraction of equity between entry and its stop, so portfolio risk does not depend
   on which instruments happen to be volatile that month.

2. **Equity-curve drawdown brake.** When the panel equity falls a set fraction below its own
   running peak, new positions are scaled down; the deeper the drawdown, the smaller the
   scale. This is the standard "equity-curve trading" overlay, and it is the only one of the
   two that can shorten a prolonged underwater stretch, because it acts on the whole book
   rather than on a single trade.

Causality
---------
The brake is derived from the panel equity curve and then fed back into it, which looks
circular. It is not: ``scale[i]`` is a function of ``equity[<= i]`` only, so at the moment a
position is taken the brake has seen no future information. Because positions influence the
path they are measured on, the panel is re-simulated a few times (a bounded fixed-point
iteration) and the per-pass residual is stored. A run that has not converged is reported as
such rather than quietly presented.

The panel grid
--------------
``equity``, ``scale`` and every symbol's ``Result.timestamps`` all live on the *union* of
the symbols' bar timestamps for one timeframe. That grid depends only on the bar series, not
on which trades happened, so it is **identical on every pass** — the fixed-point iteration
therefore compares like with like and cannot drift by a bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import Costs, Result, Sizing, StopSpec, run

#: Nanoseconds per millisecond — the metric layer expects epoch-ms timestamps.
_NS_PER_MS = 1_000_000


def to_ms(ts: np.ndarray) -> np.ndarray:
    """Epoch-ms int64 from anything datetime-like.

    :mod:`research.emf_adl.metrics` casts timestamps with ``datetime64[ms]``, so passing
    nanoseconds would silently produce dates ~55,000 years in the future and corrupt any
    per-period segmentation.
    """
    a = np.asarray(ts)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ns]").astype(np.int64) // _NS_PER_MS
    return a.astype(np.int64)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive run of True in ``mask``."""
    if mask.size == 0 or not mask.any():
        return 0
    pad = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(pad.astype(np.int8)))
    return int((edges[1::2] - edges[0::2]).max())


def _longest_run_bounds(mask: np.ndarray) -> tuple[int, int]:
    """``(start, stop)`` of the longest consecutive run of True — stop is exclusive."""
    if mask.size == 0 or not mask.any():
        return 0, 0
    pad = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(pad.astype(np.int8)))
    starts, stops = edges[0::2], edges[1::2]
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


@dataclass(frozen=True)
class BrakeSpec:
    """Equity-curve drawdown brake.

    ``dd_on``    portfolio drawdown (positive fraction) at which de-risking starts.
    ``dd_full``  drawdown at which the scale reaches ``floor``.
    ``floor``    minimum scale. Keeping it above zero lets a recovery still be captured; a
                 zero floor turns the brake into a flat switch, which whipsaws at the worst
                 possible moment.
    ``dd_off``   drawdown at which the brake fully releases. ``dd_off < dd_on`` gives
                 hysteresis so the brake cannot flicker on and off around the threshold.
    """

    dd_on: float = 0.10
    dd_full: float = 0.25
    floor: float = 0.25
    dd_off: float = 0.05

    @property
    def enabled(self) -> bool:
        return self.dd_on > 0 and self.floor < 1.0


#: A brake that never engages — the control arm.
NO_BRAKE = BrakeSpec(dd_on=0.0, dd_full=0.0, floor=1.0, dd_off=0.0)


def panel_grid(contexts: dict) -> np.ndarray:
    """Union of every symbol's bar timestamps for this timeframe, ascending, epoch-ms.

    Position-independent by construction, which is what makes the fixed-point iteration
    well defined.
    """
    grid: np.ndarray | None = None
    for ctx in contexts.values():
        b = ctx.bars["timestamp"].to_numpy(np.int64)
        grid = b if grid is None else np.union1d(grid, b)
    return np.zeros(0, dtype=np.int64) if grid is None else np.asarray(grid, np.int64)


def panel_equity(results: dict[str, Result], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-weight panel equity on ``grid``, weighting only the tickers live at each bar.

    Dropping any timestamp where one ticker is missing would bias the panel toward coins
    with the longest history; averaging over the live subset keeps every bar usable.
    Returns ``(equity, grid, live_count)``; ``equity`` has ``len(grid) + 1`` points because
    it starts from 1.0 before the first bar's return.
    """
    n = len(grid)
    if n == 0 or not results:
        return np.zeros(0), grid, np.zeros(0, dtype=np.int64)

    # returns[k] = pct change of equity on bar k of the grid, NaN where the ticker is dead.
    rets = np.full((len(results), n), np.nan, dtype=float)
    for row, (sym, r) in enumerate(sorted(results.items())):
        eq = np.asarray(r.equity, dtype=float)
        if len(eq) < 2:
            continue
        step = np.diff(eq) / np.where(eq[:-1] == 0, np.nan, eq[:-1])
        pos = np.searchsorted(grid, to_ms(r.timestamps[1:]))
        ok = (pos >= 0) & (pos < n)
        rets[row, pos[ok]] = step[ok]

    live = np.isfinite(rets).sum(axis=0)
    with np.errstate(invalid="ignore"):
        port = np.nansum(rets, axis=0) / np.where(live == 0, np.nan, live)
    port = np.nan_to_num(port, nan=0.0, posinf=0.0, neginf=0.0)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + port)])
    return eq, grid, live


def drawdown_stats(eq: np.ndarray, ts: np.ndarray | None = None) -> dict:
    """Max drawdown, its duration, and duration-aware drawdown metrics.

    ``ulcer`` is the RMS drawdown — it penalises depth *and* persistence together, which
    ``max_dd`` alone cannot: a −30% dip that recovers in a month and a −30% dip that
    persists for two years score identically on max drawdown and completely differently on
    the ulcer index. ``martin`` is the excess return per unit of that risk. These two are
    the honest answer to "avoid extended periods in a deep drawdown".
    """
    if len(eq) < 3:
        return {"max_dd": 0.0, "max_dd_bars": 0, "underwater_bars": 0, "time_underwater": 0.0,
                "worst_dd_start": 0, "worst_dd_end": 0, "n_dd_gt_20": 0, "n_dd_gt_10": 0,
                "frac_dd_gt_10": 0.0, "frac_dd_gt_20": 0.0, "frac_dd_gt_5": 0.0,
                "longest_dd_gt_5_bars": 0, "longest_dd_gt_5_days": 0.0,
                "longest_dd_gt_10_bars": 0, "longest_dd_gt_10_days": 0.0,
                "longest_dd_gt_10_bars_legacy": 0,
                "n_recovery_gt_180_bars": 0, "max_recovery_bars": 0,
                "longest_underwater_bars": 0, "ulcer": 0.0, "martin": 0.0,
                "worst_year_dd": 0.0, "max_dd_days": 0.0,
                "longest_underwater_days": 0.0}
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    trough = int(np.argmin(dd))
    start = int(np.argmax(eq[: trough + 1])) if trough > 0 else 0
    underwater = dd < -1e-12
    # Recovery durations: bars from each trough back to the prior peak.
    rec: list[int] = []
    idx = np.flatnonzero(~underwater)
    if len(idx):
        j = 0
        for i in np.flatnonzero(underwater):
            while j < len(idx) and idx[j] < i:
                j += 1
            if j < len(idx):
                rec.append(int(idx[j] - i))
    max_rec = int(max(rec)) if rec else 0
    # The single longest stretch spent below any prior peak — the honest "prolonged
    # drawdown" number. Counting troughs whose recovery exceeds N bars is not useful: on a
    # low-drawdown curve almost every tiny dip qualifies, so the count approaches the bar
    # count and stops discriminating.
    longest_uw = _longest_run(underwater)
    # Wall-clock durations, when real timestamps are available. "6 months underwater" and
    # "6 months of tiny dips" are the same bar count and very different experiences, so the
    # calendar number is reported rather than derived from the bar count by a fixed ratio.
    dd_days = 0.0
    uw_days = 0.0
    if ts is not None and len(ts) == len(eq) and len(eq) > 1:
        t = to_ms(ts).astype(np.float64) / 86_400_000.0
        dd_days = float(t[trough] - t[start])
        a, b = _longest_run_bounds(underwater)
        if b > a:
            uw_days = float(t[b - 1] - t[a])
    ulcer = float(np.sqrt(np.mean(np.square(dd))))
    # Depth-banded persistence. ``longest_underwater_bars`` counts a dip of one basis point
    # the same as a 30% collapse, so on a low-drawdown curve it is dominated by micro-dips
    # and cannot answer "how long was the portfolio in a *deep* hole?". These bands can.
    def _band(thr: float) -> tuple[int, float]:
        mask = dd < -thr
        a, b = _longest_run_bounds(mask)
        days = float(t[b - 1] - t[a]) if (b > a and t is not None and len(t) == len(eq)) else 0.0
        return int(b - a), days

    gt5_bars, gt5_days = _band(0.05)
    gt10_bars, gt10_days = _band(0.10)
    years = max(len(eq) / (365.25 * 6), 1e-9)  # 6 bars/day == 4H; only used for Martin's scale
    cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if eq[0] > 0 and eq[-1] > 0 else -1.0
    return {
        "max_dd": float(-dd.min()),
        "max_dd_bars": int(trough - start),
        "underwater_bars": int(underwater.sum()),
        "time_underwater": float(underwater.mean()),
        "worst_dd_start": int(start),
        "worst_dd_end": int(trough),
        "n_dd_gt_20": int((dd < -0.20).sum()),
        "n_dd_gt_10": int((dd < -0.10).sum()),
        "frac_dd_gt_10": float((dd < -0.10).mean()),
        "frac_dd_gt_20": float((dd < -0.20).mean()),
        "frac_dd_gt_5": float((dd < -0.05).mean()),
        "longest_dd_gt_5_bars": gt5_bars,
        "longest_dd_gt_5_days": gt5_days,
        "longest_dd_gt_10_bars": gt10_bars,
        "longest_dd_gt_10_days": gt10_days,
        #: Denominated in bars for the original report; kept so nothing downstream breaks.
        "longest_dd_gt_10_bars_legacy": _longest_run(dd < -0.10),
        "n_recovery_gt_180_bars": int(sum(1 for x in rec if x > 180)),
        "max_recovery_bars": max_rec,
        "longest_underwater_bars": longest_uw,
        "ulcer": ulcer,
        "martin": float((cagr - 0.02) / ulcer) if ulcer > 1e-9 else 0.0,
        "worst_year_dd": _worst_year_dd(eq, ts),
        "max_dd_days": dd_days,
        "longest_underwater_days": uw_days,
    }


def _worst_year_dd(eq: np.ndarray, ts: np.ndarray | None) -> float:
    """Deepest intra-year drawdown, on real calendar years when timestamps are given.

    A fixed-width slice would cut a year in half and understate a December-to-February
    slide, which is exactly the kind of stretch this metric exists to catch.
    """
    if len(eq) < 2:
        return 0.0
    if ts is None or len(ts) != len(eq):
        step = max(len(eq) // 12, 1)
        groups = [eq[a: a + step + 1] for a in range(0, len(eq), step)]
    else:
        years = (to_ms(ts) // (365 * 24 * 3600 * 1000)).astype(np.int64)
        groups = [eq[years == y] for y in np.unique(years)]
    out = 0.0
    for seg in groups:
        if len(seg) < 2:
            continue
        pk = np.maximum.accumulate(seg)
        out = max(out, float(-(seg / pk - 1.0).min()))
    return out


def brake_scale(eq: np.ndarray, spec: BrakeSpec) -> np.ndarray:
    """Per-bar position scale read off the *already realised* equity path. Causal.

    ``scale[i]`` uses ``eq[: i + 1]`` only, so applying it at bar ``i`` introduces no
    lookahead. Hysteresis via ``dd_off`` stops the brake flickering around the threshold.
    """
    n = len(eq)
    if not spec.enabled or n == 0:
        return np.ones(max(n, 0), dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / peak
    on_ = False
    out = np.ones(n, dtype=float)
    span = max(spec.dd_full - spec.dd_on, 1e-9)
    for i in range(n):
        if not on_:
            if dd[i] >= spec.dd_on:
                on_ = True
        elif dd[i] <= spec.dd_off:
            on_ = False
        if on_:
            frac = (dd[i] - spec.dd_on) / span
            out[i] = float(np.clip(1.0 - frac * (1.0 - spec.floor), spec.floor, 1.0))
    return out


@dataclass
class PortfolioRun:
    equity: np.ndarray
    timestamps: np.ndarray          # epoch-ms int64, length == len(equity)
    live: np.ndarray
    scale: np.ndarray               # per-bar, length == len(equity)
    results: dict[str, Result]
    stats: dict = field(default_factory=dict)
    convergence: list[dict] = field(default_factory=list)
    gross_exposure: np.ndarray | None = None

    @property
    def converged(self) -> bool:
        return bool(self.convergence) and self.convergence[-1]["residual"] < 1e-4


def vol_target_scale(bars: pd.DataFrame, *, target_bar_vol: float, window: int = 100,
                     lo: float = 0.25, hi: float = 4.0) -> np.ndarray:
    """Per-bar multiplier that equalises risk contribution across a heterogeneous universe.

    Crypto perps have wildly different volatilities (BTC ~1.2% per 4H bar, DOGE ~4%).
    Sizing them identically means the book's risk is whatever the noisiest name did, and a
    single DOGE swing sets the portfolio drawdown. Scaling each symbol to a common
    volatility target makes every name contribute comparably, which is a variance-reduction
    argument (diversification), not a fitted one.

    Causality: the multiplier applied at bar ``i`` uses closes only up to ``i - 1``. The
    fill happens at bar ``i``'s open, so bar ``i``'s close is genuinely unknown.
    """
    close = bars["close"].to_numpy(dtype=float)
    n = len(close)
    if n == 0:
        return np.zeros(0)
    r = pd.Series(close).pct_change()
    rv = r.rolling(window, min_periods=max(window // 3, 20)).std().shift(1).to_numpy()
    out = np.full(n, np.nan)
    ok = np.isfinite(rv) & (rv > 0)
    out[ok] = np.clip(target_bar_vol / rv[ok], lo, hi)
    # Warm-up and degenerate-vol bars fall back to unit scale rather than a guess.
    out[~ok] = 1.0
    return out


def run_portfolio(
    contexts: dict,
    *,
    signals_for,
    stops_for,
    sizing: Sizing,
    costs: Costs,
    brake: BrakeSpec = NO_BRAKE,
    windows=None,
    vol_for=None,
    max_passes: int = 6,
    tol: float = 1e-4,
) -> PortfolioRun:
    """Simulate every per-symbol book, then apply the portfolio drawdown brake.

    The panel is simulated; the brake is read off that panel's own equity curve; the panel
    is simulated again with the brake applied; repeat until ``scale`` stops moving. The
    per-pass residual is returned so a non-converged run can be rejected instead of
    reported as if it were stable.
    """
    contexts = dict(contexts)
    grid = panel_grid(contexts)
    if len(grid) == 0:
        return PortfolioRun(np.zeros(0), np.zeros(0, np.int64), np.zeros(0),
                            np.zeros(0), {}, {}, [], None)
    # An inactive brake is constant 1.0, so the fixed point is already solved: one pass.
    if not brake.enabled:
        max_passes = 1

    scale = np.ones(len(grid) + 1, dtype=float)
    prev: np.ndarray | None = None
    convergence: list[dict] = []
    results: dict[str, Result] = {}
    eq = np.ones(1)
    live = np.zeros(0, dtype=np.int64)

    for p in range(max_passes):
        idx_in_grid = {sym: np.searchsorted(grid, ctx.bars["timestamp"].to_numpy(np.int64))
                       for (sym, _), ctx in contexts.items()}
        for (sym, tf), ctx in sorted(contexts.items()):
            sig, _frames = signals_for(ctx)
            if windows is not None:
                keep = windows(ctx)
                sig = type(sig)(
                    entry_long=sig.entry_long & keep,
                    entry_short=sig.entry_short & keep,
                    exit_long=sig.exit_long, exit_short=sig.exit_short,
                    atr=sig.atr, warmup=sig.warmup, label=sig.label,
                )
            bar_scale = scale[np.clip(idx_in_grid[sym], 0, len(scale) - 1)]
            s = Sizing(risk_per_trade=sizing.risk_per_trade,
                       max_leverage=sizing.max_leverage,
                       bar_scale=bar_scale,
                       vol_scale=vol_for(ctx) if vol_for is not None else None,
                       fallback_stop_pct=sizing.fallback_stop_pct)
            results[sym] = run(
                ctx.bars, sig, symbol=sym, timeframe=ctx.timeframe,
                tf_hours=ctx.tf_hours, costs=costs, stops=stops_for(ctx),
                funding=ctx.funding, sizing=s,
            )

        eq, grid, live = panel_equity(results, grid)
        new_scale = brake_scale(eq, brake)
        residual = (float(np.max(np.abs(new_scale - prev))) if prev is not None
                    and len(prev) == len(new_scale) else float("inf"))
        convergence.append({"pass": p + 1, "residual": residual,
                            "mean_scale": float(new_scale.mean()),
                            "min_scale": float(new_scale.min())})
        prev, scale = new_scale, new_scale
        if residual < tol and p > 0:
            break

    # Gross exposure: |notional| summed across live positions, per bar, as a multiple of
    # equity. Reported because risk-based sizing caps risk per *trade*, not the aggregate —
    # with ~10 concurrent books the gross book can exceed 1x even at 1x per position.
    notional_sum = np.zeros(len(grid), dtype=float)
    for r in results.values():
        pos = np.searchsorted(grid, to_ms(r.timestamps))
        for t in r.trades:
            a, b = int(t.entry_bar), int(t.exit_bar)
            if b <= a:
                continue
            sl = slice(max(a, 0), min(b, len(pos)))
            seg = pos[sl]
            ok = (seg >= 0) & (seg < len(grid))
            np.add.at(notional_sum, seg[ok], float(t.notional_at_entry))
    gross = notional_sum / np.where(eq[1:] == 0, np.nan, eq[1:])
    gross = np.nan_to_num(gross, nan=0.0, posinf=0.0, neginf=0.0)

    ts_ms = to_ms(grid)
    ts = np.concatenate([[ts_ms[0] - 1 if len(ts_ms) else 0], ts_ms]).astype(np.int64)
    return PortfolioRun(equity=eq, timestamps=ts, live=live, scale=scale,
                        results=results, stats=drawdown_stats(eq, ts),
                        convergence=convergence, gross_exposure=gross)
