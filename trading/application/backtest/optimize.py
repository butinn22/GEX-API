"""Adaptive parameter search — the optimisation half of the win/loss loop.

Method
------
A small grid sweep over the strategy's tunables, scored **out of sample**:
bars are split 70 / 30 into train / validation; each candidate is backtested
on both, and ranked by the ``profit_win`` composite — validation profit
gates the pick and win rate scales it — discounted when the train and
validation Sharpe disagree (a candidate that only works in-sample is worse
than a slightly weaker but consistent one) and when the trade count is too
low to be statistically meaningful.

The search is deliberately a plain grid over a handful of interpretable
knobs — not a black-box optimiser — because the result must stay explainable
in the UI ("min_confluence 2→3, zone_atr 0.5→0.3") and cheap enough to run
behind a cancellable HTTP endpoint.

The whole sweep is **synchronous** so the API layer can offload it with
``asyncio.to_thread`` and keep serving ``/backtest/cancel`` while it runs.
"""
from __future__ import annotations

import asyncio
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from trading.application.cancellation import CancelToken
from trading.application.strategy_factory import build_strategy
from trading.domain import Bar

from .engine import BacktestConfig, BacktestResult, run_backtest
from .trade_analysis import TradeAnalysis, analyze_trades, recommend_adjustments

__all__ = [
    "OptimizeCandidate",
    "OptimizeResult",
    "DEFAULT_GRID",
    "UNIFIED_GRID",
    "OPTIMIZATION_OBJECTIVES",
    "SWEEP_MIN_REFRESH",
    "MAX_GRID_COMBOS",
    "optimize_strategy",
    "metric_value",
]

#: Default sweep for ``trend_confluence`` (64 combos; the request can override
#: with its own grid). Besides the confluence-zone knobs it sweeps the
#: risk-management params (hard stop / take-profit / trail) and the
#: confirmation-candle filter, because those are what the profit×win-rate
#: objective trades off.
DEFAULT_GRID: dict[str, list[Any]] = {
    "zone_atr": [0.3, 0.8],
    "min_confluence": [2, 3],
    "stop_atr": [1.5, 2.5],
    "tp_r": [0.0, 2.0],
    "atr_trail_mult": [2.0, 3.0],
    "need_rejection": [True, False],
}

#: Default sweep for the unified strategy. Grid keys may use dotted paths to
#: reach nested blocks — ``"emf.atr_tp_mult"`` sets ``params["emf"]
#: ["atr_tp_mult"]`` — so EMF+ADL fields are optimizable exactly like the
#: flat Trend-Confluence knobs. Kept at 64 combos so the default run stays
#: interactive; the request can override any axis.
UNIFIED_GRID: dict[str, list[Any]] = {
    "zone_atr": [0.5, 0.8],
    "min_confluence": [2, 3],
    "emf_mode": ["require", "bonus"],
    "momentum_mode": ["gate", "bonus"],
    "emf.atr_tp_mult": [2.0],
    "atr_trail_mult": [3.0],
    "stop_atr": [1.5, 2.5],
    "need_rejection": [True, False],
}

#: Default sweep for ``trend_confluence_pine`` (16 combos). The Pine knobs the
#: user tunes most (TP %, trailing %, zone width, confluence count) — the
#: console can sweep any other schema parameter by sending an explicit grid.
PINE_GRID: dict[str, list[Any]] = {
    "zone_atr": [0.5, 0.8],
    "min_confluence": [2, 3],
    "tp_percent": [1.5, 3.0],
    "trailing_percent": [0.8, 2.0],
}

#: Optimization objectives the sweep can rank by (applied to the validation
#: split). ``max_drawdown`` is maximised as ``-max_drawdown`` so "bigger is
#: better" holds for every objective.
#:
#: ``profit_win`` (the default) is the composite the strategy refocuses on:
#: **profitability first** (validation total return gates — a candidate that
#: loses money on validation is disqualified outright, never merely "least
#: bad"), **win rate second** (a multiplier in [0.5, 1.5]) and **drawdown
#: third** (a discount that saturates at ``DD_PENALTY_CAP``), with the usual
#: overfitting and thin-sample discounts on top.
OPTIMIZATION_OBJECTIVES: tuple[str, ...] = (
    "profit_win",
    "sharpe",
    "sortino",
    "calmar",
    "total_return",
    "profit_factor",
    "win_rate",
    "max_drawdown",
)

#: Drawdown fraction at which the profit_win penalty saturates (50 % DD costs
#: the full ``DD_PENALTY_WEIGHT`` of the score; 25 % costs half of it).
DD_PENALTY_CAP = 0.50
#: Maximum score discount the drawdown penalty applies (at the cap).
DD_PENALTY_WEIGHT = 0.5
#: Validation-trade count at which the thin-sample ramp reaches full credit.
SAMPLE_FULL_TRADES = 15.0

#: Sweep speed guard: the trendline rebuild in ``prepare()`` dominates the
#: per-bar cost, so sweep legs (baseline + train/validation) never refresh the
#: trendlines more often than every ``SWEEP_MIN_REFRESH`` bars — even when the
#: caller pinned a denser cadence (the Pine console ships ``trendline_refresh``
#: default 5, and 1 makes a single 1 000-bar backtest ~6× slower, turning a
#: modest grid into a multi-minute sweep that looks like a hang). The final
#: best-candidate re-run on the full history still honors the caller's value.
SWEEP_MIN_REFRESH = 10

#: Hard cap on grid combinations. Beyond this the sweep would run for many
#: minutes (worst case hours) with no visible progress; a 400 with the planned
#: count beats a request that appears frozen. Interactive default grids are
#: ≤ 64 combos (asserted in tests), so the cap only rejects runaway requests.
MAX_GRID_COMBOS = 512


def metric_value(m, objective: str) -> float:
    """The objective's value for a metrics object (``-inf`` when unusable).

    ``m`` is any object with the ``BacktestMetrics`` field names.
    """
    if objective == "profit_win":
        # The composite's profitability leg; the win-rate leg and the gates
        # live in ``_score`` where both train and validation are in scope.
        value = float(m.total_return)
    elif objective == "max_drawdown":
        value = -float(m.max_drawdown)
    else:
        try:
            value = float(getattr(m, objective))
        except AttributeError as exc:
            raise ValueError(
                f"unknown objective '{objective}' (known: {', '.join(OPTIMIZATION_OBJECTIVES)})"
            ) from exc
    return value if math.isfinite(value) else -1e18


@dataclass
class OptimizeCandidate:
    params: dict[str, Any]
    train_sharpe: float
    validation_sharpe: float
    validation_return: float
    validation_trades: int
    score: float
    validation_win_rate: float = 0.0
    #: Why this candidate scored what it scored (F8): the composite's legs for
    #: the winner, or ``{"reason": …, "score": -1e9}`` when disqualified.
    score_breakdown: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "train_sharpe": round(self.train_sharpe, 3),
            "validation_sharpe": round(self.validation_sharpe, 3),
            "validation_return": round(self.validation_return, 4),
            "validation_trades": self.validation_trades,
            "validation_win_rate": round(self.validation_win_rate, 4),
            "score": round(self.score, 3),
            "score_breakdown": self.score_breakdown,
        }


def c_value(v: str) -> Any:
    """Restore a grid value's original Python type from its string form."""
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _direction(best: Any, worst: Any) -> str:
    """Human-readable direction: is a bigger value better?"""
    if not isinstance(best, (int, float)) or isinstance(best, bool) or \
            not isinstance(worst, (int, float)) or isinstance(worst, bool):
        return f"prefer {best}"
    if best > worst:
        return "higher is better"
    if best < worst:
        return "lower is better"
    return "flat"


def parameter_impact(
    candidates: Sequence[OptimizeCandidate],
    keys: Sequence[str],
    *,
    top_share: float = 0.25,
) -> list[dict[str, Any]]:
    """Rank the swept parameters by how much they actually move the score.

    For each parameter the candidates are grouped by value and summarised three
    ways, because a raw mean hides what a user needs to know:

    * ``valid_share`` — how often that value produced a candidate that passed
      the objective's gates (e.g. "profitable on validation");
    * ``mean_score`` / ``top_mean_score`` — average score of the group, and of
      only its best ``top_share`` (robust against one lucky combination);
    * ``impact`` — spread between the best and worst group by ``top_mean``.

    ``impact_share`` normalises that spread across the swept parameters, so the
    console can say "trailing_percent moves the result 3× more than ema_fast".
    Disqualified candidates (``score <= -1e8``) count towards ``valid_share``
    but not towards the means.
    """
    valid_floor = -1e8
    rows: list[dict[str, Any]] = []
    for k in keys:
        groups: dict[str, list[float]] = {}
        totals: dict[str, int] = {}
        for c in candidates:
            if k not in c.params:
                continue
            v = str(c.params[k])
            groups.setdefault(v, []).append(float(c.score))
            totals[v] = totals.get(v, 0) + 1
        if not groups:
            continue
        values: list[dict[str, Any]] = []
        for v, scores in groups.items():
            ok = sorted((s for s in scores if s > valid_floor), reverse=True)
            keep = max(1, int(len(ok) * top_share))
            top = ok[:keep]
            values.append({
                "value": c_value(v),
                "n": totals[v],
                "valid_share": len(ok) / totals[v] if totals[v] else 0.0,
                "mean_score": (sum(ok) / len(ok)) if ok else None,
                "top_mean_score": (sum(top) / len(top)) if top else None,
                "best_score": ok[0] if ok else None,
            })
        scored = [x for x in values if x["top_mean_score"] is not None]
        if not scored:
            rows.append({
                "param": k,
                "values": sorted(values, key=lambda x: -x["valid_share"]),
                "recommended": None, "impact": 0.0, "impact_share": 0.0,
                "direction": "none",
                "note": "no candidate passed the objective gates",
            })
            continue
        best_v = max(scored, key=lambda x: (x["top_mean_score"], x["valid_share"]))
        worst_v = min(scored, key=lambda x: x["top_mean_score"])
        impact = float(best_v["top_mean_score"] - worst_v["top_mean_score"])
        rows.append({
            "param": k,
            "values": sorted(values, key=lambda x: -(x["top_mean_score"] or -1e18)),
            "recommended": best_v["value"],
            "worst": worst_v["value"],
            "impact": impact,
            "impact_share": 0.0,  # normalised below
            "direction": _direction(best_v["value"], worst_v["value"]),
            "note": "",
        })
    total = sum(r["impact"] for r in rows)
    for r in rows:
        r["impact_share"] = (r["impact"] / total) if total > 0 else 0.0
    rows.sort(key=lambda r: r["impact"], reverse=True)
    return rows


@dataclass
class OptimizeResult:
    symbol: str
    strategy: str
    n_candidates: int
    baseline: dict[str, Any]
    best_params: dict[str, Any]
    best: dict[str, Any]
    leaderboard: list[OptimizeCandidate] = field(default_factory=list)
    trade_analysis: TradeAnalysis | None = None
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    impact: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "n_candidates": self.n_candidates,
            "baseline": self.baseline,
            "best_params": self.best_params,
            "best": self.best,
            "leaderboard": [c.as_dict() for c in self.leaderboard],
            "trade_analysis": self.trade_analysis.as_dict() if self.trade_analysis else None,
            "recommendations": self.recommendations,
            "impact": self.impact,
        }


def _metrics_brief(res: BacktestResult) -> dict[str, Any]:
    m = res.metrics
    return {
        "total_return": round(m.total_return, 4),
        "sharpe": round(m.sharpe, 3),
        "max_drawdown": round(m.max_drawdown, 4),
        "win_rate": round(m.win_rate, 4),
        "n_trades": len(res.trades),
    }


def _bt(name: str, symbol: str, params: Mapping[str, Any],
        bars: Sequence[Bar], cfg: BacktestConfig) -> BacktestResult:
    """Synchronous one-shot backtest (the sweep runs inside ``to_thread``)."""
    strategy = build_strategy(name, symbol, params)
    return asyncio.run(run_backtest(strategy, bars, cfg))


def _score(train: BacktestResult, val: BacktestResult,
           objective: str = "profit_win") -> float:
    """Validation objective, discounted for overfitting and thin samples.

    ``profit_win`` (default) is the profit-and-win-rate composite with an
    explicit drawdown penalty: disqualifying when the validation split loses
    money or has no trades, otherwise
    ``val_return × (0.5 + win_rate) × dd_mult × trend × sample`` — profit
    sets the magnitude, win rate scales it ±50 %, drawdown discounts it up to
    50 % (a 25 % validation DD costs a quarter of the score; ≥50 % costs
    half), times an overfit discount (train must agree on the sign of the
    PnL) and a thin-sample ramp (15 trades for full credit).
    """
    score, _ = _score_with_breakdown(train, val, objective)
    return score


def _score_with_breakdown(
    train: BacktestResult, val: BacktestResult, objective: str = "profit_win"
) -> tuple[float, dict[str, Any]]:
    """``_score`` plus the per-leg breakdown the console shows (F8)."""
    if objective == "profit_win":
        v_ret = float(val.metrics.total_return)
        if v_ret <= 0 or not val.trades:
            reason = "validation_loss" if v_ret <= 0 else "no_trades"
            return -1e9, {"reason": reason, "score": -1e9}  # never crown a loser
        win = float(val.metrics.win_rate)
        max_dd = max(0.0, float(val.metrics.max_drawdown))
        win_mult = 0.5 + win
        dd_mult = 1.0 - DD_PENALTY_WEIGHT * min(1.0, max_dd / DD_PENALTY_CAP)
        trend = 1.0 if float(train.metrics.total_return) > 0 else 0.3
        sample = min(1.0, len(val.trades) / SAMPLE_FULL_TRADES)
        score = v_ret * win_mult * dd_mult * trend * sample
        breakdown = {
            "validation_return": round(v_ret, 4),
            "win_rate": round(win, 4),
            "win_multiplier": round(win_mult, 4),
            "max_drawdown": round(max_dd, 4),
            "dd_multiplier": round(dd_mult, 4),
            "trend_multiplier": trend,
            "sample_multiplier": round(sample, 4),
            "score": round(score, 4),
        }
        return score, breakdown
    s_val = metric_value(val.metrics, objective)
    if s_val <= -1e17:  # non-finite objective (e.g. no trades → profit factor)
        return -1e9, {"reason": "no_trades", "score": -1e9}
    # Overfit penalty: train says great but validation disagrees.
    penalty = 1.0
    s_train = metric_value(train.metrics, objective)
    if s_train > 0 and s_val < 0:
        penalty = 0.25
    elif s_train > 0:
        penalty = min(1.0, 0.5 + 0.5 * (s_val / s_train)) if s_val < s_train else 1.0
    # Thin-sample discount: fewer than 5 validation trades is anecdote.
    sample = min(1.0, len(val.trades) / 5.0) if val.trades else 0.0
    score = s_val * penalty * sample
    return score, {
        "objective": objective,
        "overfit_penalty": round(penalty, 4),
        "sample_multiplier": round(sample, 4),
        "score": round(score, 4),
    }


def _expand_params(
    base: Mapping[str, Any], keys: Sequence[str], combo: Sequence[Any]
) -> dict[str, Any]:
    """Merge a grid combination onto ``base``, expanding dotted grid keys.

    ``"emf.atr_tp_mult"`` targets the nested ``emf`` block (copied, never
    mutated in place) so EMF+ADL parameters sweep like flat ones.
    """
    params = dict(base)
    for k, v in zip(keys, combo):
        if "." in k:
            outer, inner = k.split(".", 1)
            block = dict(params.get(outer) or {})
            block[inner] = v
            params[outer] = block
        else:
            params[k] = v
    return params


def _default_grid(strategy: str) -> dict[str, list[Any]]:
    if strategy == "trend_confluence_pine":
        return dict(PINE_GRID)
    return dict(UNIFIED_GRID) if strategy == "trend_confluence_unified" else dict(DEFAULT_GRID)


def optimize_strategy(
    name: str,
    symbol: str,
    bars: Sequence[Bar],
    *,
    base_params: Mapping[str, Any] | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    cfg: BacktestConfig | None = None,
    train_fraction: float = 0.7,
    cancel: CancelToken | None = None,
    top_n: int = 10,
    objective: str = "profit_win",
) -> OptimizeResult:
    """Grid-search ``name``'s parameters on ``bars`` with a train/validation split.

    Candidates are ranked by ``objective`` (default: the ``profit_win``
    composite — validation profit gates the pick and win rate scales it;
    see :data:`OPTIMIZATION_OBJECTIVES`), discounted for train/validation
    disagreement (overfitting) and thin trade samples. Grid keys may use
    dotted paths (``"emf.atr_tp_mult"``) to sweep nested parameter blocks.

    The sweep is deterministic: a plain ``itertools.product`` over the grid
    in declared key order — no randomness anywhere.

    Raises :class:`RunCancelled` (via the cancel token) between candidates.
    """
    if objective not in OPTIMIZATION_OBJECTIVES:
        raise ValueError(
            f"unknown objective '{objective}' (known: {', '.join(OPTIMIZATION_OBJECTIVES)})"
        )
    bars = sorted(bars, key=lambda b: b.timestamp)
    if len(bars) < 100:
        raise ValueError(f"optimization needs >= 100 bars, got {len(bars)}")
    cfg = cfg or BacktestConfig()
    base = dict(base_params or {})
    # An explicit empty grid must not silently fall back to the (much larger)
    # strategy default: only ``grid=None`` means "use the default sweep".
    if grid is None:
        grid = _default_grid(name)
    grid = {k: list(v) for k, v in grid.items() if v}
    if not grid:
        raise ValueError(
            "empty sweep grid — pass at least one parameter with 2+ candidate "
            "values, or omit 'grid' to use the strategy's default sweep"
        )
    planned = 1
    for values in grid.values():
        planned *= len(values)
    if planned > MAX_GRID_COMBOS:
        raise ValueError(
            f"sweep grid too large: {planned} combinations (max {MAX_GRID_COMBOS}) "
            "— remove axes or candidate values"
        )
    # Sweep speed guard (see SWEEP_MIN_REFRESH): denser trendline cadences are
    # honored for the final best-candidate re-run but clamped for the sweep
    # itself, whose cost is planned-combos × 2 backtests.
    raw_refresh = base.get("trendline_refresh")
    try:
        user_refresh = int(raw_refresh) if raw_refresh is not None else None
    except (TypeError, ValueError):
        user_refresh = None
    if user_refresh is None or user_refresh < SWEEP_MIN_REFRESH:
        base["trendline_refresh"] = SWEEP_MIN_REFRESH

    split = max(50, int(len(bars) * train_fraction))
    split = min(split, len(bars) - 50)
    train_bars, val_bars = bars[:split], bars[split:]

    if cancel:
        cancel.check()
    baseline_res = _bt(name, symbol, base, bars, cfg)

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) or [()]
    candidates: list[OptimizeCandidate] = []
    for combo in combos:
        if cancel:
            cancel.check()
        params = _expand_params(base, keys, combo)
        try:
            train_res = _bt(name, symbol, params, train_bars, cfg)
            val_res = _bt(name, symbol, params, val_bars, cfg)
        except Exception:  # a degenerate combo must not sink the sweep
            continue
        score, breakdown = _score_with_breakdown(train_res, val_res, objective)
        candidates.append(OptimizeCandidate(
            params=dict(zip(keys, combo)),
            train_sharpe=train_res.metrics.sharpe,
            validation_sharpe=val_res.metrics.sharpe,
            validation_return=val_res.metrics.total_return,
            validation_trades=len(val_res.trades),
            validation_win_rate=val_res.metrics.win_rate,
            score=score,
            score_breakdown=breakdown,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    impact = parameter_impact(candidates, keys)
    best_combo = candidates[0].params if candidates else {}
    active_keys = [k for k in keys if k in best_combo]
    best_params = _expand_params(base, active_keys, [best_combo[k] for k in active_keys])
    # The winner is re-run (and persisted) with the caller's own cadence, not
    # the sweep-only clamp applied above.
    if user_refresh is not None:
        best_params["trendline_refresh"] = user_refresh
    if cancel:
        cancel.check()
    best_res = _bt(name, symbol, best_params, bars, cfg)
    analysis = analyze_trades(best_res.trades)

    return OptimizeResult(
        symbol=symbol,
        strategy=name,
        n_candidates=len(candidates),
        baseline={"params": base, "metrics": _metrics_brief(baseline_res)},
        best_params=best_params,
        best={
            "metrics": _metrics_brief(best_res),
            "objective": objective,
            # F8: mirror the winning candidate's validation legs so the UI can
            # show *why* it won (profit / win rate / drawdown components).
            "score_breakdown": candidates[0].score_breakdown if candidates else {},
        },
        leaderboard=candidates[:top_n],
        trade_analysis=analysis,
        recommendations=recommend_adjustments(analysis, best_params),
        impact=impact,
    )
