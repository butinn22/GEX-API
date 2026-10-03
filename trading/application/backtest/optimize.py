"""Adaptive parameter search — the optimisation half of the win/loss loop.

Method
------
A small grid sweep over the strategy's tunables, scored **out of sample**:
bars are split 70 / 30 into train / validation; each candidate is backtested
on both, and ranked by validation Sharpe, discounted when the train and
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
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from trading.application.cancellation import CancelToken, RunCancelled
from trading.application.strategy_factory import build_strategy
from trading.domain import Bar

from .engine import BacktestConfig, BacktestResult, run_backtest
from .trade_analysis import TradeAnalysis, analyze_trades, recommend_adjustments

__all__ = ["OptimizeCandidate", "OptimizeResult", "DEFAULT_GRID", "optimize_strategy"]

#: Default sweep for ``trend_confluence`` (18 combos — a couple of minutes on
#: 1,000 daily bars; the request can override with its own grid).
DEFAULT_GRID: dict[str, list[Any]] = {
    "zone_atr": [0.3, 0.5, 0.8],
    "min_confluence": [2, 3],
    "atr_trail_mult": [2.0, 3.0, 4.0],
}


@dataclass
class OptimizeCandidate:
    params: dict[str, Any]
    train_sharpe: float
    validation_sharpe: float
    validation_return: float
    validation_trades: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "train_sharpe": round(self.train_sharpe, 3),
            "validation_sharpe": round(self.validation_sharpe, 3),
            "validation_return": round(self.validation_return, 4),
            "validation_trades": self.validation_trades,
            "score": round(self.score, 3),
        }


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


def _score(train: BacktestResult, val: BacktestResult) -> float:
    """Validation Sharpe, discounted for overfitting and thin samples."""
    s_val = val.metrics.sharpe
    if not math.isfinite(s_val):
        return -1e9
    # Overfit penalty: train says great but validation disagrees.
    penalty = 1.0
    s_train = train.metrics.sharpe
    if math.isfinite(s_train) and s_train > 0 and s_val < 0:
        penalty = 0.25
    elif math.isfinite(s_train) and s_train > 0:
        penalty = min(1.0, 0.5 + 0.5 * (s_val / s_train)) if s_val < s_train else 1.0
    # Thin-sample discount: fewer than 5 validation trades is anecdote.
    sample = min(1.0, len(val.trades) / 5.0) if val.trades else 0.0
    return s_val * penalty * sample


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
) -> OptimizeResult:
    """Grid-search ``name``'s parameters on ``bars`` with a train/validation split.

    Raises :class:`RunCancelled` (via the cancel token) between candidates.
    """
    bars = sorted(bars, key=lambda b: b.timestamp)
    if len(bars) < 100:
        raise ValueError(f"optimization needs >= 100 bars, got {len(bars)}")
    cfg = cfg or BacktestConfig()
    base = dict(base_params or {})
    grid = {k: list(v) for k, v in (grid or DEFAULT_GRID).items() if v}
    # Speed the sweep up unless the caller pinned the cadence: trendlines every
    # 10 bars changes little but halves the per-run cost.
    base.setdefault("trendline_refresh", 10)

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
        params = {**base, **dict(zip(keys, combo))}
        try:
            train_res = _bt(name, symbol, params, train_bars, cfg)
            val_res = _bt(name, symbol, params, val_bars, cfg)
        except Exception:  # a degenerate combo must not sink the sweep
            continue
        candidates.append(OptimizeCandidate(
            params=dict(zip(keys, combo)),
            train_sharpe=train_res.metrics.sharpe,
            validation_sharpe=val_res.metrics.sharpe,
            validation_return=val_res.metrics.total_return,
            validation_trades=len(val_res.trades),
            score=_score(train_res, val_res),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    best_params = {**base, **(candidates[0].params if candidates else {})}
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
        best={"metrics": _metrics_brief(best_res)},
        leaderboard=candidates[:top_n],
        trade_analysis=analysis,
        recommendations=recommend_adjustments(analysis, best_params),
    )
