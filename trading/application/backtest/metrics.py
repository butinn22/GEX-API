"""Backtest performance metrics and bootstrap confidence intervals.

Everything works on numpy arrays (no pandas) so the metrics are usable both by
the event-driven engine and by the Monte-Carlo path simulators.

Conventions
-----------
* ``equity`` is a 1-D array of portfolio equity sampled at bar close.
* Returns are *simple* returns: ``r_t = (e_t - e_{t-1}) / e_{t-1}``.
* Risk-free rate is assumed 0 (configurable later).
* ``periods_per_year`` annualises Sharpe/Sortino and the CAGR (252 for daily).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BacktestMetrics",
    "compute_metrics",
    "returns_from_equity",
    "calmar",
    "bootstrap_equity_ci",
    "bootstrap_trade_ci",
]

_EPS = 1e-12


def returns_from_equity(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, dtype=float)
    if e.ndim != 1 or e.size < 2:
        return np.array([], dtype=float)
    if e[0] <= 0:
        raise ValueError("equity must start at a positive value")
    return np.diff(e) / e[:-1]


def total_return(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if e.size < 2 or e[0] <= 0:
        return 0.0
    return float(e[-1] / e[0] - 1.0)


def annualized_return(equity: np.ndarray, periods_per_year: int) -> float:
    tr = total_return(equity)
    n = max(len(equity) - 1, 1)
    if tr <= -1.0:
        return -1.0
    return float((1.0 + tr) ** (periods_per_year / n) - 1.0)


def sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < _EPS:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: int) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    # Semi-deviation is defined over *all* periods, with the positive returns
    # clipped to zero — not over only the losing observations. Averaging over
    # just the losers inflates the denominator whenever losses are a small,
    # severe tail, understating Sortino.
    downside = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd < _EPS:
        return float("inf") if r.mean() > 0 else 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / np.where(peak > 0, peak, 1.0)
    return float(dd.max())


def calmar(equity: np.ndarray, periods_per_year: int) -> float:
    """Calmar ratio: annualized return / max drawdown (drawdown as a positive fraction)."""
    mdd = max_drawdown(equity)
    if mdd < _EPS:
        return float("inf") if annualized_return(equity, periods_per_year) > 0 else 0.0
    return float(annualized_return(equity, periods_per_year) / mdd)


def value_at_risk(returns: np.ndarray, q: float = 0.05) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.percentile(r, q * 100.0))


def conditional_var(returns: np.ndarray, q: float = 0.05) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    v = value_at_risk(r, q)
    tail = r[r <= v]
    return float(tail.mean()) if tail.size else v


def win_rate(trade_pnls: Sequence[float]) -> float:
    pnls = list(trade_pnls)
    if not pnls:
        return 0.0
    return float(sum(1 for p in pnls if p > 0) / len(pnls))


def profit_factor(trade_pnls: Sequence[float]) -> float:
    pnls = list(trade_pnls)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss < _EPS:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_periods: int


def compute_metrics(
    equity: np.ndarray,
    trade_pnls: Sequence[float] | None = None,
    *,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute the standard metric set from an equity curve + trade PnLs."""
    e = np.asarray(equity, dtype=float)
    r = returns_from_equity(e)
    pnls = list(trade_pnls) if trade_pnls is not None else []
    return BacktestMetrics(
        total_return=total_return(e),
        annualized_return=annualized_return(e, periods_per_year),
        sharpe=sharpe(r, periods_per_year),
        sortino=sortino(r, periods_per_year),
        calmar=calmar(e, periods_per_year),
        max_drawdown=max_drawdown(e),
        var_95=value_at_risk(r, 0.05),
        cvar_95=conditional_var(r, 0.05),
        win_rate=win_rate(pnls),
        profit_factor=profit_factor(pnls),
        n_trades=len(pnls),
        n_periods=int(r.size),
    )


# ── Bootstrap confidence intervals ─────────────────────────────────────


def _block_bootstrap_returns(
    returns: np.ndarray, n_boot: int, block_size: int, seed: int | None
) -> list[np.ndarray]:
    """Return ``n_boot`` block-resampled return vectors (length preserved)."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    n = r.size
    out: list[np.ndarray] = []
    for _ in range(n_boot):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.integers(0, max(n - block_size + 1, 1)))
            sample.extend(r[start : start + block_size].tolist())
        out.append(np.asarray(sample[:n], dtype=float))
    return out


def bootstrap_equity_ci(
    metric_fn: Callable[[np.ndarray], float],
    equity: np.ndarray,
    *,
    n_boot: int = 1000,
    block_size: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for an equity-based metric via block bootstrap of returns.

    Returns are block-resampled (to preserve serial correlation), compounded back
    into an equity curve, and ``metric_fn`` is recomputed per path.
    """
    e = np.asarray(equity, dtype=float)
    r = returns_from_equity(e)
    if r.size == 0:
        return (0.0, 0.0)
    boot = _block_bootstrap_returns(r, n_boot, block_size, seed)
    stats = []
    for br in boot:
        eq = np.concatenate([[1.0], np.cumprod(1.0 + br)])
        stats.append(metric_fn(eq))
    lo, hi = np.percentile(stats, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def bootstrap_trade_ci(
    metric_fn: Callable[[Sequence[float]], float],
    trade_pnls: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for a trade-based metric via i.i.d. resampling of trade PnLs."""
    pnls = np.asarray(list(trade_pnls), dtype=float)
    if pnls.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, pnls.size, size=pnls.size)
        stats.append(metric_fn(pnls[idx].tolist()))
    lo, hi = np.percentile(stats, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)
