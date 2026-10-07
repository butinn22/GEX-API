"""Monte-Carlo backtest engine.

Takes a **realised return series** (e.g. a strategy's per-bar returns from a
historical backtest) and generates ``n_paths`` forward scenarios under one of
four models, then summarises the *distribution* of outcomes:

* percentile equity bands → the fan chart;
* a histogram of final returns;
* per-path performance metrics (total/annualised return, Sharpe, Sortino, max DD)
  aggregated to a mean **and a 95 % confidence interval**;
* tail risk (VaR/CVaR) and the probability of profit.

All heavy lifting is vectorised on ``(n_paths, n_steps + 1)`` numpy arrays — a
10 000-path run is a handful of array ops, so it is fast enough to call in the
request path and trivially parallelisable in Celery for much larger N.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .metrics import returns_from_equity
from .simulators import block_bootstrap, bootstrap_residuals, geometric_brownian_motion

if TYPE_CHECKING:  # pragma: no cover
    from trading.application.cancellation import CancelToken

__all__ = [
    "MC_METHODS",
    "CANCEL_CHUNK_PATHS",
    "MonteCarloConfig",
    "Histogram",
    "MonteCarloResult",
    "run_monte_carlo",
    "run_monte_carlo_from_equity",
]

#: Supported path-generation models.
MC_METHODS: tuple[str, ...] = ("gbm", "bootstrap", "block_bootstrap", "historical")

#: Paths generated per cancellable block. Small enough that a stop feels
#: instant, large enough that per-block overhead is invisible. Runs at or below
#: this size take the original single-shot path, so results are unchanged.
CANCEL_CHUNK_PATHS = 4_096

_BANDS: tuple[int, ...] = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class MonteCarloConfig:
    n_paths: int = 10_000
    n_steps: int | None = None  # defaults to the length of the input series
    method: str = "gbm"
    block_size: int = 5
    dt: float = 1.0
    seed: int | None = 0
    initial_equity: float = 100_000.0
    periods_per_year: int = 252
    histogram_bins: int = 40
    max_histogram_samples: int = 5_000


@dataclass(frozen=True)
class Histogram:
    counts: tuple[int, ...]
    centers: tuple[float, ...]
    bin_edges: tuple[float, ...]


@dataclass
class MonteCarloResult:
    method: str
    n_paths: int
    n_steps: int
    initial_equity: float
    steps: tuple[int, ...]
    bands: dict[str, tuple[float, ...]]  # "p5".."p95" → equity per step
    mean_path: tuple[float, ...]
    final_percentiles: dict[str, float]  # percentiles of final *equity*
    final_return_percentiles: dict[str, float]  # percentiles of final *return*
    mean_return: float
    histogram: Histogram
    metrics_mean: dict[str, float]
    metrics_ci: dict[str, tuple[float, float]]
    prob_profit: float
    var_95: float
    cvar_95: float
    best_return: float
    worst_return: float


def _simulate_block(
    returns: np.ndarray, cfg: MonteCarloConfig, n_steps: int, n_paths: int, seed: int | None
) -> np.ndarray:
    """Generate ``n_paths`` equity-multiplier paths (index 0 starts at 1.0)."""
    r = np.asarray(returns, dtype=float)
    method = cfg.method
    if method in ("bootstrap", "historical"):
        return bootstrap_residuals(r, n_steps, n_paths, seed=seed)
    if method == "block_bootstrap":
        return block_bootstrap(r, n_steps, cfg.block_size, n_paths, seed=seed)
    if method == "gbm":
        logret = np.log1p(r)
        mu = float(logret.mean()) if logret.size else 0.0
        sigma = float(logret.std(ddof=1)) if logret.size > 1 else 0.0
        return geometric_brownian_motion(
            1.0, mu, sigma, n_steps, cfg.dt, n_paths=n_paths, seed=seed
        )
    raise ValueError(f"unknown Monte-Carlo method '{method}' (known: {', '.join(MC_METHODS)})")


def _block_seed(seed: int | None, block: int) -> int | None:
    """Deterministic per-block seed, so chunking never changes reproducibility.

    Block 0 reuses ``seed`` exactly, which keeps every run small enough to fit in
    a single block bit-identical to the pre-chunking behaviour.
    """
    if seed is None:
        return None
    return seed + block * 1_000_003


def _simulate(
    returns: np.ndarray,
    cfg: MonteCarloConfig,
    n_steps: int,
    cancel: CancelToken | None = None,
) -> np.ndarray:
    """Return an ``(n_paths, n_steps + 1)`` array of equity multipliers (start 1.0).

    Long runs are generated in blocks so ``cancel`` is observed promptly instead
    of after the whole simulation.
    """
    n = cfg.n_paths
    if n <= CANCEL_CHUNK_PATHS:
        return _simulate_block(returns, cfg, n_steps, n, cfg.seed)
    blocks: list[np.ndarray] = []
    done = 0
    i = 0
    while done < n:
        if cancel is not None:
            cancel.check()
        take = min(CANCEL_CHUNK_PATHS, n - done)
        blocks.append(
            _simulate_block(returns, cfg, n_steps, take, _block_seed(cfg.seed, i))
        )
        done += take
        i += 1
    return blocks[0] if len(blocks) == 1 else np.vstack(blocks)


def _path_metrics(equity: np.ndarray, periods_per_year: int) -> dict[str, np.ndarray]:
    """Vectorised per-path metric arrays for an ``(P, S+1)`` equity matrix."""
    start = equity[:, :1]
    total = equity[:, -1:] / start - 1.0
    n = max(equity.shape[1] - 1, 1)
    annual = np.where(total <= -1.0, -1.0, np.power(np.clip(1.0 + total, 0.0, None),
                                                     periods_per_year / n) - 1.0)

    peak = np.maximum.accumulate(equity, axis=1)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    max_dd = dd.max(axis=1)

    rets = equity[:, 1:] / equity[:, :-1] - 1.0
    sd = rets.std(axis=1, ddof=1) if rets.shape[1] > 1 else np.zeros(equity.shape[0])
    mean = rets.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 1e-12, mean / sd * np.sqrt(periods_per_year), 0.0)

    downside = np.where(rets < 0, rets, 0.0)
    dd_std = np.sqrt((downside**2).mean(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino = np.where(dd_std > 1e-12, mean / dd_std * np.sqrt(periods_per_year), 0.0)

    return {
        "total_return": np.squeeze(total, axis=1),
        "annualized_return": np.squeeze(annual, axis=1),
        "sharpe": np.nan_to_num(sharpe),
        "sortino": np.nan_to_num(sortino),
        "max_drawdown": max_dd,
    }


def _histogram(values: np.ndarray, bins: int, cap: int) -> Histogram:
    v = _finite(np.asarray(values, dtype=float))
    if v.size == 0:
        return Histogram(counts=(), centers=(), bin_edges=())
    if v.size > cap:
        rng = np.random.default_rng(0)
        v = v[rng.choice(v.size, size=cap, replace=False)]
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    # A (nearly) constant sample — e.g. a flat return series, where every
    # bootstrap path lands on the same number — leaves numpy with a zero-width
    # range and it refuses to build finite bins. Widen the window ourselves,
    # relative to the magnitude of the data, before bucketing.
    scale = max(abs(lo), abs(hi), 1.0)
    pad = scale * 0.05 if not np.isfinite(span) or span < scale * 1e-09 else span * 0.02
    counts, edges = np.histogram(v, bins=bins, range=(lo - pad, hi + pad))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return Histogram(
        counts=tuple(int(c) for c in counts),
        centers=tuple(float(c) for c in centers),
        bin_edges=tuple(float(e) for e in edges),
    )


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def run_monte_carlo(
    returns: Sequence[float] | np.ndarray,
    config: MonteCarloConfig | None = None,
    *,
    cancel: CancelToken | None = None,
) -> MonteCarloResult:
    """Run a Monte-Carlo simulation over a simple-return series.

    Pass ``cancel`` to make a long run interruptible; the engine raises
    :class:`~trading.application.cancellation.RunCancelled` between blocks.
    """
    cfg = config or MonteCarloConfig()
    r = np.asarray(list(returns), dtype=float) if not isinstance(returns, np.ndarray) else returns.astype(float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("Monte-Carlo requires at least 2 finite returns")
    if cfg.n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    if cancel is not None:
        cancel.check()
    n_steps = cfg.n_steps or r.size
    mult = _simulate(r, cfg, n_steps, cancel)
    equity = mult * cfg.initial_equity
    if cancel is not None:
        cancel.check()

    steps = tuple(range(n_steps + 1))
    bands = {f"p{p}": tuple(float(x) for x in np.percentile(equity, p, axis=0)) for p in _BANDS}
    mean_path = tuple(float(x) for x in equity.mean(axis=0))
    if cancel is not None:
        cancel.check()

    final_equity = equity[:, -1]
    final_pct = {f"p{p}": float(np.percentile(final_equity, p)) for p in _BANDS}
    final_return = final_equity / cfg.initial_equity - 1.0
    final_return_pct = {f"p{p}": float(np.percentile(final_return, p)) for p in _BANDS}

    per_path = _path_metrics(equity, cfg.periods_per_year)
    if cancel is not None:
        cancel.check()
    metrics_mean: dict[str, float] = {}
    metrics_ci: dict[str, tuple[float, float]] = {}
    for key, arr in per_path.items():
        finite = _finite(arr)
        metrics_mean[key] = float(finite.mean()) if finite.size else 0.0
        if finite.size:
            lo, hi = np.percentile(finite, [5, 95])
            metrics_ci[key] = (float(lo), float(hi))
        else:
            metrics_ci[key] = (0.0, 0.0)

    var_95 = float(np.percentile(final_return, 5))
    tail = final_return[final_return <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95

    return MonteCarloResult(
        method=cfg.method,
        n_paths=cfg.n_paths,
        n_steps=n_steps,
        initial_equity=cfg.initial_equity,
        steps=steps,
        bands=bands,
        mean_path=mean_path,
        final_percentiles=final_pct,
        final_return_percentiles=final_return_pct,
        mean_return=float(final_return.mean()),
        histogram=_histogram(final_return, cfg.histogram_bins, cfg.max_histogram_samples),
        metrics_mean=metrics_mean,
        metrics_ci=metrics_ci,
        prob_profit=float((final_return > 0).mean()),
        var_95=var_95,
        cvar_95=cvar_95,
        best_return=float(final_return.max()),
        worst_return=float(final_return.min()),
    )


def run_monte_carlo_from_equity(
    equity: Sequence[float] | np.ndarray,
    config: MonteCarloConfig | None = None,
    *,
    cancel: CancelToken | None = None,
) -> MonteCarloResult:
    """Convenience wrapper: derive returns from an equity curve, then simulate."""
    if cancel is not None:
        cancel.check()
    return run_monte_carlo(
        returns_from_equity(np.asarray(equity, dtype=float)), config, cancel=cancel
    )
