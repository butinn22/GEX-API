"""Monte-Carlo path simulators.

Each simulator returns a 2-D array of shape ``(n_paths, n_steps + 1)`` of
price-like paths (index 0 is the starting level). They are used to stress-test a
strategy's return distribution and to put confidence intervals on metrics.

* ``geometric_brownian_motion`` — log-normal diffusion with drift ``mu`` / vol ``sigma``.
* ``bootstrap_residuals`` — resample mean-removed historical returns (i.i.d.).
* ``block_bootstrap`` — resample contiguous blocks (preserves serial correlation).
* ``historical_resampling`` — alias of ``bootstrap_residuals``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "geometric_brownian_motion",
    "bootstrap_residuals",
    "block_bootstrap",
    "historical_resampling",
]


def geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float,
    n_paths: int = 1,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Geometric Brownian motion: dS = mu*S*dt + sigma*S*dW (Euler–Maruyama)."""
    if s0 <= 0 or sigma < 0 or n_steps < 0 or n_paths < 1:
        raise ValueError("invalid GBM parameters")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_paths, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt) * z
    log_path = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(drift + shock, axis=1)], axis=1
    )
    return s0 * np.exp(log_path)


def _residual_returns(returns: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw ``size`` returns by resampling mean-removed residuals with replacement."""
    r = np.asarray(returns, dtype=float)
    residuals = r - r.mean()
    idx = rng.integers(0, residuals.size, size=size)
    return residuals[idx] + r.mean()


def bootstrap_residuals(
    returns: np.ndarray,
    n_steps: int,
    n_paths: int = 1,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Resample historical returns i.i.d. and compound them into paths (start 1.0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or n_steps < 0 or n_paths < 1:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    sampled = np.stack([_residual_returns(r, rng, n_steps) for _ in range(n_paths)])
    return np.concatenate([np.ones((n_paths, 1)), np.cumprod(1.0 + sampled, axis=1)], axis=1)


def block_bootstrap(
    returns: np.ndarray,
    n_steps: int,
    block_size: int,
    n_paths: int = 1,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Resample contiguous blocks of (mean-removed) returns, compound into paths."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or n_steps < 0 or n_paths < 1 or block_size < 1:
        raise ValueError("invalid block-bootstrap parameters")
    residuals = r - r.mean()
    n = residuals.size
    rng = np.random.default_rng(seed)

    def _one_path() -> np.ndarray:
        sample: list[float] = []
        while len(sample) < n_steps:
            start = int(rng.integers(0, max(n - block_size + 1, 1)))
            sample.extend(residuals[start : start + block_size].tolist())
        return np.asarray(sample[:n_steps], dtype=float)

    sampled = np.stack([_one_path() for _ in range(n_paths)])
    return np.concatenate([np.ones((n_paths, 1)), np.cumprod(1.0 + sampled, axis=1)], axis=1)


def historical_resampling(returns, n_steps, n_paths=1, *, seed=None) -> np.ndarray:
    """Alias of :func:`bootstrap_residuals` (resample historical returns i.i.d.)."""
    return bootstrap_residuals(returns, n_steps, n_paths, seed=seed)
