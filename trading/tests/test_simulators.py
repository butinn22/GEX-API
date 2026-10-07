"""Tests for Monte-Carlo path simulators."""
from __future__ import annotations

import numpy as np
import pytest

from trading.application.backtest.simulators import (
    block_bootstrap,
    bootstrap_residuals,
    geometric_brownian_motion,
    historical_resampling,
)


def test_gbm_shape_and_positivity():
    paths = geometric_brownian_motion(100.0, 0.1, 0.2, 50, 1 / 252, n_paths=10, seed=0)
    assert paths.shape == (10, 51)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_gbm_drift_and_vol_statistics():
    # With many paths the sample mean/std of log-returns should match the theory.
    n_paths, n_steps, dt = 20_000, 100, 1 / 252
    mu, sigma = 0.1, 0.2
    paths = geometric_brownian_motion(1.0, mu, sigma, n_steps, dt, n_paths=n_paths, seed=0)
    log_returns = np.diff(np.log(paths), axis=1)
    total_log_return = log_returns.sum(axis=1)  # per-path total log return
    expected_mean = (mu - 0.5 * sigma**2) * n_steps * dt
    expected_std = sigma * np.sqrt(n_steps * dt)
    assert total_log_return.mean() == pytest.approx(expected_mean, rel=0.05)
    assert total_log_return.std(ddof=1) == pytest.approx(expected_std, rel=0.05)


def test_gbm_rejects_bad_params():
    with pytest.raises(ValueError):
        geometric_brownian_motion(-1.0, 0.1, 0.2, 10, 1 / 252)
    with pytest.raises(ValueError):
        geometric_brownian_motion(1.0, 0.1, -0.2, 10, 1 / 252)


def test_bootstrap_residuals_shape_and_seed():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.02, 100)
    p1 = bootstrap_residuals(returns, 30, n_paths=5, seed=3)
    p2 = bootstrap_residuals(returns, 30, n_paths=5, seed=3)
    assert p1.shape == (5, 31)
    assert np.all(p1[:, 0] == 1.0)
    assert np.allclose(p1, p2)  # deterministic given seed


def test_bootstrap_residuals_mean_preserved():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.02, 500)
    paths = bootstrap_residuals(returns, 200, n_paths=100, seed=0)
    sampled = np.diff(paths, axis=1)
    # resampled returns stay close to the historical mean
    assert sampled.mean() == pytest.approx(returns.mean(), abs=0.01)


def test_block_bootstrap_shape():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.02, 100)
    paths = block_bootstrap(returns, 40, block_size=10, n_paths=8, seed=0)
    assert paths.shape == (8, 41)
    assert np.all(paths[:, 0] == 1.0)
    assert np.all(np.isfinite(paths))


def test_block_bootstrap_preserves_historical_drift():
    # Regression: the mean was removed to build residuals and never added back,
    # so every block-bootstrap path was driftless even for a strongly positive
    # strategy. The resampled per-step returns must track the historical mean.
    rng = np.random.default_rng(3)
    returns = rng.normal(0.01, 0.02, 400)
    paths = block_bootstrap(returns, 300, block_size=20, n_paths=200, seed=0)
    per_step = paths[:, 1:] / paths[:, :-1] - 1.0
    assert per_step.mean() == pytest.approx(returns.mean(), abs=0.005)


def test_historical_resampling_alias():
    returns = np.array([0.01, -0.02, 0.03, 0.0])
    assert np.allclose(
        historical_resampling(returns, 10, n_paths=2, seed=0),
        bootstrap_residuals(returns, 10, n_paths=2, seed=0),
    )
