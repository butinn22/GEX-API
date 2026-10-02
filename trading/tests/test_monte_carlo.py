"""Tests for the Monte-Carlo engine (paths, bands, CIs, tail risk)."""
from __future__ import annotations

import numpy as np
import pytest

from trading.application.backtest.metrics import returns_from_equity
from trading.application.backtest.monte_carlo import (
    MC_METHODS,
    MonteCarloConfig,
    run_monte_carlo,
    run_monte_carlo_from_equity,
)


def _returns(n: int = 300, mu: float = 0.0005, sigma: float = 0.02, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


# ── configuration / validation ─────────────────────────────────────────


@pytest.mark.parametrize("method", MC_METHODS)
def test_all_methods_produce_well_formed_paths(method):
    res = run_monte_carlo(
        _returns(), MonteCarloConfig(n_paths=500, method=method, seed=1)
    )
    assert res.method == method
    assert res.n_paths == 500
    assert res.n_steps == 300
    for band in ("p5", "p25", "p50", "p75", "p95"):
        assert len(res.bands[band]) == res.n_steps + 1
    assert len(res.mean_path) == res.n_steps + 1
    assert len(res.steps) == res.n_steps + 1


def test_custom_n_steps_overrides_input_length():
    res = run_monte_carlo(_returns(200), MonteCarloConfig(n_paths=100, n_steps=50, method="gbm"))
    assert res.n_steps == 50
    assert len(res.bands["p50"]) == 51


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown Monte-Carlo method"):
        run_monte_carlo(_returns(), MonteCarloConfig(n_paths=10, method="bogus"))


def test_too_few_returns_raises():
    with pytest.raises(ValueError, match="at least 2 finite returns"):
        run_monte_carlo([0.01], MonteCarloConfig(n_paths=10))


def test_non_finite_returns_are_dropped():
    r = np.concatenate([_returns(100), [np.nan, np.inf, -np.inf]])
    res = run_monte_carlo(r, MonteCarloConfig(n_paths=100, method="bootstrap"))
    assert res.n_steps == 100  # the 3 bad values were dropped


def test_zero_paths_raises():
    with pytest.raises(ValueError, match="n_paths"):
        run_monte_carlo(_returns(), MonteCarloConfig(n_paths=0))


# ── band ordering / determinism ────────────────────────────────────────


@pytest.mark.parametrize("method", MC_METHODS)
def test_percentile_bands_are_ordered(method):
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=1000, method=method, seed=7))
    for i in range(res.n_steps + 1):
        vals = [res.bands[b][i] for b in ("p5", "p25", "p50", "p75", "p95")]
        assert vals == sorted(vals), f"bands out of order at step {i}: {vals}"


@pytest.mark.parametrize("method", MC_METHODS)
def test_equal_across_runs_with_same_seed(method):
    a = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=200, method=method, seed=42))
    b = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=200, method=method, seed=42))
    assert a.bands["p50"] == b.bands["p50"]
    assert a.var_95 == b.var_95


def test_different_seeds_give_different_paths():
    a = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=200, method="gbm", seed=1))
    b = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=200, method="gbm", seed=2))
    assert a.bands["p50"] != b.bands["p50"]


def test_paths_start_at_initial_equity():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=100, method="gbm",
                                                       initial_equity=50_000.0))
    assert res.bands["p50"][0] == pytest.approx(50_000.0)
    assert res.initial_equity == pytest.approx(50_000.0)


# ── distribution statistics ────────────────────────────────────────────


def test_final_percentiles_ordered_and_consistent():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=2000, method="bootstrap", seed=3))
    fp = res.final_percentiles
    assert fp["p5"] <= fp["p25"] <= fp["p50"] <= fp["p75"] <= fp["p95"]
    fr = res.final_return_percentiles
    assert fr["p5"] <= fr["p25"] <= fr["p50"] <= fr["p75"] <= fr["p95"]


def test_headline_stats_are_finite_and_bounded():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=1000))
    assert 0.0 <= res.prob_profit <= 1.0
    assert res.worst_return <= res.best_return
    assert res.cvar_95 <= res.var_95  # CVaR is the mean of the tail beyond VaR (worse)
    assert res.worst_return <= res.var_95
    assert np.isfinite([res.prob_profit, res.var_95, res.cvar_95, res.mean_return]).all()


def test_positive_drift_has_median_above_start_for_gbm():
    # strong positive drift, small vol → median final equity should exceed the start
    res = run_monte_carlo(
        np.full(300, 0.002), MonteCarloConfig(n_paths=2000, method="gbm", seed=5)
    )
    assert res.final_percentiles["p50"] > res.initial_equity
    assert res.prob_profit > 0.5


def test_negative_drift_produces_losses():
    res = run_monte_carlo(
        np.full(300, -0.003), MonteCarloConfig(n_paths=2000, method="gbm", seed=6)
    )
    assert res.final_percentiles["p50"] < res.initial_equity
    assert res.prob_profit < 0.5


def test_zero_vol_gives_deterministic_paths():
    res = run_monte_carlo(np.full(252, 0.001), MonteCarloConfig(n_paths=50, method="gbm", seed=8))
    # no randomness → every percentile is identical
    assert res.bands["p5"] == res.bands["p95"]
    assert res.prob_profit == pytest.approx(1.0)


# ── metric confidence intervals ────────────────────────────────────────


def test_metric_cis_present_and_ordered():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=1500, method="block_bootstrap", seed=9))
    for key in ("total_return", "annualized_return", "sharpe", "sortino", "max_drawdown"):
        assert key in res.metrics_mean
        assert key in res.metrics_ci
        lo, hi = res.metrics_ci[key]
        assert lo <= hi, f"{key}: ci not ordered ({lo}, {hi})"
        assert np.isfinite([lo, hi, res.metrics_mean[key]]).all()


def test_max_drawdown_ci_is_non_negative():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=800, method="bootstrap", seed=10))
    lo, hi = res.metrics_ci["max_drawdown"]
    assert lo >= 0.0 and hi >= 0.0
    assert hi <= 1.0 + 1e-9


def test_ci_is_wider_for_noisier_input():
    calm = run_monte_carlo(_returns(300, 0.0, 0.005), MonteCarloConfig(n_paths=1500, seed=11))
    wild = run_monte_carlo(_returns(300, 0.0, 0.05), MonteCarloConfig(n_paths=1500, seed=11))
    calm_w = calm.metrics_ci["total_return"][1] - calm.metrics_ci["total_return"][0]
    wild_w = wild.metrics_ci["total_return"][1] - wild.metrics_ci["total_return"][0]
    assert wild_w > calm_w


# ── histogram ──────────────────────────────────────────────────────────


def test_histogram_shape_and_invariants():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=2000, histogram_bins=30))
    h = res.histogram
    assert len(h.counts) == 30
    assert len(h.centers) == 30
    assert len(h.bin_edges) == 31
    assert sum(h.counts) == 2000
    assert h.centers == tuple(sorted(h.centers))
    assert h.bin_edges[0] <= res.worst_return
    assert h.bin_edges[-1] >= res.best_return


def test_histogram_bin_edges_are_monotonic():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=1000, histogram_bins=25))
    e = res.histogram.bin_edges
    assert all(e[i] <= e[i + 1] for i in range(len(e) - 1))


def test_histogram_center_matches_edge_midpoint():
    res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=1000, histogram_bins=15))
    h = res.histogram
    for i, c in enumerate(h.centers):
        assert c == pytest.approx((h.bin_edges[i] + h.bin_edges[i + 1]) / 2.0)


# ── equity wrapper ─────────────────────────────────────────────────────


def test_from_equity_derives_returns_then_simulates():
    equity = np.concatenate([[100_000.0], 100_000.0 * np.cumprod(1.0 + _returns(200, seed=12))])
    res = run_monte_carlo_from_equity(equity, MonteCarloConfig(n_paths=500, method="bootstrap"))
    assert res.n_steps == 200
    assert res.initial_equity == pytest.approx(100_000.0)
    assert len(res.bands["p50"]) == 201


def test_from_equity_returns_match_manual_returns():
    equity = np.array([1.0, 1.1, 1.21, 1.331])
    r = returns_from_equity(equity)
    res = run_monte_carlo_from_equity(equity, MonteCarloConfig(n_paths=50, method="bootstrap", seed=13))
    assert res.n_steps == len(r)
