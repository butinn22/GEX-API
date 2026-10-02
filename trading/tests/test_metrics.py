"""Tests for backtest metrics and bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pytest

from trading.application.backtest.metrics import (
    bootstrap_equity_ci,
    bootstrap_trade_ci,
    compute_metrics,
    conditional_var,
    max_drawdown,
    profit_factor,
    returns_from_equity,
    total_return,
    value_at_risk,
    win_rate,
)


def test_total_return_and_drawdown():
    eq = np.array([100.0, 110.0, 99.0, 118.8])
    assert total_return(eq) == pytest.approx(0.188)
    assert max_drawdown(eq) == pytest.approx(0.1)  # (110-99)/110


def test_var_and_cvar():
    r = np.array([0.1, -0.1, 0.2])
    assert value_at_risk(r, 0.05) == pytest.approx(-0.08)
    assert conditional_var(r, 0.05) == pytest.approx(-0.1)  # only -0.1 <= -0.08


def test_win_rate_and_profit_factor():
    pnls = [50.0, -20.0, 30.0]
    assert win_rate(pnls) == pytest.approx(2 / 3)
    assert profit_factor(pnls) == pytest.approx((50 + 30) / 20)


def test_profit_factor_no_losses():
    assert profit_factor([10.0, 5.0]) == float("inf")
    assert profit_factor([0.0, 0.0]) == 0.0


def test_sharpe_and_sortino_match_reference():
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    exp_sharpe = r.mean() / r.std(ddof=1) * np.sqrt(1)
    downside = r[r < 0]
    exp_sortino = r.mean() / np.sqrt(np.mean(downside**2)) * np.sqrt(1)
    m = compute_metrics(np.concatenate([[100.0], 100.0 * np.cumprod(1 + r)]),
                        [1.0, -2.0], periods_per_year=1)
    assert m.sharpe == pytest.approx(exp_sharpe)
    assert m.sortino == pytest.approx(exp_sortino)


def test_returns_from_equity():
    assert np.allclose(returns_from_equity(np.array([100.0, 110.0, 99.0])),
                       [0.1, -0.1])


def test_compute_metrics_shape():
    eq = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    m = compute_metrics(eq, [1.0, -0.5], periods_per_year=252)
    assert m.n_periods == 4
    assert m.n_trades == 2
    assert np.isfinite(m.sharpe)
    assert np.isfinite(m.max_drawdown)


def test_bootstrap_equity_ci_bounds():
    rng = np.random.default_rng(42)
    eq = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, 300))
    eq = np.concatenate([[100.0], eq])
    lo, hi = bootstrap_equity_ci(total_return, eq, n_boot=200, block_size=5, seed=0)
    assert lo <= hi
    assert np.isfinite(lo) and np.isfinite(hi)


def test_bootstrap_ci_seed_reproducible():
    eq = np.linspace(100.0, 120.0, 100)
    a = bootstrap_equity_ci(total_return, eq, n_boot=100, seed=7)
    b = bootstrap_equity_ci(total_return, eq, n_boot=100, seed=7)
    assert a == b


def test_bootstrap_trade_ci_bounds():
    lo, hi = bootstrap_trade_ci(win_rate, [10.0, -5.0, 3.0, -2.0, 8.0], n_boot=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
