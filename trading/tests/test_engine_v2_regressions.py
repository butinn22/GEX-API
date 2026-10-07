"""Backtest-engine v2 regression tests (correctness fixes that change results).

See ``ENGINE_VERSION`` in ``trading.application.backtest.engine`` — 2.0.0 fixed:
exit sizing, block-bootstrap drift, the GBM Itô correction, the Sortino
semi-deviation and the optimizer's treatment of a perfect (``+inf``) objective.
The tests that only exercise one module live next to it; these cover the
cross-module contracts.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from trading.application.backtest.engine import ENGINE_VERSION
from trading.application.backtest.monte_carlo import MonteCarloConfig, _simulate_block
from trading.application.backtest.optimize import metric_value


def test_engine_version_is_present_and_forward():
    assert isinstance(ENGINE_VERSION, str)
    major = int(ENGINE_VERSION.split(".")[0])
    assert major >= 2


def test_gbm_drift_matches_the_historical_log_mean():
    """The GBM caller must not subtract the Itô correction twice.

    ``_simulate_block`` is handed mean *log* returns; the simulator applies
    ``mu - 0.5*sigma**2`` internally, so the caller has to invert that. Before
    the fix the simulated drift was low by ``0.5*sigma**2`` per step.
    """
    rng = np.random.default_rng(11)
    returns = rng.normal(0.001, 0.02, 1000)
    n_steps = 50
    cfg = MonteCarloConfig(n_paths=20_000, method="gbm", seed=3)
    paths = _simulate_block(returns, cfg, n_steps=n_steps, n_paths=20_000, seed=3)
    total_log = np.log(paths[:, -1] / paths[:, 0])
    expected = float(np.log1p(returns).mean()) * n_steps
    # The old double correction misses by 0.5*sigma**2*n_steps = 0.01, ~10 SE.
    assert total_log.mean() == pytest.approx(expected, abs=0.003)


def test_optimizer_metric_value_treats_infinite_objective_as_best():
    m = types.SimpleNamespace(
        profit_factor=float("inf"),
        sortino=float("-inf"),
        calmar=float("nan"),
        total_return=1.0,
        max_drawdown=0.0,
    )
    assert metric_value(m, "profit_factor") > 1e17      # perfect split = best
    assert metric_value(m, "sortino") < -1e17           # -inf = worst
    assert metric_value(m, "calmar") < -1e17            # nan = unusable
    assert metric_value(m, "max_drawdown") == 0.0
