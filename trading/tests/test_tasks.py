"""Tests for Celery tasks (eager mode — no broker needed)."""
from __future__ import annotations

from trading.application.cancellation import run_registry
from trading.tasks import (
    celery_app,
    cleanup_old_backtests_task,
    reconcile_positions_task,
    run_backtest_task,
    run_monte_carlo_task,
    run_portfolio_backtest_task,
)

celery_app.conf.task_always_eager = True


def test_run_backtest_task():
    result = run_backtest_task.delay("sma_crossover", "SYNTH", fast=5, slow=20)
    assert result.successful()
    data = result.get()
    assert data["strategy"] == "sma_crossover"
    assert "sharpe" in data and "max_drawdown" in data
    assert len(data["equity_curve"]) > 0


def test_run_monte_carlo_task():
    result = run_monte_carlo_task.delay("SYNTH", n_paths=500, n_steps=50)
    data = result.get()
    assert "mean" in data and "p5" in data and "p95" in data
    assert data["p5"] <= data["mean"] <= data["p95"]


def test_placeholder_reconcile_and_cleanup():
    assert reconcile_positions_task.delay().get()["reconciled"] == 0
    assert cleanup_old_backtests_task.delay().get()["deleted"] == 0


# ── cancellation on the worker side ────────────────────────────────────


def test_monte_carlo_task_is_cancellable_before_it_starts():
    """A job cancelled while still queued must not burn a worker for minutes."""
    run_registry.cancel("task-cancel-mc")
    data = run_monte_carlo_task.delay(
        "SYNTH", source="synthetic", n_paths=20_000, n_steps=1_000,
        method="block_bootstrap", run_token="task-cancel-mc",
    ).get()
    assert data.get("cancelled") is True
    assert data["run_token"] == "task-cancel-mc"


def test_portfolio_task_is_cancellable_before_it_starts():
    run_registry.cancel("task-cancel-pf")
    data = run_portfolio_backtest_task.delay(
        [{"symbol": "AAA", "source": "synthetic", "limit": 300}],
        run_token="task-cancel-pf",
    ).get()
    assert data.get("cancelled") is True
    assert data["n_tickers"] == 1


def test_task_token_is_released_after_use():
    """`clear()` in the finally-block keeps tokens reusable across runs."""
    run_registry.cancel("task-reuse")
    assert run_monte_carlo_task.delay(
        "SYNTH", source="synthetic", n_paths=20_000, n_steps=1_000,
        method="block_bootstrap", run_token="task-reuse",
    ).get()["cancelled"] is True

    again = run_monte_carlo_task.delay(
        "SYNTH", source="synthetic", n_paths=300, n_steps=40, run_token="task-reuse",
    ).get()
    assert not again.get("cancelled")
    assert again["n_paths"] == 300
    # The registry is a process-wide singleton shared with other tests, so only
    # assert about this token.
    assert "task-reuse" not in run_registry.active_tokens()
    assert run_registry.is_cancelled("task-reuse") is False


def test_uncancelled_task_returns_a_normal_payload():
    data = run_monte_carlo_task.delay(
        "SYNTH", source="synthetic", n_paths=300, n_steps=40,
        method="gbm", run_token="task-no-cancel",
    ).get()
    assert not data.get("cancelled")
    assert data["n_paths"] == 300
    assert 0.0 <= data["prob_profit"] <= 1.0
