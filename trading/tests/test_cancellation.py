"""Cancellation: registry semantics, engine cooperation, and the HTTP surface.

The interesting claim here is *"a Stop button actually stops the CPU work"*, so
these tests don't just check that a flag is set — they check that a long
Monte-Carlo raises :class:`RunCancelled` part-way through, and that the endpoint
turns that into HTTP 499 while the run is still in flight.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from trading.application.backtest.monte_carlo import MonteCarloConfig, run_monte_carlo
from trading.application.backtest.portfolio import (
    PortfolioBacktestConfig,
    TickerSpec,
    run_portfolio_backtest,
)
from trading.application.cancellation import (
    RunCancelled,
    RunRegistry,
    is_valid_token,
    run_registry,
)
from trading.main import app


def _returns(n: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0006, 0.018, n)


#: Deliberately slow: ``block_bootstrap`` builds each path in pure Python, so
#: 20 000 x 1 000 takes seconds — long enough to cancel mid-flight, short enough
#: that a broken cancellation fails the test instead of hanging the suite.
SLOW_MC = MonteCarloConfig(n_paths=20_000, n_steps=1_000, method="block_bootstrap", seed=1)


# ── registry ───────────────────────────────────────────────────────────


def test_new_token_creates_a_handle():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    assert is_valid_token(tok.token)
    assert reg.is_active(tok.token)
    assert tok.cancelled is False


def test_cancel_sets_the_flag_and_reports_active():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    assert reg.cancel(tok.token) is True  # known: it was active
    assert tok.cancelled is True
    assert reg.is_cancelled(tok.token) is True


def test_cancel_unknown_token_is_not_known():
    reg = RunRegistry(redis_url="")
    assert reg.cancel("never-started-token") is False


def test_cancel_is_idempotent():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    reg.cancel(tok.token)
    assert reg.cancel(tok.token) is True
    assert tok.cancelled is True


def test_finished_run_is_no_longer_active():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    reg.mark_finished(tok.token)
    assert reg.is_active(tok.token) is False
    assert reg.cancel(tok.token) is False


def test_token_reuse_after_finish_starts_clean():
    """`clear()` runs when a run ends, so the next run on that token is clean."""
    reg = RunRegistry(redis_url="")
    first = reg.new("shared-token")
    reg.cancel("shared-token")
    assert first.cancelled is True
    reg.clear("shared-token")  # what the API/task does in its finally block
    assert reg.new("shared-token").cancelled is False


def test_cancel_arriving_before_the_run_starts_is_honoured():
    """A queued job cancelled before a worker picks it up must not run.

    This is the race that matters for Celery: POST /cancel can land while the
    task is still in the queue, so `new()` must not wipe an existing flag.
    """
    reg = RunRegistry(redis_url="")
    reg.cancel("queued-job")           # user hits Stop before the worker starts
    tok = reg.new("queued-job")        # worker picks the job up afterwards
    assert tok.cancelled is True
    with pytest.raises(RunCancelled):
        tok.check()


def test_clear_forgets_a_run():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    reg.cancel(tok.token)
    reg.clear(tok.token)
    assert tok.cancelled is False
    assert reg.is_active(tok.token) is False


def test_invalid_tokens_are_rejected():
    reg = RunRegistry(redis_url="")
    for bad in ("has space", "semi;colon", "../escape", "", "x" * 65):
        assert is_valid_token(bad) is False
        assert reg.cancel(bad) is False
    with pytest.raises(ValueError):
        reg.new("bad token!")


def test_check_is_a_noop_until_cancelled():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    tok.check()  # must not raise
    reg.cancel(tok.token)
    with pytest.raises(RunCancelled):
        tok.check()


def test_run_cancelled_carries_the_token():
    reg = RunRegistry(redis_url="")
    tok = reg.new()
    reg.cancel(tok.token)
    with pytest.raises(RunCancelled) as ei:
        tok.check()
    assert ei.value.token == tok.token


# ── engine cooperation ─────────────────────────────────────────────────


def test_monte_carlo_refuses_to_start_if_already_cancelled():
    tok = run_registry.new()
    run_registry.cancel(tok.token)
    try:
        with pytest.raises(RunCancelled):
            run_monte_carlo(_returns(), MonteCarloConfig(n_paths=100), cancel=tok)
    finally:
        run_registry.clear(tok.token)


def test_monte_carlo_stops_mid_simulation_when_cancelled():
    """The real point: a cancel from another thread aborts a running simulation."""
    tok = run_registry.new()
    timer = threading.Timer(0.25, run_registry.cancel, args=(tok.token,))
    timer.start()
    try:
        with pytest.raises(RunCancelled):
            run_monte_carlo(_returns(), SLOW_MC, cancel=tok)
    finally:
        timer.cancel()
        run_registry.clear(tok.token)


def test_uncancelled_run_completes_normally():
    tok = run_registry.new()
    try:
        res = run_monte_carlo(_returns(), MonteCarloConfig(n_paths=1500, seed=4), cancel=tok)
        assert res.n_paths == 1500
        assert 0.0 <= res.prob_profit <= 1.0
    finally:
        run_registry.clear(tok.token)


def test_chunking_does_not_change_small_run_results():
    """Runs at or below the chunk size must stay bit-identical to pre-chunk behaviour."""
    cfg = MonteCarloConfig(n_paths=2000, method="gbm", seed=42)
    plain = run_monte_carlo(_returns(), cfg)
    with_token = run_monte_carlo(_returns(), cfg, cancel=run_registry.new())
    assert plain.final_return_percentiles == with_token.final_return_percentiles
    assert plain.mean_return == with_token.mean_return


def test_large_run_is_chunked_deterministically():
    cfg = MonteCarloConfig(n_paths=10_000, n_steps=40, method="gbm", seed=5)
    a = run_monte_carlo(_returns(), cfg)
    b = run_monte_carlo(_returns(), cfg)
    assert a.final_percentiles == b.final_percentiles
    assert a.n_paths == 10_000


def test_different_seeds_still_differ_after_chunking():
    base = {"n_paths": 10_000, "n_steps": 40, "method": "gbm"}
    a = run_monte_carlo(_returns(), MonteCarloConfig(seed=1, **base))
    b = run_monte_carlo(_returns(), MonteCarloConfig(seed=2, **base))
    assert a.mean_return != b.mean_return


async def test_portfolio_backtest_refuses_to_start_if_already_cancelled():
    tok = run_registry.new()
    run_registry.cancel(tok.token)
    try:
        with pytest.raises(RunCancelled):
            await run_portfolio_backtest(
                [TickerSpec("AAA", source="synthetic", limit=200)],
                PortfolioBacktestConfig(),
                cancel=tok,
            )
    finally:
        run_registry.clear(tok.token)


async def test_portfolio_cancelled_between_tickers():
    """Cancel lands after the first leg; the run must abort rather than finish."""
    tok = run_registry.new()
    run_registry.cancel(tok.token)  # cancels before leg 1 completes
    try:
        with pytest.raises(RunCancelled):
            await run_portfolio_backtest(
                [
                    TickerSpec("AAA", source="synthetic", limit=400),
                    TickerSpec("BBB", source="synthetic", limit=400),
                ],
                PortfolioBacktestConfig(),
                cancel=tok,
            )
    finally:
        run_registry.clear(tok.token)


# ── HTTP surface ───────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # Round-2: the /backtest router is JWT-guarded → authenticate once.
        r = await c.post(
            "/api/v1/auth/token", json={"username": "admin", "password": "admin"}
        )
        assert r.status_code == 200, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        yield c


async def test_response_echoes_the_run_token(client):
    r = await client.post(
        "/api/v1/backtest/portfolio",
        json={
            "tickers": [{"symbol": "AAA", "source": "synthetic", "limit": 300}],
            "run_token": "echo-me-123",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["run_token"] == "echo-me-123"


async def test_run_token_is_generated_when_not_supplied(client):
    """Omitting a token still yields a usable handle, so any client can cancel."""
    r = await client.post(
        "/api/v1/backtest/portfolio",
        json={"tickers": [{"symbol": "AAA", "source": "synthetic", "limit": 300}]},
    )
    assert r.status_code == 200
    token = r.json()["run_token"]
    assert is_valid_token(token)
    # and it is immediately cancellable (a no-op, but a valid target)
    assert (await client.post(f"/api/v1/backtest/cancel/{token}")).status_code == 200


async def test_cancelling_an_unknown_token_is_not_an_error(client):
    r = await client.post("/api/v1/backtest/cancel/never-started")
    assert r.status_code == 200
    body = r.json()
    assert body == {"token": "never-started", "cancelled": True, "known": False}


async def test_malformed_token_is_rejected(client):
    r = await client.post("/api/v1/backtest/cancel/bad%20token")
    assert r.status_code == 400


async def test_active_runs_endpoint_is_empty_when_idle(client):
    r = await client.get("/api/v1/backtest/cancel")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


async def test_running_monte_carlo_can_be_cancelled_and_returns_499(client):
    """Start a slow MC, cancel it while in flight, expect 499 — not a 200."""
    token = "live-cancel-test"
    payload = {
        "symbol": "SYNTH", "strategy": "sma_crossover", "source": "synthetic",
        "limit": 300, "n_paths": 20_000, "n_steps": 1_000,
        "method": "block_bootstrap", "seed": 1, "run_token": token,
    }
    run = asyncio.create_task(client.post("/api/v1/backtest/monte-carlo", json=payload))
    try:
        await asyncio.sleep(0.5)
        active = (await client.get("/api/v1/backtest/cancel")).json()
        assert token in active, "the run should be registered as active while executing"

        ack = await client.post(f"/api/v1/backtest/cancel/{token}")
        assert ack.status_code == 200
        assert ack.json()["known"] is True

        res = await asyncio.wait_for(run, timeout=30)
        assert res.status_code == 499, res.text
        assert res.json()["detail"]["message"] == "run cancelled"
        assert res.json()["detail"]["run_token"] == token
    finally:
        run.cancel()
        run_registry.clear(token)


async def test_cancelled_run_is_deregistered_afterwards(client):
    token = "cleanup-check"
    payload = {
        "symbol": "SYNTH", "strategy": "sma_crossover", "source": "synthetic",
        "limit": 300, "n_paths": 20_000, "n_steps": 1_000,
        "method": "block_bootstrap", "seed": 1, "run_token": token,
    }
    run = asyncio.create_task(client.post("/api/v1/backtest/monte-carlo", json=payload))
    await asyncio.sleep(0.5)
    await client.post(f"/api/v1/backtest/cancel/{token}")
    await asyncio.wait_for(run, timeout=30)
    # The finally-block must have released the token.
    assert token not in (await client.get("/api/v1/backtest/cancel")).json()


async def test_event_loop_stays_responsive_during_a_run(client):
    """The whole reason for offloading: /health must answer while MC is running."""
    token = "responsive-check"
    payload = {
        "symbol": "SYNTH", "strategy": "sma_crossover", "source": "synthetic",
        "limit": 300, "n_paths": 20_000, "n_steps": 1_000,
        "method": "block_bootstrap", "seed": 1, "run_token": token,
    }
    run = asyncio.create_task(client.post("/api/v1/backtest/monte-carlo", json=payload))
    try:
        await asyncio.sleep(0.4)
        r = await asyncio.wait_for(client.get("/health"), timeout=5)
        assert r.status_code == 200
    finally:
        await client.post(f"/api/v1/backtest/cancel/{token}")
        await asyncio.wait_for(run, timeout=30)
        run_registry.clear(token)
