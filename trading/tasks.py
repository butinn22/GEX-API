"""Celery app + task definitions + beat schedule.

Long-running trading work (backtests, Monte-Carlo, bulk data fetching,
reconciliation) runs on Celery so the API never blocks and external rate limits
are respected. The broker is Redis (the same instance the rate limiter uses);
results go to a separate Redis DB.

Reliability (see the Celery quality checklist)
----------------------------------------------
* ``task_acks_late`` + ``reject_on_worker_lost`` → a task is re-queued if a worker
  dies mid-flight, so a long Monte-Carlo is never silently lost.
* ``autoretry_for=(DataFetchError,)`` + ``retry_backoff``/``retry_jitter`` →
  transient upstream failures are retried with exponential backoff instead of
  hammering an exchange (protects API access).
* Hard ``time_limit`` + ``soft_time_limit`` → a runaway simulation cannot pin a
  worker forever.
* Tasks are idempotent and pure (same inputs → same outputs; ``seed`` is explicit).

Test with ``task_always_eager=True`` (no broker needed).
"""
from __future__ import annotations

import asyncio
from typing import Any

from celery import Celery
from celery.schedules import crontab

from trading.config import settings
from trading.application.cancellation import RunCancelled, run_registry
from trading.domain import DataFetchError

__all__ = [
    "celery_app",
    "run_backtest_task",
    "run_monte_carlo_task",
    "run_portfolio_backtest_task",
    "fetch_market_data_task",
    "reconcile_positions_task",
    "cleanup_old_backtests_task",
    "broker_health_task",
]

celery_app = Celery(
    "gex_trading",
    broker=settings.redis_url,
    backend=settings.redis_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    broker_transport_options={"visibility_timeout": 3600},
)

celery_app.conf.beat_schedule = {
    "reconcile-positions-every-5-min": {
        "task": "trading.tasks.reconcile_positions_task",
        "schedule": crontab(minute="*/5"),
    },
    "fetch-market-data-hourly": {
        "task": "trading.tasks.fetch_market_data_task",
        "schedule": crontab(minute="0"),
    },
    "cleanup-backtests-daily": {
        "task": "trading.tasks.cleanup_old_backtests_task",
        "schedule": crontab(hour=3, minute=0),
    },
    "broker-health-every-5-min": {
        "task": "trading.tasks.broker_health_task",
        "schedule": crontab(minute="*/5"),
    },
}


# ── helpers ────────────────────────────────────────────────────────────


def _mc_payload(mc) -> dict[str, Any]:
    """Flatten a :class:`MonteCarloResult` into a JSON-safe dict (summary)."""
    return {
        "method": mc.method,
        "n_paths": mc.n_paths,
        "n_steps": mc.n_steps,
        "mean": mc.mean_return,
        "p5": mc.final_return_percentiles["p5"],
        "p95": mc.final_return_percentiles["p95"],
        "median": mc.final_return_percentiles["p50"],
        "prob_profit": mc.prob_profit,
        "var_95": mc.var_95,
        "cvar_95": mc.cvar_95,
        "best_return": mc.best_return,
        "worst_return": mc.worst_return,
        "metrics_mean": mc.metrics_mean,
        "metrics_ci": {k: [lo, hi] for k, (lo, hi) in mc.metrics_ci.items()},
        "bands": {k: list(v) for k, v in mc.bands.items()},
        "steps": list(mc.steps),
        "final_percentiles": mc.final_percentiles,
        "histogram": {
            "counts": list(mc.histogram.counts),
            "centers": list(mc.histogram.centers),
            "bin_edges": list(mc.histogram.bin_edges),
        },
    }


def _mc_config(opts: dict[str, Any], *, initial_equity: float, ppy: int):
    from trading.application.backtest.monte_carlo import MonteCarloConfig

    return MonteCarloConfig(
        n_paths=int(opts.get("n_paths", 10_000)),
        n_steps=opts.get("n_steps"),
        method=opts.get("method", "gbm"),
        block_size=int(opts.get("block_size", 5)),
        seed=int(opts.get("seed", 0)),
        initial_equity=initial_equity,
        periods_per_year=ppy,
    )


def _cancel_token(run_token: str | None):
    """Open a cancellable run, or ``None`` when the caller didn't ask for one.

    The API worker and the Celery worker share Redis, so a
    ``POST /backtest/cancel/{token}`` served by the web process is observed here
    by the engine's block-level check.
    """
    return run_registry.new(run_token) if run_token else None


def _cancelled_payload(token, **extra: Any) -> dict[str, Any]:
    return {"cancelled": True, "run_token": token.token if token else "", **extra}


def _finish(token) -> None:
    """Release a run token: drops the flag so a reused token starts clean."""
    if token is not None:
        run_registry.clear(token.token)


# ── tasks ──────────────────────────────────────────────────────────────


@celery_app.task(name="trading.tasks.run_backtest_task", bind=True)
def run_backtest_task(self, strategy: str, symbol: str, bars: list | None = None,
                      initial_cash: float = 100_000.0, **kwargs) -> dict:
    """Run a single-symbol backtest (async) and return the metrics summary."""
    from trading.application.backtest.engine import BacktestConfig, run_backtest
    from trading.application.backtest.portfolio import TickerSpec, load_bars
    from trading.application.strategy_factory import build_strategy
    from trading.domain import Bar

    async def _run() -> dict:
        if bars:
            data = [Bar(**b) for b in bars]
        else:
            data = await load_bars(
                TickerSpec(symbol=symbol, source="synthetic", limit=300)
            )
        params = {k: v for k, v in kwargs.items() if v is not None}
        strat = build_strategy(strategy, symbol, params)
        cfg = BacktestConfig(initial_cash=initial_cash)
        result = await run_backtest(strat, data, cfg)
        m = result.metrics
        return {
            "strategy": strategy, "symbol": symbol, "n_trades": m.n_trades,
            "total_return": m.total_return, "sharpe": m.sharpe,
            "max_drawdown": m.max_drawdown, "win_rate": m.win_rate,
            "profit_factor": m.profit_factor if m.profit_factor != float("inf") else None,
            "equity_curve": result.equity_curve.tolist(),
        }

    return asyncio.run(_run())


@celery_app.task(
    name="trading.tasks.run_monte_carlo_task",
    autoretry_for=(DataFetchError,), retry_backoff=True, retry_jitter=True, max_retries=3,
    soft_time_limit=settings.monte_carlo_time_limit,
    time_limit=settings.monte_carlo_time_limit + 300,
)
def run_monte_carlo_task(symbol: str = "SYNTH", strategy: str = "sma_crossover",
                         params: dict | None = None, source: str = "auto",
                         timeframe: str = "1d", limit: int = 1000,
                         initial_cash: float = 100_000.0, n_paths: int = 10_000,
                         n_steps: int | None = None, method: str = "gbm",
                         block_size: int = 5, seed: int = 0,
                         run_token: str | None = None) -> dict:
    """Backtest the strategy, then Monte-Carlo its return distribution.

    Pass ``run_token`` to make the job cancellable from the API; on cancellation
    the task returns ``{"cancelled": True, ...}`` instead of a result payload.
    """
    from trading.application.backtest.engine import BacktestConfig, run_backtest
    from trading.application.backtest.monte_carlo import run_monte_carlo_from_equity
    from trading.application.backtest.portfolio import TickerSpec, load_bars
    from trading.application.strategy_factory import build_strategy

    token = _cancel_token(run_token)

    async def _run() -> dict:
        bars = await load_bars(
            TickerSpec(symbol=symbol, source=source, timeframe=timeframe, limit=limit)
        )
        strat = build_strategy(strategy, symbol, params or {})
        result = await run_backtest(strat, bars, BacktestConfig(initial_cash=initial_cash))
        mc = run_monte_carlo_from_equity(
            result.equity_curve,
            _mc_config(
                {"n_paths": n_paths, "n_steps": n_steps, "method": method,
                 "block_size": block_size, "seed": seed},
                initial_equity=initial_cash, ppy=252,
            ),
            cancel=token,
        )
        payload = _mc_payload(mc)
        payload.update({"symbol": symbol, "strategy": strategy, "n_trades": result.metrics.n_trades})
        return payload

    try:
        return asyncio.run(_run())
    except RunCancelled:
        return _cancelled_payload(token, symbol=symbol, strategy=strategy)
    finally:
        _finish(token)


@celery_app.task(
    name="trading.tasks.run_portfolio_backtest_task",
    autoretry_for=(DataFetchError,), retry_backoff=True, retry_jitter=True, max_retries=3,
    soft_time_limit=settings.monte_carlo_time_limit,
    time_limit=settings.monte_carlo_time_limit + 300,
)
def run_portfolio_backtest_task(tickers: list[dict], initial_cash: float = 100_000.0,
                                fee_rate: float = 0.001, slippage: float = 0.0005,
                                periods_per_year: int = 252,
                                monte_carlo: dict | None = None,
                                run_token: str | None = None) -> dict:
    """Backtest a multi-ticker basket (each ticker with its own strategy/settings).

    ``tickers`` is a list of ``{symbol, strategy, params, weight, capital, source,
    timeframe, limit, enabled}`` — the same shape the REST layer accepts, so the
    API can queue exactly what a client submitted. ``run_token`` makes the job
    cancellable from the API.
    """
    from trading.application.backtest.monte_carlo import run_monte_carlo_from_equity
    from trading.application.backtest.portfolio import (
        PortfolioBacktestConfig,
        TickerSpec,
        run_portfolio_backtest,
    )

    token = _cancel_token(run_token)

    async def _run() -> dict:
        specs = [
            TickerSpec(
                symbol=t["symbol"],
                strategy=t.get("strategy", "sma_crossover"),
                params=t.get("params") or {},
                weight=float(t.get("weight", 1.0)),
                capital=t.get("capital"),
                source=t.get("source", "auto"),
                timeframe=t.get("timeframe", "1d"),
                limit=int(t.get("limit", 5000)),
                enabled=bool(t.get("enabled", True)),
            )
            for t in tickers
        ]
        cfg = PortfolioBacktestConfig(
            initial_cash=initial_cash, fee_rate=fee_rate, slippage=slippage,
            periods_per_year=periods_per_year,
        )
        result = await run_portfolio_backtest(specs, cfg, cancel=token)
        m = result.metrics
        payload: dict[str, Any] = {
            "n_tickers": result.n_tickers,
            "initial_cash": result.initial_cash,
            "final_equity": float(result.equity_curve[-1]),
            "total_return": m.total_return, "sharpe": m.sharpe,
            "max_drawdown": m.max_drawdown, "win_rate": m.win_rate,
            "tickers": [
                {"symbol": t.symbol, "strategy": t.strategy, "weight": t.weight,
                 "capital": t.capital, "total_return": t.total_return,
                 "n_trades": len(t.result.trades)}
                for t in result.tickers
            ],
            "errors": list(result.errors),
            "equity_curve": result.equity_curve.tolist(),
        }
        if monte_carlo:
            payload["monte_carlo"] = _mc_payload(
                run_monte_carlo_from_equity(
                    result.equity_curve,
                    _mc_config(monte_carlo, initial_equity=result.initial_cash,
                               ppy=periods_per_year),
                    cancel=token,
                )
            )
        return payload

    try:
        return asyncio.run(_run())
    except RunCancelled:
        return _cancelled_payload(token, n_tickers=len(tickers))
    finally:
        _finish(token)


@celery_app.task(name="trading.tasks.fetch_market_data_task")
def fetch_market_data_task(symbol: str, timeframe: str = "1d", limit: int = 500,
                           source: str = "auto") -> dict:
    """Bulk (throttled) market-data fetch via the real fetcher registry."""
    from trading.application.backtest.portfolio import TickerSpec, load_bars

    async def _run() -> dict:
        bars = await load_bars(
            TickerSpec(symbol=symbol, source=source, timeframe=timeframe, limit=limit)
        )
        return {"symbol": symbol, "bars": len(bars),
                "first": bars[0].timestamp.isoformat(), "last": bars[-1].timestamp.isoformat()}

    return asyncio.run(_run())


@celery_app.task(name="trading.tasks.reconcile_positions_task")
def reconcile_positions_task() -> dict:
    """Periodic position reconciliation (sync local DB with broker)."""
    # Placeholder until the position-keeper persists to Postgres; returns an
    # empty report so the schedule is exercised without side effects.
    return {"reconciled": 0, "note": "position persistence not wired yet"}


@celery_app.task(name="trading.tasks.cleanup_old_backtests_task")
def cleanup_old_backtests_task(days: int = 30) -> dict:
    """Delete backtest results older than ``days`` (best-effort)."""
    return {"deleted": 0, "note": "backtest result store not wired yet"}


@celery_app.task(name="trading.tasks.broker_health_task")
def broker_health_task() -> dict:
    """Ping every configured broker; reflect reachability into ``BROKER_STATUS``.

    Without this beat entry the gauge is never set and the ``TradingBrokerDown``
    alert can never fire. The task builds one broker per configured exchange and
    runs :class:`BrokerHealthMonitor` over them.
    """
    from trading.adapters.brokers.bingx import BingxBroker, BingxClient
    from trading.adapters.brokers.tbank import TbankBroker
    from trading.adapters.persistence import database as db
    from trading.application.broker_health import BrokerHealthMonitor
    from trading.application.keys_service import KeysService
    from trading.domain import Exchange

    async def _run() -> dict:
        db.configure()
        svc = KeysService(settings.encryption_secret)
        monitor = BrokerHealthMonitor()
        results: dict[str, bool] = {}
        async with db.session_factory()() as session:
            seen: set[str] = set()
            for row in await svc.list_keys(session):
                if row.exchange in seen:
                    continue
                seen.add(row.exchange)
                creds = await svc.resolve_credentials(session, row.exchange)
                if creds is None:
                    continue
                if row.exchange == "bingx":
                    broker = BingxBroker(BingxClient(
                        creds["api_key"], creds["api_secret"],
                        base_url=settings.bingx_base_url,
                    ))
                    exchange = Exchange.BINGX
                elif row.exchange == "tbank":
                    broker = TbankBroker(
                        creds["api_key"], creds["extra"].get("account_id", ""),
                        sandbox=settings.tbank_sandbox,
                    )
                    exchange = Exchange.TBANK
                else:
                    continue
                results[row.exchange] = await monitor.check(exchange, broker)
        return {"checked": results}

    return asyncio.run(_run())
