"""Backtest endpoints.

* ``POST /backtest``                 — single-symbol event-driven backtest
* ``POST /backtest/monte-carlo``     — Monte-Carlo over a strategy's returns
* ``POST /backtest/portfolio``       — multi-ticker basket, **per-ticker settings**
* ``POST /backtest/portfolio/monte-carlo`` — basket + Monte-Carlo distribution
* ``POST /backtest/portfolio/report``      — self-contained HTML report with charts
* ``POST /backtest/cancel/{token}``  — stop a running backtest / Monte-Carlo

The heavy lifting lives in ``trading.application.backtest``; this module only
validates input, resolves tickers, and maps domain results to response models.

Long runs are cancellable and must not block the event loop
-----------------------------------------------------------
Monte-Carlo is pure CPU work. Running it inline in an ``async def`` handler
freezes the worker's event loop, which means the ``POST /backtest/cancel``
request *cannot even be received* until the run it is meant to stop has already
finished. So every long-running call is offloaded with :func:`asyncio.to_thread`
and handed a :class:`CancelToken` the engine polls between blocks of work. The
run then raises :class:`RunCancelled`, which this layer maps to **HTTP 499**
(nginx's "client closed request") so a client can tell "I stopped it" apart from
"I hit an error".
"""
from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.backtest.metrics import BacktestMetrics
from trading.application.backtest.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo,
    run_monte_carlo_from_equity,
)
from trading.application.backtest.portfolio import (
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    TickerSpec,
    load_bars,
    run_portfolio_backtest,
)
from trading.application.backtest.reporter import MonteCarloReporter, PortfolioReporter
from trading.application.backtest.optimize import optimize_strategy
from trading.application.backtest.trade_analysis import analyze_trades, recommend_adjustments
from trading.application.cancellation import (
    CancelToken,
    RunCancelled,
    is_valid_token,
    run_registry,
)
from trading.application.instruments import select_universe
from trading.application.strategy_factory import build_strategy
from trading.domain import Bar, DataFetchError, StrategyError
from trading.ports import Strategy

from ..schemas import (
    AnalyzeResponse,
    AutoTuneRequest,
    BacktestMetricsOut,
    BacktestRequest,
    BacktestResponse,
    CancelOut,
    CorrelationOut,
    HistogramOut,
    MonteCarloRequest,
    MonteCarloRunOptions,
    MonteCarloSummaryOut,
    OptimizeRequest,
    OptimizeResponse,
    PortfolioBacktestRequest,
    PortfolioBacktestResponse,
    PortfolioMonteCarloRequest,
    TickerConfig,
    TickerResultOut,
)

router = APIRouter(prefix="/backtest", tags=["backtest"])

#: Charts don't need every bar; cap payload size while keeping both endpoints.
MAX_POINTS = 1000

#: HTTP status for "the run was stopped by the client", not an error.
CANCELLED_STATUS = 499


# ── helpers ────────────────────────────────────────────────────────────


def _open_run(token: str | None) -> CancelToken:
    """Register a cancellable run and return its token."""
    return run_registry.new(token)


def _cancelled(exc: RunCancelled) -> HTTPException:
    return HTTPException(
        status_code=CANCELLED_STATUS,
        detail={"message": "run cancelled", "run_token": exc.token},
    )


async def _run_mc(
    equity_or_returns,
    cfg: MonteCarloConfig,
    *,
    cancel: CancelToken | None,
    from_equity: bool = True,
) -> MonteCarloResult:
    """Run Monte-Carlo off the event loop so the worker can still serve /cancel."""
    fn = run_monte_carlo_from_equity if from_equity else run_monte_carlo
    return await asyncio.to_thread(fn, equity_or_returns, cfg, cancel=cancel)


def _finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _metrics_out(m: BacktestMetrics) -> BacktestMetricsOut:
    return BacktestMetricsOut(
        total_return=m.total_return,
        annualized_return=m.annualized_return,
        sharpe=m.sharpe,
        sortino=_finite_or_none(m.sortino),
        calmar=_finite_or_none(m.calmar),
        max_drawdown=m.max_drawdown,
        var_95=m.var_95,
        cvar_95=m.cvar_95,
        win_rate=m.win_rate,
        profit_factor=_finite_or_none(m.profit_factor),
        n_trades=m.n_trades,
        n_periods=m.n_periods,
    )


def _sample_indices(n: int, cap: int = MAX_POINTS) -> list[int]:
    if n <= cap:
        return list(range(n))
    step = (n - 1) / (cap - 1)
    idx = sorted({int(round(i * step)) for i in range(cap)})
    idx[0], idx[-1] = 0, n - 1
    return idx


def _downsample(seq: list[float], idx: list[int]) -> list[float]:
    return [float(seq[i]) for i in idx]


def _spec_from_config(t: TickerConfig) -> TickerSpec:
    return TickerSpec(
        symbol=t.symbol,
        strategy=t.strategy,
        params=t.folded_params(),
        weight=t.weight,
        capital=t.capital,
        source=t.source,
        timeframe=t.timeframe,
        limit=t.limit,
        enabled=t.enabled,
    )


def _specs_from_request(req: PortfolioBacktestRequest) -> list[TickerSpec]:
    specs = [_spec_from_config(t) for t in req.tickers]
    if req.n_tickers > 0:
        try:
            picked = select_universe(req.category, req.n_tickers)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not picked:
            raise HTTPException(400, f"no instruments in category '{req.category}'")
        for inst in picked:
            specs.append(
                TickerSpec(
                    symbol=inst["symbol"],
                    strategy=req.default_strategy,
                    params=dict(req.default_params),
                    source=req.default_source,
                    timeframe=req.default_timeframe,
                    limit=req.default_limit,
                )
            )
    return specs


def _mc_config(opts: MonteCarloRunOptions, *, initial_equity: float, ppy: int) -> MonteCarloConfig:
    return MonteCarloConfig(
        n_paths=opts.n_paths,
        n_steps=opts.n_steps,
        method=opts.method,
        block_size=opts.block_size,
        seed=opts.seed,
        initial_equity=initial_equity,
        periods_per_year=ppy,
    )


def _mc_out(mc: MonteCarloResult, label: str = "", run_token: str = "") -> MonteCarloSummaryOut:
    return MonteCarloSummaryOut(
        label=label,
        run_token=run_token,
        method=mc.method,
        n_paths=mc.n_paths,
        n_steps=mc.n_steps,
        initial_equity=mc.initial_equity,
        steps=list(mc.steps),
        bands={k: list(v) for k, v in mc.bands.items()},
        mean_path=list(mc.mean_path),
        final_percentiles=mc.final_percentiles,
        final_return_percentiles=mc.final_return_percentiles,
        mean_return=mc.mean_return,
        mean=mc.mean_return,
        p5=mc.final_return_percentiles["p5"],
        p95=mc.final_return_percentiles["p95"],
        histogram=HistogramOut(
            counts=list(mc.histogram.counts),
            centers=list(mc.histogram.centers),
            bin_edges=list(mc.histogram.bin_edges),
        ),
        metrics_mean=mc.metrics_mean,
        metrics_ci={k: [lo, hi] for k, (lo, hi) in mc.metrics_ci.items()},
        prob_profit=mc.prob_profit,
        var_95=mc.var_95,
        cvar_95=mc.cvar_95,
        best_return=mc.best_return,
        worst_return=mc.worst_return,
    )


async def _bars_for(
    symbol: str, source: str, timeframe: str, limit: int, *, refresh: bool = False
) -> list[Bar]:
    try:
        bars = await load_bars(
            TickerSpec(symbol=symbol, source=source, timeframe=timeframe, limit=limit),
            refresh=refresh,
        )
    except (DataFetchError, StrategyError, ValueError) as exc:
        raise HTTPException(502, f"no data for '{symbol}' (source={source}): {exc}") from exc
    bars = sorted(bars, key=lambda b: b.timestamp)
    if len(bars) < 2:
        raise HTTPException(400, f"need at least 2 bars for '{symbol}', got {len(bars)}")
    return bars


def _portfolio_response(
    result: PortfolioBacktestResult,
    mc: MonteCarloResult | None,
    run_token: str = "",
) -> PortfolioBacktestResponse:
    idx = _sample_indices(len(result.times))
    return PortfolioBacktestResponse(
        n_tickers=result.n_tickers,
        initial_cash=result.initial_cash,
        final_equity=float(result.equity_curve[-1]),
        metrics=_metrics_out(result.metrics),
        equity_curve=_downsample(list(result.equity_curve), idx),
        times=[result.times[i].isoformat() for i in idx],
        run_token=run_token,
        tickers=[
            TickerResultOut(
                symbol=t.symbol,
                strategy=t.strategy,
                source=t.source,
                timeframe=t.timeframe,
                weight=t.weight,
                capital=t.capital,
                metrics=_metrics_out(t.metrics),
                equity_curve=[
                    float(result.aligned_equity[k][i])
                    for i in idx
                ] if k < len(result.aligned_equity) else _downsample(list(t.equity_curve), idx),
                n_trades=len(t.result.trades),
            )
            for k, t in enumerate(result.tickers)
        ],
        correlation=(
            CorrelationOut(symbols=list(result.correlation.symbols),
                           matrix=[list(r) for r in result.correlation.matrix])
            if result.correlation else None
        ),
        errors=list(result.errors),
        monte_carlo=_mc_out(mc, "portfolio", run_token) if mc is not None else None,
    )


async def _persist_result(request: BacktestRequest, result) -> int | None:
    """Store metrics + the granular trade-event ledger for later export.

    Persistence is best-effort: a database hiccup must not fail a completed
    backtest, so any error degrades to ``result_id=None``.
    """
    try:
        from trading.adapters.persistence.bulk import TaskResultStore
        from trading.adapters.persistence.database import _session_factory, init_db

        await init_db()
        async with _session_factory() as s:
            row = await TaskResultStore(s).save_backtest(
                request.strategy, request.symbol,
                {"total_return": result.metrics.total_return,
                 "sharpe": result.metrics.sharpe,
                 "max_drawdown": result.metrics.max_drawdown,
                 "win_rate": result.metrics.win_rate,
                 "n_trades": len(result.trades)},
                trades=[e.as_dict() for e in result.events],
            )
            return row.id
    except Exception:
        return None


# ── single-symbol backtest ─────────────────────────────────────────────


@router.post("", response_model=BacktestResponse)
async def run(request: BacktestRequest) -> BacktestResponse:
    if request.bars:
        bars = [
            Bar(timestamp=b.timestamp, open=b.open, high=b.high, low=b.low,
                close=b.close, volume=b.volume)
            for b in request.bars
        ]
    else:
        bars = await _bars_for(
            request.symbol, request.source, request.timeframe, request.limit,
            refresh=request.refresh_data,
        )

    bars = sorted(bars, key=lambda b: b.timestamp)
    if len(bars) < 2:
        raise HTTPException(400, "need at least 2 bars")

    params = dict(request.params or {})
    params.update({
        "fast": request.fast, "slow": request.slow, "period": request.period,
        "long": request.long.model_dump() if request.long else None,
        "short": request.short.model_dump() if request.short else None,
        "settings": request.settings,
    })
    try:
        strategy: Strategy = build_strategy(
            request.strategy, request.symbol, {k: v for k, v in params.items() if v is not None}
        )
    except StrategyError as exc:
        raise HTTPException(400, str(exc)) from exc

    cfg = BacktestConfig(
        initial_cash=request.initial_cash,
        fee_rate=request.fee_rate,
        slippage=request.slippage,
        position_fraction=request.position_fraction,
        periods_per_year=request.periods_per_year,
    )
    result = await run_backtest(strategy, bars, cfg)
    idx = _sample_indices(len(bars))
    result_id = await _persist_result(request, result)
    return BacktestResponse(
        strategy=request.strategy,
        symbol=request.symbol,
        metrics=_metrics_out(result.metrics),
        equity_curve=_downsample(list(result.equity_curve), idx),
        times=[bars[i].timestamp.isoformat() for i in idx],
        result_id=result_id,
        n_trades=len(result.trades),
    )


# ── Monte-Carlo (single symbol) ────────────────────────────────────────


@router.post("/monte-carlo", response_model=MonteCarloSummaryOut)
async def monte_carlo(request: MonteCarloRequest) -> MonteCarloSummaryOut:
    """Backtest the strategy, then simulate its return distribution N times."""
    token = _open_run(request.run_token)
    try:
        bars = await _bars_for(
            request.symbol, request.source, request.timeframe, request.limit,
            refresh=request.refresh_data,
        )
        try:
            strategy = build_strategy(request.strategy, request.symbol, request.folded_params())
        except StrategyError as exc:
            raise HTTPException(400, str(exc)) from exc
        cfg = BacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate, slippage=request.slippage,
            position_fraction=request.position_fraction, periods_per_year=request.periods_per_year,
        )
        result = await run_backtest(strategy, bars, cfg)
        mc = await _run_mc(
            result.equity_curve,
            MonteCarloConfig(
                n_paths=request.n_paths, n_steps=request.n_steps, method=request.method,
                block_size=request.block_size, seed=request.seed,
                initial_equity=request.initial_cash, periods_per_year=request.periods_per_year,
            ),
            cancel=token,
        )
        return _mc_out(mc, request.symbol, token.token)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


# ── portfolio backtest ─────────────────────────────────────────────────


@router.post("/portfolio", response_model=PortfolioBacktestResponse)
async def portfolio(request: PortfolioBacktestRequest) -> PortfolioBacktestResponse:
    """Backtest a basket; each ticker runs its own strategy and settings."""
    token = _open_run(request.run_token)
    try:
        specs = _specs_from_request(request)
        cfg = PortfolioBacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate, slippage=request.slippage,
            position_fraction=request.position_fraction, periods_per_year=request.periods_per_year,
        )
        try:
            result = await run_portfolio_backtest(specs, cfg, cancel=token,
                                                  refresh=request.refresh_data)
        except DataFetchError as exc:
            raise HTTPException(502, str(exc)) from exc
        mc = None
        if request.monte_carlo and request.monte_carlo.enabled:
            mc = await _run_mc(
                result.equity_curve,
                _mc_config(request.monte_carlo, initial_equity=result.initial_cash,
                           ppy=request.periods_per_year),
                cancel=token,
            )
        return _portfolio_response(result, mc, token.token)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


@router.post("/portfolio/monte-carlo", response_model=MonteCarloSummaryOut)
async def portfolio_monte_carlo(request: PortfolioMonteCarloRequest) -> MonteCarloSummaryOut:
    token = _open_run(request.run_token or request.portfolio.run_token)
    try:
        specs = _specs_from_request(request.portfolio)
        cfg = PortfolioBacktestConfig(
            initial_cash=request.portfolio.initial_cash, fee_rate=request.portfolio.fee_rate,
            slippage=request.portfolio.slippage, position_fraction=request.portfolio.position_fraction,
            periods_per_year=request.portfolio.periods_per_year,
        )
        try:
            result = await run_portfolio_backtest(
                specs, cfg, cancel=token, refresh=request.portfolio.refresh_data
            )
        except DataFetchError as exc:
            raise HTTPException(502, str(exc)) from exc
        mc = await _run_mc(
            result.equity_curve,
            _mc_config(request.monte_carlo, initial_equity=result.initial_cash,
                       ppy=request.portfolio.periods_per_year),
            cancel=token,
        )
        return _mc_out(mc, "portfolio", token.token)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


@router.post("/portfolio/report", response_class=HTMLResponse)
async def portfolio_report(request: PortfolioBacktestRequest) -> HTMLResponse:
    """Self-contained HTML report (equity, drawdown, correlation, per-ticker)."""
    token = _open_run(request.run_token)
    try:
        specs = _specs_from_request(request)
        cfg = PortfolioBacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate, slippage=request.slippage,
            position_fraction=request.position_fraction, periods_per_year=request.periods_per_year,
        )
        try:
            result = await run_portfolio_backtest(specs, cfg, cancel=token,
                                                  refresh=request.refresh_data)
        except DataFetchError as exc:
            raise HTTPException(502, str(exc)) from exc
        html = PortfolioReporter(result).to_html()
        if request.monte_carlo and request.monte_carlo.enabled:
            mc = await _run_mc(
                result.equity_curve,
                _mc_config(request.monte_carlo, initial_equity=result.initial_cash,
                           ppy=request.periods_per_year),
                cancel=token,
            )
            html = html.replace("</body>", MonteCarloReporter(mc).to_html() + "</body>")
        return HTMLResponse(html)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


# ── trade analysis + adaptive optimization ─────────────────────────────


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: BacktestRequest) -> AnalyzeResponse:
    """Backtest one symbol, then break the trade ledger down win/loss.

    The response carries the usual metrics plus a full trade analysis
    (expectancy, payoff, streaks, per-side stats, PnL histogram) and a list of
    rule-based parameter recommendations — the diagnostic half of the
    strategy-improvement loop.
    """
    bars = await _bars_for(
        request.symbol, request.source, request.timeframe, request.limit,
        refresh=request.refresh_data,
    )
    params = dict(request.params or {})
    params.update({
        "fast": request.fast, "slow": request.slow, "period": request.period,
        "long": request.long.model_dump() if request.long else None,
        "short": request.short.model_dump() if request.short else None,
        "settings": request.settings,
    })
    try:
        strategy = build_strategy(
            request.strategy, request.symbol, {k: v for k, v in params.items() if v is not None}
        )
    except StrategyError as exc:
        raise HTTPException(400, str(exc)) from exc
    cfg = BacktestConfig(
        initial_cash=request.initial_cash, fee_rate=request.fee_rate,
        slippage=request.slippage, position_fraction=request.position_fraction,
        periods_per_year=request.periods_per_year,
    )
    result = await run_backtest(strategy, bars, cfg)
    analysis = analyze_trades(result.trades)
    flat_params = {k: v for k, v in params.items() if v is not None}
    return AnalyzeResponse(
        strategy=request.strategy,
        symbol=request.symbol,
        metrics=_metrics_out(result.metrics),
        n_trades=len(result.trades),
        analysis=analysis.as_dict(),
        recommendations=recommend_adjustments(analysis, flat_params),
    )


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Grid-search a strategy's parameters with a 70/30 train/validation split.

    Candidates are ranked by validation Sharpe, discounted for train/validation
    disagreement (overfitting) and thin trade samples. The best candidate is
    re-run on the full history and returned with its win/loss analysis and
    recommendations — the adaptive half of the improvement loop.

    The sweep is synchronous CPU work and runs in a thread so the event loop
    stays free to serve ``POST /backtest/cancel/{token}``.
    """
    token = _open_run(request.run_token)
    try:
        bars = await _bars_for(
            request.symbol, request.source, request.timeframe, request.limit,
            refresh=request.refresh_data,
        )
        cfg = BacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate,
            slippage=request.slippage, position_fraction=request.position_fraction,
            periods_per_year=request.periods_per_year,
        )
        try:
            result = await asyncio.to_thread(
                optimize_strategy,
                request.strategy, request.symbol, bars,
                base_params=request.params, grid=request.grid, cfg=cfg, cancel=token,
            )
        except StrategyError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        payload = result.as_dict()
        payload["run_token"] = token.token
        return OptimizeResponse(**payload)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


@router.post("/autotune")
async def autotune_endpoint(request: AutoTuneRequest) -> dict:
    """Pre-live tuning: parameter search + volatility-adaptive SL/TP targets.

    Runs the train/validation grid search (like ``/optimize``) and layers the
    selected risk profile on top: Stop Loss / Take Profit in ATR multiples
    adapted to the asset's measured volatility; the High (breakout) profile
    also reports EMA50 trend confirmation.
    """
    from trading.application.autotune import RiskProfile, autotune

    token = _open_run(request.run_token)
    try:
        bars = await _bars_for(
            request.symbol, request.source, request.timeframe, request.limit,
            refresh=request.refresh_data,
        )
        cfg = BacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate,
            slippage=request.slippage, position_fraction=request.position_fraction,
            periods_per_year=request.periods_per_year,
        )
        try:
            result = await asyncio.to_thread(
                autotune,
                request.symbol, request.strategy, bars,
                RiskProfile(request.risk_profile),
                base_params=request.params, grid=request.grid, cfg=cfg, cancel=token,
            )
        except (StrategyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        payload = result.as_dict()
        payload["run_token"] = token.token
        return payload
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


# ── cancellation ───────────────────────────────────────────────────────


@router.post("/cancel/{token}", response_model=CancelOut)
async def cancel_run(token: str) -> CancelOut:
    """Stop a running backtest / Monte-Carlo identified by ``token``.

    Idempotent and safe to call after the run has finished: ``cancelled`` means a
    cancel flag is now in effect for that token, and ``known`` says whether a run
    was still executing here when the request arrived. The engine observes the
    flag between blocks of work and responds with HTTP 499.
    """
    if not is_valid_token(token):
        raise HTTPException(400, "invalid run token")
    known = run_registry.cancel(token)
    return CancelOut(token=token, cancelled=True, known=known)


@router.get("/cancel", response_model=list[str])
async def active_runs() -> list[str]:
    """Tokens of runs currently executing in this process (debug/ops aid)."""
    return run_registry.active_tokens()
