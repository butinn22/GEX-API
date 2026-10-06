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
import logging
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
from trading.application.presets import PresetService
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
    GlobalOptimizeRequest,
    GlobalOptimizeStatus,
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

log = logging.getLogger(__name__)

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


def _default_assignment(t: TickerConfig) -> dict:
    """Transparency info for a ticker without a preset (never optimized)."""
    explicit = bool(t.params) or ({"fast", "slow", "period"} & t.model_fields_set) \
        or t.long is not None or t.short is not None or t.settings is not None
    return {
        "strategy_name": "",
        "preset_id": None,
        "preset_version": None,
        "params": t.folded_params(),
        "assignment": "adhoc" if explicit else "default",
        "optimized": False,
    }


async def _spec_from_preset_config(t: TickerConfig) -> tuple[TickerSpec, dict]:
    """Resolve a preset-bound ticker into a spec + transparency info.

    The stored preset row is the source of truth: its strategy + params are
    the baseline (explicit request params still override per key — the
    existing convention). The returned info reports exactly the params the
    engine will run with, plus the preset's provenance for the
    ``assignment`` / ``optimized`` markers.
    """
    try:
        row = await _load_preset(t.preset_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            # A bad assignment is a request error, not a missing resource:
            # name the ticker and the id so the UI can mark the offending row.
            raise HTTPException(
                400, f"ticker '{t.symbol}': preset {t.preset_id} not found"
            ) from exc
        raise
    params = PresetService.params_of(row)
    overrides = dict(t.params or {})
    overrides.update(t.explicit_overrides())
    params.update({k: v for k, v in overrides.items() if v is not None})
    spec = TickerSpec(
        symbol=t.symbol,
        strategy=row.strategy,
        params=params,
        weight=t.weight,
        capital=t.capital,
        source=t.source,
        timeframe=t.timeframe,
        limit=t.limit,
        enabled=t.enabled,
    )
    info = {
        "strategy_name": row.strategy_name or "",
        "preset_id": row.id,
        "preset_version": row.version,
        "params": dict(params),
        "assignment": "preset",
        "optimized": bool(row.optimizer_run_id or row.backtest_ref),
    }
    return spec, info


async def _specs_from_request(
    req: PortfolioBacktestRequest,
) -> tuple[list[TickerSpec], dict[str, dict]]:
    """Build the engine specs plus the per-symbol assignment map.

    The assignment map drives the response transparency fields
    (``TickerResultOut.strategy_name/preset_*/params/assignment/optimized``);
    auto-selected tickers are reported as ``default``.
    """
    specs: list[TickerSpec] = []
    assignments: dict[str, dict] = {}
    for t in req.tickers:
        if t.preset_id:
            spec, info = await _spec_from_preset_config(t)
            specs.append(spec)
            assignments[t.symbol] = info
        else:
            specs.append(_spec_from_config(t))
            assignments.setdefault(t.symbol, _default_assignment(t))
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
            assignments.setdefault(
                inst["symbol"],
                {
                    "strategy_name": "",
                    "preset_id": None,
                    "preset_version": None,
                    "params": dict(req.default_params),
                    "assignment": "default",
                    "optimized": False,
                },
            )
    return specs, assignments


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
    assignments: dict[str, dict] | None = None,
) -> PortfolioBacktestResponse:
    idx = _sample_indices(len(result.times))
    assign = assignments or {}

    def _ticker_out(k: int, t) -> TickerResultOut:
        info = assign.get(t.symbol, {})
        return TickerResultOut(
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
            strategy_name=info.get("strategy_name", ""),
            preset_id=info.get("preset_id"),
            preset_version=info.get("preset_version"),
            params=info.get("params", {}),
            assignment=info.get("assignment", "default"),
            optimized=info.get("optimized", False),
        )

    return PortfolioBacktestResponse(
        n_tickers=result.n_tickers,
        initial_cash=result.initial_cash,
        final_equity=float(result.equity_curve[-1]),
        metrics=_metrics_out(result.metrics),
        equity_curve=_downsample(list(result.equity_curve), idx),
        times=[result.times[i].isoformat() for i in idx],
        run_token=run_token,
        tickers=[
            _ticker_out(k, t) for k, t in enumerate(result.tickers)
        ],
        correlation=(
            CorrelationOut(symbols=list(result.correlation.symbols),
                           matrix=[list(r) for r in result.correlation.matrix])
            if result.correlation else None
        ),
        errors=list(result.errors),
        monte_carlo=_mc_out(mc, "portfolio", run_token) if mc is not None else None,
    )


def _log_planned_combos(strategy: str, grid: dict[str, list] | None) -> None:
    """Log the planned sweep size before it starts.

    A sweep's wall time is ``combos × 2 backtests``; logging the planned count
    makes "the UI hung on Run optimization" diagnosable from the server log
    alone (an oversized grid shows up here instead of as silence).
    """
    grid = grid or {}
    planned = 1
    for values in grid.values():
        planned *= max(1, len(values))
    log.info(
        "optimize %s starting: %d planned combinations from %d grid axis(es)%s",
        strategy, planned, len(grid),
        "" if grid else " (default sweep — request grid was empty/absent)",
    )


async def _persist_result(request: BacktestRequest, result, *,
                          strategy: str | None = None,
                          symbol: str | None = None) -> int | None:
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
                strategy or request.strategy, symbol or request.symbol,
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


async def _load_preset(preset_id: int) -> StrategyPresetRow:
    """Load a saved strategy version for ``preset_id``-driven runs.

    The stored row is the source of truth: its strategy/symbol/params are used
    verbatim (explicit request ``params`` keys still override — the existing
    override convention).
    """
    from trading.adapters.persistence import database
    from trading.adapters.persistence.models import StrategyPresetRow

    await database.init_db()
    async with database._session_factory() as session:
        row = await PresetService(session).get(preset_id)
        if row is None:
            raise HTTPException(404, f"preset {preset_id} not found")
        return row


# ── single-symbol backtest ─────────────────────────────────────────────


@router.post("", response_model=BacktestResponse)
async def run(request: BacktestRequest) -> BacktestResponse:
    # A saved strategy version is the source of truth when preset_id is set:
    # its strategy/symbol/params are used verbatim (explicit request params
    # still override on a per-key basis).
    preset_row = await _load_preset(request.preset_id) if request.preset_id else None
    strategy_name = preset_row.strategy if preset_row else request.strategy
    symbol = preset_row.symbol if preset_row else request.symbol
    if request.bars:
        bars = [
            Bar(timestamp=b.timestamp, open=b.open, high=b.high, low=b.low,
                close=b.close, volume=b.volume)
            for b in request.bars
        ]
    else:
        bars = await _bars_for(
            symbol, request.source, request.timeframe, request.limit,
            refresh=request.refresh_data,
        )

    bars = sorted(bars, key=lambda b: b.timestamp)
    if len(bars) < 2:
        raise HTTPException(400, "need at least 2 bars")

    params: dict = {}
    if preset_row is not None:
        params.update(PresetService.params_of(preset_row))
    params.update(dict(request.params or {}))
    # The ergonomic fast/slow/period fields carry defaults, so with a preset
    # loaded only the *explicitly given* ones may override the stored params
    # (``model_fields_set`` separates "sent" from "defaulted"). Not-given keys
    # are simply absent — never written as None over the preset's values.
    explicit = request.model_fields_set if preset_row is not None else None

    def _given(key: str) -> bool:
        return explicit is None or key in explicit

    if _given("fast"):
        params["fast"] = request.fast
    if _given("slow"):
        params["slow"] = request.slow
    if _given("period"):
        params["period"] = request.period
    if request.long is not None:
        params["long"] = request.long.model_dump()
    if request.short is not None:
        params["short"] = request.short.model_dump()
    if request.settings is not None and _given("settings"):
        params["settings"] = request.settings
    try:
        strategy: Strategy = build_strategy(
            strategy_name, symbol, {k: v for k, v in params.items() if v is not None}
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
    result_id = await _persist_result(request, result,
                                      strategy=strategy_name, symbol=symbol)
    return BacktestResponse(
        strategy=strategy_name,
        symbol=symbol,
        metrics=_metrics_out(result.metrics),
        equity_curve=_downsample(list(result.equity_curve), idx),
        times=[bars[i].timestamp.isoformat() for i in idx],
        result_id=result_id,
        preset_id=request.preset_id,
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
        specs, assignments = await _specs_from_request(request)
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
        return _portfolio_response(result, mc, token.token, assignments)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


@router.post("/portfolio/monte-carlo", response_model=MonteCarloSummaryOut)
async def portfolio_monte_carlo(request: PortfolioMonteCarloRequest) -> MonteCarloSummaryOut:
    token = _open_run(request.run_token or request.portfolio.run_token)
    try:
        specs, _assignments = await _specs_from_request(request.portfolio)
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
        specs, _assignments = await _specs_from_request(request)
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

    Candidates are ranked by the chosen ``objective`` (default: the
    ``profit_win`` composite — validation profit gates the pick, win rate
    scales it), discounted for train/validation disagreement (overfitting) and
    thin trade samples. The best candidate is re-run on the full history and
    returned with its win/loss analysis and recommendations — the adaptive
    half of the improvement loop. With ``save_preset`` the winner is persisted
    as the ticker's default preset (source=``optimizer``).

    The sweep is synchronous CPU work and runs in a thread so the event loop
    stays free to serve ``POST /backtest/cancel/{token}``.
    """
    token = _open_run(request.run_token)
    try:
        # A saved strategy version seeds the sweep when preset_id is set: its
        # strategy/symbol/base params are used verbatim (explicit request
        # params still override on a per-key basis).
        preset_row = (await _load_preset(request.preset_id)
                      if request.preset_id else None)
        strategy_name = preset_row.strategy if preset_row else request.strategy
        symbol = preset_row.symbol if preset_row else request.symbol
        base_params: dict = {}
        if preset_row is not None:
            base_params.update(PresetService.params_of(preset_row))
        base_params.update(dict(request.params))
        bars = await _bars_for(
            symbol, request.source, request.timeframe, request.limit,
            refresh=request.refresh_data,
        )
        cfg = BacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate,
            slippage=request.slippage, position_fraction=request.position_fraction,
            periods_per_year=request.periods_per_year,
        )
        try:
            _log_planned_combos(strategy_name, request.grid)
            result = await asyncio.to_thread(
                optimize_strategy,
                strategy_name, symbol, bars,
                base_params=base_params, grid=request.grid, cfg=cfg,
                cancel=token, objective=request.objective,
            )
            log.info(
                "optimize %s/%s finished: %d candidates (planned grid %s)",
                strategy_name, symbol, result.n_candidates, request.grid or "default",
            )
        except StrategyError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if request.save_preset:
            await _save_optimized_preset(
                symbol, strategy_name, result.best_params,
                strategy_version=_strategy_version(strategy_name),
                optimizer_run_id=token.token,
                metrics=(result.best or {}).get("metrics") or None,
                strategy_name=((preset_row.strategy_name if preset_row else "")
                               or request.strategy_name),
                timeframe=request.timeframe,
            )
        payload = result.as_dict()
        payload["run_token"] = token.token
        return OptimizeResponse(**payload)
    except RunCancelled as exc:
        raise _cancelled(exc) from exc
    finally:
        run_registry.clear(token.token)


async def _save_optimized_preset(
    symbol: str,
    strategy: str,
    best_params: dict,
    *,
    strategy_version: str,
    optimizer_run_id: str,
    metrics: dict | None = None,
    strategy_name: str = "",
    timeframe: str = "",
) -> None:
    """Persist an optimization winner as the group's next version.

    Best-effort by design (mirrors ``_persist_result``): a database hiccup
    must not fail a completed optimization. The winner carries its metrics
    snapshot together with its provenance (``backtest_ref =
    "optimizer:<run_token>"``) — never fabricated.
    """
    try:
        from trading.adapters.persistence.database import _session_factory, init_db
        from trading.application.presets import PresetService

        await init_db()
        async with _session_factory() as session:
            await PresetService(session).save_optimization(
                symbol=symbol,
                strategy=strategy,
                strategy_version=strategy_version,
                best_params=best_params,
                optimizer_run_id=optimizer_run_id,
                metrics=metrics,
                strategy_name=strategy_name,
                timeframe=timeframe,
            )
    except Exception:
        pass


def _strategy_version(strategy_name: str) -> str:
    from trading.application.strategies.trend_confluence_unified import (
        UNIFIED_STRATEGY_VERSION,
    )

    if strategy_name == "trend_confluence_unified":
        return UNIFIED_STRATEGY_VERSION
    return ""


# ── global (all-tickers) optimization ─────────────────────────────────


@router.post("/optimize/global", response_model=GlobalOptimizeStatus)
async def optimize_global(request: GlobalOptimizeRequest) -> GlobalOptimizeStatus:
    """Start optimization for every ticker (explicit list or a universe).

    Returns immediately with a ``run_id``; poll ``GET /optimize/global/{id}``
    for progress (current ticker, completed/failed counts, ETA, results).
    Failures are isolated per ticker and reported in ``errors`` — one bad
    ticker never aborts the run.
    """
    from trading.application.global_optimize import global_optimize_runner

    try:
        symbols = global_optimize_runner.resolve_symbols(
            request.symbols, request.category, request.n_tickers,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    token = _open_run(request.run_token)
    try:
        cfg = BacktestConfig(
            initial_cash=request.initial_cash, fee_rate=request.fee_rate,
            slippage=request.slippage, position_fraction=request.position_fraction,
            periods_per_year=request.periods_per_year,
        )
        state = await global_optimize_runner.start(
            run_id=token.token,
            symbols=symbols,
            strategy=request.strategy,
            base_params=dict(request.params),
            grid=request.grid,
            objective=request.objective,
            source=request.source,
            timeframe=request.timeframe,
            limit=request.limit,
            cfg=cfg,
            refresh=request.refresh_data,
            save_preset=request.save_preset,
            session_factory=_session_factory_value(),
            cancel=token,
        )
        return GlobalOptimizeStatus(**state.as_dict())
    except ValueError as exc:
        run_registry.clear(token.token)
        raise HTTPException(400, str(exc)) from exc


def _session_factory_value():
    from trading.adapters.persistence import database

    if database._session_factory is None:
        database.configure()
    return database._session_factory


@router.get("/optimize/global/{run_id}", response_model=GlobalOptimizeStatus)
async def optimize_global_status(run_id: str) -> GlobalOptimizeStatus:
    """Progress/result of a global optimization run."""
    from trading.application.global_optimize import global_optimize_runner

    state = global_optimize_runner.status(run_id)
    if state is None:
        raise HTTPException(404, f"unknown optimization run '{run_id}'")
    return GlobalOptimizeStatus(**state.as_dict())


@router.post("/optimize/global/{run_id}/cancel", response_model=CancelOut)
async def optimize_global_cancel(run_id: str) -> CancelOut:
    """Stop a running global optimization at the next ticker boundary."""
    from trading.application.global_optimize import global_optimize_runner

    if not global_optimize_runner.cancel_run(run_id):
        raise HTTPException(404, f"no running optimization '{run_id}'")
    return CancelOut(token=run_id, cancelled=True, known=True)


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
