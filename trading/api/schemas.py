"""Pydantic v2 request/response models for the trading API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "LoginRequest",
    "TokenResponse",
    "ApiKeyCreate",
    "ApiKeyOut",
    "ApiKeySettingsUpdate",
    "AccountRouteOut",
    "OhlcvBar",
    "BacktestRequest",
    "BacktestMetricsOut",
    "BacktestResponse",
    "StrategyInfo",
    "TickerConfig",
    "PortfolioBacktestRequest",
    "TickerResultOut",
    "CorrelationOut",
    "PortfolioBacktestResponse",
    "MonteCarloRequest",
    "MonteCarloRunOptions",
    "MonteCarloSummaryOut",
    "HistogramOut",
    "PortfolioMonteCarloRequest",
    "UniverseTicker",
    "UniverseResponse",
    "CancelOut",
    "AnalyzeResponse",
    "OptimizeRequest",
    "OptimizeResponse",
    "AutoTuneRequest",
    # ── presets (per-ticker unified-strategy configurations) ──
    "PresetCreate",
    "PresetUpdate",
    "PresetOut",
    "PresetFromBacktestRequest",
    "PresetValidateOut",
    # ── global optimization ──
    "GlobalOptimizeRequest",
    "GlobalOptimizeStatus",
    # ── signal API keys ──
    "SignalKeyCreate",
    "SignalKeyOut",
    "SignalKeyGenerateReport",
    # ── basket export / deploy (complete basket transfer) ──
    "BasketTickerIn",
    "BasketExportRequest",
    "BasketTickerOut",
    "BasketExportResponse",
    "BasketDeployRequest",
    "BasketDeployResponse",
]

#: A client-supplied handle for a cancellable run. Restricted so it can be used
#: safely as a Redis key suffix.
RUN_TOKEN_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


# ── Auth ───────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── API keys ───────────────────────────────────────────────────────────


class ApiKeyCreate(BaseModel):
    exchange: Literal["bingx", "tbank"]
    label: str = ""
    api_key: str
    api_secret: str = ""
    account_id: str = ""  # TBANK account id (stored in extra)


class ApiKeySettingsUpdate(BaseModel):
    """Per-account routing/risk patch (multi-account 'wallet' settings)."""

    instruments: list[str] | None = None  # empty list = all instruments
    risk_profile: Literal["low", "medium", "high"] | None = None
    max_position_pct: float | None = Field(default=None, gt=0, le=1.0)
    leverage: float | None = Field(default=None, ge=1.0)
    enabled: bool | None = None


class ApiKeyOut(BaseModel):
    id: int
    exchange: str
    label: str
    api_key_masked: str
    created_at: datetime | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class AccountRouteOut(BaseModel):
    """One account that would receive orders for a symbol (routing preview)."""

    key_id: int
    exchange: str
    label: str
    settings: dict[str, Any]


# ── Backtest ───────────────────────────────────────────────────────────


class OhlcvBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class SideSettings(BaseModel):
    """Per-side (long/short) strategy settings for the dual strategy."""

    enabled: bool = True
    fast: int = Field(default=10, ge=1)
    slow: int = Field(default=20, ge=2)
    period: int = Field(default=20, ge=1)
    strength: float = Field(default=1.0, gt=0, le=1.0)


class BacktestRequest(BaseModel):
    strategy: Literal[
        "sma_crossover", "buy_and_hold", "mean_reversion", "momentum", "sma_crossover_ls", "gex_emf",
        "trend_confluence", "trend_confluence_unified",
    ] = "sma_crossover"
    symbol: str = "SYNTH"
    bars: list[OhlcvBar] | None = None
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)
    fast: int = Field(default=20, ge=1)
    slow: int = Field(default=50, ge=2)
    period: int = Field(default=20, ge=1)
    long: SideSettings | None = None
    short: SideSettings | None = None
    settings: dict | None = None  # GEX strategy (EMAFilterTrendStrategy) settings
    #: Free-form per-strategy params (trend_confluence knobs, options walls, …).
    params: dict[str, Any] | None = None
    source: str = "auto"  # "auto" (resolve from symbol) | "synthetic" | moex/yfinance/bybit
    timeframe: str = "1d"
    limit: int = Field(default=5000, ge=60, le=10000)
    refresh_data: bool = False  # bypass the OHLCV cache for this run
    #: Load the run's strategy + params from a saved strategy version
    #: (Strategy Hub). The stored params are used verbatim; explicit ``params``
    #: keys still override (the existing override convention).
    preset_id: int | None = None


class BacktestMetricsOut(BaseModel):
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float | None
    calmar: float | None
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float | None
    n_trades: int
    n_periods: int


class BacktestResponse(BaseModel):
    strategy: str
    symbol: str
    metrics: BacktestMetricsOut
    equity_curve: list[float]
    times: list[str]
    n_trades: int
    #: Persisted run id (for /export/backtest/{id}/trades.csv|xlsx); None when
    #: persistence was unavailable for this run.
    result_id: int | None = None
    #: The saved strategy version the run loaded (None for a free-form run).
    preset_id: int | None = None


# ── Strategies ─────────────────────────────────────────────────────────


class StrategyInfo(BaseModel):
    name: str
    params: list[str]


# ── Orders ─────────────────────────────────────────────────────────────


class OrderCreate(BaseModel):
    exchange: Literal["bingx", "tbank"]
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"] = "market"
    quantity: float = Field(gt=0)
    price: float | None = None
    strategy: str | None = None
    reason: str | None = None


class OrderOut(BaseModel):
    id: str
    exchange: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    status: str
    strategy: str | None = None
    reason: str | None = None


class BulkIdsRequest(BaseModel):
    """Bulk operation over a list of string ids (orders)."""

    ids: list[str] = Field(min_length=1, max_length=1000)


class BulkIntIdsRequest(BaseModel):
    """Bulk operation over a list of integer ids (stored runs, signal keys)."""

    ids: list[int] = Field(min_length=1, max_length=1000)


class SignalKeyBulkRequest(BaseModel):
    """Bulk enable/disable/revoke over signal-key ids."""

    ids: list[int] = Field(min_length=1, max_length=1000)
    action: Literal["enable", "disable", "revoke"]


class MonteCarloRequest(BaseModel):
    """Monte-Carlo simulation over a strategy's realised returns on one symbol."""

    symbol: str = "SYNTH"
    strategy: str = "sma_crossover"
    params: dict[str, Any] = Field(default_factory=dict)
    long: SideSettings | None = None
    short: SideSettings | None = None
    settings: dict[str, Any] | None = None
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = Field(default=1000, ge=60, le=10000)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)
    n_paths: int = Field(default=10_000, ge=1, le=100_000)
    n_steps: int | None = Field(default=None, ge=1, le=10_000)
    method: Literal["gbm", "bootstrap", "block_bootstrap", "historical"] = "gbm"
    block_size: int = Field(default=5, ge=1, le=500)
    seed: int = 0
    refresh_data: bool = False  # bypass the OHLCV cache for this run
    run_token: str | None = Field(
        default=None, pattern=RUN_TOKEN_PATTERN,
        description="optional handle to cancel this run via POST /backtest/cancel/{token}",
    )

    def folded_params(self) -> dict[str, Any]:
        return _fold_params(self.params, self.long, self.short, self.settings)


# ── Multi-ticker portfolio backtest ────────────────────────────────────


class TickerConfig(BaseModel):
    """Per-ticker settings — every ticker can trade a different strategy."""

    symbol: str
    strategy: str = "sma_crossover"
    #: Load this ticker's strategy + params from a saved strategy version
    #: (Strategy Hub). The stored params are the baseline; explicit ``params``
    #: keys still override (the existing override convention).
    preset_id: int | None = None
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0, description="relative allocation weight")
    capital: float | None = Field(default=None, gt=0, description="absolute capital (overrides weight)")
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = Field(default=5000, ge=60, le=10000)
    params: dict[str, Any] = Field(default_factory=dict)
    # convenience shortcuts (merged into params unless already present)
    fast: int | None = Field(default=None, ge=1)
    slow: int | None = Field(default=None, ge=2)
    period: int | None = Field(default=None, ge=1)
    long: SideSettings | None = None
    short: SideSettings | None = None
    settings: dict[str, Any] | None = None

    def folded_params(self) -> dict[str, Any]:
        p = _fold_params(self.params, self.long, self.short, self.settings)
        if self.fast is not None:
            p.setdefault("fast", self.fast)
        if self.slow is not None:
            p.setdefault("slow", self.slow)
        if self.period is not None:
            p.setdefault("period", self.period)
        return p

    def explicit_overrides(self) -> dict[str, Any]:
        """Only the explicitly-sent ergonomic fields (never the defaults).

        With a preset as the params baseline, defaulted fields must not leak
        over the stored values — only keys the client actually sent may
        override (same convention as the single-symbol ``preset_id`` path).
        """
        out: dict[str, Any] = {}
        sent = self.model_fields_set
        if "fast" in sent:
            out["fast"] = self.fast
        if "slow" in sent:
            out["slow"] = self.slow
        if "period" in sent:
            out["period"] = self.period
        if self.long is not None:
            out["long"] = self.long.model_dump()
        if self.short is not None:
            out["short"] = self.short.model_dump()
        if self.settings is not None:
            out["settings"] = self.settings
        return out


class MonteCarloRunOptions(BaseModel):
    """Optional Monte-Carlo block embedded in a portfolio request."""

    enabled: bool = True
    n_paths: int = Field(default=10_000, ge=1, le=100_000)
    n_steps: int | None = Field(default=None, ge=1, le=10_000)
    method: Literal["gbm", "bootstrap", "block_bootstrap", "historical"] = "gbm"
    block_size: int = Field(default=5, ge=1, le=500)
    seed: int = 0


class PortfolioBacktestRequest(BaseModel):
    """Backtest a basket of tickers, each with its own strategy + settings.

    Provide an explicit ``tickers`` list, **or** set ``n_tickers`` to auto-select
    that many from ``category`` (the shared ``default_*`` fields seed them).
    """

    tickers: list[TickerConfig] = Field(default_factory=list)
    n_tickers: int = Field(default=0, ge=0, le=50)
    category: Literal["us", "crypto", "fx", "ru", "sectors", "all"] = "all"
    default_strategy: str = "sma_crossover"
    default_params: dict[str, Any] = Field(default_factory=dict)
    default_source: str = "auto"
    default_timeframe: str = "1d"
    default_limit: int = Field(default=5000, ge=60, le=10000)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)
    monte_carlo: MonteCarloRunOptions | None = None
    refresh_data: bool = Field(
        default=False,
        description="bypass the OHLCV cache and re-download bars for this run",
    )
    run_token: str | None = Field(
        default=None, pattern=RUN_TOKEN_PATTERN,
        description="optional handle to cancel this run via POST /backtest/cancel/{token}",
    )

    @model_validator(mode="after")
    def _need_targets(self) -> PortfolioBacktestRequest:
        if not self.tickers and self.n_tickers < 1:
            raise ValueError("provide either 'tickers' or 'n_tickers' >= 1")
        return self


class TickerResultOut(BaseModel):
    symbol: str
    strategy: str
    source: str
    timeframe: str
    weight: float
    capital: float
    metrics: BacktestMetricsOut
    equity_curve: list[float]
    n_trades: int
    # ── result transparency (per-ticker assignment audit) ──
    #: user-facing strategy name of the assigned preset ("" = none)
    strategy_name: str = ""
    #: the saved strategy version the ticker ran with (None = not preset-driven)
    preset_id: int | None = None
    preset_version: int | None = None
    #: the complete params the engine actually ran with (not a client echo)
    params: dict[str, Any] = Field(default_factory=dict)
    #: preset = assigned saved strategy | adhoc = explicit params | default = strategy defaults
    assignment: Literal["preset", "adhoc", "default"] = "default"
    #: True only for preset-driven tickers whose version carries real-run
    #: provenance (optimizer_run_id / backtest_ref); never fabricated.
    optimized: bool = False


class CorrelationOut(BaseModel):
    symbols: list[str]
    matrix: list[list[float]]


class PortfolioBacktestResponse(BaseModel):
    n_tickers: int
    initial_cash: float
    final_equity: float
    metrics: BacktestMetricsOut
    equity_curve: list[float]
    times: list[str]
    tickers: list[TickerResultOut]
    correlation: CorrelationOut | None = None
    errors: list[dict[str, str]] = Field(default_factory=list)
    monte_carlo: MonteCarloSummaryOut | None = None
    run_token: str = ""


class PortfolioMonteCarloRequest(BaseModel):
    """Standalone Monte-Carlo for a basket (re-runs the portfolio then simulates)."""

    portfolio: PortfolioBacktestRequest
    monte_carlo: MonteCarloRunOptions = Field(default_factory=MonteCarloRunOptions)
    run_token: str | None = Field(default=None, pattern=RUN_TOKEN_PATTERN)


# ── Monte-Carlo response ───────────────────────────────────────────────


class HistogramOut(BaseModel):
    counts: list[int]
    centers: list[float]
    bin_edges: list[float]


class MonteCarloSummaryOut(BaseModel):
    label: str = ""
    method: str
    n_paths: int
    n_steps: int
    initial_equity: float
    steps: list[int]
    bands: dict[str, list[float]]
    mean_path: list[float]
    final_percentiles: dict[str, float]
    final_return_percentiles: dict[str, float]
    mean_return: float
    #: headline final-return stats (convenience aliases of the p5/p95 band)
    mean: float
    p5: float
    p95: float
    histogram: HistogramOut
    metrics_mean: dict[str, float]
    metrics_ci: dict[str, list[float]]
    prob_profit: float
    var_95: float
    cvar_95: float
    best_return: float
    worst_return: float
    run_token: str = ""


# ── Cancellation ───────────────────────────────────────────────────────


class CancelOut(BaseModel):
    """Acknowledgement for ``POST /backtest/cancel/{token}``."""

    token: str
    cancelled: bool
    #: ``False`` means the run had already finished (or never existed) — the
    #: cancel is a no-op, which is not an error for the client.
    known: bool = False


# ── Trade analysis / adaptive optimization ─────────────────────────────


class AnalyzeResponse(BaseModel):
    """Backtest + win/loss diagnostics + adaptation suggestions."""

    strategy: str
    symbol: str
    metrics: BacktestMetricsOut
    n_trades: int
    analysis: dict[str, Any]
    recommendations: list[dict[str, Any]]


class OptimizeRequest(BaseModel):
    """Adaptive parameter search for one symbol (train/validation split)."""

    strategy: str = "trend_confluence"
    symbol: str = "SYNTH"
    params: dict[str, Any] = Field(default_factory=dict)
    #: Base params from a saved strategy version (Strategy Hub) — the stored
    #: params are the sweep's starting point, verbatim.
    preset_id: int | None = None
    #: Group name for ``save_preset`` — defaults to the preset's own name (or
    #: the unnamed legacy group).
    strategy_name: str = Field(default="", max_length=64)
    grid: dict[str, list[Any]] | None = Field(
        default=None,
        description="parameter → candidates; defaults to the strategy's sweep",
    )
    #: Ranking objective applied to the validation split.
    #: ``profit_win`` = profitability first (validation return gates the pick),
    #: win rate second (scales the score); the default.
    objective: Literal[
        "profit_win", "sharpe", "sortino", "calmar", "total_return",
        "profit_factor", "win_rate", "max_drawdown",
    ] = "profit_win"
    #: Persist the winner as the ticker's default preset (source=optimizer).
    save_preset: bool = False
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = Field(default=1000, ge=200, le=3000)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)
    refresh_data: bool = False
    run_token: str | None = Field(default=None, pattern=RUN_TOKEN_PATTERN)


class OptimizeResponse(BaseModel):
    symbol: str
    strategy: str
    n_candidates: int
    baseline: dict[str, Any]
    best_params: dict[str, Any]
    best: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    trade_analysis: dict[str, Any] | None
    recommendations: list[dict[str, Any]]
    #: Per-parameter impact ranking (see
    #: :func:`trading.application.backtest.optimize.parameter_impact`).
    impact: list[dict[str, Any]] = Field(default_factory=list)
    run_token: str = ""


class StrategyParamSchema(BaseModel):
    """Editable-parameter schema for one strategy (console form + sweep)."""

    name: str
    groups: list[str]
    params: list[dict[str, Any]]
    defaults: dict[str, Any]
    sweep: dict[str, list[Any]]


class AutoTuneRequest(BaseModel):
    """Pre-live tuning: parameter search + volatility-adaptive SL/TP targets."""

    strategy: str = "trend_confluence"
    symbol: str = "SYNTH"
    risk_profile: Literal["low", "medium", "high"] = "medium"
    params: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] | None = None
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = Field(default=1000, ge=200, le=3000)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)
    refresh_data: bool = False
    run_token: str | None = Field(default=None, pattern=RUN_TOKEN_PATTERN)


# ── Universe selection ─────────────────────────────────────────────────


class UniverseTicker(BaseModel):
    symbol: str
    name: str
    category: str


class UniverseResponse(BaseModel):
    category: str
    n: int
    available: int
    tickers: list[UniverseTicker]


# ── shared helpers ─────────────────────────────────────────────────────


def _fold_params(
    params: dict[str, Any] | None,
    long: SideSettings | None,
    short: SideSettings | None,
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge the ergonomic ``long``/``short``/``settings`` blocks into one params dict."""
    p: dict[str, Any] = dict(params or {})
    if long is not None:
        p["long"] = long.model_dump()
    if short is not None:
        p["short"] = short.model_dump()
    if settings is not None:
        p["settings"] = settings
    return p


# ── Presets (per-ticker unified-strategy configurations) ────────────────


class PresetCreate(BaseModel):
    """Save a parameter preset version for one ticker (source: manual default).

    Every create is the group's **next version** — rows are never overwritten.
    ``metrics`` (headline snapshot of a real run) requires provenance
    (``optimizer_run_id`` or ``backtest_ref``); metrics are never fabricated.
    """

    symbol: str = Field(min_length=1, max_length=32)
    strategy: str = "trend_confluence_unified"
    strategy_version: str = "1.0.0"
    #: user-facing strategy name; "" = the legacy/unnamed group
    strategy_name: str = Field(default="", max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    source: Literal["manual", "backtest", "optimizer"] = "manual"
    optimizer_run_id: str | None = Field(default=None, max_length=64)
    #: bar interval the version was validated on (audit/UX)
    timeframe: str = Field(default="", max_length=16)
    #: {total_return, sharpe, max_drawdown, win_rate, n_trades} from a real run
    metrics: dict[str, Any] | None = None
    #: "optimizer:<run_token>" | "backtest:<backtest_results.id>"
    backtest_ref: str | None = Field(default=None, max_length=64)
    is_default: bool = True
    notes: str = ""


class PresetUpdate(BaseModel):
    """Edit a preset's parameters (and optionally its notes).

    A ``params`` edit creates the group's **next version** (source=manual)
    instead of mutating the row — history is never rewritten.
    """

    params: dict[str, Any] | None = None
    notes: str | None = None
    set_default: bool = False


class PresetOut(BaseModel):
    """One saved strategy version (Strategy Hub ``PresetOut`` v2)."""

    id: int
    symbol: str
    strategy: str
    strategy_version: str
    #: user-facing strategy name; "" = the legacy/unnamed group
    strategy_name: str = ""
    #: monotonic per (symbol, strategy, strategy_name)
    version: int = 1
    params: dict[str, Any] = Field(default_factory=dict)
    #: bar interval the version was validated on
    timeframe: str = ""
    #: headline snapshot written only from real backtest/optimize results
    metrics: dict[str, Any] = Field(default_factory=dict)
    #: backtest_only | live_enabled
    status: str = "backtest_only"
    backtest_ref: str | None = None
    source: str
    optimizer_run_id: str | None = None
    is_default: bool
    notes: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PresetValidateOut(BaseModel):
    """Go-live gate verdict for one version (see F6)."""

    ok: bool
    reasons: list[dict[str, Any]] = Field(default_factory=list)


class PresetFromBacktestRequest(BaseModel):
    """Generate the default preset via a real backtest run (source of truth)."""

    strategy: str = "trend_confluence_unified"
    params: dict[str, Any] = Field(default_factory=dict)
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = Field(default=1000, ge=60, le=10000)
    notes: str = ""


# ── Global optimization ────────────────────────────────────────────────


class GlobalOptimizeRequest(BaseModel):
    """Optimize every ticker (explicit list or a universe category)."""

    strategy: str = "trend_confluence_unified"
    symbols: list[str] = Field(default_factory=list)
    category: Literal["us", "crypto", "fx", "ru", "sectors", "all"] = "all"
    n_tickers: int = Field(default=0, ge=0, le=200,
                           description="0 = all symbols given, or all in category")
    params: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] | None = None
    objective: str = "profit_win"
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = Field(default=1000, ge=200, le=3000)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)
    refresh_data: bool = False
    save_preset: bool = True
    run_token: str | None = Field(default=None, pattern=RUN_TOKEN_PATTERN)


class GlobalOptimizeStatus(BaseModel):
    """Progress/result of a global (all-tickers) optimization run."""

    run_id: str
    state: Literal["running", "done", "failed", "cancelled"]
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_symbol: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    eta_seconds: float | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


# ── Signal API keys ─────────────────────────────────────────────────────


class SignalKeyCreate(BaseModel):
    """Create a signal key for one broker referencing the full config."""

    exchange: Literal["bingx", "tbank"]
    label: str = ""
    strategy: str = "trend_confluence_unified"
    strategy_version: str = "1.0.0"
    tickers: list[str] = Field(min_length=1)
    #: per-ticker params; missing tickers fall back to their default preset
    params_by_ticker: dict[str, dict[str, Any]] = Field(default_factory=dict)
    timeframe: str = "1d"
    source: str = "auto"
    limit: int = Field(default=1000, ge=60, le=10000)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)


class SignalKeyOut(BaseModel):
    id: int
    key: str
    exchange: str
    label: str
    active: bool
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SignalKeyGenerateReport(BaseModel):
    """Outcome of one signal-generation run for a key."""

    key: str
    exchange: str
    generated_at: datetime
    n_signals: int = 0
    n_trades: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)


# ── Basket export / deploy (complete basket transfer) ──────────────────


class BasketTickerIn(TickerConfig):
    """One ticker of an outgoing basket.

    Same shape as :class:`TickerConfig` but ``strategy`` is optional: a
    ticker bound to a saved strategy version (``preset_id``) takes the
    strategy and params from the preset store instead.
    """

    strategy: str = ""  # optional — preset_id may carry it


class BasketExportRequest(BaseModel):
    """A complete basket to validate and export (or deploy to live signals)."""

    tickers: list[BasketTickerIn] = Field(min_length=1)
    initial_cash: float = Field(default=100_000.0, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    position_fraction: float = Field(default=0.95, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, gt=0)


class BasketTickerOut(BaseModel):
    """One validated ticker of an exported basket.

    ``params`` is the complete, server-resolved effective set (from the
    preset store for preset-driven tickers) — never a client echo — and is
    guaranteed to survive a round-trip back into ``/backtest/portfolio``.
    """

    symbol: str
    strategy: str
    strategy_name: str = ""
    preset_id: int | None = None
    preset_version: int | None = None
    #: audit-only provenance (ignored when the payload is fed back)
    preset_source: str | None = None
    backtest_ref: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    #: preset (saved strategy) | adhoc (explicit params, not optimized)
    assignment: Literal["preset", "adhoc"] = "adhoc"
    optimized: bool = False
    weight: float = 1.0
    capital: float | None = None
    source: str = "auto"
    timeframe: str = "1d"
    limit: int = 5000
    enabled: bool = True
    warnings: list[str] = Field(default_factory=list)


class BasketExportResponse(BaseModel):
    """Parameter-lossless export payload (round-trip safe).

    ``costs.*`` map onto the top-level fields of ``PortfolioBacktestRequest``
    and each ticker, with its provenance fields stripped, is a valid
    ``TickerConfig`` — see :meth:`to_backtest_request`.
    """

    schema_version: str = "1"
    exported_at: datetime
    costs: dict[str, Any] = Field(default_factory=dict)
    tickers: list[BasketTickerOut]

    def to_backtest_request(self) -> dict[str, Any]:
        """The ``POST /backtest/portfolio`` body that replays this basket.

        Provenance fields (``strategy_name / preset_* / backtest_ref /
        assignment / optimized / warnings``) are dropped — they are audit-only
        and never re-read.
        """
        replay_keys = (
            "symbol", "strategy", "params", "weight", "capital",
            "source", "timeframe", "limit", "enabled",
        )
        return {
            **self.costs,
            "tickers": [
                {k: v for k, v in t.model_dump().items() if k in replay_keys}
                for t in self.tickers
            ],
        }


class BasketDeployRequest(BaseModel):
    """Deploy a (previously exportable) basket to the live signal pipeline."""

    basket: BasketExportRequest
    exchange: Literal["bingx", "tbank"]
    label: str = ""


class BasketDeployResponse(BaseModel):
    """The created signal key plus any broker-routing warnings."""

    id: int
    key: str
    exchange: str
    label: str
    warnings: list[str] = Field(default_factory=list)
    dashboard: str = ""


# Resolve forward references (models defined after their first use).
for _model in (PortfolioBacktestResponse, PortfolioMonteCarloRequest,
               BasketExportResponse):
    _model.model_rebuild()
del _model
