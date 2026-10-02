"""Pydantic v2 request/response models for the trading API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "LoginRequest",
    "TokenResponse",
    "ApiKeyCreate",
    "ApiKeyOut",
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


class ApiKeyOut(BaseModel):
    id: int
    exchange: str
    label: str
    api_key_masked: str
    created_at: datetime | None = None


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
        "sma_crossover", "buy_and_hold", "mean_reversion", "momentum", "sma_crossover_ls", "gex_emf"
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
    source: str = "auto"  # "auto" (resolve from symbol) | "synthetic" | moex/yfinance/bybit
    timeframe: str = "1d"
    limit: int = Field(default=5000, ge=60, le=10000)
    refresh_data: bool = False  # bypass the OHLCV cache for this run


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
    def _need_targets(self) -> "PortfolioBacktestRequest":
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


# Resolve forward references (models defined after their first use).
for _model in (PortfolioBacktestResponse, PortfolioMonteCarloRequest):
    _model.model_rebuild()
del _model
