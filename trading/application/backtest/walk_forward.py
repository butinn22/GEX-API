"""Walk-forward analysis and parameter sensitivity scans."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from trading.domain import Bar

from .engine import BacktestConfig, BacktestResult, run_backtest

__all__ = ["walk_forward", "parameter_sensitivity"]


async def walk_forward(
    strategy_factory: Callable[[], object],
    bars: Sequence[Bar],
    *,
    n_windows: int = 4,
    config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    """Rolling test windows: backtest on each sequential window (walk-forward).

    ``strategy_factory`` is called fresh per window so strategy state is reset.
    The train portion (everything before the test window) is available to the
    caller's factory for parameter selection; this helper runs the test only.
    """
    if n_windows < 2:
        raise ValueError("n_windows must be >= 2")
    bars = sorted(bars, key=lambda b: b.timestamp)
    size = max(len(bars) // n_windows, 1)
    results: list[BacktestResult] = []
    for w in range(1, n_windows):
        test = bars[w * size : (w + 1) * size]
        if len(test) < 2:
            continue
        results.append(await run_backtest(strategy_factory(), test, config))
    return results


async def parameter_sensitivity(
    strategy_factory: Callable[[float], object],
    bars: Sequence[Bar],
    param_name: str,
    values: Sequence[float],
    *,
    metric: str = "sharpe",
    config: BacktestConfig | None = None,
) -> list[tuple[float, float]]:
    """Run the strategy over a parameter grid; return [(param_value, metric)]."""
    out: list[tuple[float, float]] = []
    for v in values:
        result = await run_backtest(strategy_factory(v), bars, config)
        out.append((float(v), float(getattr(result.metrics, metric))))
    return out
