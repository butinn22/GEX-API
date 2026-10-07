"""Vectorized backtest (fast path for long/flat strategies).

Assumes: BUY enters at the next bar, SELL exits at the next bar, a fixed
fraction of equity in the market, mark-to-market at close. This path has no
per-trade ledger (trades are empty), so it is used for fast optimisation /
sensitivity scans; the event-driven engine remains the source of truth for
trade-level metrics.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trading.domain import Bar, Side

from .engine import BacktestConfig, BacktestResult
from .metrics import compute_metrics

__all__ = ["run_backtest_vectorized"]


async def run_backtest_vectorized(
    strategy, bars: Sequence[Bar], config: BacktestConfig | None = None
) -> BacktestResult:
    cfg = config or BacktestConfig()
    bars = sorted(bars, key=lambda b: b.timestamp)
    if not bars:
        raise ValueError("backtest requires at least one bar")

    closes = np.array([b.close for b in bars], dtype=float)
    ts_to_idx = {b.timestamp: i for i, b in enumerate(bars)}
    position = np.zeros(len(bars), dtype=float)  # fraction of equity in market
    in_market = False

    for sig in await strategy.generate_signals(bars):
        idx = ts_to_idx.get(sig.timestamp)
        if idx is None:
            continue
        entry_idx = min(idx + 1, len(bars) - 1)
        if sig.side is Side.BUY and not in_market:
            position[entry_idx:] = cfg.position_fraction * sig.strength
            in_market = True
        elif sig.side is Side.SELL and in_market:
            position[entry_idx:] = 0.0
            in_market = False

    equity = np.empty(len(bars), dtype=float)
    equity[0] = cfg.initial_cash
    for t in range(1, len(bars)):
        ret = position[t - 1] * (closes[t] / closes[t - 1] - 1.0)
        dpos = position[t] - position[t - 1]
        cost = abs(dpos) * (cfg.fee_rate + cfg.slippage) if dpos != 0 else 0.0
        equity[t] = equity[t - 1] * (1.0 + ret - cost)

    metrics = compute_metrics(equity, None, periods_per_year=cfg.periods_per_year)
    return BacktestResult(equity_curve=equity, trades=(), metrics=metrics)
