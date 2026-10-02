"""Standalone backtest verification: prints real metrics + a Monte-Carlo run.

Run:  .venv/Scripts/python.exe scripts/verify_backtest.py
"""
from __future__ import annotations

import asyncio

import numpy as np

from trading.adapters.fetchers.synthetic import SyntheticFetcher
from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.backtest.metrics import bootstrap_equity_ci, total_return
from trading.application.backtest.simulators import geometric_brownian_motion
from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.application.strategies.sma_crossover import SmaCrossover


async def main() -> None:
    bars = await SyntheticFetcher(seed=0).get_ohlcv("SYNTH", "1d", limit=500)
    cfg = BacktestConfig(initial_cash=100_000.0, fee_rate=0.001, slippage=0.0005)

    for strat in (BuyAndHold("SYNTH"), SmaCrossover("SYNTH", fast=20, slow=50)):
        r = await run_backtest(strat, bars, cfg)
        m = r.metrics
        print(f"=== {strat.name} ({m.n_periods} bars) ===")
        print(f"  total_return   {m.total_return:+.4f}")
        print(f"  annualized     {m.annualized_return:+.4f}")
        print(f"  sharpe         {m.sharpe:+.3f}")
        print(f"  sortino        {m.sortino:+.3f}")
        print(f"  max_drawdown   {m.max_drawdown:+.4f}")
        print(f"  var_95         {m.var_95:+.4f}")
        print(f"  cvar_95        {m.cvar_95:+.4f}")
        print(f"  win_rate       {m.win_rate:.3f}")
        print(f"  profit_factor  {m.profit_factor:.3f}")
        print(f"  n_trades       {m.n_trades}")
        lo, hi = bootstrap_equity_ci(total_return, r.equity_curve, n_boot=1000, block_size=5, seed=0)
        print(f"  total_return 95% CI: [{lo:+.4f}, {hi:+.4f}]")
        print()

    # Monte-Carlo GBM on the fitted log-returns.
    closes = np.array([b.close for b in bars])
    logret = np.diff(np.log(closes))
    mu, sigma = logret.mean(), logret.std(ddof=1)
    paths = geometric_brownian_motion(closes[-1], mu, sigma, n_steps=252, dt=1.0,
                                      n_paths=10_000, seed=0)
    total_returns = paths[:, -1] / closes[-1] - 1.0
    lo, hi = np.percentile(total_returns, [5, 95])
    print(f"=== Monte-Carlo GBM (10,000 paths, 252 steps) ===")
    print(f"  fitted mu={mu:.5f}  sigma={sigma:.5f}")
    print(f"  mean total return  {total_returns.mean():+.4f}")
    print(f"  90% interval       [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    asyncio.run(main())
