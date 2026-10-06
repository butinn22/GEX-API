"""Tests for the redesigned optimizer: objectives, nested grids, global runs."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

import pytest

from trading.application.backtest.engine import BacktestConfig
from trading.application.backtest.metrics import compute_metrics
from trading.application.backtest.optimize import (
    OPTIMIZATION_OBJECTIVES,
    UNIFIED_GRID,
    metric_value,
    optimize_strategy,
)
from trading.application.backtest.portfolio import TickerSpec
from trading.application.cancellation import run_registry
from trading.application.global_optimize import GlobalOptimizeRunner
from trading.application.presets import PresetService
from trading.domain import Bar


def _trending_bars(n: int = 400, *, start: float = 100.0, drift: float = 0.0015) -> list[Bar]:
    bars: list[Bar] = []
    price = start
    t0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ret = drift + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        bars.append(Bar(timestamp=t0 + timedelta(days=i), open=prev,
                        high=max(prev, price) * 1.004, low=min(prev, price) * 0.996,
                        close=price, volume=1000.0))
    return bars


# ── objectives ─────────────────────────────────────────────────────────


def test_metric_value_maps_objectives() -> None:
    equity = [100.0 * (1 + 0.001) ** i for i in range(300)]
    m = compute_metrics(equity)
    assert metric_value(m, "sharpe") == m.sharpe
    assert metric_value(m, "total_return") == m.total_return
    assert metric_value(m, "max_drawdown") == -m.max_drawdown  # maximised
    with pytest.raises(ValueError):
        metric_value(m, "nonsense")
    assert set(OPTIMIZATION_OBJECTIVES) >= {
        "sharpe", "sortino", "calmar", "total_return",
        "profit_factor", "win_rate", "max_drawdown",
    }


def test_optimize_unknown_objective_rejected() -> None:
    bars = _trending_bars(150)
    with pytest.raises(ValueError):
        optimize_strategy("trend_confluence", "SYNTH", bars, objective="nonsense")


# ── sweep guards: empty grid, runaway grid, trendline cadence clamp ────


def test_optimize_empty_grid_rejected_not_silently_defaulted() -> None:
    """An explicit empty grid must 400, not silently run the 64-combo default.

    Before the fix ``grid or _default_grid(name)`` treated ``{}`` as falsy and
    a request with ``grid={}`` (or all-empty candidate lists) kicked off the
    full default sweep — one of the "looks like a hang" reports.
    """
    bars = _trending_bars(150)
    with pytest.raises(ValueError, match="empty sweep grid"):
        optimize_strategy("trend_confluence", "SYNTH", bars, grid={})
    with pytest.raises(ValueError, match="empty sweep grid"):
        optimize_strategy("trend_confluence", "SYNTH", bars,
                          grid={"zone_atr": [], "min_confluence": []})


def test_optimize_runaway_grid_rejected() -> None:
    """Grids beyond MAX_GRID_COMBOS are refused before any backtest runs."""
    bars = _trending_bars(150)
    runaway = {f"axis_{i}": [1, 2] for i in range(10)}  # 2**10 = 1024 combos
    with pytest.raises(ValueError, match="grid too large"):
        optimize_strategy("trend_confluence", "SYNTH", bars, grid=runaway)


def test_optimize_sweep_clamps_dense_trendline_refresh_but_winner_keeps_user_value() -> None:
    """Sweep legs run at >= SWEEP_MIN_REFRESH; the final winner keeps the user's.

    The Pine console always sends ``trendline_refresh`` (default 5, min 1);
    the old ``base.setdefault("trendline_refresh", 10)`` was a no-op for it,
    so sweeps ran up to ~6× slower than designed and looked frozen.
    """
    bars = _trending_bars(150)
    res = optimize_strategy(
        "trend_confluence", "SYNTH", bars,
        base_params={"trendline_refresh": 1},
        grid={"zone_atr": [0.3, 0.5]},
    )
    from trading.application.backtest.optimize import SWEEP_MIN_REFRESH
    assert res.baseline["params"]["trendline_refresh"] == SWEEP_MIN_REFRESH
    assert res.best_params["trendline_refresh"] == 1
    # When the caller pins a cadence at or above the floor it is honored everywhere.
    res2 = optimize_strategy(
        "trend_confluence", "SYNTH", bars,
        base_params={"trendline_refresh": 20},
        grid={"zone_atr": [0.3, 0.5]},
    )
    assert res2.baseline["params"]["trendline_refresh"] == 20
    assert res2.best_params["trendline_refresh"] == 20


def test_optimize_alternate_objective_ranks_differently_but_runs() -> None:
    bars = _trending_bars(300)
    res = optimize_strategy(
        "trend_confluence", "SYNTH", bars,
        grid={"zone_atr": [0.3, 0.5]}, objective="total_return",
    )
    assert res.n_candidates == 2
    assert res.best["objective"] == "total_return"
    assert res.best_params["zone_atr"] in (0.3, 0.5)


# ── unified strategy optimization ──────────────────────────────────────


def test_optimize_unified_with_nested_emf_grid() -> None:
    bars = _trending_bars(300)
    res = optimize_strategy(
        "trend_confluence_unified", "SYNTH", bars,
        grid={
            "zone_atr": [0.5, 0.8],
            "emf_mode": ["require", "bonus"],
            "emf.atr_tp_mult": [1.5, 2.5],
        },
        cfg=BacktestConfig(),
    )
    assert res.n_candidates == 8
    # nested grid key must land inside the emf block of the best params
    assert res.best_params["emf"]["atr_tp_mult"] in (1.5, 2.5)
    assert res.best_params["emf_mode"] in ("require", "bonus")
    # leaderboard entries record the dotted key as given
    assert set(res.leaderboard[0].params) == {"zone_atr", "emf_mode", "emf.atr_tp_mult"}


def test_unified_default_grid_is_small_and_covers_all_three() -> None:
    n = 1
    for values in UNIFIED_GRID.values():
        n *= len(values)
    assert n <= 64  # interactive default
    keys = set(UNIFIED_GRID)
    assert {"zone_atr", "min_confluence", "atr_trail_mult"} <= keys  # TC core
    assert "emf_mode" in keys and any(k.startswith("emf.") for k in keys)  # EMF+ADL
    assert "momentum_mode" in keys  # momentum


def test_optimize_deterministic() -> None:
    bars = _trending_bars(250)
    a = optimize_strategy("trend_confluence", "SYNTH", bars,
                           grid={"zone_atr": [0.3, 0.5], "min_confluence": [2, 3]})
    b = optimize_strategy("trend_confluence", "SYNTH", bars,
                          grid={"zone_atr": [0.3, 0.5], "min_confluence": [2, 3]})
    assert a.best_params == b.best_params
    assert [c.score for c in a.leaderboard] == [c.score for c in b.leaderboard]


# ── global optimization runner ────────────────────────────────────────


def test_resolve_symbols_dedupes_and_uppercases() -> None:
    out = GlobalOptimizeRunner.resolve_symbols(
        ["nvda", "NVDA", " btc ", ""], "all", 0
    )
    assert out == ["NVDA", "BTC"]
    # all-blank symbols → invalid
    with pytest.raises(ValueError):
        GlobalOptimizeRunner.resolve_symbols(["  "], "all", 0)
    # empty list → the whole category universe
    universe = GlobalOptimizeRunner.resolve_symbols([], "us", 0)
    assert universe and all(isinstance(s, str) for s in universe)


async def test_global_run_completes_and_saves_presets(tmp_path):
    from trading.adapters.persistence import database as db

    await db.init_db()
    runner = GlobalOptimizeRunner()
    symbols = ["AAA", "BBB"]
    bars = {s: _trending_bars(220, drift=0.002 if s == "AAA" else 0.001)
            for s in symbols}
    token = run_registry.new("test-global-1")

    async with db._session_factory() as s:
        from sqlalchemy import delete
        from trading.adapters.persistence.models import StrategyPresetRow

        await s.execute(delete(StrategyPresetRow))
        await s.commit()

    state = await runner.start(
        run_id=token.token,
        symbols=symbols,
        strategy="trend_confluence_unified",
        base_params={},
        grid={"zone_atr": [0.5, 0.8]},
        objective="sharpe",
        source="auto",
        timeframe="1d",
        limit=300,
        cfg=BacktestConfig(),
        refresh=False,
        save_preset=True,
        session_factory=db._session_factory,
        cancel=token,
        bars_by_symbol=bars,
    )
    # wait for the background job
    for _ in range(600):
        await asyncio.sleep(0.05)
        if state.state != "running":
            break
    assert state.state == "done"
    assert state.completed == 2 and state.failed == 0
    assert {r["symbol"] for r in state.results} == {"AAA", "BBB"}

    async with db._session_factory() as s:
        svc = PresetService(s)
        for sym in symbols:
            row = await svc.get_default(sym, "trend_confluence_unified")
            assert row is not None, f"preset missing for {sym}"
            assert row.source == "optimizer"
            assert row.optimizer_run_id == "test-global-1"
    assert runner.status("test-global-1") is state
    assert runner.status("nope") is None


async def test_global_run_isolates_failures(tmp_path):
    from trading.adapters.persistence import database as db

    await db.init_db()
    runner = GlobalOptimizeRunner()
    # BAD has too few bars → optimize_strategy raises ValueError, isolated.
    bars = {"GOOD": _trending_bars(220), "BAD": _trending_bars(50)}
    token = run_registry.new("test-global-2")
    state = await runner.start(
        run_id=token.token,
        symbols=["GOOD", "BAD"],
        strategy="trend_confluence_unified",
        base_params={},
        grid={"zone_atr": [0.5, 0.8]},
        objective="sharpe",
        source="auto",
        timeframe="1d",
        limit=300,
        cfg=BacktestConfig(),
        refresh=False,
        save_preset=False,
        session_factory=db._session_factory,
        cancel=token,
        bars_by_symbol=bars,
    )
    for _ in range(600):
        await asyncio.sleep(0.05)
        if state.state != "running":
            break
    assert state.state == "done"  # the failure did not abort the run
    assert state.completed == 1 and state.failed == 1
    assert state.errors and state.errors[0]["symbol"] == "BAD"
