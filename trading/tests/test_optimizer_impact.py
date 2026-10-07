"""Tests for the optimizer's per-parameter impact ranking.

The console uses this to tell a user *which* knob matters: the ranking must be
orderable, the recommended value must be the best group, and a parameter whose
values all fail the objective gates must not be silently ranked first.
"""
from __future__ import annotations

from datetime import UTC

import pytest

from trading.application.backtest.optimize import (
    OptimizeCandidate,
    c_value,
    optimize_strategy,
    parameter_impact,
)


def _cand(params: dict, score: float, ret: float = 0.0, win: float = 0.0) -> OptimizeCandidate:
    return OptimizeCandidate(params=params, train_sharpe=0.0, validation_sharpe=0.0,
                             validation_return=ret, validation_trades=10,
                             validation_win_rate=win, score=score)


def test_impact_ranks_the_parameter_that_moves_the_score() -> None:
    cands = []
    # tp_percent swings the score hard; ema_fast barely matters
    for tp in (1.0, 3.0):
        for ema in (10, 20):
            cands.append(_cand({"tp_percent": tp, "ema_fast": ema},
                               score=0.30 if tp == 3.0 else 0.02,
                               ret=0.05 if tp == 3.0 else 0.0))
    rows = parameter_impact(cands, ["tp_percent", "ema_fast"])
    assert [r["param"] for r in rows] == ["tp_percent", "ema_fast"]
    assert rows[0]["recommended"] == 3.0
    assert rows[0]["direction"] == "higher is better"
    assert rows[0]["impact_share"] > rows[1]["impact_share"]
    assert rows[1]["impact"] == pytest.approx(0.0)


def test_disqualified_candidates_lower_valid_share_only() -> None:
    """A value that always fails the gates must not win on a lucky mean."""
    cands = [
        _cand({"trailing_percent": 0.5}, score=-1e9),
        _cand({"trailing_percent": 0.5}, score=-1e9),
        _cand({"trailing_percent": 2.0}, score=-1e9),
        _cand({"trailing_percent": 2.0}, score=0.10),
    ]
    rows = parameter_impact(cands, ["trailing_percent"])
    row = rows[0]
    assert row["recommended"] == 2.0
    by_val = {str(v["value"]): v for v in row["values"]}
    assert by_val["0.5"]["valid_share"] == 0.0
    assert by_val["2.0"]["valid_share"] == 0.5


def test_all_failing_returns_note_not_crash() -> None:
    cands = [_cand({"tp_percent": 1.0}, score=-1e9),
             _cand({"tp_percent": 3.0}, score=-1e9)]
    rows = parameter_impact(cands, ["tp_percent"])
    assert rows[0]["recommended"] is None
    assert rows[0]["note"]


def test_c_value_restores_types() -> None:
    assert c_value("2") == 2 and isinstance(c_value("2"), int)
    assert c_value("2.5") == 2.5
    assert c_value("True") is True and c_value("False") is False
    assert c_value("bonus") == "bonus"


def test_optimize_returns_impact_for_pine_defaults() -> None:
    import math
    from datetime import datetime, timedelta

    from trading.application.backtest.engine import BacktestConfig
    from trading.domain import Bar

    bars: list[Bar] = []
    price = 100.0
    t0 = datetime(2022, 1, 1, tzinfo=UTC)
    for i in range(320):
        ret = 0.0015 + 0.012 * math.sin(i / 9.0) - 0.006
        prev = price
        price = max(1.0, price * (1 + ret))
        bars.append(Bar(timestamp=t0 + timedelta(days=i), open=prev,
                        high=max(prev, price) * 1.004, low=min(prev, price) * 0.996,
                        close=price, volume=1000.0))
    res = optimize_strategy(
        "trend_confluence_pine", "SYNTH", bars,
        grid={"tp_percent": [1.5, 3.0], "trailing_percent": [0.8, 2.0]},
        cfg=BacktestConfig(), objective="profit_win",
    )
    assert res.impact, "impact ranking must be returned alongside the leaderboard"
    names = {r["param"] for r in res.impact}
    assert names == {"tp_percent", "trailing_percent"}
    assert all(0.0 <= r["impact_share"] <= 1.0 for r in res.impact)
    payload = res.as_dict()
    assert payload["impact"] == res.impact
    assert all("validation_win_rate" in c for c in payload["leaderboard"])
