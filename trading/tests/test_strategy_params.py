"""Tests for the editable-parameter schema served to the console.

The schema is the contract between the strategy and the settings UI: every key
must be a parameter the strategy actually accepts, and every default must match
the strategy's own default — otherwise the console would silently run a
different configuration than it shows.
"""
from __future__ import annotations

import pytest

from trading.application.strategy_params import (
    GROUPS,
    defaults_for,
    grid_for,
    schema_for,
)
from trading.application.strategies.trend_confluence_pine import (
    PINE_PARAM_NAMES,
    PineConfluenceParams,
)
from gex.strategy.settings import StrategySettings

PINE = "trend_confluence_pine"


def test_unknown_strategy_has_empty_schema() -> None:
    assert schema_for("nope") == []
    assert defaults_for("nope") == {}
    assert grid_for("nope") == {}


def test_every_key_is_a_real_strategy_parameter() -> None:
    allowed_flat = set(PINE_PARAM_NAMES)
    emf_fields = set(StrategySettings().__dict__)
    for p in schema_for(PINE):
        if p["key"].startswith("emf."):
            assert p["key"].split(".", 1)[1] in emf_fields, p["key"]
        else:
            assert p["key"] in allowed_flat, p["key"]


def test_defaults_match_the_strategy() -> None:
    strat = PineConfluenceParams.from_dict(None)
    settings = StrategySettings()
    for p in schema_for(PINE):
        if p["key"].startswith("emf."):
            actual = getattr(settings, p["key"].split(".", 1)[1])
        else:
            actual = getattr(strat, p["key"])
        assert actual == p["default"], f"{p['key']}: schema {p['default']} vs {actual}"


def test_groups_and_kinds_are_known() -> None:
    params = schema_for(PINE)
    assert {p["group"] for p in params} <= set(GROUPS)
    assert {p["kind"] for p in params} <= {"int", "float", "bool", "enum"}
    # the five families the console shows must all be populated
    assert {p["group"] for p in params} == set(GROUPS)


def test_numeric_bounds_are_sane() -> None:
    for p in schema_for(PINE):
        if p["kind"] in ("int", "float"):
            assert p["min"] is not None and p["max"] is not None, p["key"]
            assert p["min"] <= p["default"] <= p["max"], p["key"]


def test_sweep_candidates_exist_and_build_a_grid() -> None:
    grid = grid_for(PINE)
    assert grid, "the console needs at least one sweepable parameter"
    # every sweepable param must accept its own candidates
    PineConfluenceParams.from_dict({k: v[0] for k, v in grid.items()})
    assert all(isinstance(v, list) and v for v in grid.values())


def test_strategy_accepts_schema_defaults() -> None:
    """Re-feeding the schema defaults must reproduce the same strategy."""
    from trading.application.strategy_factory import build_strategy

    nested = {}
    for k, v in defaults_for(PINE).items():
        if k.startswith("emf."):
            nested.setdefault("emf", {})[k.split(".", 1)[1]] = v
        else:
            nested[k] = v
    strat = build_strategy(PINE, "SYNTH", nested)
    assert strat._pp.tp_percent == 2.0
    assert strat._pp.trailing_percent == 1.0
    assert strat._emf_settings.damping == 0.9
