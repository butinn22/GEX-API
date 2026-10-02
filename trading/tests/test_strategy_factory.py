"""Tests for the strategy factory (one builder for every registered strategy)."""
from __future__ import annotations

import pytest

from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.application.strategies.dual_sma_crossover import DualSmaCrossover
from trading.application.strategies.gex_emf import GexEMFStrategy
from trading.application.strategies.mean_reversion import MeanReversion
from trading.application.strategies.momentum import Momentum
from trading.application.strategies.sma_crossover import SmaCrossover
from trading.application.strategy_factory import STRATEGY_NAMES, build_strategy
from trading.domain import StrategyError


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sma_crossover", SmaCrossover),
        ("sma_crossover_ls", DualSmaCrossover),
        ("buy_and_hold", BuyAndHold),
        ("mean_reversion", MeanReversion),
        ("momentum", Momentum),
        ("gex_emf", GexEMFStrategy),
    ],
)
def test_builds_every_registered_strategy(name, expected):
    assert build_strategy(name, "SYNTH", {}) is not None
    assert isinstance(build_strategy(name, "SYNTH", {}), expected)


def test_strategy_names_cover_the_registry():
    from trading.application.strategy_registry import STRATEGY_REGISTRY

    assert set(STRATEGY_NAMES) == set(STRATEGY_REGISTRY.names())


def test_unknown_strategy_raises_domain_error():
    with pytest.raises(StrategyError, match="unknown strategy"):
        build_strategy("nope", "X", {})


# ── sma_crossover ──────────────────────────────────────────────────────


def test_sma_crossover_params():
    s = build_strategy("sma_crossover", "X", {"fast": 7, "slow": 21})
    assert (s.fast, s.slow) == (7, 21)


def test_sma_crossover_rejects_fast_ge_slow():
    with pytest.raises(StrategyError, match="fast < slow"):
        build_strategy("sma_crossover", "X", {"fast": 30, "slow": 10})


def test_sma_crossover_bad_type_is_domain_error():
    with pytest.raises(StrategyError, match="must be an integer"):
        build_strategy("sma_crossover", "X", {"fast": "abc", "slow": 20})


# ── dual (per-side) ────────────────────────────────────────────────────


def test_dual_flat_params_and_defaults():
    s = build_strategy("sma_crossover_ls", "X", {})
    assert (s.long_fast, s.long_slow) == (10, 20)
    assert (s.short_fast, s.short_slow) == (10, 20)
    assert s.long_enabled and s.short_enabled


def test_dual_nested_long_short_blocks():
    s = build_strategy(
        "sma_crossover_ls", "X",
        {
            "long": {"enabled": True, "fast": 5, "slow": 15, "strength": 0.5},
            "short": {"enabled": False, "fast": 8, "slow": 21, "strength": 0.25},
        },
    )
    assert (s.long_fast, s.long_slow, s.long_strength) == (5, 15, 0.5)
    assert (s.short_fast, s.short_slow, s.short_strength) == (8, 21, 0.25)
    assert s.long_enabled is True
    assert s.short_enabled is False


def test_dual_accepts_flat_long_fast_keys():
    s = build_strategy("sma_crossover_ls", "X", {"long_fast": 4, "long_slow": 9,
                                                 "short_fast": 6, "short_slow": 12})
    assert (s.long_fast, s.long_slow) == (4, 9)
    assert (s.short_fast, s.short_slow) == (6, 12)


def test_dual_rejects_bad_windows():
    with pytest.raises(Exception):  # StrategyError or the ctor's ValueError
        build_strategy("sma_crossover_ls", "X", {"long": {"fast": 30, "slow": 10},
                                                 "short": {"fast": 8, "slow": 21}})


def test_dual_strength_must_be_positive():
    with pytest.raises(StrategyError, match="strength"):
        build_strategy("sma_crossover_ls", "X", {"long": {"strength": 0}})


# ── period-based ───────────────────────────────────────────────────────


def test_period_strategies():
    assert build_strategy("momentum", "X", {"period": 5}).period == 5
    assert build_strategy("mean_reversion", "X", {"period": 30}).period == 30


def test_period_must_be_at_least_two():
    with pytest.raises(StrategyError, match="period"):
        build_strategy("momentum", "X", {"period": 0})


def test_buy_and_hold_ignores_params():
    assert isinstance(build_strategy("buy_and_hold", "X", {"junk": 1}), BuyAndHold)


# ── gex_emf settings ───────────────────────────────────────────────────


def test_gex_emf_accepts_nested_settings():
    s = build_strategy("gex_emf", "X", {"settings": {"atr_length": 20, "length_adl": 10}})
    assert isinstance(s, GexEMFStrategy)
    assert s._settings.atr_length == 20
    assert s._settings.length_adl == 10


def test_gex_emf_accepts_flat_settings():
    s = build_strategy("gex_emf", "X", {"atr_length": 33, "length_adl": 11})
    assert s._settings.atr_length == 33
    assert s._settings.length_adl == 11


def test_gex_emf_with_no_settings_uses_defaults():
    s = build_strategy("gex_emf", "X", {})
    assert isinstance(s, GexEMFStrategy)


def test_bool_coercion_from_strings():
    s = build_strategy("sma_crossover_ls", "X", {"long": {"enabled": "false"},
                                                 "short": {"enabled": "true"}})
    assert s.long_enabled is False
    assert s.short_enabled is True
