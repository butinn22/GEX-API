"""Tests for the multi-ticker portfolio backtest engine."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pytest

from trading.application.backtest.portfolio import (
    PortfolioBacktestConfig,
    TickerSpec,
    load_bars,
    resolve_exchanges,
    run_portfolio_backtest,
)
from trading.domain import Bar, DataFetchError, Exchange


def _bars(symbol: str, n: int = 200, drift: float = 0.001, seed: int = 0) -> list[Bar]:
    """Deterministic geometric random walk of bars."""
    rng = np.random.default_rng(seed)
    price = 100.0
    out: list[Bar] = []
    t0 = datetime(2024, 1, 1)
    for i in range(n):
        price *= 1.0 + drift + rng.normal(0, 0.01)
        out.append(
            Bar(timestamp=t0 + timedelta(days=i), open=price * 0.999, high=price * 1.01,
                low=price * 0.99, close=price, volume=1000.0)
        )
    return out


def _run(specs, *, cfg=None, bars=None):
    return asyncio.run(run_portfolio_backtest(specs, cfg, bars_by_symbol=bars))


# ── data resolution ────────────────────────────────────────────────────


def test_resolve_exchanges_auto_uses_listing_exchange():
    exchanges, sym = resolve_exchanges("SBER", "auto")  # MOEX ticker
    assert exchanges == [Exchange.MOEX] and sym == "SBER"


def test_resolve_exchanges_crypto_maps_to_bybit():
    exchanges, sym = resolve_exchanges("BTC", "auto")
    assert exchanges == [Exchange.BYBIT] and "USDT" in sym


def test_resolve_exchanges_explicit_source():
    exchanges, _ = resolve_exchanges("AAPL", "yfinance")
    assert Exchange.YFINANCE in exchanges


def test_resolve_exchanges_unknown_source_raises():
    with pytest.raises(DataFetchError, match="unknown source"):
        resolve_exchanges("AAPL", "nope")


def test_resolve_exchanges_synthetic_is_none():
    exchanges, sym = resolve_exchanges("SYNTH", "synthetic")
    assert exchanges is None and sym == "SYNTH"


def test_load_bars_synthetic_symbol_is_offline_and_deterministic():
    spec = TickerSpec(symbol="SYNTH", source="auto", limit=120)
    a = asyncio.run(load_bars(spec))
    b = asyncio.run(load_bars(spec))
    assert len(a) == 120
    assert [x.close for x in a] == [x.close for x in b]  # deterministic


# ── basic engine behaviour ─────────────────────────────────────────────


def test_single_ticker_portfolio_matches_standalone_backtest():
    from trading.application.backtest.engine import BacktestConfig, run_backtest
    from trading.application.strategy_factory import build_strategy

    bars = _bars("A", 250, seed=1)
    res = _run(
        [TickerSpec("A", strategy="sma_crossover", params={"fast": 5, "slow": 20}, source="synthetic")],
        bars={"A": bars},
    )
    single = asyncio.run(
        run_backtest(build_strategy("sma_crossover", "A", {"fast": 5, "slow": 20}), bars,
                     BacktestConfig(initial_cash=100_000.0))
    )
    assert res.n_tickers == 1
    np.testing.assert_allclose(res.equity_curve, single.equity_curve)
    assert res.metrics.total_return == pytest.approx(single.metrics.total_return)


def test_curve_starts_at_initial_cash():
    bars = {"A": _bars("A", 150, seed=2), "B": _bars("B", 150, seed=3)}
    res = _run(
        [TickerSpec("A", source="synthetic", weight=1), TickerSpec("B", source="synthetic", weight=1)],
        bars=bars,
    )
    assert res.equity_curve[0] == pytest.approx(100_000.0, rel=1e-6)


def test_weights_split_capital():
    bars = {"A": _bars("A", 120, seed=4), "B": _bars("B", 120, seed=5)}
    res = _run(
        [TickerSpec("A", source="synthetic", weight=3), TickerSpec("B", source="synthetic", weight=1)],
        bars=bars,
    )
    caps = {t.symbol: t.capital for t in res.tickers}
    assert caps["A"] == pytest.approx(75_000.0)
    assert caps["B"] == pytest.approx(25_000.0)


def test_explicit_capital_overrides_weight():
    bars = {"A": _bars("A", 120, seed=6), "B": _bars("B", 120, seed=7)}
    res = _run(
        [TickerSpec("A", source="synthetic", capital=10_000.0, weight=99),
         TickerSpec("B", source="synthetic", weight=1)],
        bars=bars,
    )
    caps = {t.symbol: t.capital for t in res.tickers}
    assert caps["A"] == pytest.approx(10_000.0)
    assert caps["B"] == pytest.approx(90_000.0)


def test_per_ticker_individual_settings_are_honoured():
    bars = {"A": _bars("A", 200, seed=8), "B": _bars("B", 200, seed=9)}
    res = _run(
        [
            TickerSpec("A", strategy="sma_crossover", params={"fast": 5, "slow": 10}, source="synthetic"),
            TickerSpec("B", strategy="momentum", params={"period": 30}, source="synthetic"),
        ],
        bars=bars,
    )
    by_sym = {t.symbol: t for t in res.tickers}
    assert by_sym["A"].strategy == "sma_crossover"
    assert by_sym["B"].strategy == "momentum"
    # each leg's capital is its own sub-account, starting at 50k
    assert by_sym["A"].equity_curve[0] == pytest.approx(50_000.0)


def test_alignment_forward_fills_uneven_time_grids():
    a = _bars("A", 100, seed=10)
    b = _bars("B", 100, seed=11)[::2]  # half as many bars (same span)
    res = _run(
        [TickerSpec("A", source="synthetic"), TickerSpec("B", source="synthetic")],
        bars={"A": a, "B": b},
    )
    assert len(res.equity_curve) == 100  # union == the longer grid
    assert len(res.aligned_equity) == 2
    assert all(len(c) == 100 for c in res.aligned_equity)


def test_combined_curve_is_sum_of_aligned_legs_plus_cash():
    bars = {"A": _bars("A", 140, seed=12), "B": _bars("B", 140, seed=13)}
    res = _run(
        [TickerSpec("A", source="synthetic", weight=1), TickerSpec("B", source="synthetic", weight=1)],
        bars=bars,
    )
    total = np.sum(np.vstack(res.aligned_equity), axis=0)
    np.testing.assert_allclose(res.equity_curve, total)


def test_disabled_tickers_are_skipped():
    bars = {"A": _bars("A", 120, seed=14)}
    res = _run(
        [TickerSpec("A", source="synthetic"), TickerSpec("B", source="synthetic", enabled=False)],
        bars=bars,
    )
    assert res.n_tickers == 1
    assert {t.symbol for t in res.tickers} == {"A"}


def test_no_enabled_tickers_raises():
    with pytest.raises(ValueError, match="at least one enabled"):
        _run([TickerSpec("A", enabled=False)], bars={"A": _bars("A")})


# ── error isolation ────────────────────────────────────────────────────


def test_failed_ticker_is_isolated_and_reported():
    def _boom(spec):
        raise DataFetchError("exchange down")

    bars = {"A": _bars("A", 120, seed=15)}
    res = _run(
        [TickerSpec("A", source="synthetic"), TickerSpec("MISSING", source="nope")],
        bars=bars,
    )
    # 'MISSING' fails (unknown source) but 'A' still produces a portfolio
    assert res.n_tickers == 1
    assert any(e["symbol"] == "MISSING" for e in res.errors)


def test_bad_strategy_params_isolated_per_ticker():
    bars = {"A": _bars("A", 150, seed=16), "B": _bars("B", 150, seed=17)}
    res = _run(
        [
            TickerSpec("A", source="synthetic"),
            TickerSpec("B", strategy="sma_crossover", params={"fast": 50, "slow": 10}, source="synthetic"),
        ],
        bars=bars,
    )
    assert res.n_tickers == 1
    assert any(e["symbol"] == "B" for e in res.errors)


def test_too_few_bars_is_reported_not_raised():
    res = _run(
        [TickerSpec("A", source="synthetic"), TickerSpec("B", source="synthetic")],
        bars={"A": _bars("A", 120, seed=18), "B": _bars("B", 1, seed=19)},
    )
    assert res.n_tickers == 1
    assert any(e["symbol"] == "B" and "bars" in e["error"] for e in res.errors)


def test_failed_capital_stays_as_cash():
    res = _run(
        [TickerSpec("A", source="synthetic", weight=1), TickerSpec("Z", source="nope", weight=1)],
        bars={"A": _bars("A", 120, seed=20)},
    )
    # A gets 50k; Z's 50k remains as cash → the curve still starts at 100k
    assert res.tickers[0].capital == pytest.approx(50_000.0)
    assert res.equity_curve[0] == pytest.approx(100_000.0)


def test_all_tickers_failing_raises():
    with pytest.raises(DataFetchError, match="no ticker produced a result"):
        _run([TickerSpec("Z", source="nope"), TickerSpec("Y", source="bogus")], bars={})


# ── correlation ────────────────────────────────────────────────────────


def test_correlation_matrix_computed_for_multiple_tickers():
    bars = {
        "A": _bars("A", 200, seed=30),
        "B": _bars("B", 200, seed=31),
        "C": _bars("C", 200, seed=32),
    }
    res = _run(
        [TickerSpec(s, source="synthetic") for s in ("A", "B", "C")], bars=bars
    )
    corr = res.correlation
    assert corr is not None
    assert list(corr.symbols) == ["A", "B", "C"]
    for i, row in enumerate(corr.matrix):
        assert row[i] == pytest.approx(1.0, abs=1e-9)  # diagonal
        assert all(-1.0 <= x <= 1.0 + 1e-9 for x in row)
    # symmetric
    for i in range(3):
        for j in range(3):
            assert corr.matrix[i][j] == pytest.approx(corr.matrix[j][i], abs=1e-9)


def test_correlation_none_for_single_ticker():
    res = _run([TickerSpec("A", source="synthetic")], bars={"A": _bars("A", 150, seed=33)})
    assert res.correlation is None


def test_identical_curves_are_highly_correlated():
    bars = _bars("A", 200, seed=34)
    res = _run(
        [TickerSpec("A", source="synthetic"), TickerSpec("A2", source="synthetic")],
        bars={"A": bars, "A2": bars},
    )
    corr = res.correlation
    assert corr is not None
    assert abs(corr.matrix[0][1]) > 0.99


# ── metrics ────────────────────────────────────────────────────────────


def test_metrics_are_finite_and_consistent():
    bars = {"A": _bars("A", 260, seed=40), "B": _bars("B", 260, seed=41)}
    res = _run(
        [TickerSpec("A", source="synthetic"), TickerSpec("B", source="synthetic")], bars=bars
    )
    m = res.metrics
    assert m.n_periods == len(res.equity_curve) - 1
    assert np.isfinite([m.sharpe, m.max_drawdown, m.var_95, m.cvar_95, m.win_rate]).all()
    assert 0.0 <= m.max_drawdown <= 1.0
    assert m.cvar_95 <= m.var_95  # CVaR (tail mean) is at or below the VaR percentile
    assert res.initial_cash == pytest.approx(100_000.0)


def test_config_fees_and_slippage_reduce_returns():
    bars = {"A": _bars("A", 250, seed=42, drift=0.002)}
    cheap = _run([TickerSpec("A", source="synthetic")], bars=bars,
                 cfg=PortfolioBacktestConfig(fee_rate=0.0, slippage=0.0))
    costly = _run([TickerSpec("A", source="synthetic")], bars=bars,
                  cfg=PortfolioBacktestConfig(fee_rate=0.02, slippage=0.01))
    assert costly.metrics.total_return < cheap.metrics.total_return


def test_per_ticker_return_accessor():
    bars = {"A": _bars("A", 200, seed=43), "B": _bars("B", 200, seed=44)}
    res = _run([TickerSpec("A", source="synthetic"), TickerSpec("B", source="synthetic")], bars=bars)
    assert set(res.per_ticker_return) == {"A", "B"}
    for t in res.tickers:
        assert res.per_ticker_return[t.symbol] == pytest.approx(t.total_return)


def test_weight_property_reflects_capital_share():
    bars = {"A": _bars("A", 150, seed=45), "B": _bars("B", 150, seed=46)}
    res = _run(
        [TickerSpec("A", source="synthetic", weight=1), TickerSpec("B", source="synthetic", weight=3)],
        bars=bars,
    )
    w = {t.symbol: t.weight for t in res.tickers}
    assert w["A"] == pytest.approx(0.25)
    assert w["B"] == pytest.approx(0.75)


def test_many_tickers_run():
    """The 'choose N tickers' path: 8 legs, each with its own settings."""
    syms = [f"S{i}" for i in range(8)]
    bars = {s: _bars(s, 150, seed=50 + i) for i, s in enumerate(syms)}
    strategies = ["sma_crossover", "momentum", "mean_reversion", "buy_and_hold"]
    specs = [
        TickerSpec(s, strategy=strategies[i % len(strategies)], source="synthetic")
        for i, s in enumerate(syms)
    ]
    res = _run(specs, bars=bars)
    assert res.n_tickers == 8
    assert len(res.aligned_equity) == 8
    assert np.isfinite(res.equity_curve).all()
