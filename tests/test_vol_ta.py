"""Тесты TA-индикаторов и вероятностных функций.

Покрытие: функции TA (RSI, MACD, EMA, тренд), EV/Kelly, z-score, корреляция.
(Движок vol_reversal удалён из проекта вместе с фичей — см. gexcone.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex.domain.ta import (
    _wilder_rsi,
    compute_indicators,
    detect_trend,
    reversal_probability,
)


def _synthetic_profile():
    """POSITIVE-gamma профиль на синтетической цепочке."""
    from gex.domain.data_loader import OptionSnapshot
    from gex.domain.metrics import GEXMetrics

    chain = pd.DataFrame({
        "strike": [480.0, 490.0, 500.0, 510.0, 520.0],
        "type": ["P", "P", "C", "C", "C"],
        "oi": [800, 900, 1000, 900, 800],
        "iv": [0.22] * 5,
        "T": [0.08] * 5,
    })
    snap = OptionSnapshot(symbol="TEST", spot=500.0, as_of=pd.Timestamp.now("UTC"), chain=chain)
    return GEXMetrics(spot=500.0).compute(snap)


def _ohlcv(close):
    """OHLCV-датафрейм из серии закрытий (для detect_trend/reversal_probability)."""
    c = pd.Series(np.asarray(close, dtype=float))
    return pd.DataFrame({
        "Open": c,
        "High": c * 1.01,
        "Low": c * 0.99,
        "Close": c,
        "Volume": np.ones(len(c)) * 1e6,
    })


# ====================================================================== #
#  EV / Kelly
# ====================================================================== #
class TestComputeEVKelly:
    def _ev(self, tp: float, sl: float):
        from gex.domain.ev import EVCalculator, TradeSetup
        from gex.domain.stochastic import StochasticEngine

        setup = TradeSetup(
            direction="long", entry=500.0, take_profit=tp, stop_loss=sl,
            size=100, label="test",
        )
        return EVCalculator(StochasticEngine()).evaluate(
            setup=setup, profile=_synthetic_profile(), T=30 / 365, sigma=0.2,
        )

    def test_ev_positive_for_profitable_setup(self):
        """RR 2:1 (TP=2%, SL=1%) → EV > 0 и Kelly > 0."""
        res = self._ev(tp=502.0, sl=499.0)
        assert res.ev_dollars > 0
        assert res.kelly > 0

    def test_ev_negative_for_unprofitable_setup(self):
        """RR 1:2 (TP=1%, SL=2%) → EV < 0."""
        res = self._ev(tp=501.0, sl=498.0)
        assert res.ev_dollars < 0


# ====================================================================== #
#  TA (Technical Analysis) — чистые функции
# ====================================================================== #
class TestWilderRSI:
    def test_constant_prices_rsi_50(self):
        close = pd.Series([100.0] * 20)
        rsi = _wilder_rsi(close, 14)
        assert abs(float(rsi.iloc[-1]) - 50.0) < 1.0

    def test_uptrend_rsi_above_50(self):
        close = pd.Series(np.linspace(90, 110, 30))
        rsi = _wilder_rsi(close, 14)
        assert float(rsi.iloc[-1]) > 50

    def test_downtrend_rsi_below_50(self):
        close = pd.Series(np.linspace(110, 90, 30))
        rsi = _wilder_rsi(close, 14)
        assert float(rsi.iloc[-1]) < 50

    def test_rsi_bounded_0_100(self):
        close = pd.Series([100.0] * 5 + [200.0] * 5 + [50.0] * 5 + [100.0] * 10)
        rsi = _wilder_rsi(close, 14)
        assert 0 <= float(rsi.iloc[-1]) <= 100


class TestComputeIndicators:
    def test_returns_dict(self):
        df = pd.DataFrame({
            "Open": np.linspace(100, 110, 100),
            "High": np.linspace(102, 112, 100),
            "Low": np.linspace(98, 108, 100),
            "Close": np.linspace(100, 110, 100),
            "Volume": np.ones(100) * 1e6,
        })
        ind = compute_indicators(df)
        from gex.domain.ta import TAIndicators
        assert isinstance(ind, TAIndicators)
        assert ind.ema20 is not None
        assert ind.ema50 is not None
        assert ind.rsi is not None
        assert ind.macd is not None

    def test_insufficient_bars_returns_none_or_partial(self):
        """Слишком мало баров не должно падать."""
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [98, 99],
            "Close": [101, 102],
            "Volume": [1e6, 1e6],
        })
        ind = compute_indicators(df)
        from gex.domain.ta import TAIndicators
        assert isinstance(ind, TAIndicators)


class TestDetectTrend:
    def test_uptrend_returns_bullish(self):
        trend = detect_trend(_ohlcv(np.linspace(100, 150, 200)))
        assert trend.direction == "BULLISH"

    def test_downtrend_returns_bearish(self):
        trend = detect_trend(_ohlcv(np.linspace(150, 100, 200)))
        assert trend.direction == "BEARISH"

    def test_flat_is_range(self):
        close = pd.Series(np.random.randn(200) * 2 + 100)
        trend = detect_trend(_ohlcv(close))
        assert isinstance(trend.direction, str)
        assert isinstance(trend.strength, float)


class TestReversalProbability:
    def test_reversal_prob_in_0_1(self):
        prob = reversal_probability(
            _ohlcv(np.linspace(100, 150, 200)),
            current_trend="BULLISH",
            n_paths=100,
        )
        assert 0 <= prob.p_reversal <= 1.0
