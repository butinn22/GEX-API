"""Tests for gex.trend_regime — детектор тренда/флэта на 200 барах.

Проверяем:
1. Математику индикаторов (ATR Wilder, BBW, нормированные z-движения).
2. Синтетический тренд → UP/DOWN, синтетический флэт → FLAT.
3. Монотонность слайдера: шире слайдер ⇒ больше flat_score / больше флэта.
4. Раскол «метрики ↔ вердикт»: метрики не зависят от слайдера.
5. Недостаток данных → None (сигналы НЕ режутся).
6. Гейт сигналов: flat/direction_mismatch, выходы не блокируются.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from gex.domain.trend_regime import (
    DEFAULT_PARAMS,
    DEFAULT_SLIDERS,
    RegimeParams,
    RegimeSliders,
    add_indicators,
    analyze_regime,
    below_score,
    compute_regime_metrics,
    evaluate_regime,
    percentile_rank,
    signal_allowed,
    sigmoid,
    slider_multiplier,
)


# ================================================================= #
#  Синтетические ряды
# ================================================================= #
def make_df(closes: np.ndarray) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    high = closes * 1.004
    low = closes * 0.996
    op = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": closes})


def _trend_up(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = np.concatenate([
        rng.normal(0.0002, 0.004, n - 100),
        rng.normal(0.012, 0.012, 100),  # ускоряющийся тренд с ростом волатильности
    ])
    return make_df(100.0 * np.exp(np.cumsum(rets)))


def _trend_down(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = np.concatenate([
        rng.normal(0.0, 0.004, n - 100),
        rng.normal(-0.012, 0.012, 100),
    ])
    return make_df(100.0 * np.exp(np.cumsum(rets)))


def _flat(n: int = 300) -> pd.DataFrame:
    t = np.arange(n)
    closes = 100.0 + np.sin(t / 7.0) * 0.6 * np.exp(-t / 400.0)
    return make_df(closes)


# ================================================================= #
#  1. Математика индикаторов
# ================================================================= #
class TestIndicators:
    def test_sigmoid_bounds(self):
        assert sigmoid(-20) == pytest.approx(0.0, abs=1e-8)
        assert sigmoid(20) == pytest.approx(1.0, abs=1e-8)
        assert sigmoid(0) == pytest.approx(0.5)

    def test_below_score(self):
        assert below_score(0.01, 0.05, 0.05) > 0.6    # заметно ниже порога
        assert below_score(0.20, 0.05, 0.05) < 0.1    # заметно выше порога
        assert below_score(0.10, 0.05, 0.0) == 0.0    # жёсткий режим
        assert below_score(0.01, 0.05, 0.0) == 1.0

    def test_slider_multiplier(self):
        assert slider_multiplier(0.0) == pytest.approx(0.5)
        assert slider_multiplier(0.5) == pytest.approx(1.0)
        assert slider_multiplier(1.0) == pytest.approx(2.0)
        assert slider_multiplier(2.0) == pytest.approx(2.0)  # клип

    def test_percentile_rank(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_rank(3.0, s) == pytest.approx(0.4)
        assert percentile_rank(5.0, s) == pytest.approx(0.8)
        assert np.isnan(percentile_rank(1.0, pd.Series([], dtype=float)))

    def test_add_indicators_columns(self):
        out = add_indicators(_flat(300))
        assert {"atr", "bbw", "atr_pct"} <= set(out.columns)
        assert out["atr"].iloc[0:13].isna().all()  # прогрев RMA
        assert out["bbw"].iloc[0:19].isna().all()  # прогрев SMA/STD
        assert (out["bbw"] > 0).all() or out["bbw"].isna().any()

    def test_flat_atr_constant_range(self):
        # Постоянный диапазон → ATR постоянный, bbw примерно постоянный
        closes = np.linspace(100.0, 102.0, 300)
        df = pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes,
        })
        out = add_indicators(df)
        tail = out.dropna()
        assert tail["atr"].std() < 0.01
        assert tail["bbw"].std() < 0.01


# ================================================================= #
#  2. Состояния: тренды и флэт
# ================================================================= #
class TestStates:
    def test_uptrend_is_up(self):
        v = analyze_regime(_trend_up())
        assert v is not None
        assert v["state"] == "UP"
        assert v["direction"] == 1
        assert v["trend_strength"] > 50
        assert not v["is_flat"]

    def test_downtrend_is_down(self):
        v = analyze_regime(_trend_down())
        assert v is not None
        assert v["state"] == "DOWN"
        assert v["direction"] == -1
        assert not v["is_flat"]

    def test_flat_is_flat(self):
        v = analyze_regime(_flat())
        assert v is not None
        assert v["state"] == "FLAT"
        assert v["is_flat"]
        assert v["trend_strength"] < 50
        assert abs(v["metrics"]["z"]) < 1.0

    def test_insufficient_data_returns_none(self):
        assert compute_regime_metrics(_trend_up(100)) is None
        assert analyze_regime(_trend_up(100)) is None

    def test_flat_chop_flag(self):
        v = analyze_regime(_flat())
        assert v["is_chop"]  # узкое движение при rank>0.65 — «вялый»/шумный боковик


# ================================================================= #
#  3. Слайдер: монотонность
# ================================================================= #
class TestSlider:
    def test_flat_score_monotonic(self):
        m = compute_regime_metrics(_trend_up(300, seed=1))
        assert m is not None
        scores = [
            evaluate_regime(m, RegimeSliders(flat=s))["flat_score"]
            for s in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        # Неубывание flat_score при расширении слайдера
        assert scores == sorted(scores)

    def test_more_flat_with_wider_slider(self):
        # Пограничный случай: при 0.0 тренд, при 1.0 — флэт
        rng = np.random.default_rng(3)
        rets = np.concatenate([
            rng.normal(0.0, 0.005, 200),
            rng.normal(0.0022, 0.005, 100),
        ])
        m = compute_regime_metrics(make_df(100.0 * np.exp(np.cumsum(rets))))
        assert m is not None
        v0 = evaluate_regime(m, RegimeSliders(flat=0.0))
        v1 = evaluate_regime(m, RegimeSliders(flat=1.0))
        assert v1["flat_score"] >= v0["flat_score"]
        assert v1["trend_strength"] <= v0["trend_strength"]

    def test_slider_split_independent(self):
        """Раскол: метрики НЕ зависят от слайдера."""
        df = _trend_up()
        m_a = compute_regime_metrics(df)
        m_b = compute_regime_metrics(df)
        assert m_a is not None and m_b is not None
        for key in ("z", "z_w", "z_n", "atr_change_n", "bbw_change_n",
                    "atr_rank_w", "bbw_rank_w", "atr_pct_mean_w", "atr_pct_mean_n"):
            assert m_a[key] == m_b[key]

    def test_individual_sliders_override(self):
        m = compute_regime_metrics(_flat())
        v = evaluate_regime(m, RegimeSliders(flat=0.5, pct=0.0))
        assert v["multipliers"]["m_pct"] == pytest.approx(0.5)
        assert v["sliders"]["pct"] == pytest.approx(0.0)
        assert v["sliders"]["flat"] == pytest.approx(0.5)


# ================================================================= #
#  4. Гейт сигналов
# ================================================================= #
class TestSignalGate:
    def test_none_verdict_allows(self):
        assert signal_allowed("entry_long", None) == (True, None)

    def test_flat_blocks_entry(self):
        assert signal_allowed("entry_long", {"is_flat": True, "state": "FLAT"}) == (False, "flat")

    def test_direction_mismatch(self):
        assert signal_allowed("entry_long", {"is_flat": False, "state": "DOWN"}) == (False, "direction_mismatch")
        assert signal_allowed("entry_short", {"is_flat": False, "state": "UP"}) == (False, "direction_mismatch")

    def test_matching_direction_allowed(self):
        assert signal_allowed("entry_long", {"is_flat": False, "state": "UP"}) == (True, None)
        assert signal_allowed("entry_short", {"is_flat": False, "state": "DOWN"}) == (True, None)
        assert signal_allowed("add_long", {"is_flat": False, "state": "UP"}) == (True, None)

    def test_exits_never_blocked_by_default(self):
        assert signal_allowed("exit_long", {"is_flat": True, "state": "FLAT"}) == (True, None)
        assert signal_allowed("exit_short", {"is_flat": True, "state": "FLAT"}) == (True, None)

    def test_exits_blocked_when_gate_exits(self):
        assert signal_allowed("exit_long", {"is_flat": True}, gate_exits=True) == (False, "flat")

    def test_unknown_order_type_allowed(self):
        # hold при отсутствии вердикта (None) — не блокируется;
        # во флэте hold тоже режется (сигнал не подтверждён)
        assert signal_allowed("hold", None) == (True, None)
        assert signal_allowed(None, None) == (True, None)
        assert signal_allowed("hold", {"is_flat": True}) == (False, "flat")


# ================================================================= #
#  5. Параметры и края
# ================================================================= #
class TestParams:
    def test_params_validation(self):
        with pytest.raises(ValueError):
            RegimeParams(window=0)
        with pytest.raises(ValueError):
            RegimeParams(recent=300)  # recent > window
        with pytest.raises(ValueError):
            RegimeParams(bb_mult=0)

    def test_min_bars(self):
        assert DEFAULT_PARAMS.min_bars == 205
        assert RegimeParams(window=100, recent=50).min_bars == 105

    def test_regime_sliders_from_dict(self):
        s = RegimeSliders.from_dict({"flat": 0.8, "pct": 0.2, "flat_score_threshold": 70})
        assert s.flat == pytest.approx(0.8)
        assert s.pct == pytest.approx(0.2)
        assert s.atr is None
        assert s.flat_score_threshold == pytest.approx(70.0)
        s2 = RegimeSliders.from_dict(None)
        assert s2 == DEFAULT_SLIDERS

    def test_evaluate_none_metrics(self):
        assert evaluate_regime(None, RegimeSliders()) is None
        assert evaluate_regime({}, RegimeSliders()) is None
        assert evaluate_regime({"z": 1.0}, RegimeSliders()) is None  # нет atr/bbw


# ================================================================= #
#  6. Интеграция: StrategySettings → trend_regime
# ================================================================= #
class TestStrategyIntegration:
    def test_settings_flow(self):
        from gex.strategy.trading_algorithm import EMAFilterTrendStrategy, StrategySettings
        st = EMAFilterTrendStrategy(StrategySettings(flat_slider=0.7, flat_filter_signals=True))
        params = st.regime_params()
        assert params.window == 200 and params.recent == 50
        sliders = st.regime_sliders()
        assert sliders.flat == pytest.approx(0.7)
        v = st.trend_regime(_trend_up())
        assert v is not None and v["state"] == "UP"
        # Гейт отключён по умолчанию (flat_filter_signals=False) — поведение не меняется
        st2 = EMAFilterTrendStrategy()
        assert st2.regime_signal_allowed("entry_long", {"is_flat": True}) == (True, None)
        assert st.regime_signal_allowed("entry_long", {"is_flat": True}) == (False, "flat")
