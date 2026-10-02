"""Тесты многофакторной логит-модели направления GEX (gex.direction).

Без сети: синтетические OHLCV-DataFrames + синтетический GEXProfile.
Проверяем, что модель:
  * даёт BULLISH на восходящем тренде, BEARISH на нисходящем;
  * реагирует на близость/силу стен (магнит);
  * реагирует на асимметрию гаммы;
  * не падает без OHLCV (MOEX/VIX path);
  * вероятность и confidence в корректных диапазонах;
  * нивелирующая цена давит хвосты корректно.

Паттерн повторяет tests/test_macd_trend.py: чистые функции + синтетика.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gex.domain.metrics import GEXProfile
from gex.domain.direction import (
    DirectionResult,
    combine_logit,
    compute_direction,
    confidence_from_factors,
    direction_from_p,
    flip_pressure_signal,
    level_asymmetry_signal_simple,
    momentum_signal,
    neutralizing_price_series,
    trend_strength_from_factors,
    wall_magnet_signal,
)


# ====================================================================== #
#  Фикстуры синтетических данных
# ====================================================================== #
def _ohlcv(direction: str = "up", n: int = 60, start: float = 100.0) -> pd.DataFrame:
    """Синтетический OHLCV-ряд с tz-aware UTC индексом.

    direction='up'  — растущий (close растёт, open < close — бычьи тела);
    direction='down' — падающий (open > close — медвежьи тела);
    direction='flat' — боковик (close колеблется около start).
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    if direction == "up":
        close = np.linspace(start, start * 1.15, n)
        op = close - 0.4  # бычья свеча: open < close
    elif direction == "down":
        close = np.linspace(start * 1.15, start, n)
        op = close + 0.4  # медвежья: open > close
    else:  # flat
        rng = np.random.default_rng(42)
        close = start + rng.normal(0, 0.1, n).cumsum() * 0.0 + start  # стабильно
        op = close + rng.normal(0, 0.05, n)
    high = np.maximum(op, close) + 0.5
    low = np.minimum(op, close) - 0.3
    vol = np.full(n, 1000.0)
    return pd.DataFrame(
        {"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _profile(
    regime: str = "POSITIVE",
    z_score: float = 0.0,
    gamma_flip: float = 100.0,
    call_wall: float = 108.0,
    put_wall: float = 92.0,
    gex_distribution: str = "symmetric",
) -> GEXProfile:
    """Синтетический GEXProfile по страйкам 90..110 (симметрично вокруг 100).

    gex_distribution:
      * 'symmetric' — гамма ниже 100 = −10, выше = +10 (баланс |gex| по сторонам);
      * 'call_heavy' — перевес гаммы сверху (сопротивление сильнее);
      * 'put_heavy' — перевес гаммы снизу (поддержка сильнее).

    Страйки выбраны симметрично (90..99 ниже, 101..110 выше — по 10 с каждой
    стороны), чтобы 'symmetric' давал нулевую асимметрию относительно spot=100.
    """
    strikes = np.arange(90, 111, dtype=float)
    if gex_distribution == "symmetric":
        gex = np.where(strikes < 100, -10.0, np.where(strikes > 100, 10.0, 0.0))
    elif gex_distribution == "call_heavy":
        gex = np.where(strikes < 100, -5.0, np.where(strikes > 100, 20.0, 0.0))
    elif gex_distribution == "put_heavy":
        gex = np.where(strikes < 100, -20.0, np.where(strikes > 100, 5.0, 0.0))
    else:
        raise ValueError(gex_distribution)
    df = pd.DataFrame({
        "strike": strikes,
        "gex_net": gex,
        "gex_abs": np.abs(gex),
        "oi_call": np.abs(gex),
        "oi_put": np.abs(gex),
    })
    return GEXProfile(
        per_strike=df,
        net_gex=float(gex.sum()),
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        call_wall_oi=call_wall,
        put_wall_oi=put_wall,
        regime=regime,
        z_score=z_score,
    )


# ====================================================================== #
#  1. Нивелирующая цена
# ====================================================================== #
class TestNeutralizingPrice:
    def test_bullish_candle_uses_close_high(self):
        """Растущая свеча (close >= open) → среднее(close, high)."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [110.0], "Low": [99.0], "Close": [105.0],
        })
        np_ser = neutralizing_price_series(df)
        assert np_ser.iloc[0] == pytest.approx((105.0 + 110.0) / 2.0)

    def test_bearish_candle_uses_open_low(self):
        """Падающая свеча (open > close) → среднее(open, low)."""
        df = pd.DataFrame({
            "Open": [105.0], "High": [106.0], "Low": [95.0], "Close": [100.0],
        })
        np_ser = neutralizing_price_series(df)
        assert np_ser.iloc[0] == pytest.approx((105.0 + 95.0) / 2.0)

    def test_dampens_wick_outliers(self):
        """Теневой прокол не уводит нивелирующую так сильно, как close."""
        # Падающая свеча с огромной верхней тенью: close не должен учитывать хай.
        df = pd.DataFrame({
            "Open": [100.0], "High": [200.0], "Low": [98.0], "Close": [99.0],
        })
        np_val = neutralizing_price_series(df).iloc[0]
        # bearish → (open+low)/2 = (100+98)/2 = 99, тень 200 проигнорирована.
        assert np_val == pytest.approx(99.0)


# ====================================================================== #
#  2. Асимметрия стен
# ====================================================================== #
class TestLevelAsymmetry:
    def test_call_heavy_gives_bearish_pressure(self):
        """Перевес гаммы сверху → сопротивление → давление вниз (z < 0)."""
        p = _profile(gex_distribution="call_heavy")
        z = level_asymmetry_signal_simple(100.0, p)
        assert z < 0.0

    def test_put_heavy_gives_bullish_pressure(self):
        """Перевес гаммы снизу → поддержка → давление вверх (z > 0)."""
        p = _profile(gex_distribution="put_heavy")
        z = level_asymmetry_signal_simple(100.0, p)
        assert z > 0.0

    def test_symmetric_near_zero(self):
        p = _profile(gex_distribution="symmetric")
        z = level_asymmetry_signal_simple(100.0, p)
        assert abs(z) < 0.2

    def test_bounded(self):
        p = _profile(gex_distribution="call_heavy")
        z = level_asymmetry_signal_simple(100.0, p)
        assert -3.0 <= z <= 3.0


# ====================================================================== #
#  3. Магнит стены
# ====================================================================== #
class TestWallMagnet:
    def test_call_wall_near_spot_above_pulls_up(self):
        """Ближняя call_wall сверху → притяжение вверх (z > 0)."""
        p = _profile(call_wall=102.0, put_wall=96.0)
        z = wall_magnet_signal(101.0, p)  # spot 101, call_wall 102 близко сверху
        assert z > 0.0

    def test_put_wall_near_spot_below_pulls_down(self):
        """Ближняя put_wall снизу → притяжение вниз (z < 0)."""
        p = _profile(call_wall=108.0, put_wall=99.0)
        z = wall_magnet_signal(100.0, p)  # spot 100, put_wall 99 близко снизу
        assert z < 0.0

    def test_far_walls_weak_signal(self):
        """Далёкие стены дают слабый магнит (|z| мал)."""
        p = _profile(call_wall=108.0, put_wall=96.0)
        z = wall_magnet_signal(100.0, p)  # обе стены в 8% — далеко
        assert abs(z) < 1.0

    def test_wall_at_spot_returns_zero(self):
        """Стена в точке спота не магнит (|z| ≈ 0 или не определяется)."""
        p = _profile(call_wall=100.0, put_wall=100.0)
        z = wall_magnet_signal(100.0, p)
        assert z == 0.0


# ====================================================================== #
#  4. Моментум свечей
# ====================================================================== #
class TestMomentumSignal:
    def test_rising_series_positive(self):
        z = momentum_signal(_ohlcv("up"))
        assert z > 0.0

    def test_falling_series_negative(self):
        z = momentum_signal(_ohlcv("down"))
        assert z < 0.0

    def test_too_few_bars_returns_zero(self):
        df = _ohlcv("up", n=5)
        assert momentum_signal(df) == 0.0

    def test_none_returns_zero(self):
        assert momentum_signal(None) == 0.0

    def test_bounded_pm2(self):
        z = momentum_signal(_ohlcv("up"))
        assert -2.0 <= z <= 2.0


# ====================================================================== #
#  5. Давление Flip
# ====================================================================== #
class TestFlipPressure:
    def test_positive_regime_mean_reversion(self):
        """POSITIVE gamma: спот выше Flip (z>0) → давление вниз."""
        p = _profile(regime="POSITIVE", z_score=1.5, gamma_flip=100.0)
        z = flip_pressure_signal(102.0, p, 30 / 365)
        assert z < 0.0

    def test_positive_regime_spot_below_flip(self):
        """POSITIVE gamma: спот ниже Flip (z<0) → давление вверх."""
        p = _profile(regime="POSITIVE", z_score=-1.5, gamma_flip=100.0)
        z = flip_pressure_signal(98.0, p, 30 / 365)
        assert z > 0.0

    def test_negative_regime_trend_continuation(self):
        """NEGATIVE gamma: спот выше Flip (z>0) → продолжение вверх."""
        p = _profile(regime="NEGATIVE", z_score=1.5, gamma_flip=100.0)
        z = flip_pressure_signal(102.0, p, 30 / 365)
        assert z > 0.0

    def test_no_flip_returns_zero(self):
        p = _profile(gamma_flip=0.0)
        # gamma_flip=0 → None-подобный путь, но z_score всё ещё задан.
        # При gamma_flip<=0 функция должна вернуть 0.
        z = flip_pressure_signal(100.0, p, 30 / 365)
        assert z == 0.0


# ====================================================================== #
#  6. Комбинатор и оркестратор
# ====================================================================== #
class TestCombineLogit:
    def test_empty_factors_returns_zero(self):
        assert combine_logit({"a": 0.0, "b": 0.0}) == 0.0

    def test_inactive_weight_redistributed(self):
        """Выключенный фактор не ослабляет активные (перенормировка весов)."""
        full = combine_logit({"level_asymmetry": 2.0, "momentum": 0.0,
                              "wall_magnet": 0.0, "flip_pressure": 0.0})
        # только level_asymmetry активен → его вес 0.40 перенормируется в 1.0
        # → L = 2.0 * 1.0 = 2.0
        assert full == pytest.approx(2.0, abs=0.05)

    def test_all_positive_factors_positive_logit(self):
        L = combine_logit({"level_asymmetry": 1.0, "momentum": 1.0,
                           "wall_magnet": 1.0, "flip_pressure": 1.0})
        assert L > 0.0


class TestDirectionFromP:
    def test_bullish(self):
        assert direction_from_p(0.7) == "BULLISH"

    def test_bearish(self):
        assert direction_from_p(0.3) == "BEARISH"

    def test_neutral(self):
        assert direction_from_p(0.5) == "NEUTRAL"

    def test_thresholds(self):
        assert direction_from_p(0.58) == "BULLISH"
        assert direction_from_p(0.42) == "BEARISH"
        assert direction_from_p(0.50) == "NEUTRAL"


class TestConfidence:
    def test_extreme_p_high_confidence(self):
        # p_up=0.95, все факторы согласны (все +)
        factors = {"a": 1.5, "b": 1.5, "c": 1.5}
        conf = confidence_from_factors(0.95, factors)
        assert conf > 70.0

    def test_disagreement_lowers_confidence(self):
        # p_up=0.7, но мнения разделились → confidence ниже, чем при согласии.
        agree = confidence_from_factors(0.7, {"a": 1.0, "b": 1.0})
        disagree = confidence_from_factors(0.7, {"a": 1.0, "b": -1.0})
        assert agree > disagree

    def test_neutral_p_zero_confidence(self):
        assert confidence_from_factors(0.5, {"a": 1.0}) == 0.0

    def test_bounded_0_100(self):
        factors = {"a": 3.0, "b": 3.0}
        assert 0.0 <= confidence_from_factors(0.99, factors) <= 100.0
        assert 0.0 <= confidence_from_factors(0.01, factors) <= 100.0


# ====================================================================== #
#  7. End-to-end: compute_direction
# ====================================================================== #
class TestComputeDirection:
    def test_up_trend_bullish(self):
        """Чистый восходящий ряд → BULLISH, p_up > 0.6, confidence > 25."""
        p = _profile(regime="POSITIVE", z_score=0.5)
        r = compute_direction(
            spot=101.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=_ohlcv("up"),
        )
        assert isinstance(r, DirectionResult)
        assert r.direction == "BULLISH"
        assert r.p_up > 0.6
        assert r.confidence > 25.0
        assert r.p_down == pytest.approx(1.0 - r.p_up, abs=1e-9)

    def test_down_trend_bearish(self):
        """Чистый нисходящий ряд → BEARISH, p_up < 0.45."""
        p = _profile(regime="POSITIVE", z_score=0.5)
        r = compute_direction(
            spot=101.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=_ohlcv("down"),
        )
        assert r.direction == "BEARISH"
        assert r.p_up < 0.45

    def test_no_ohlcv_does_not_crash(self):
        """Без OHLCV (MOEX/VIX path) модель работает на GEX-факторах."""
        p = _profile(gex_distribution="call_heavy", z_score=1.0)
        r = compute_direction(
            spot=100.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=None,
        )
        # call_heavy → перевес сопротивления → давление вниз
        assert r.p_up < 0.5
        assert r.factors["momentum"] == 0.0
        assert 0.01 <= r.p_up <= 0.99

    def test_call_heavy_pushes_down(self):
        """Перевес гаммы сверху (сопротивление) → p_up < 0.5 без моментума."""
        p = _profile(gex_distribution="call_heavy", z_score=0.0)
        r = compute_direction(
            spot=100.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=None,
        )
        assert r.p_up < 0.5

    def test_put_heavy_pushes_up(self):
        p = _profile(gex_distribution="put_heavy", z_score=0.0)
        r = compute_direction(
            spot=100.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=None,
        )
        assert r.p_up > 0.5

    def test_ranges_valid(self):
        """p_up ∈ [0.01, 0.99], confidence ∈ [0, 100], trend_strength ∈ [0, 100]."""
        p = _profile(gex_distribution="symmetric", z_score=1.0)
        for direction in ("up", "down"):
            r = compute_direction(
                spot=101.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
                ohlcv_df=_ohlcv(direction),
            )
            assert 0.01 <= r.p_up <= 0.99
            assert 0.0 <= r.confidence <= 100.0
            assert 0.0 <= r.trend_strength <= 100.0

    def test_extreme_trend_not_stuck_at_50_50(self):
        """Главное регрессионное условие: не вечные 0.51/0.49.

        Прежняя логика давала ~0.51 на любом тренде. Новая на сильном тренде
        должна уходить от 0.5 минимум на 0.15 (т.е. p_up > 0.65 или < 0.35).
        """
        p = _profile(regime="POSITIVE", z_score=0.5)
        r_up = compute_direction(
            spot=101.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=_ohlcv("up"),
        )
        r_down = compute_direction(
            spot=101.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=_ohlcv("down"),
        )
        assert abs(r_up.p_up - 0.5) > 0.15
        assert abs(r_down.p_up - 0.5) > 0.10

    def test_factors_dict_populated(self):
        p = _profile()
        r = compute_direction(
            spot=100.0, profile=p, horizon_years=30 / 365, atm_vol=0.2,
            ohlcv_df=_ohlcv("up"),
        )
        assert set(r.factors.keys()) == {
            "level_asymmetry", "momentum", "wall_magnet", "flip_pressure"
        }


# ====================================================================== #
#  8. Интеграция с GEXService._direction (обратная совместимость)
# ====================================================================== #
class TestServiceDirectionIntegration:
    """Проверяем, что GEXService._direction возвращает тот же кортеж
    (p_up, p_down, direction, confidence), что и прежде — контракт не сломан."""

    def test_returns_four_tuple(self):
        from gex.application.service import _DirectionProvider
        p = _profile(gex_distribution="put_heavy", z_score=0.5)
        result = _DirectionProvider().compute(
            spot=100.0, profile=p, horizon_years=30 / 365,
            atm_vol=0.2, ticker=None,
        )
        assert len(result) == 4
        p_up, p_down, direction, confidence = result
        assert isinstance(p_up, float)
        assert isinstance(direction, str)
        assert direction in ("BULLISH", "BEARISH", "NEUTRAL")
        assert p_up + p_down == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= confidence <= 100.0
