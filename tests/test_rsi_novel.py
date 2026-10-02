"""Тесты RSI Novel Candles: Pine-примитивы, алгоритм, сериализация, схемы."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gex.application.novel_candles import NovelCandlesService
from gex.application.rsi_novel import (
    RsiNovelService,
    calc_linreg_custom,
    calc_mad_custom,
    pine_bb,
    pine_ema,
    pine_linreg,
    pine_rma,
    pine_sma,
    pine_stdev,
)
from gex.schemas.rsi_novel import RsiNovelBar, RsiNovelResponse


# ══════════════════════════════════════════════════════════════════════
#  Фикстуры
# ══════════════════════════════════════════════════════════════════════
def _make_ohlcv_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """OHLCV DataFrame: списки (Open, High, Low, Close)."""
    dates = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    records = [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1e6}
               for o, h, l, c in rows]
    return pd.DataFrame(records, index=dates)


def _uptrend_5bars() -> pd.DataFrame:
    """5 растущих баров: 100→101→102→103→104."""
    return _make_ohlcv_df([
        (100.0, 100.5, 99.5, 100.0),
        (101.0, 102.0, 100.5, 101.5),
        (102.0, 103.0, 101.5, 102.5),
        (103.0, 104.0, 102.5, 103.5),
        (104.0, 105.0, 103.5, 104.5),
    ])


def _synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Синтетический OHLCV: тренд + синусоида + шум."""
    rng = np.random.default_rng(seed)
    base = 100 + np.arange(n) * 0.05 + 3 * np.sin(np.arange(n) / 12)
    o = base + rng.normal(0, 0.4, n)
    c = np.roll(base, -1) + rng.normal(0, 0.3, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.5, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n))
    return _make_ohlcv_df(list(zip(o, h, l, c)))


def _novel_of(df: pd.DataFrame) -> pd.DataFrame:
    return NovelCandlesService.compute_novel_candles(df)


# ══════════════════════════════════════════════════════════════════════
#  Pine-примитивы
# ══════════════════════════════════════════════════════════════════════
class TestPinePrimitives:
    def test_rma_seed_sma_and_recursion(self):
        """RMA: сид = SMA первых length, далее alpha*value + (1-alpha)*prev."""
        r = pine_rma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
        assert math.isnan(r.iloc[0]) and math.isnan(r.iloc[1])
        assert abs(r.iloc[2] - 2.0) < 1e-9            # SMA(1,2,3)
        assert abs(r.iloc[3] - (4 + 2 * 2) / 3) < 1e-9  # alpha=1/3
        assert abs(r.iloc[4] - (5 + 2 * 8 / 3) / 3) < 1e-9

    def test_rma_with_nan_gap(self):
        """Пропуск в начале: сид берётся по первым length валидным."""
        s = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
        r = pine_rma(s, 3)
        assert math.isnan(r.iloc[0])
        assert abs(r.iloc[3] - 2.0) < 1e-9  # SMA(1,2,3) на позиции 3
        assert abs(r.iloc[4] - (4 + 2 * 2) / 3) < 1e-9

    def test_rma_short_series_all_nan(self):
        r = pine_rma(pd.Series([1.0, 2.0]), 5)
        assert r.isna().all()

    def test_ema_seed_and_recursion(self):
        r = pine_ema(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
        assert abs(r.iloc[2] - 2.0) < 1e-9
        # alpha = 2/(3+1) = 0.5
        assert abs(r.iloc[3] - (0.5 * 4 + 0.5 * 2)) < 1e-9

    def test_sma(self):
        r = pine_sma(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
        assert math.isnan(r.iloc[1])
        assert abs(r.iloc[2] - 2.0) < 1e-9
        assert abs(r.iloc[3] - 3.0) < 1e-9

    def test_stdev_population(self):
        """Pine stdev по умолчанию biased (ddof=0)."""
        r = pine_stdev(pd.Series([1.0, 2.0, 3.0]), 3)
        assert abs(r.iloc[2] - math.sqrt(2 / 3)) < 1e-9

    def test_bb(self):
        basis, upper, lower = pine_bb(pd.Series([1.0, 2.0, 3.0, 4.0]), 3, 2.0)
        assert abs(basis.iloc[3] - 3.0) < 1e-9
        expected_dev = 2.0 * pine_stdev(pd.Series([1.0, 2.0, 3.0, 4.0]), 3).iloc[3]
        assert abs(upper.iloc[3] - (3.0 + expected_dev)) < 1e-9
        assert abs(lower.iloc[3] - (3.0 - expected_dev)) < 1e-9

    def test_linreg_linear_series(self):
        """ta.linreg линейного ряда = текущее значение."""
        s = pd.Series(np.arange(1.0, 11.0))
        r = pine_linreg(s, 5, 0)
        assert abs(r.iloc[9] - 10.0) < 1e-9

    def test_linreg_offset(self):
        """offset=1 → значение на один бар раньше (x = length-1-offset)."""
        s = pd.Series(np.arange(1.0, 11.0))
        r = pine_linreg(s, 5, 1)
        assert abs(r.iloc[9] - 9.0) < 1e-9

    def test_linreg_short(self):
        r = pine_linreg(pd.Series([1.0, 2.0]), 5)
        assert r.isna().all()

    def test_calc_linreg_custom_exact_fit(self):
        x = np.arange(100, dtype=float)[::-1]  # 99..0, как в Pine
        y = 2.0 * x + 3.0
        slope, intercept = calc_linreg_custom(x, y)
        assert abs(slope - 2.0) < 1e-9
        assert abs(intercept - 3.0) < 1e-9

    def test_calc_linreg_custom_empty(self):
        slope, intercept = calc_linreg_custom([], [])
        assert math.isnan(slope) and math.isnan(intercept)

    def test_calc_mad_custom(self):
        x = np.arange(10, dtype=float)
        y = 5.0 * x + 1.0
        assert abs(calc_mad_custom(x, y, 5.0, 1.0)) < 1e-12
        # Сдвиг всех точек на 2 → MAD = 2
        assert abs(calc_mad_custom(x, y + 2.0, 5.0, 1.0) - 2.0) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  compute_rsi_novel
# ══════════════════════════════════════════════════════════════════════
class TestComputeRsiNovel:
    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="Missing required columns"):
            RsiNovelService.compute_rsi_novel(pd.DataFrame({"Open": [1], "Close": [2]}))

    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        with pytest.raises(ValueError, match="пуст"):
            RsiNovelService.compute_rsi_novel(df)

    def test_rsi_in_range(self):
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(200)))
        rc = rdf["RSI_close"].dropna()
        assert len(rc) > 50
        assert rc.between(0, 100).all()
        for col in ["RSI_open", "RSI_high", "RSI_low"]:
            assert rdf[col].dropna().between(0, 100).all()

    def test_uptrend_rsi_100(self):
        df = _make_ohlcv_df([(100 + i, 102 + i, 99 + i, 101 + i) for i in range(60)])
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(df))
        assert abs(rdf["RSI_close"].iloc[-1] - 100.0) < 1e-6

    def test_downtrend_rsi_0(self):
        df = _make_ohlcv_df([(100 - i, 101 - i, 99 - i, 99.5 - i) for i in range(60)])
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(df))
        assert abs(rdf["RSI_close"].iloc[-1]) < 1e-6

    def test_high_low_fixed_invariants(self):
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(150)))
        finite = rdf["RSI_high_fixed"].notna() & rdf["RSI_open"].notna() & rdf["RSI_close"].notna()
        assert (rdf.loc[finite, "RSI_high_fixed"] >= rdf.loc[finite, "RSI_open"]).all()
        assert (rdf.loc[finite, "RSI_high_fixed"] >= rdf.loc[finite, "RSI_close"]).all()
        assert (rdf.loc[finite, "RSI_low_fixed"] <= rdf.loc[finite, "RSI_open"]).all()
        assert (rdf.loc[finite, "RSI_low_fixed"] <= rdf.loc[finite, "RSI_close"]).all()

    def test_candle_direction(self):
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(100)))
        dirs = rdf["candle_direction"].dropna()
        assert set(dirs.unique()) <= {1, -1}
        up = rdf["RSI_close"] > rdf["RSI_close"].shift(1)
        mask = up.notna()
        assert (rdf.loc[mask, "candle_direction"] == np.where(up[mask], 1, -1)).all()

    def test_wicks_false_runs(self):
        rdf_on = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(120)), wicks=True)
        rdf_off = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(120)), wicks=False)
        assert rdf_off["RSI_close"].dropna().between(0, 100).all()
        # Wicks влияет на open/high/low, но не на close
        assert np.allclose(rdf_on["RSI_close"].dropna(), rdf_off["RSI_close"].dropna(), equal_nan=True)

    def test_short_df(self):
        # lenn=2: RMA сидуется на 2 валидных gain → RSI появляется с 3-го бара
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_uptrend_5bars()), lenn=2)
        assert len(rdf) == 5
        assert rdf["RSI_close"].isna().iloc[0]  # первый бар: нет change
        assert len(rdf["RSI_close"].dropna()) == 3

    def test_single_bar_all_rsi_nan(self):
        df = _make_ohlcv_df([(100.0, 102.0, 99.0, 101.0)])
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(df))
        assert rdf["RSI_close"].isna().all()
        assert rdf["novel_close"].notna().all()

    def test_custom_regression_warmup_and_finite(self):
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(200)), period100=50)
        # Прогрев: сначала NaN (нужны rsiMA3 и окно period100)
        assert rdf["lin_reg"].iloc[:49].isna().all()
        # В конце — конечные значения
        assert rdf["lin_reg"].iloc[-50:].notna().all()
        assert rdf["custom_slope"].iloc[-50:].notna().all()
        assert rdf["custom_mad"].iloc[-50:].notna().all()

    def test_linear_reg_curve_formula(self):
        """curve = avg(linear_reg, rsiMA3, linear_reg, rsiMA3, lin_reg)."""
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(150)))
        expected = (
            2.0 * rdf["linear_reg"] + 2.0 * rdf["rsiMA3"] + rdf["lin_reg"]
        ) / 5.0
        mask = rdf["linear_reg_curve"].notna() & expected.notna()
        assert np.allclose(rdf.loc[mask, "linear_reg_curve"], expected[mask])

    def test_resistance_support_mid_finite(self):
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(200)))
        for col in ["resistance", "support", "mid"]:
            vals = rdf[col].dropna()
            assert len(vals) > 100
        # support <= resistance на конечных значениях
        mask = rdf["support"].notna() & rdf["resistance"].notna()
        assert (rdf.loc[mask, "support"] <= rdf.loc[mask, "resistance"] + 1e-9).all()

    def test_mas_shapes(self):
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(120)))
        for col in ["rsiMA", "rsiMA3", "rsiMAfast", "trend_ma"]:
            assert len(rdf[col]) == 120
            assert rdf[col].dropna().between(0, 100).all()


# ══════════════════════════════════════════════════════════════════════
#  Сериализация и сервис
# ══════════════════════════════════════════════════════════════════════
class TestSerialize:
    def _result(self, **overrides) -> dict:
        svc = RsiNovelService()
        rdf = RsiNovelService.compute_rsi_novel(_novel_of(_synthetic_df(150)))
        novel = _novel_of(_synthetic_df(150))
        params = dict(ob_level=75.0, os_level=25.0, om_level=50.0, lenn=14)
        params.update(overrides)
        return RsiNovelService._serialize("TEST", "1d", "stock", novel, rdf, **params)

    def test_keys_and_lengths(self):
        res = self._result()
        assert res["ticker"] == "TEST"
        assert res["n_bars"] == 150
        assert len(res["bars"]) == 150
        for key in ["rsi_close", "rsi_open", "rsi_avg", "rsi_ma_fast", "rsi_ma",
                    "rsi_ma3", "trend_ma", "linear_reg_curve", "resistance",
                    "support", "mid"]:
            assert key in res, f"missing {key}"
            assert len(res[key]) == 150

    def test_no_nan_in_json(self):
        res = self._result()
        for key in ["rsi_close", "rsi_open", "rsi_avg", "rsi_ma_fast", "rsi_ma",
                    "rsi_ma3", "trend_ma", "linear_reg_curve", "resistance",
                    "support", "mid"]:
            for v in res[key]:
                assert v is None or isinstance(v, float), f"{key}: {v!r}"
                if v is not None:
                    assert math.isfinite(v)
        for b in res["bars"]:
            for k in ["open", "high", "low", "close"]:
                assert b[k] is None or isinstance(b[k], float)

    def test_signal_logic(self):
        # Uptrend → RSI=100 → overbought
        df = _make_ohlcv_df([(100 + i, 102 + i, 99 + i, 101 + i) for i in range(60)])
        novel = _novel_of(df)
        rdf = RsiNovelService.compute_rsi_novel(novel)
        res = RsiNovelService._serialize("X", "1d", "stock", novel, rdf, 75.0, 25.0, 50.0, 14)
        assert res["signal"] == "overbought"
        assert res["last_rsi"] == 100.0

        # Downtrend → RSI=0 → oversold
        df = _make_ohlcv_df([(100 - i, 101 - i, 99 - i, 99.5 - i) for i in range(60)])
        novel = _novel_of(df)
        rdf = RsiNovelService.compute_rsi_novel(novel)
        res = RsiNovelService._serialize("X", "1d", "stock", novel, rdf, 75.0, 25.0, 50.0, 14)
        assert res["signal"] == "oversold"
        assert res["last_rsi"] == 0.0

    def test_fetch_and_analyze_monkeypatched(self, monkeypatch):
        """Полный конвейер: фетч подменён синтетикой (без сети)."""
        df = _synthetic_df(120)

        svc = RsiNovelService()
        monkeypatch.setattr(svc._novel_svc, "_fetch_ohlcv", lambda *a, **k: df)

        res = svc.fetch_and_analyze("TEST", timeframe="1d", limit=120)
        assert res["ticker"] == "TEST"
        assert res["n_bars"] == 120
        assert res["levels"] == {"ob": 75.0, "os": 25.0, "om": 50.0}
        assert res["rsi_length"] == 14
        assert len(res["bars"]) == 120

    def test_fetch_and_analyze_bad_timeframe(self, monkeypatch):
        svc = RsiNovelService()
        with pytest.raises(ValueError, match="таймфрейм"):
            svc.fetch_and_analyze("TEST", timeframe="3d")

    def test_fetch_and_analyze_no_data(self, monkeypatch):
        import pandas as pd
        svc = RsiNovelService()
        monkeypatch.setattr(svc._novel_svc, "_fetch_ohlcv", lambda *a, **k: pd.DataFrame())
        with pytest.raises(ValueError, match="Нет OHLCV"):
            svc.fetch_and_analyze("TEST", timeframe="1d")


# ══════════════════════════════════════════════════════════════════════
#  Pydantic-схемы
# ══════════════════════════════════════════════════════════════════════
class TestSchemas:
    def test_bar_valid(self):
        bar = RsiNovelBar(time="2026-01-01T00:00:00+00:00", open=50.0, high=60.0, low=40.0, close=55.0)
        assert bar.high == 60.0

    def test_bar_extra_forbidden(self):
        with pytest.raises(Exception):
            RsiNovelBar(time="t", open=1, high=2, low=0.5, close=1.5, extra="nope")

    def test_response_minimal(self):
        from gex.schemas.rsi_novel import RsiNovelLevels
        resp = RsiNovelResponse(
            ticker="SPY", timeframe="1d", asset_type="stock", n_bars=0,
            rsi_length=14, levels=RsiNovelLevels(ob=75.0, os=25.0, om=50.0),
        )
        assert resp.bars == []
        assert resp.signal is None

    def test_response_with_data(self):
        from gex.schemas.rsi_novel import RsiNovelLevels
        resp = RsiNovelResponse(
            ticker="SPY", timeframe="1d", asset_type="stock", n_bars=1,
            bars=[RsiNovelBar(time="t", open=50.0, high=60.0, low=40.0, close=55.0)],
            rsi_close=[55.0], rsi_open=[50.0], rsi_avg=[51.25],
            rsi_ma_fast=[None], rsi_ma=[None], rsi_ma3=[None],
            trend_ma=[None], linear_reg_curve=[None],
            resistance=[None], support=[None], mid=[None],
            slope=0.01, mad=2.5, last_rsi=55.0, signal="neutral",
            rsi_length=14, levels=RsiNovelLevels(ob=75.0, os=25.0, om=50.0),
        )
        assert len(resp.bars) == 1
        assert resp.signal == "neutral"
