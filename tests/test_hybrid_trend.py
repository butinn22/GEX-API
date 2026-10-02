"""Тесты модуля gex.hybrid_trend: фракталы Билла Уильямса по гибридным сериям,
маркировка HH/HL/LH/LL, строгий тренд, novel-фильтр, score, зигзаг, стадия рынка.

Критерии приёмки (п.12 ТЗ):
- восходящий синтетический ряд → trend_strict=+1 после двух подтверждённых волн;
- нисходящий → trend_strict=-1;
- сигнал не раньше бара подтверждения фрактала;
- равные хаи/лои без превышения порога → нет ложных HH/HL/LH/LL;
- novelsrc против тренда при use_novel_filter=True → trend_strict=0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex.application.hybrid_trend import (
    HybridTrendParams,
    HybridTrendService,
    build_trend,
    compute_atr,
    compute_heikin_ashi,
    compute_hybrid_features,
    detect_fractals,
    select_fractal_series,
)
from gex.application.novel_candles import NovelCandlesService
from gex.schemas.hybrid_trend import HybridStructureResponse

RESULT_COLUMNS = [
    "ha_open", "ha_close",
    "median_top", "median_bottom",
    "hybrid_open", "hybrid_close",
    "candle_top", "candle_bottom", "avg_candle",
    "hlcc4", "sourceformas", "novelsrc",
    "atr", "fractal_top", "fractal_bottom",
    "fractal_high", "fractal_low",
    "fractal_high_confirmed", "fractal_low_confirmed",
    "fractal_high_accepted", "fractal_low_accepted",
    "hh_event", "lh_event", "hl_event", "ll_event",
    "last_high_label", "last_low_label",
    "trend_strict", "trend_score", "trend_score_dir", "trend_final",
    "novel_momentum",
    "stage", "zigzag_high", "zigzag_low",
]


# ══════════════════════════════════════════════════════════════════════
#  Фикстуры
# ══════════════════════════════════════════════════════════════════════
def _mk(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """DataFrame OHLC: Open = предыдущий Close, Close = (High+Low)/2."""
    closes = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    rows = []
    prev_c: float | None = None
    for i, (h, l) in enumerate(zip(highs, lows)):
        o = prev_c if prev_c is not None else closes[i]
        rows.append({"Open": o, "High": h, "Low": l, "Close": closes[i]})
        prev_c = closes[i]
    dates = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=dates)


def _std_params(**overrides) -> HybridTrendParams:
    """Стандартные параметры для детерминированных структурных тестов."""
    base = dict(
        fractal_left=2,
        fractal_right=2,
        fractal_source="standard",
        strict_fractals=True,
        atr_period=14,
        atr_mult=0.25,
        min_change=0.0,
        max_event_age=120,
        use_novel_filter=False,
    )
    base.update(overrides)
    return HybridTrendParams(**base)


def _uptrend_stair() -> pd.DataFrame:
    """Ступенчатый восходящий ряд (20 баров).

    Пики: 109@3, 113@11, 117@19 (не подтверждён — за пределами данных).
    Впадины: 101@7, 105@15.
    Подтверждения (right=2): H@5, L@9, H@13, L@17.
    События: HH@13 (113 vs 109), HL@17 (105 vs 101) → strict=+1 с бара 17.
    """
    highs = [100, 103, 106, 109, 107, 105, 103, 101, 104, 107, 110, 113,
             111, 109, 107, 105, 108, 111, 114, 117]
    lows = [99, 100, 102, 104, 104, 103, 102, 101, 104, 106, 108, 110,
            109, 107, 106, 105, 108, 110, 112, 114]
    return _mk(highs, lows)


def _downtrend_stair() -> pd.DataFrame:
    """Ступенчатый нисходящий ряд (20 баров).

    Впадины: 87@3, 83@11, 79@19 (не подтверждена).
    Пики: 99@7, 95@15.
    События: LL@13 (83 vs 87), LH@17 (95 vs 99) → strict=-1 с бара 17.
    """
    highs = [100, 97, 94, 91, 93, 95, 97, 99, 96, 93, 90, 87,
             89, 91, 93, 95, 92, 89, 86, 83]
    lows = [96, 93, 90, 87, 90, 92, 94, 96, 92, 89, 86, 83,
            86, 88, 90, 92, 88, 85, 82, 79]
    return _mk(highs, lows)


def _equal_swings() -> pd.DataFrame:
    """24 бара: равные пики 109 и равные впадины 101 — события не возникают."""
    highs = [100, 103, 106, 109, 107, 105, 103, 101, 104, 106, 108, 109,
             107, 105, 103, 101, 104, 106, 108, 109, 107, 105, 103, 101]
    lows = [99, 100, 102, 104, 104, 103, 102, 101, 104, 105, 107, 108,
            106, 104, 103, 101, 104, 106, 108, 110, 108, 106, 105, 101]
    return _mk(highs, lows)


def _reversal_mixed() -> pd.DataFrame:
    """Восходящие ноги 1-3 (HH@13), затем глубокая впадина 99@15 (LL@17).

    last_high=+1, last_low=-1 → стадия REVERSAL.
    """
    highs = [100, 103, 106, 109, 107, 105, 103, 101, 104, 107, 110, 113,
             111, 108, 105, 102, 105, 108, 111, 114]
    lows = [99, 100, 102, 104, 104, 103, 102, 101, 104, 106, 108, 110,
            107, 104, 101, 99, 103, 106, 109, 112]
    return _mk(highs, lows)


def _uptrend_then_collapse() -> pd.DataFrame:
    """Восходящая лестница + обвал (бары 20-23) для теста novel-фильтра."""
    highs = [100, 103, 106, 109, 107, 105, 103, 101, 104, 107, 110, 113,
             111, 109, 107, 105, 108, 111, 114, 117, 114, 111, 108, 105]
    lows = [99, 100, 102, 104, 104, 103, 102, 101, 104, 106, 108, 110,
            109, 107, 106, 105, 108, 110, 112, 114, 110, 107, 104, 101]
    return _mk(highs, lows)


# ══════════════════════════════════════════════════════════════════════
#  Параметры
# ══════════════════════════════════════════════════════════════════════
class TestParams:
    def test_defaults(self):
        p = HybridTrendParams()
        assert p.fractal_left == 2 and p.fractal_right == 2
        assert p.fractal_source == "median_candle"
        assert p.alpha == 0.5
        assert p.atr_period == 14 and p.atr_mult == 0.25
        assert p.use_novel_filter is True

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError):
            HybridTrendParams(alpha=1.5)
        with pytest.raises(ValueError):
            HybridTrendParams(alpha=-0.1)

    def test_negative_fractal_window(self):
        with pytest.raises(ValueError):
            HybridTrendParams(fractal_left=-1)
        with pytest.raises(ValueError):
            HybridTrendParams(fractal_right=-1)


# ══════════════════════════════════════════════════════════════════════
#  Heikin-Ashi + гибридные серии (формулы ТЗ)
# ══════════════════════════════════════════════════════════════════════
class TestHybridFeatures:
    def _two_bars(self) -> pd.DataFrame:
        return _mk([14.0, 16.0], [8.0, 10.0])  # Open=12,Close=12 / Open=12,Close=13

    def test_heikin_ashi_formulas(self):
        df = pd.DataFrame({
            "Open": [10.0, 12.0], "High": [14.0, 16.0],
            "Low": [8.0, 10.0], "Close": [12.0, 14.0],
        })
        ha_open, ha_close = compute_heikin_ashi(df)
        assert np.allclose(ha_close, [11.0, 13.0])
        assert np.allclose(ha_open, [11.0, 11.0])

    def test_hybrid_formulas_bar0(self):
        df = pd.DataFrame({
            "Open": [10.0, 12.0], "High": [14.0, 16.0],
            "Low": [8.0, 10.0], "Close": [12.0, 14.0],
        })
        out = compute_hybrid_features(df)
        assert np.isclose(out["ha_open"].iloc[0], 11.0)
        assert np.isclose(out["ha_close"].iloc[0], 11.0)
        assert np.isclose(out["median_top"].iloc[0], 11.5)
        assert np.isclose(out["median_bottom"].iloc[0], 10.5)
        assert np.isclose(out["hybrid_open"].iloc[0], 10.5)
        assert np.isclose(out["hybrid_close"].iloc[0], 11.5)
        assert np.isclose(out["candle_top"].iloc[0], 11.5)
        assert np.isclose(out["candle_bottom"].iloc[0], 10.5)
        assert np.isclose(out["avg_candle"].iloc[0], 11.0)
        assert np.isclose(out["hlcc4"].iloc[0], 11.5)
        assert np.isclose(out["sourceformas"].iloc[0], 13.0)  # O<=C → (C+H)/2
        assert np.isclose(out["novelsrc"].iloc[0], 35.5 / 3.0)

    def test_hybrid_formulas_bar1(self):
        df = pd.DataFrame({
            "Open": [10.0, 12.0], "High": [14.0, 16.0],
            "Low": [8.0, 10.0], "Close": [12.0, 14.0],
        })
        out = compute_hybrid_features(df)
        assert np.isclose(out["median_top"].iloc[1], 13.5)
        assert np.isclose(out["median_bottom"].iloc[1], 11.5)
        assert np.isclose(out["hybrid_close"].iloc[1], 13.5)
        assert np.isclose(out["avg_candle"].iloc[1], 12.5)
        assert np.isclose(out["novelsrc"].iloc[1], 41.0 / 3.0)

    def test_sourceformas_bearish_bar(self):
        # Open > Close → (Open + Low) / 2
        df = pd.DataFrame({
            "Open": [14.0], "High": [15.0], "Low": [8.0], "Close": [10.0],
        })
        out = compute_hybrid_features(df)
        assert np.isclose(out["sourceformas"].iloc[0], 11.0)

    def test_atr_wilder(self):
        df = pd.DataFrame({
            "Open": [10.0, 12.0], "High": [14.0, 16.0],
            "Low": [8.0, 10.0], "Close": [12.0, 14.0],
        })
        atr = compute_atr(df, period=14)
        # TR0 = 6; TR1 = max(6, |16-12|, |10-12|) = 6 → ATR остаётся 6.
        assert np.allclose(atr, [6.0, 6.0])


# ══════════════════════════════════════════════════════════════════════
#  Выбор источника фракталов
# ══════════════════════════════════════════════════════════════════════
class TestFractalSeries:
    def test_standard_mode(self):
        df = _uptrend_stair()
        out = compute_hybrid_features(df)
        top, bottom = select_fractal_series(out, _std_params(fractal_source="standard"))
        assert np.allclose(top, out["High"].to_numpy())
        assert np.allclose(bottom, out["Low"].to_numpy())

    def test_candle_mode(self):
        out = compute_hybrid_features(_uptrend_stair())
        top, bottom = select_fractal_series(out, _std_params(fractal_source="candle"))
        assert np.allclose(top, out["candle_top"].to_numpy())
        assert np.allclose(bottom, out["candle_bottom"].to_numpy())

    def test_median_mode(self):
        out = compute_hybrid_features(_uptrend_stair())
        top, bottom = select_fractal_series(out, _std_params(fractal_source="median"))
        assert np.allclose(top, out["median_top"].to_numpy())
        assert np.allclose(bottom, out["median_bottom"].to_numpy())

    def test_median_candle_alpha(self):
        out = compute_hybrid_features(_uptrend_stair())
        top, bottom = select_fractal_series(out, _std_params(fractal_source="median_candle", alpha=0.5))
        expected_top = 0.5 * out["median_top"].to_numpy() + 0.5 * out["candle_top"].to_numpy()
        expected_bottom = 0.5 * out["median_bottom"].to_numpy() + 0.5 * out["candle_bottom"].to_numpy()
        assert np.allclose(top, expected_top)
        assert np.allclose(bottom, expected_bottom)

    def test_invalid_source(self):
        out = compute_hybrid_features(_uptrend_stair())
        with pytest.raises(ValueError):
            select_fractal_series(out, _std_params(fractal_source="bogus"))


# ══════════════════════════════════════════════════════════════════════
#  Фракталы
# ══════════════════════════════════════════════════════════════════════
class TestDetectFractals:
    def test_strict_mode_rejects_plateau(self):
        top = np.array([2.0, 3.0, 3.0, 2.0])
        bottom = np.array([0.0, 1.0, 1.0, 0.0])
        fh, fl = detect_fractals(top, bottom, left=1, right=1, strict=True)
        assert not fh.any()
        assert not fl.any()

    def test_non_strict_accepts_plateau_but_not_flat(self):
        top = np.array([2.0, 3.0, 3.0, 2.0])
        bottom = np.array([0.0, 1.0, 1.0, 0.0])
        fh, _ = detect_fractals(top, bottom, left=1, right=1, strict=False)
        assert fh[1] and fh[2]
        # Полностью плоская серия не даёт фракталов даже в нестрогом режиме
        flat = np.full(5, 1.0)
        fh2, fl2 = detect_fractals(flat, flat, left=2, right=2, strict=False)
        assert not fh2.any() and not fl2.any()

    def test_too_short_series(self):
        fh, fl = detect_fractals(np.arange(3.0), np.arange(3.0), left=2, right=2, strict=True)
        assert not fh.any() and not fl.any()


# ══════════════════════════════════════════════════════════════════════
#  build_trend: критерии приёмки
# ══════════════════════════════════════════════════════════════════════
class TestBuildTrend:
    def test_empty_df_returns_empty_with_columns(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        out = build_trend(df, _std_params())
        assert len(out) == 0
        assert set(out.columns) == {"Open", "High", "Low", "Close"} | set(RESULT_COLUMNS)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Open": [1.0]})
        with pytest.raises(ValueError):
            build_trend(df, _std_params())

    def test_output_columns(self):
        out = build_trend(_uptrend_stair(), _std_params())
        assert set(RESULT_COLUMNS) <= set(out.columns)
        # все расчётные колонки конечны
        for col in RESULT_COLUMNS:
            if out[col].dtype == bool:
                continue
            assert out[col].notna().all(), f"NaN в колонке {col}"

    def test_index_preserved(self):
        df = _uptrend_stair()
        out = build_trend(df, _std_params())
        pd.testing.assert_index_equal(out.index, df.index)

    # --- Приёмка: восходящий ряд ---
    def test_uptrend_strict_positive(self):
        out = build_trend(_uptrend_stair(), _std_params())
        assert out["trend_strict"].iloc[17] == 1
        assert (out["trend_strict"].iloc[:17] == 0).all()
        assert out["stage"].iloc[-1] == "UPTREND"

    # --- Приёмка: нисходящий ряд ---
    def test_downtrend_strict_negative(self):
        out = build_trend(_downtrend_stair(), _std_params())
        assert out["trend_strict"].iloc[17] == -1
        assert (out["trend_strict"].iloc[:17] == 0).all()
        assert out["stage"].iloc[-1] == "DOWNTREND"

    # --- Приёмка: нет lookahead ---
    def test_no_signal_before_confirmation(self):
        out = build_trend(_uptrend_stair(), _std_params())
        # Пик A на баре 3 → подтверждение только на баре 5
        assert out["fractal_high"].iloc[3]
        assert not out["fractal_high_confirmed"].iloc[4]
        assert out["fractal_high_confirmed"].iloc[5]
        assert (out["fractal_high_confirmed"].iloc[:5] == False).all()  # noqa: E712
        # Первый фрактал (бар 5) не даёт события (нет предыдущего для сравнения)
        assert not out["hh_event"].iloc[5]
        assert not out["lh_event"].iloc[5]
        # HH появляется только в момент подтверждения пика C (бар 13)
        assert (out["hh_event"].iloc[:13] == False).all()  # noqa: E712
        assert out["hh_event"].iloc[13]
        # HL — в момент подтверждения впадины D (бар 17)
        assert (out["hl_event"].iloc[:17] == False).all()  # noqa: E712
        assert out["hl_event"].iloc[17]

    # --- Приёмка: равные хаи/лои без порога ---
    def test_equal_swings_no_false_events(self):
        out = build_trend(_equal_swings(), _std_params())
        assert out["hh_event"].sum() == 0
        assert out["lh_event"].sum() == 0
        assert out["hl_event"].sum() == 0
        assert out["ll_event"].sum() == 0
        assert (out["last_high_label"] == 0).all()
        assert (out["last_low_label"] == 0).all()
        assert (out["trend_strict"] == 0).all()
        assert out["stage"].iloc[-1] == "RANGE"

    # --- Приёмка: novel-фильтр гасит противоречащий тренд ---
    def test_novel_filter_kills_counter_momentum(self):
        df = _uptrend_then_collapse()
        with_filter = build_trend(df, _std_params(use_novel_filter=True))
        without = build_trend(df, _std_params(use_novel_filter=False))
        # Структура всё ещё восходящая (события живы), но импульс novelsrc отрицательный
        assert without["trend_strict"].iloc[23] == 1
        assert with_filter["novel_momentum"].iloc[23] < 0
        assert with_filter["trend_strict"].iloc[23] == 0
        assert without["stage"].iloc[-1] == "UPTREND"

    def test_novel_filter_does_not_kill_uptrend(self):
        out = build_trend(_uptrend_stair(), _std_params(use_novel_filter=True))
        assert out["novel_momentum"].iloc[17] >= 0
        assert out["trend_strict"].iloc[17] == 1

    # --- Стадия REVERSAL ---
    def test_reversal_stage(self):
        out = build_trend(_reversal_mixed(), _std_params())
        assert out["hh_event"].iloc[13]
        assert out["ll_event"].iloc[17]
        assert out["trend_strict"].iloc[17] == 0
        assert out["stage"].iloc[-1] == "REVERSAL"

    # --- Зигзаг: чередующиеся пивоты ---
    def test_zigzag_alternates(self):
        out = build_trend(_uptrend_stair(), _std_params())
        zh = out.index[out["zigzag_high"]].tolist()
        zl = out.index[out["zigzag_low"]].tolist()
        assert [out.index.get_loc(i) for i in zh] == [3, 11]
        assert [out.index.get_loc(i) for i in zl] == [7, 15]
        # Пивоты чередуются по времени
        order = sorted(
            [(out.index.get_loc(i), "high") for i in zh]
            + [(out.index.get_loc(i), "low") for i in zl]
        )
        kinds = [k for _, k in order]
        assert kinds == ["high", "low", "high", "low"]

    # --- Score-каналы заполнены ---
    def test_score_channels(self):
        out = build_trend(_uptrend_stair(), _std_params())
        assert set(["trend_score", "trend_score_dir", "trend_final", "novel_momentum"]) <= set(out.columns)
        assert out["trend_score_dir"].isin([-1, 0, 1]).all()
        assert out["trend_final"].isin([-1, 0, 1]).all()

    def test_allow_score_fallback(self):
        # В развороте strict=0; score может дать направление, если разрешено
        out = build_trend(_reversal_mixed(), _std_params(allow_score_fallback=True))
        assert out["trend_final"].isin([-1, 0, 1]).all()
        # без fallback: final == strict
        out2 = build_trend(_reversal_mixed(), _std_params(allow_score_fallback=False))
        assert (out2["trend_final"] == out2["trend_strict"]).all()

    def test_novel_atr_mult_zero_is_safe(self):
        # novel_atr_mult=0 не должен давать NaN/Inf в momentum и score
        out = build_trend(_uptrend_stair(), _std_params(novel_atr_mult=0.0))
        assert np.isfinite(out["novel_momentum"]).all()
        assert np.isfinite(out["trend_score"]).all()
        assert (out["novel_momentum"] == 0.0).all()


# ══════════════════════════════════════════════════════════════════════
#  Фильтр минимального расстояния между фракталами
# ══════════════════════════════════════════════════════════════════════
class TestMinFractalDistance:
    KEY_COLUMNS = (
        "fractal_high_confirmed", "fractal_low_confirmed",
        "hh_event", "lh_event", "hl_event", "ll_event",
        "last_high_label", "last_low_label",
        "trend_strict", "trend_score", "trend_score_dir", "trend_final",
        "stage", "zigzag_high", "zigzag_low",
    )

    def test_negative_distance_rejected(self):
        with pytest.raises(ValueError):
            HybridTrendParams(min_fractal_distance=-1)

    def test_distance_zero_matches_default(self):
        # Дефолт 0 обязан давать идентичное поведение (нулевая регрессия)
        base = build_trend(_uptrend_stair(), _std_params())
        out = build_trend(_uptrend_stair(), _std_params(min_fractal_distance=0))
        for col in self.KEY_COLUMNS:
            assert (out[col] == base[col]).all(), f"столбец {col} разошёлся при distance=0"

    def test_distance_4_accepts_all_on_stair(self):
        # Пивоты 3/7/11/15 разнесены ровно на 4 бара → все приняты
        out = build_trend(_uptrend_stair(), _std_params(min_fractal_distance=4))
        assert out["fractal_high_accepted"].sum() == 2
        assert out["fractal_low_accepted"].sum() == 2
        assert out["hh_event"].iloc[13]
        assert out["hl_event"].iloc[17]
        assert out["trend_strict"].iloc[17] == 1
        assert out["stage"].iloc[-1] == "UPTREND"

    def test_distance_5_rejects_closer_pivots(self):
        # N=5: low@7 (7-3=4) и low@15 (15-11=4) отсечены; high@3 и high@11 приняты
        out = build_trend(_uptrend_stair(), _std_params(min_fractal_distance=5))
        assert out["fractal_high_accepted"].iloc[5]     # high@3 принят (первый всегда)
        assert out["fractal_high_accepted"].iloc[13]    # high@11: 11-3=8 ≥ 5
        assert not out["fractal_low_accepted"].any()    # оба лоу отсечены
        # События: HH есть (113 vs 109), HL нет
        assert out["hh_event"].iloc[13]
        assert not out["hl_event"].any()
        assert (out["trend_strict"] == 0).all()         # нет HL → строгого тренда нет
        assert out["stage"].iloc[-1] == "RANGE"
        # Зигзаг: принятые high@3 и high@11 однотипны подряд → в зигзаге только первый
        assert out["zigzag_high"].sum() == 1
        assert not out["zigzag_low"].any()

    def test_accepted_columns_present_in_empty_df(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        out = build_trend(df, _std_params(min_fractal_distance=5))
        assert len(out) == 0
        assert "fractal_high_accepted" in out.columns
        assert "fractal_low_accepted" in out.columns

    def test_event_age_expiry(self):
        # Очень короткий max_event_age: события протухают → тренд гаснет
        out = build_trend(_uptrend_stair(), _std_params(max_event_age=1))
        # HH подтверждён на баре 13: age 0 и 1 — валидны, age 2 — уже нет
        assert out["last_high_label"].iloc[13] == 1
        assert out["last_high_label"].iloc[14] == 1
        assert out["last_high_label"].iloc[15] == 0


# ══════════════════════════════════════════════════════════════════════
#  Сервис + сериализация
# ══════════════════════════════════════════════════════════════════════
class TestService:
    def test_fetch_and_analyze(self, monkeypatch):
        df = _uptrend_stair()

        def fake_fetch(self, ticker, timeframe, asset_type, limit):
            return df

        monkeypatch.setattr(NovelCandlesService, "_fetch_ohlcv", fake_fetch)
        svc = HybridTrendService()
        res = svc.fetch_and_analyze(
            "SPY", timeframe="1d", limit=20,
            params=_std_params(), with_trendlines=False,
        )
        assert res["ticker"] == "SPY"
        assert res["n_bars"] == 20
        assert len(res["bars"]) == 20
        assert len(res["novel_bars"]) == 20
        assert res["stage"] == "UPTREND"
        assert res["last_close"] is not None and res["atr"] is not None

        # События: HH@13 (пивот 11), HL@17 (пивот 15)
        labels = [e["label"] for e in res["events"]]
        assert labels == ["HH", "HL"]
        assert res["events"][0]["index"] == 13
        assert res["events"][0]["pivot_index"] == 11
        assert res["events"][1]["index"] == 17
        assert res["events"][1]["pivot_index"] == 15

        # Фракталы: 4 подтверждённых, чередование high/low, цена с бара пивота
        assert len(res["fractals"]) == 4
        assert [f["kind"] for f in res["fractals"]] == ["high", "low", "high", "low"]
        fr_hi = res["fractals"][0]
        assert fr_hi["index"] == 5 and fr_hi["pivot_index"] == 3
        assert fr_hi["price"] == pytest.approx(109.0)  # High на баре пивота (3), не подтверждения (5)
        fr_lo = res["fractals"][1]
        assert fr_lo["index"] == 9 and fr_lo["pivot_index"] == 7
        assert fr_lo["price"] == pytest.approx(101.0)  # Low на баре пивота (7)

        # Зигзаг: чередующиеся пивоты на барах возникновения
        assert [z["kind"] for z in res["zigzag"]] == ["high", "low", "high", "low"]
        assert [z["index"] for z in res["zigzag"]] == [3, 7, 11, 15]

        # Трендовые каналы
        assert res["trend_strict"][17] == 1
        assert len(res["trend_final"]) == 20

        # Ответ валиден по Pydantic-схеме
        schema = HybridStructureResponse(**res)
        assert schema.stage == "UPTREND"
        assert len(schema.bars) == 20
        assert len(schema.events) == 2

    def test_fetch_empty_raises(self, monkeypatch):
        monkeypatch.setattr(
            NovelCandlesService, "_fetch_ohlcv", lambda self, *a, **k: None
        )
        svc = HybridTrendService()
        with pytest.raises(ValueError):
            svc.fetch_and_analyze("SPY")

    def test_bad_timeframe_raises(self):
        svc = HybridTrendService()
        with pytest.raises(ValueError):
            svc.fetch_and_analyze("SPY", timeframe="3d")


# ══════════════════════════════════════════════════════════════════════
#  Pydantic-схемы
# ══════════════════════════════════════════════════════════════════════
class TestSchemas:
    def test_response_minimal(self):
        resp = HybridStructureResponse(
            ticker="SPY", timeframe="1d", asset_type="stock", n_bars=0,
        )
        assert resp.stage == "RANGE"
        assert resp.bars == []
        assert resp.trendlines is None

    def test_stage_literal(self):
        with pytest.raises(Exception):
            HybridStructureResponse(
                ticker="SPY", timeframe="1d", asset_type="stock", n_bars=0,
                stage="MOON",
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            HybridStructureResponse(
                ticker="SPY", timeframe="1d", asset_type="stock", n_bars=0,
                bogus=1,
            )
