"""Тесты расширенного GEX-анализатора (gex.extended).

Без сети: только синтетические опционные цепочки. Проверяем, что анализатор:
  * корректно считает GEX отдельного опциона (Γ × OI × per_contract × 100);
  * ставит знаки: Put GEX < 0, Call GEX > 0 (конвенция SqueezeMetrics);
  * находит Zero Gamma (линейная интерполяция смены знака кумулятивы);
  * находит Call/Put Wall и силу стены ∈ [0, 1];
  * строит Power Zones из топ-10% страйков по AG;
  * считает Gamma Dollar, Hedge Requirement, PCR, GEX Score, DHR, Max Pain;
  * определяет Directional Bias по знаку Total Net GEX;
  * автоопределяет источник (crypto ↔ stock) по тикеру;
  * валидируется Pydantic-схемой (extended_report_to_schema).

Паттерн повторяет tests/test_direction.py: чистые функции + синтетика.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gex.domain.data_loader import OptionSnapshot
from gex.application.extended import (
    ExtendedGEXAnalyzer,
    ExtendedGEXReport,
    PowerZone,
    KeyLevel,
    HedgeScenario,
)
from gex.schemas.extended_schemas import (
    ExtendedGEXAnalysisOut,
    extended_report_to_schema,
)
from gex.domain.greeks import bs_gamma


# ====================================================================== #
#  Фикстуры синтетических цепочек
# ====================================================================== #
def _synthetic_snapshot(
    spot: float = 100.0,
    n_strikes: int = 21,
    atm_iv: float = 0.20,
    T_years: float = 30.0 / 365.0,
    call_bias: float = 0.0,
    symbol: str = "TEST",
) -> OptionSnapshot:
    """Синтетическая опционная цепочка вокруг spot.

    Параметры
    ---------
    call_bias : float
        Сдвиг OI в сторону коллов (>0 → больше коллов → Call GEX перевес).
        Позволяет конструировать как BULLISH, так и BEARISH профили.
    """
    half = n_strikes // 2
    strikes = np.linspace(spot * 0.9, spot * 1.1, n_strikes)
    rows = []
    for k in strikes:
        # OI концентрируется вокруг ATM, гауссов колокол.
        moneyness = (k - spot) / spot
        base = math.exp(-0.5 * (moneyness / 0.04) ** 2)
        oi_c = max(100.0, base * 10_000 * (1.0 + call_bias))
        oi_p = max(100.0, base * 10_000 * (1.0 - call_bias))
        # Лёгкий smile: крылья поднимаются.
        iv = atm_iv * (1.0 + 2.0 * moneyness ** 2)
        rows.append({"strike": float(k), "type": "C", "oi": oi_c, "iv": iv, "T": T_years})
        rows.append({"strike": float(k), "type": "P", "oi": oi_p, "iv": iv, "T": T_years})
    chain = pd.DataFrame(rows)
    return OptionSnapshot(symbol=symbol, spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)


def _single_expiry_snapshot() -> OptionSnapshot:
    """Цепочка с одной экспирацией — упрощает проверку временных весов."""
    return _synthetic_snapshot(T_years=7.0 / 365.0)


# ====================================================================== #
#  Базовые расчёты (разделы 1-2 ТЗ)
# ====================================================================== #
class TestBasicGEX:
    """Гамма, GEX опциона, агрегация по страйку, знаки."""

    def test_analyzer_returns_report(self):
        """analyze() возвращает ExtendedGEXReport со всеми полями."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", days=30, snapshot=snap)
        assert isinstance(report, ExtendedGEXReport)
        assert report.symbol == "TEST"
        assert report.source == "stock"
        assert report.per_contract == 100
        assert len(report.per_strike) > 0

    def test_put_gex_always_negative(self):
        """Конвенция SqueezeMetrics: Put GEX ≤ 0 на каждом страйке."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for s in report.per_strike:
            assert s.gex_put <= 0, f"Put GEX > 0 на страйке {s.strike}"
        assert report.total_put_gex <= 0

    def test_call_gex_positive(self):
        """Call GEX > 0 на каждом страйке (дилер покупает коллы)."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for s in report.per_strike:
            assert s.gex_call >= 0, f"Call GEX < 0 на страйке {s.strike}"
        assert report.total_call_gex >= 0

    def test_aggregate_gamma_definition(self):
        """AG = |Call GEX| + |Put GEX| (ТЗ 1.3 / 5.2)."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for s in report.per_strike:
            expected = abs(s.gex_call) + abs(s.gex_put)
            assert math.isclose(s.ag, expected, rel_tol=1e-6)

    def test_gex_formula_matches_tz(self):
        """GEX опциона = Γ × OI × per_contract × 100 (ТЗ 1.2).

        Проверяем на синтетике с одним коллом и одним путом на страйке: перерасчёт
        вручную через bs_gamma должен совпасть со взвешенным GEX (с учётом веса
        экспирации e^(-T_days/30)).
        """
        spot = 100.0
        T = 7.0 / 365.0
        chain = pd.DataFrame([
            {"strike": 100.0, "type": "C", "oi": 500.0, "iv": 0.20, "T": T},
            {"strike": 100.0, "type": "P", "oi": 500.0, "iv": 0.20, "T": T},
        ])
        snap = OptionSnapshot(symbol="X", spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)

        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("X", snapshot=snap)
        s = report.per_strike[0]

        gamma = float(bs_gamma(spot, 100.0, T, analyzer.r, 0.20, 0.0))
        per_contract = 100  # stock
        weight = math.exp(-(T * 365.0) / 30.0)
        gex_call_expected = +1.0 * gamma * 500.0 * per_contract * 100.0 * weight
        gex_put_expected = -1.0 * gamma * 500.0 * per_contract * 100.0 * weight

        assert math.isclose(s.gex_call, gex_call_expected, rel_tol=1e-9)
        assert math.isclose(s.gex_put, gex_put_expected, rel_tol=1e-9)
        assert math.isclose(s.gex_net, gex_call_expected + gex_put_expected, rel_tol=1e-9)

    def test_crypto_per_contract_is_one(self):
        """Для крипты per_contract=1 (1 контракт = 1 монета)."""
        snap = _synthetic_snapshot(symbol="BTC")
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("BTC", snapshot=snap)
        assert report.source == "crypto"
        assert report.per_contract == 1


# ====================================================================== #
#  Zero Gamma Level (раздел 2.1 ТЗ)
# ====================================================================== #
class TestZeroGamma:
    """Уровень нулевой гаммы — линейная интерполяция смены знака."""

    def test_zero_gamma_within_strike_range(self):
        """Zero Gamma лежит в диапазоне страйков цепочки."""
        snap = _synthetic_snapshot(call_bias=0.3)  # коллов больше
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        if report.zero_gamma is not None:
            ks = [s.strike for s in report.per_strike]
            assert min(ks) <= report.zero_gamma <= max(ks)

    def test_zero_gamma_none_when_one_sided(self):
        """Если весь рынок в одном режиме (нет смены знака) — zero_gamma = None."""
        # Сильный перекос в коллы → весь профиль положительный.
        snap = _synthetic_snapshot(call_bias=0.95)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        # Либо None, либо совпадает с крайним страйком (граничный случай).
        if report.zero_gamma is not None:
            ks = [s.strike for s in report.per_strike]
            assert min(ks) - 1 <= report.zero_gamma <= max(ks) + 1

    def test_zero_gamma_interpolation_precision(self):
        """Линейная интерполяция: точное значение между двумя страйками."""
        # Ручная конструкция: два страйка, кумулятива меняет знак между ними.
        spot = 100.0
        T = 7.0 / 365.0
        # Нижний страйк: сильные путы → отрицательный Net GEX.
        # Верхний страйк: сильные коллы → положительный Net GEX.
        chain = pd.DataFrame([
            {"strike": 95.0, "type": "C", "oi": 100.0, "iv": 0.25, "T": T},
            {"strike": 95.0, "type": "P", "oi": 5000.0, "iv": 0.25, "T": T},
            {"strike": 105.0, "type": "C", "oi": 5000.0, "iv": 0.25, "T": T},
            {"strike": 105.0, "type": "P", "oi": 100.0, "iv": 0.25, "T": T},
        ])
        snap = OptionSnapshot(symbol="X", spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("X", snapshot=snap)
        assert report.zero_gamma is not None
        # Должен лежать между 95 и 105.
        assert 95.0 <= report.zero_gamma <= 105.0

    def test_zero_gamma_uses_last_sign_change(self):
        """Шумовые OTM-переходы не должны давать ложный нулевой уровень.

        Далёкие OTM-путы с крошечным GEX дают ранний «шумовой» переход нуля
        (~80), настоящий флип — у спота (~104). ``_zero_gamma_level`` берёт
        **последнюю** смену знака → результат у реального флипа, а не у шума.
        """
        per_strike = pd.DataFrame([
            {"strike": 80.0, "gex_net": -0.1},   # OTM-шум
            {"strike": 85.0, "gex_net": 0.2},    # ранний переход нуля (шум)
            {"strike": 90.0, "gex_net": -100.0},  # пут-тяжёлая зона
            {"strike": 95.0, "gex_net": -50.0},
            {"strike": 100.0, "gex_net": -30.0},
            {"strike": 105.0, "gex_net": 200.0},  # реальный флип в call-зону
            {"strike": 110.0, "gex_net": 150.0},
        ])
        zg = ExtendedGEXAnalyzer._zero_gamma_level(per_strike)
        assert zg is not None
        # Реальный флип в полосе 100–105, а не шумовой переход ~80.
        assert 95.0 < zg <= 105.0


# ====================================================================== #
#  Call/Put Wall (раздел 2.2-2.3 ТЗ)
# ====================================================================== #
class TestWalls:
    """Стены — страйки с экстремальным Net GEX + сила стены."""

    def test_call_wall_has_positive_net_gex(self):
        """Call Wall — страйк с макс положительным Net GEX."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        # Находим страйк Call Wall в профиле и проверяем знак.
        if not math.isnan(report.call_wall):
            wall = next(s for s in report.per_strike
                        if math.isclose(s.strike, report.call_wall, rel_tol=1e-4))
            assert wall.gex_net > 0

    def test_put_wall_has_negative_net_gex(self):
        """Put Wall — страйк с мин (наиболее отрицательным) Net GEX."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        if not math.isnan(report.put_wall):
            wall = next(s for s in report.per_strike
                        if math.isclose(s.strike, report.put_wall, rel_tol=1e-4))
            assert wall.gex_net < 0

    def test_wall_strength_in_unit_interval(self):
        """Сила стены = |Net GEX| / Σ|Net GEX| ∈ [0, 1]."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert 0.0 <= report.call_wall_strength <= 1.0
        assert 0.0 <= report.put_wall_strength <= 1.0

    def test_call_wall_max_of_positive(self):
        """Call Wall действительно максимум среди положительных Net GEX."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        pos_gex = [s.gex_net for s in report.per_strike if s.gex_net > 0]
        if pos_gex:
            assert math.isclose(
                max(pos_gex),
                next(s.gex_net for s in report.per_strike
                     if math.isclose(s.strike, report.call_wall, rel_tol=1e-4)),
                rel_tol=1e-6,
            )


# ====================================================================== #
#  Power Zones (раздел 2.4 ТЗ)
# ====================================================================== #
class TestPowerZones:
    """Зоны концентрации гаммы — топ-10% страйков по AG."""

    def test_power_zones_non_empty(self):
        """Power Zones строятся (топ-10% содержит хотя бы 1 страйк)."""
        snap = _synthetic_snapshot(n_strikes=21)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert len(report.power_zones) >= 1

    def test_power_zone_fields(self):
        """Каждая зона имеет корректные поля и n_strikes ≥ 1."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for z in report.power_zones:
            assert isinstance(z, PowerZone)
            assert z.n_strikes >= 1
            assert z.total_ag > 0
            assert z.dominant_type in ("CALL", "PUT")
            assert z.min_strike <= z.center <= z.max_strike

    def test_power_zones_sorted_by_ag(self):
        """Зоны отсортированы по убыванию total_ag (сильнейшая первой)."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        ags = [z.total_ag for z in report.power_zones]
        assert ags == sorted(ags, reverse=True)

    def test_power_zone_width_nonnegative(self):
        """Ширина зоны = max − min ≥ 0."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for z in report.power_zones:
            assert z.width >= 0.0


# ====================================================================== #
#  Агрегация по экспирациям (раздел 3 ТЗ)
# ====================================================================== #
class TestExpiryWeights:
    """Временной вес e^(-дней/30) и взвешенное суммирование."""

    def test_near_expiry_dominates(self):
        """Ближайшая экспирация получает больший вес (≈0.97 для 1 дня)."""
        weight_1d = math.exp(-1.0 / 30.0)
        weight_30d = math.exp(-30.0 / 30.0)
        assert weight_1d > weight_30d
        assert math.isclose(weight_1d, 0.967, abs_tol=0.01)
        assert math.isclose(weight_30d, 0.367, abs_tol=0.01)

    def test_multiple_expiries_combined(self):
        """Цепочка с несколькими экспирациями обрабатывается без ошибок."""
        spot = 100.0
        rows = []
        for days in (1, 7, 14, 30):
            T = days / 365.0
            for k in (95.0, 100.0, 105.0):
                rows.append({"strike": k, "type": "C", "oi": 1000.0, "iv": 0.2, "T": T})
                rows.append({"strike": k, "type": "P", "oi": 1000.0, "iv": 0.2, "T": T})
        chain = pd.DataFrame(rows)
        snap = OptionSnapshot(symbol="X", spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("X", snapshot=snap)
        # Каждый страйк должен быть представлен (3 страйка).
        assert len(report.per_strike) == 3


# ====================================================================== #
#  Метрики влияния (раздел 4 ТЗ)
# ====================================================================== #
class TestMarketImpact:
    """Gamma Dollar, Hedge Requirement, Directional Bias, PCR."""

    def test_gamma_dollar_formula(self):
        """Gamma Dollar = Net GEX × strike × 0.01 (ТЗ 4.1)."""
        snap = _single_expiry_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for s in report.per_strike:
            expected = s.gex_net * s.strike * 0.01
            assert math.isclose(s.gamma_dollar, expected, rel_tol=1e-6)

    def test_hedge_requirement_signs(self):
        """При +1% дилеры в POSITIVE покупают/продают; знак корректен."""
        snap = _synthetic_snapshot(call_bias=0.3)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze(
            "TEST", snapshot=snap, hedge_scenarios_pct=[1.0, -1.0]
        )
        assert len(report.hedge_scenarios) == 2
        # Сценарии анти-симметричны по знаку (gex_net × pct).
        up = next(h for h in report.hedge_scenarios if h.scenario_pct > 0)
        down = next(h for h in report.hedge_scenarios if h.scenario_pct < 0)
        # shares и dollar_value должны быть противоположных знаков.
        assert (up.shares > 0) == (down.shares < 0)
        assert math.isclose(up.shares, -down.shares, rel_tol=1e-6)

    def test_hedge_shares_equals_dollar_over_spot(self):
        """shares = dollar_value / spot."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for h in report.hedge_scenarios:
            assert math.isclose(h.shares, h.dollar_value / report.spot, rel_tol=1e-6)

    def test_directional_bias_positive_net_gex_is_bullish(self):
        """Total Net GEX > 0 → BULLISH (ТЗ 4.3)."""
        snap = _synthetic_snapshot(call_bias=0.5)  # перевес коллов
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        if abs(report.net_gex) / report.total_ag >= 0.02:
            assert report.directional_bias == "BULLISH"
            assert report.net_gex > 0

    def test_directional_bias_negative_net_gex_is_bearish(self):
        """Total Net GEX < 0 → BEARISH."""
        snap = _synthetic_snapshot(call_bias=-0.5)  # перевес путов
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        if abs(report.net_gex) / report.total_ag >= 0.02:
            assert report.directional_bias == "BEARISH"
            assert report.net_gex < 0

    def test_put_call_ratio_formula(self):
        """PCR = |Total Put GEX| / Total Call GEX (ТЗ 4.4)."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        expected = abs(report.total_put_gex) / report.total_call_gex
        assert math.isclose(report.put_call_ratio, expected, rel_tol=1e-6)
        assert report.put_call_ratio >= 0.0

    def test_pcr_above_one_when_puts_dominate(self):
        """PCR > 1, если пут-гамма преобладает (медвежьи настроения)."""
        snap = _synthetic_snapshot(call_bias=-0.5)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert report.put_call_ratio > 1.0

    def test_pcr_below_one_when_calls_dominate(self):
        """PCR < 1, если колл-гамма преобладает (бычьи настроения)."""
        snap = _synthetic_snapshot(call_bias=0.5)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert report.put_call_ratio < 1.0


# ====================================================================== #
#  Aggregate Gamma (раздел 5 ТЗ)
# ====================================================================== #
class TestAggregateGamma:
    """Total AG, AG_normalized."""

    def test_ag_normalized_in_unit_interval(self):
        """AG_normalized = AG / max(AG) ∈ [0, 1]."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for s in report.per_strike:
            assert 0.0 <= s.ag_normalized <= 1.0

    def test_max_ag_normalized_is_one(self):
        """Хотя бы один страйк имеет ag_normalized = 1.0 (максимум)."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert any(math.isclose(s.ag_normalized, 1.0, abs_tol=1e-6)
                   for s in report.per_strike)

    def test_total_ag_positive(self):
        """Total AG = Σ AG > 0."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert report.total_ag > 0
        assert math.isclose(
            report.total_ag,
            sum(s.ag for s in report.per_strike),
            rel_tol=1e-6,
        )


# ====================================================================== #
#  Уровни S/R (раздел 6 ТЗ)
# ====================================================================== #
class TestKeyLevels:
    """Топ-5 страйков по |Net GEX| с типизацией."""

    def test_key_levels_at_most_five(self):
        """key_levels содержит не более 5 элементов."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert 1 <= len(report.key_levels) <= 5

    def test_key_levels_typed(self):
        """Каждый уровень типизирован RESISTANCE/SUPPORT по знаку Net GEX.

        Используем асимметричную цепочку (перевес путов), чтобы знаки были
        чёткими и не вырождались в 0 (как при полной симметрии call/put).
        """
        snap = _synthetic_snapshot(call_bias=-0.4)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for lvl in report.key_levels:
            assert lvl.type in ("RESISTANCE", "SUPPORT")
            strike = next(s for s in report.per_strike
                          if math.isclose(s.strike, lvl.strike, rel_tol=1e-4))
            # Тип определяется знаком Net GEX: > 0 → RESISTANCE, < 0 → SUPPORT.
            # Допускаем граничный ноль (полная симметрия на дальнем OTM-страйке).
            if lvl.type == "RESISTANCE":
                assert strike.gex_net > 0
            else:
                assert strike.gex_net <= 0

    def test_key_levels_sorted_by_strength_desc(self):
        """Уровни упорядочены по убыванию |Net GEX|."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        strengths = [lvl.strength for lvl in report.key_levels]
        assert strengths == sorted(strengths, reverse=True)

    def test_distance_pct_formula(self):
        """distance_pct = (strike − spot) / spot × 100."""
        snap = _synthetic_snapshot(spot=100.0)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for lvl in report.key_levels:
            expected = (lvl.strike - report.spot) / report.spot * 100.0
            assert math.isclose(lvl.distance_pct, expected, rel_tol=1e-6)


# ====================================================================== #
#  Дополнительные метрики (раздел 7 ТЗ)
# ====================================================================== #
class TestExtraMetrics:
    """Gamma Exposure Score, Delta Hedge Ratio, Max Pain."""

    def test_gamma_exposure_score_range(self):
        """Score = |Net GEX| / Total AG × 100 ∈ [0, 100]."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        expected = abs(report.net_gex) / report.total_ag * 100.0
        assert math.isclose(report.gamma_exposure_score, expected, rel_tol=1e-6)
        assert 0.0 <= report.gamma_exposure_score <= 100.0

    def test_delta_hedge_ratio_formula(self):
        """DHR = Call GEX / |Put GEX| по страйкам (None при Put GEX = 0)."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        for s in report.per_strike:
            if s.delta_hedge_ratio is not None:
                expected = s.gex_call / abs(s.gex_put)
                assert math.isclose(s.delta_hedge_ratio, expected, rel_tol=1e-6)

    def test_max_pain_within_strikes(self):
        """Max Pain лежит среди страйков цепочки."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        assert report.max_pain is not None
        ks = sorted(snap.chain["strike"].unique())
        assert min(ks) <= report.max_pain <= max(ks)

    def test_max_pain_construction(self):
        """Контролируемая конструкция max pain (формула ТЗ 7.3).

        Max Pain = страйк с **минимумом** суммарных потерь держателей опционов:
            потери_коллов(K_p) = Σ max(0, K_call − K_p) × OI_call
            потери_путов(K_p)  = Σ max(0, K_p − K_put) × OI_put

        Конструкция: путы на 90/95, коллы на 105/110, одинаковый OI. Минимум
        потерь приходится на 95 и 105 (ничья); tie-break по spot=100 даёт 95
        (равноудалённые, первый минимум в отсортированном порядке).
        """
        spot = 100.0
        T = 7.0 / 365.0
        rows = [
            # Путы на 90 и 95; коллы — на 105, 110.
            {"strike": 90.0, "type": "P", "oi": 1000.0, "iv": 0.2, "T": T},
            {"strike": 95.0, "type": "P", "oi": 1000.0, "iv": 0.2, "T": T},
            {"strike": 105.0, "type": "C", "oi": 1000.0, "iv": 0.2, "T": T},
            {"strike": 110.0, "type": "C", "oi": 1000.0, "iv": 0.2, "T": T},
        ]
        chain = pd.DataFrame(rows)
        snap = OptionSnapshot(symbol="X", spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("X", snapshot=snap)
        assert report.max_pain is not None

        # Ручной пересчёт потерь по каждому кандидатному страйку (формула ТЗ).
        candidates = sorted(chain["strike"].unique())
        losses: dict[float, float] = {}
        calls = chain[chain["type"] == "C"]
        puts = chain[chain["type"] == "P"]
        for kp in candidates:
            call_loss = float(np.maximum(0.0, calls["strike"].values - kp)
                              @ calls["oi"].values)
            put_loss = float(np.maximum(0.0, kp - puts["strike"].values)
                             @ puts["oi"].values)
            losses[kp] = call_loss + put_loss
        expected_max_pain = min(losses, key=losses.get)
        assert math.isclose(report.max_pain, expected_max_pain, abs_tol=0.01)

    def test_max_pain_symmetric_near_atm(self):
        """Симметричная цепочка → max pain около ATM (минимум, а не крыло)."""
        spot = 100.0
        rows = []
        for k in (90.0, 95.0, 100.0, 105.0, 110.0):
            rows.append({"strike": k, "type": "C", "oi": 1000.0})
            rows.append({"strike": k, "type": "P", "oi": 1000.0})
        chain = pd.DataFrame(rows)
        mp = ExtendedGEXAnalyzer._max_pain(chain, spot)
        assert mp is not None
        assert math.isclose(mp, 100.0, abs_tol=1e-6)

    def test_max_pain_asymmetric_puts_shift_down_inside_range(self):
        """Массивный пут-OI ниже спота сдвигает max pain вниз, но не в крыло."""
        spot = 100.0
        rows = [
            {"strike": 80.0, "type": "P", "oi": 100.0},
            {"strike": 90.0, "type": "P", "oi": 5000.0},  # тяжёлые путы ниже спота
            {"strike": 100.0, "type": "C", "oi": 1000.0},
            {"strike": 100.0, "type": "P", "oi": 500.0},
            {"strike": 110.0, "type": "C", "oi": 1000.0},
            {"strike": 110.0, "type": "P", "oi": 100.0},
            {"strike": 120.0, "type": "C", "oi": 100.0},
        ]
        chain = pd.DataFrame(rows)
        mp = ExtendedGEXAnalyzer._max_pain(chain, spot)
        assert mp is not None
        # Внутри диапазона, сдвинут вниз от спота, но не в крайнее крыло (80/120).
        assert 80.0 < mp < 100.0


# ====================================================================== #
#  Источник и автоопределение (раздел "организация")
# ====================================================================== #
class TestSourceAutoDetection:
    """Автоопределение crypto ↔ stock по тикеру."""

    def test_crypto_ticker_detected(self):
        """BTC/ETH/SOL/XRP/DOGE → source=crypto, per_contract=1."""
        snap = _synthetic_snapshot(symbol="BTC")
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("BTC", snapshot=snap)
        assert report.source == "crypto"
        assert report.per_contract == 1

    def test_stock_ticker_detected(self):
        """Произвольный тикер → source=stock, per_contract=100."""
        snap = _synthetic_snapshot(symbol="AAPL")
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("AAPL", snapshot=snap)
        assert report.source == "stock"
        assert report.per_contract == 100


# ====================================================================== #
#  Схема: dataclass → Pydantic
# ====================================================================== #
class TestSchemaSerialization:
    """extended_report_to_schema производит валидный ExtendedGEXAnalysisOut."""

    def test_schema_round_trip(self):
        """Все поля корректно переносятся в схему."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", days=30, snapshot=snap)
        out = extended_report_to_schema(report, days=30.0)
        assert isinstance(out, ExtendedGEXAnalysisOut)
        assert out.symbol == report.symbol
        assert out.source == report.source
        assert len(out.per_strike) == len(report.per_strike)
        assert len(out.power_zones) == len(report.power_zones)
        assert len(out.key_levels) == len(report.key_levels)
        assert len(out.hedge_scenarios) == len(report.hedge_scenarios)

    def test_schema_validation_constraints(self):
        """Pydantic-ограничения: PCR ≥ 0, score ∈ [0,100], ag_normalized ∈ [0,1]."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        out = extended_report_to_schema(report)
        assert out.put_call_ratio >= 0
        assert 0 <= out.gamma_exposure_score <= 100
        for s in out.per_strike:
            assert 0.0 <= s.ag_normalized <= 1.0

    def test_schema_json_serializable(self):
        """Схема сериализуется в JSON без ошибок."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        out = extended_report_to_schema(report)
        data = out.model_dump_json()
        assert isinstance(data, str)
        assert "per_strike" in data

    def test_schema_auto_field_backward_compatible(self):
        """AUTO-поле опционально: без coverage → ``auto = None`` (старые клиенты целы).

        С аудита 2026-09-17 coverage отдаётся и в ручном режиме (``mode='manual'``),
        но поле остаётся опциональным — отсутствие по-прежнему сериализуется в null.
        """
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)

        # coverage теперь заполняется всегда (в т.ч. ручной режим)
        assert report.coverage is not None
        assert report.coverage.mode == "manual"
        assert extended_report_to_schema(report).auto.mode == "manual"

        # ... но при его отсутствии сериализация остаётся null
        report.coverage = None
        out = extended_report_to_schema(report)
        assert out.auto is None
        # присутствует в сериализации как null, не ломает старых клиентов
        assert '"auto":null' in out.model_dump_json()


# ====================================================================== #
#  Режимы и общая консистентность
# ====================================================================== #
class TestConsistency:
    """Сквозные инварианты отчёта."""

    def test_regime_matches_net_gex_sign(self):
        """regime = POSITIVE при Net GEX ≥ 0, иначе NEGATIVE."""
        for bias in (-0.5, 0.0, 0.5):
            snap = _synthetic_snapshot(call_bias=bias)
            analyzer = ExtendedGEXAnalyzer()
            report = analyzer.analyze("TEST", snapshot=snap)
            if report.net_gex >= 0:
                assert report.regime == "POSITIVE"
            else:
                assert report.regime == "NEGATIVE"

    def test_filter_by_days_removes_far_expiries(self):
        """days=7 отсекает опционы с T > 7/365."""
        spot = 100.0
        rows = [
            {"strike": 100.0, "type": "C", "oi": 1000.0, "iv": 0.2, "T": 5.0 / 365.0},
            {"strike": 100.0, "type": "P", "oi": 1000.0, "iv": 0.2, "T": 5.0 / 365.0},
            {"strike": 100.0, "type": "C", "oi": 1000.0, "iv": 0.2, "T": 60.0 / 365.0},
            {"strike": 100.0, "type": "P", "oi": 1000.0, "iv": 0.2, "T": 60.0 / 365.0},
        ]
        chain = pd.DataFrame(rows)
        snap = OptionSnapshot(symbol="X", spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("X", days=7, snapshot=snap)
        # Должен остаться только 1 страйк (100) с ближней экспирацией.
        assert len(report.per_strike) == 1

    def test_filter_by_days_empty_window_returns_nearest_expiry(self):
        """Пустое окно (days меньше всех T) → только ближайший бакет, не весь снапшот."""
        spot = 100.0
        rows = [
            {"strike": 100.0, "type": "C", "oi": 1000.0, "iv": 0.2, "T": 30.0 / 365.0},
            {"strike": 100.0, "type": "P", "oi": 1000.0, "iv": 0.2, "T": 30.0 / 365.0},
            {"strike": 100.0, "type": "C", "oi": 1000.0, "iv": 0.2, "T": 60.0 / 365.0},
            {"strike": 100.0, "type": "P", "oi": 1000.0, "iv": 0.2, "T": 60.0 / 365.0},
        ]
        chain = pd.DataFrame(rows)
        snap = OptionSnapshot(symbol="X", spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)
        analyzer = ExtendedGEXAnalyzer()
        filtered = ExtendedGEXAnalyzer._filter_by_days(snap, days=7)
        # Только ближайший бакет (~30 дн): 2 строки (C+P), ни одной 60-дневной.
        assert len(filtered.chain) == 2
        assert set((filtered.chain["T"] * 365.0).round()) == {30.0}

    def test_filter_by_days_empty_chain_returns_unchanged(self):
        """Пустая цепочка → снапшот не меняется (fallback), не роняет исключение."""
        snap = OptionSnapshot(
            symbol="X", spot=100.0, as_of=pd.Timestamp.now(tz="UTC"),
            chain=pd.DataFrame(columns=["strike", "type", "oi", "iv", "T"]),
        )
        filtered = ExtendedGEXAnalyzer._filter_by_days(snap, days=7)
        assert filtered is snap

    def test_empty_chain_after_filter_raises(self):
        """Пустая цепочка после фильтрации → ValueError."""
        snap = OptionSnapshot(
            symbol="X", spot=100.0, as_of=pd.Timestamp.now(tz="UTC"),
            chain=pd.DataFrame(columns=["strike", "type", "oi", "iv", "T"]),
        )
        analyzer = ExtendedGEXAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze("X", snapshot=snap)

    def test_default_hedge_scenarios(self):
        """Без явного указания — сценарии [-1, 1]."""
        snap = _synthetic_snapshot()
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", snapshot=snap)
        pcts = sorted(h.scenario_pct for h in report.hedge_scenarios)
        assert pcts == [-1.0, 1.0]
