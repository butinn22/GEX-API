"""Тесты прогноза фундаментальных метрик и сценарного калькулятора.

Покрывают: WMA (веса и тренд), линейную регрессию, комбинированный прогноз,
CAGR/YoY, детект аномалий (порог + выход за тренд), сглаживание аномалий,
сервис-оркестратор (6 метрик, PEG, квартальный рост) и API
(GET /forecast, POST /valuation/calculator).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from gex.auth.models import User
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.schemas.sec_forecast import ValuationRequest
from gex.application.sec.sec_forecast import (
    SecForecastService,
    build_metric_forecast,
    cagr,
    calculate_valuation,
    combined_forecast,
    confidence_interval,
    detect_anomalies,
    forecast_error,
    linreg,
    linreg_forecast,
    scenario_forecast,
    smooth_anomalies,
    trend_regime,
    wma_forecast,
    wma_last,
    wma_series,
    wma_trend,
    yoy_growth,
)


# ══════════════════════════════════════════════════════════════════════
#  Фикстуры (переиспользуем фабрики фактов из test_sec_edgar)
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def db_ready():
    recreate_tables()
    yield
    recreate_tables()


@pytest.fixture
def apple_core():
    """Core-данные AAPL из синтетических фактов (без сети)."""
    from tests.test_sec_edgar import _apple_facts  # переиспользуем фабрику

    return _apple_facts()


def _apple_facts_growing_eps():
    """Apple-факты, но EPS растёт 5 лет (для PEG-тестов)."""
    from tests.test_sec_edgar import _apple_facts, _fact_row

    facts = _apple_facts()
    facts["facts"]["us-gaap"]["EarningsPerShareBasic"]["units"]["USD"] = [
        _fact_row(end="2020-09-30", start="2019-10-01", val=3.0, fy=2020),
        _fact_row(end="2021-09-30", start="2020-10-01", val=3.6, fy=2021),
        _fact_row(end="2022-09-30", start="2021-10-01", val=4.2, fy=2022),
        _fact_row(end="2023-09-30", start="2022-10-01", val=5.0, fy=2023),
        _fact_row(end="2024-09-28", start="2023-10-01", val=6.08, fy=2024, filed="2024-11-01"),
    ]
    return facts


def _patch_edgar(monkeypatch, facts):
    monkeypatch.setattr(
        "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
        lambda redis=None: {"AAPL": "0000320193"},
    )
    monkeypatch.setattr("gex.application.sec.sec_fundamentals.get_company_facts", lambda cik: facts)


# ══════════════════════════════════════════════════════════════════════
#  1. WMA
# ══════════════════════════════════════════════════════════════════════
class TestWma:
    def test_wma_last_weights(self):
        """Веса N..1: последнее значение получает максимальный вес."""
        values = [1, 2, 3, 4, 5]
        # (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = 55/15
        assert wma_last(values, 5) == pytest.approx(55 / 15)
        # Последнее значение влияет сильнее: wma(5 значений) ближе к 5
        assert wma_last([1, 1, 1, 1, 100], 5) == pytest.approx((1 + 2 + 3 + 4 + 500) / 15)

    def test_wma_last_short_series(self):
        assert wma_last([10, 20], 5) == pytest.approx((10 + 40) / 3)
        assert wma_last([], 5) is None

    def test_wma_series_progress(self):
        s = wma_series([1, 2, 3, 4, 5], 3)
        assert s[0] is None and s[1] is None
        assert s[2] == pytest.approx((1 + 4 + 9) / 6)
        assert s[4] == pytest.approx((3 + 8 + 15) / 6)

    def test_wma_trend_linear(self):
        """Линейный ряд 1..10, окно 5: тренд ≈ 1 (прирост WMA за шаг)."""
        assert wma_trend(list(range(1, 11)), 5) == pytest.approx(1.0, abs=1e-9)

    def test_wma_trend_short_fallback(self):
        """Меньше window+1 точек: fallback на среднее приращение значений."""
        trend = wma_trend([10, 12, 14], 5)
        assert trend == pytest.approx(2.0)

    def test_wma_forecast(self):
        # [1..5], окно 5: base = 55/15, тренд 1 → h1 = 55/15 + 1
        assert wma_forecast_value([1, 2, 3, 4, 5], 1) == pytest.approx(55 / 15 + 1)


def wma_forecast_value(values, horizon):
    from gex.application.sec.sec_forecast import wma_forecast

    return wma_forecast(values, horizon, 5)


# ══════════════════════════════════════════════════════════════════════
#  2. Линейная регрессия
# ══════════════════════════════════════════════════════════════════════
class TestLinreg:
    def test_linreg_exact_line(self):
        a, b = linreg([1.0, 2.0, 3.0, 4.0, 5.0])
        assert a == pytest.approx(1.0)
        assert b == pytest.approx(1.0)

    def test_linreg_forecast(self):
        assert linreg_forecast([100.0, 110.0, 120.0], 1) == pytest.approx(130.0)
        assert linreg_forecast([100.0, 110.0, 120.0], 3) == pytest.approx(150.0)
        assert linreg_forecast([], 1) is None

    def test_linreg_flat(self):
        a, b = linreg([5.0, 5.0, 5.0])
        assert b == pytest.approx(0.0)
        assert a == pytest.approx(5.0)


# ══════════════════════════════════════════════════════════════════════
#  3. Комбинированный прогноз + темпы
# ══════════════════════════════════════════════════════════════════════
class TestCombinedAndGrowth:
    def test_combined_alpha_mix(self):
        values = [100.0, 110.0, 120.0]
        w = wma_forecast(values, 1, 5)  # WMA-прогноз (база + тренд)
        lr = linreg_forecast(values, 1)
        assert combined_forecast(values, 1, alpha=0.5) == pytest.approx(0.5 * w + 0.5 * lr)
        # alpha=1 → чистый WMA-прогноз; alpha=0 → чистый LinReg
        assert combined_forecast(values, 1, alpha=1.0) == pytest.approx(w)
        assert combined_forecast(values, 1, alpha=0.0) == pytest.approx(lr)

    def test_yoy_growth(self):
        assert yoy_growth([100.0, 110.0]) == pytest.approx(0.1)
        assert yoy_growth([100.0]) is None
        assert yoy_growth([100.0, -110.0]) is None  # неположительное значение

    def test_cagr(self):
        assert cagr([100.0, 110.0, 121.0], 2) == pytest.approx(0.1)
        assert cagr([100.0, 110.0], 3) is None  # мало точек
        assert cagr([100.0, 0.0, 121.0], 2) is None


# ══════════════════════════════════════════════════════════════════════
#  4. Аномалии
# ══════════════════════════════════════════════════════════════════════
class TestAnomalies:
    def test_spike_detected(self):
        """Скачок +82% при стабильном росте ~5% — аномалия на индексе 3."""
        values = [100.0, 105.0, 110.0, 200.0, 210.0]
        assert detect_anomalies(values, threshold=0.6) == [3]

    def test_stable_series_no_anomaly(self):
        assert detect_anomalies([100.0, 105.0, 110.0, 115.0], threshold=0.6) == []

    def test_consistently_high_growth_not_anomaly(self):
        """Компания стабильно растёт на ~70% каждый год — это её тренд."""
        values = [100.0, 170.0, 289.0, 491.3]
        assert detect_anomalies(values, threshold=0.6) == []

    def test_threshold_tunable(self):
        values = [100.0, 105.0, 130.0, 136.5]  # скачок +24%
        assert detect_anomalies(values, threshold=0.6) == []
        assert detect_anomalies(values, threshold=0.2) == [2]

    def test_negative_growth_anomaly(self):
        """Падение на 67% — аномалия (пример из ТЗ)."""
        values = [100.0, 110.0, 120.0, 40.0, 42.0]
        assert detect_anomalies(values, threshold=0.6) == [3]

    def test_smooth_replaces_with_median_growth(self):
        values = [100.0, 105.0, 110.0, 200.0, 210.0]
        smoothed = smooth_anomalies(values, [3])
        # Медиана чистых темпов: (0.05 + 0.047619)/2 ≈ 0.04881
        med = (0.05 + 110 / 105 - 1) / 2
        assert smoothed[3] == pytest.approx(110.0 * (1 + med), rel=1e-6)
        # Остальные значения не тронуты
        assert smoothed[:3] == values[:3] and smoothed[4] == values[4]

    def test_smooth_no_anomalies_returns_copy(self):
        values = [100.0, 105.0]
        assert smooth_anomalies(values, []) == values


# ══════════════════════════════════════════════════════════════════════
#  5. Сборка прогноза метрики
# ══════════════════════════════════════════════════════════════════════
def _history(values: list[float]) -> list[dict]:
    base = datetime(2020, 9, 30)
    return [
        {"end": (base + timedelta(days=365 * i)).date(), "fy": 2020 + i, "value": v}
        for i, v in enumerate(values)
    ]


class TestMetricForecast:
    def test_basic_forecast(self):
        out = build_metric_forecast(_history([100.0, 110.0, 120.0, 130.0, 140.0]))
        assert out["growth"]["yoy"] == pytest.approx(140 / 130 - 1)
        # CAGR за 3 года: окно [110,120,130,140] → (140/110)^(1/3) − 1
        assert out["growth"]["cagr_3y"] == pytest.approx((140 / 110) ** (1 / 3) - 1)
        assert out["forecast"]["linreg"]["h1"] == pytest.approx(150.0)
        assert out["forecast"]["combined"]["h3"] is not None
        assert out["anomaly"] is None
        assert out["scenarios"] is None


class TestRecencyBlend:
    """Слайдер «влияние последнего отчёта»: 60% базово, до 100%."""

    def test_blend_100_pure_last_report(self):
        """last_weight=1.0: прогноз = траектория последнего отчёта (120·1.2^h)."""
        out = build_metric_forecast(_history([100.0, 100.0, 100.0, 100.0, 120.0]), last_weight=1.0)
        assert out["forecast"]["combined"]["h1"] == pytest.approx(120.0 * 1.2)
        assert out["forecast"]["combined"]["h2"] == pytest.approx(120.0 * 1.2 ** 2)
        assert out["forecast"]["combined"]["h3"] == pytest.approx(120.0 * 1.2 ** 3)

    def test_blend_60_default(self):
        """Базово 60%: 0.6·траектория последнего отчёта + 0.4·модель."""
        out = build_metric_forecast(_history([100.0, 100.0, 100.0, 100.0, 120.0]))
        model = combined_forecast([100.0, 100.0, 100.0, 100.0, 120.0], 1, 0.5, 5)
        assert out["forecast"]["combined"]["h1"] == pytest.approx(0.6 * 144.0 + 0.4 * model)

    def test_blend_80_between(self):
        out = build_metric_forecast(_history([100.0, 100.0, 100.0, 100.0, 120.0]), last_weight=0.8)
        model = combined_forecast([100.0, 100.0, 100.0, 100.0, 120.0], 1, 0.5, 5)
        assert out["forecast"]["combined"]["h1"] == pytest.approx(0.8 * 144.0 + 0.2 * model)

    def test_blend_ignored_when_last_not_positive(self):
        """prev ≤ 0 — темпа последнего отчёта нет → прогноз модели без смеси."""
        out = build_metric_forecast(_history([-10.0, -5.0, 6.0]), last_weight=1.0)
        assert out["forecast"]["combined"]["h1"] == combined_forecast([-10.0, -5.0, 6.0], 1, 0.5, 5)

    def test_quarterly_blend(self):
        """Квартальный прогноз тоже учитывает последний отчёт (q1 = 110·1.1)."""
        from datetime import date
        from gex.application.sec.sec_forecast import build_quarterly_forecast

        hist = [
            {"end": date(2023, 9, 30), "value": 100.0},
            {"end": date(2023, 12, 31), "value": 110.0},
        ]
        out = build_quarterly_forecast(hist, last_weight=1.0)
        assert out["forecast"]["combined"]["q1"] == pytest.approx(110.0 * 1.1)
        assert out["forecast"]["combined"]["q2"] == pytest.approx(110.0 * 1.1 ** 2)
        assert out["forecast"]["combined"]["q3"] == pytest.approx(110.0 * 1.1 ** 3)

    def test_anomaly_scenarios(self):
        out = build_metric_forecast(
            _history([100.0, 105.0, 110.0, 200.0, 210.0]), anomaly_threshold=0.6
        )
        assert out["anomaly"]["detected"] is True
        assert out["anomaly"]["change"] == pytest.approx(200 / 110 - 1)
        assert out["scenarios"] is not None
        # keep = прогноз на исходных данных, smooth — на сглаженных
        assert out["scenarios"]["keep"]["h1"] == out["forecast"]["combined"]["h1"]
        assert out["scenarios"]["smooth"]["h1"] is not None

    def test_short_history(self):
        out = build_metric_forecast(_history([100.0, 110.0]))
        assert out["forecast"] is not None
        assert out["anomaly"] is None  # мало точек — без детекта

    def test_empty_history_has_regime(self):
        """Пустой ряд (например, у банка нет операционной прибыли) не ломает схему."""
        out = build_metric_forecast([])
        assert out["history"] == []
        assert out["regime"]["regime"] == "unknown"
        assert out["forecast"] is None


# ══════════════════════════════════════════════════════════════════════
#  6. Калькулятор оценки
# ══════════════════════════════════════════════════════════════════════
class TestValuation:
    def test_price_and_pe_give_eps(self):
        out = calculate_valuation({"price": 100.0, "pe": 20.0})
        assert out["eps"] == pytest.approx(5.0)
        assert out["earnings"] is None

    def test_price_and_eps_give_pe(self):
        out = calculate_valuation({"price": 100.0, "eps": 5.0})
        assert out["pe"] == pytest.approx(20.0)

    def test_pe_and_eps_give_price(self):
        out = calculate_valuation({"pe": 20.0, "eps": 5.0})
        assert out["price"] == pytest.approx(100.0)

    def test_earnings_and_shares(self):
        out = calculate_valuation({"price": 100.0, "eps": 5.0, "shares": 1e9, "earnings": 5e9})
        assert out["earnings"] == pytest.approx(5e9)
        assert out["market_cap"] == pytest.approx(100e9)

    def test_peg_positive(self):
        out = calculate_valuation({"price": 100.0, "eps": 5.0, "growth": 0.15})
        assert out["peg"] == pytest.approx(20.0 / 15.0)
        assert out["peg_warning"] is None

    def test_peg_zero_growth_warning(self):
        out = calculate_valuation({"price": 100.0, "eps": 5.0, "growth": 0.0})
        assert out["peg"] is None
        assert out["peg_warning"]

    def test_request_validation_two_of_three(self):
        ValuationRequest(price=100.0, pe=20.0)  # ок
        ValuationRequest(price=100.0, eps=5.0)  # ок
        ValuationRequest(pe=20.0, eps=5.0)      # ок
        with pytest.raises(ValueError):
            ValuationRequest(price=100.0)       # мало
        with pytest.raises(ValueError):
            ValuationRequest(earnings=5e9, shares=1e9)  # нет price/pe/eps


# ══════════════════════════════════════════════════════════════════════
#  7. Сервис-оркестратор
# ══════════════════════════════════════════════════════════════════════
class TestForecastService:
    def test_forecast_quarterly_series(self, db_ready, monkeypatch, apple_core):
        """Квартальные ряды (3M-строки) строятся и попадают в ответ forecast."""
        from tests.test_sec_edgar import _fact_row

        facts = apple_core
        qrows = lambda tag, rows: facts["facts"]["us-gaap"].setdefault(tag, {}).setdefault("units", {}).setdefault("USD", []).extend(rows)
        # 4 квартала по каждой метрике (duration 89-90 дней)
        qrows("RevenueFromContractWithCustomerExcludingAssessedTax", [
            _fact_row(end="2023-12-30", start="2023-10-01", form="10-Q", fp="Q1", val=90e9),
            _fact_row(end="2024-03-30", start="2024-01-01", form="10-Q", fp="Q2", val=92e9),
            _fact_row(end="2024-06-29", start="2024-04-01", form="10-Q", fp="Q3", val=95e9),
            _fact_row(end="2024-09-28", start="2024-07-01", form="10-Q", fp="Q4", val=98e9),
        ])
        qrows("EarningsPerShareBasic", [
            _fact_row(end="2023-12-30", start="2023-10-01", form="10-Q", fp="Q1", val=1.5),
            _fact_row(end="2024-03-30", start="2024-01-01", form="10-Q", fp="Q2", val=1.6),
            _fact_row(end="2024-06-29", start="2024-04-01", form="10-Q", fp="Q3", val=1.7),
            _fact_row(end="2024-09-28", start="2024-07-01", form="10-Q", fp="Q4", val=1.8),
        ])
        qrows("NetCashProvidedByUsedInOperatingActivities", [
            _fact_row(end="2023-12-30", start="2023-10-01", form="10-Q", fp="Q1", val=28e9),
            _fact_row(end="2024-03-30", start="2024-01-01", form="10-Q", fp="Q2", val=29e9),
            _fact_row(end="2024-06-29", start="2024-04-01", form="10-Q", fp="Q3", val=30e9),
            _fact_row(end="2024-09-28", start="2024-07-01", form="10-Q", fp="Q4", val=31e9),
        ])
        qrows("PaymentsToAcquirePropertyPlantAndEquipment", [
            _fact_row(end="2023-12-30", start="2023-10-01", form="10-Q", fp="Q1", val=-2e9),
            _fact_row(end="2024-03-30", start="2024-01-01", form="10-Q", fp="Q2", val=-2.1e9),
            _fact_row(end="2024-06-29", start="2024-04-01", form="10-Q", fp="Q3", val=-2.2e9),
            _fact_row(end="2024-09-28", start="2024-07-01", form="10-Q", fp="Q4", val=-2.3e9),
        ])
        qrows("OperatingIncomeLoss", [
            _fact_row(end="2023-12-30", start="2023-10-01", form="10-Q", fp="Q1", val=30e9),
            _fact_row(end="2024-03-30", start="2024-01-01", form="10-Q", fp="Q2", val=31e9),
            _fact_row(end="2024-06-29", start="2024-04-01", form="10-Q", fp="Q3", val=32e9),
            _fact_row(end="2024-09-28", start="2024-07-01", form="10-Q", fp="Q4", val=33e9),
        ])

        _patch_edgar(monkeypatch, facts)
        svc = SecForecastService(redis_client=None)
        core = svc.get_forecast_core("AAPL")

        q = core["quarterly"]
        assert "revenue" in q and "eps" in q and "fcf" in q and "operating_margin" in q
        assert "net_debt_to_ebitda" not in q and "roe" not in q  # balance-ratio — только годовые
        # quarterly вложен и в КАЖДУЮ метрику (фронт рисует переключатель по metrics[k].quarterly)
        assert core["metrics"]["revenue"]["quarterly"]["history"]
        assert core["metrics"]["eps"]["quarterly"]["history"]
        assert "quarterly" not in core["metrics"]["net_debt_to_ebitda"]

        rev = q["revenue"]
        assert len(rev["history"]) == 4
        assert rev["history"][-1]["end"].isoformat() == "2024-09-28"
        assert rev["history"][-1]["value"] == pytest.approx(98e9)
        assert rev["forecast"]["combined"]["q1"] is not None  # 1 квартал вперёд
        assert rev["forecast"]["combined"]["q2"] is not None
        assert rev["forecast"]["combined"]["q3"] is not None

        # FCF квартальный = CFO − |CapEx|
        fcf = q["fcf"]
        assert fcf["history"][-1]["value"] == pytest.approx(31e9 - 2.3e9)
        # Margin квартальный = EBIT / Revenue
        mg = q["operating_margin"]
        assert mg["history"][-1]["value"] == pytest.approx(33e9 / 98e9)

    def test_get_forecast_core(self, db_ready, monkeypatch, apple_core):
        _patch_edgar(monkeypatch, apple_core)
        svc = SecForecastService(redis_client=None)
        core = svc.get_forecast_core("AAPL")
        assert core["ticker"] == "AAPL"
        assert core["cik"] == "0000320193"

        metrics = core["metrics"]
        assert set(metrics) == {
            "revenue", "fcf", "net_debt_to_ebitda", "eps", "operating_margin", "roe",
        }
        # Revenue: 2 точки (FY2023/FY2024) → прогноз есть, аномалий нет
        rev = metrics["revenue"]
        assert len(rev["history"]) == 2
        assert rev["forecast"]["linreg"]["h1"] is not None
        assert rev["anomaly"] is None
        assert rev["growth"]["yoy"] == pytest.approx(391_035_000_000 / 383_285_000_000 - 1)

        # FCF: производный ряд
        fcf = metrics["fcf"]
        assert fcf["history"][-1]["value"] == pytest.approx(118_254_000_000 - 9_447_000_000)

        # Net Debt / EBITDA: отношение
        nd = metrics["net_debt_to_ebitda"]
        assert nd["history"][-1]["value"] == pytest.approx(34_018_000_000 / 134_661_000_000)

        # ROE / margin — доли
        assert metrics["roe"]["history"][-1]["value"] == pytest.approx(93_736_000_000 / 56_950_000_000)
        assert metrics["operating_margin"]["history"][-1]["value"] == pytest.approx(
            123_216_000_000 / 391_035_000_000
        )

        assert core["shares"] == pytest.approx(15_419_532_000)
        assert core["eps_history"][-1] == pytest.approx(6.08)

    def test_with_price_peg_and_valuation(self, db_ready, monkeypatch):
        _patch_edgar(monkeypatch, _apple_facts_growing_eps())
        monkeypatch.setattr(
            SecForecastService, "_fetch_price_growth", lambda self, ticker: 0.12
        )
        svc = SecForecastService(redis_client=None)
        core = svc.get_forecast_core("AAPL")
        out = svc.with_price(core, 232.0)

        assert out["price"]["value"] == 232.0
        assert out["valuation"]["pe"] == pytest.approx(232.0 / 6.08)
        assert out["valuation"]["eps"] == pytest.approx(6.08)
        assert out["valuation"]["market_cap"] == pytest.approx(232.0 * 15_419_532_000)

        # EPS растёт: CAGR(3y) по окну [3.6, 4.2, 5.0, 6.08] → (6.08/3.6)^(1/3) − 1
        peg = out["peg"]
        assert peg is not None and peg["warning"] is None
        assert peg["growth"] == pytest.approx((6.08 / 3.6) ** (1 / 3) - 1)
        assert peg["pe"] == pytest.approx(232.0 / 6.08)
        assert peg["peg"] == pytest.approx(peg["pe"] / (peg["growth"] * 100))

    def test_price_forecast(self, db_ready, monkeypatch):
        """Прогноз цены: EPS(combined)×P/E×quality + поправка на 5-летний рост цены."""
        _patch_edgar(monkeypatch, _apple_facts_growing_eps())
        monkeypatch.setattr(
            SecForecastService, "_fetch_price_growth", lambda self, ticker: 0.12
        )
        svc = SecForecastService(redis_client=None)
        core = svc.get_forecast_core("AAPL")
        out = svc.with_price(core, 232.0)

        pf = out["price_forecast"]
        assert pf is not None
        assert pf["pe_base"] == pytest.approx(232.0 / 6.08)
        assert pf["price_growth"] == pytest.approx(0.12)
        # EPS-прогноз (combined) растёт → цена растёт; поправка на рост цены ≥ 1
        assert pf["h1"] is not None and pf["h1"] > 232.0 * 0.5
        assert pf["h2"] is not None and pf["h3"] is not None
        # Смесь 50/50: 0.5·(EPS_h·P/E·q) + 0.5·232·(1.12)^h
        assert pf["h1"] == pytest.approx(
            0.5 * core["metrics"]["eps"]["forecast"]["combined"]["h1"] * pf["pe_base"] * pf["quality"]["h1"]
            + 0.5 * 232.0 * 1.12,
            rel=1e-6,
        )
        # quality в узких пределах (0.85..1.15) и считается по revenue/EPS/FCF
        assert 0.85 <= pf["quality"]["h1"] <= 1.15
        assert "revenue/EPS/FCF" in pf["method"], pf["method"]

    def test_build_price_forecast_non_negative(self):
        """Отрицательный EPS-прогноз НЕ даёт отрицательную цену (исторический рост + clamp)."""
        from gex.application.sec.sec_forecast import build_price_forecast

        metrics = {"eps": {"forecast": {"combined": {"h1": -5.0, "h2": -4.0, "h3": -3.0}}}}
        out = build_price_forecast(100.0, metrics, 5.0, price_growth=0.05)
        assert out["h1"] is not None and out["h1"] >= 0
        assert out["h2"] is not None and out["h2"] >= 0
        assert out["h3"] is not None and out["h3"] >= 0
        # без поправки на рост цены отрицательный EPS → прогноз отсутствует (не отрицательный)
        out2 = build_price_forecast(100.0, metrics, 5.0, price_growth=None)
        assert out2["h1"] is None and out2["h3"] is None

    def test_build_price_forecast_growth_fallback(self):
        """Без EPS-драйвера цена идёт по историческому 5-летнему росту (≥ 0)."""
        from gex.application.sec.sec_forecast import build_price_forecast

        metrics = {"eps": {"forecast": {"combined": {"h1": None, "h2": None, "h3": None}}}}
        out = build_price_forecast(100.0, metrics, 5.0, price_growth=0.10)
        assert out["h1"] == pytest.approx(110.0)
        assert out["h3"] == pytest.approx(100.0 * 1.1 ** 3, rel=1e-9)

    def test_price_forecast_none_without_price(self, db_ready, monkeypatch):
        _patch_edgar(monkeypatch, _apple_facts_growing_eps())
        svc = SecForecastService(redis_client=None)
        out = svc.with_price(svc.get_forecast_core("AAPL"), None)
        assert out["price_forecast"] is None

    def test_with_price_none_price(self, db_ready, monkeypatch, apple_core):
        _patch_edgar(monkeypatch, apple_core)
        svc = SecForecastService(redis_client=None)
        out = svc.with_price(svc.get_forecast_core("AAPL"), None)
        assert out["price"] is None
        assert out["peg"] is None
        assert out["valuation"] is None


# ══════════════════════════════════════════════════════════════════════
#  7a2. Квартальный прогноз цены + кросс-валидация с годовым
# ══════════════════════════════════════════════════════════════════════
class TestQuarterlyPriceForecast:
    """Переключатель «Год / Квартал» в блоке «Прогноз цены акции»."""

    @staticmethod
    def _metrics(q_forecast: dict, hist: list[float] | None = None) -> dict:
        hist = hist if hist is not None else [1.5, 1.6, 1.7, 1.8]
        return {
            "eps": {
                "history": [{"end": "2025-06-30", "value": sum(hist)}],
                "forecast": {"combined": {"h1": 7.5, "h2": 8.0, "h3": 8.5}},
                "quarterly": {
                    "history": [
                        {"end": f"202{i}-03-31", "value": v} for i, v in enumerate(hist)
                    ],
                    "forecast": {"combined": q_forecast},
                },
            }
        }

    def test_horizons_and_pe_by_ttm(self):
        """q1/q2/q3 считаются, P/E — по TTM EPS последних 4 кварталов."""
        from gex.application.sec.sec_forecast import build_quarterly_price_forecast

        metrics = self._metrics({"q1": 1.9, "q2": 2.0, "q3": 2.1})
        out = build_quarterly_price_forecast(200.0, metrics, 0.10, {"h1": 240.0})
        assert out is not None
        assert out["eps_ttm"] == pytest.approx(6.6)
        assert out["pe_base"] == pytest.approx(200.0 / 6.6)
        for key in ("q1", "q2", "q3"):
            assert out[key] is not None and out[key] > 0
        assert "кросс-валидация" in out["method"]

    def test_none_when_history_short(self):
        """Меньше 4 кварталов факта — квартальный прогноз не строится."""
        from gex.application.sec.sec_forecast import build_quarterly_price_forecast

        metrics = self._metrics({"q1": 1.9, "q2": 2.0, "q3": 2.1}, hist=[1.7, 1.8, 1.9])
        assert build_quarterly_price_forecast(200.0, metrics, 0.10, {"h1": 240.0}) is None

    def test_pace_capped_by_annual(self):
        """Квартальный рост не может сильно обгонять годовую траекторию."""
        from gex.application.sec.sec_forecast import _PACE_TOLERANCE, build_quarterly_price_forecast

        # Взрывной квартальный EPS против скромного годового прогноза цены
        metrics = self._metrics({"q1": 6.0, "q2": 7.0, "q3": 8.0})
        price, annual = 200.0, {"h1": 210.0}
        out = build_quarterly_price_forecast(price, metrics, None, annual)
        assert out["cross_validation"]["adjusted"] is True
        assert "pace" in out["cross_validation"]["codes"]
        year_ret = annual["h1"] / price - 1.0
        for h in (1, 2, 3):
            implied = price * (annual["h1"] / price) ** (h / 4.0)
            limit = min((implied / price - 1.0) * _PACE_TOLERANCE, year_ret)
            assert out[f"q{h}"] <= price * (1.0 + limit) + 1e-9
            # кварталы не перерастают годовой прогноз
            assert out[f"q{h}"] <= annual["h1"] + 1e-9

    def test_direction_conflict_smoothed(self):
        """Квартальное падение при годовом росте — сглаживается к годовой траектории."""
        from gex.application.sec.sec_forecast import build_quarterly_price_forecast

        metrics = self._metrics({"q1": 0.4, "q2": 0.3, "q3": 0.2})
        price, annual = 200.0, {"h1": 240.0}
        out = build_quarterly_price_forecast(price, metrics, None, annual)
        cv = out["cross_validation"]
        assert cv["adjusted"] is True
        assert "direction" in cv["codes"] and cv["notes"]
        for h in (1, 2, 3):
            implied = price * (annual["h1"] / price) ** (h / 4.0)
            # направление не противоречит годовому (в крайнем случае — без изменения)
            assert (out[f"q{h}"] - price) * (implied - price) >= 0
            assert out[f"q{h}"] <= implied + 1e-9
    def test_no_annual_no_adjustment(self):
        """Без годового прогноза кросс-валидация не правит значения."""
        from gex.application.sec.sec_forecast import build_quarterly_price_forecast

        metrics = self._metrics({"q1": 6.0, "q2": 7.0, "q3": 8.0})
        out = build_quarterly_price_forecast(200.0, metrics, None, None)
        cv = out["cross_validation"]
        assert cv["adjusted"] is False and cv["annual_implied"] is None
        assert out["q1"] > 200.0  # сырой фундаментальный прогноз

    def test_service_attaches_quarterly(self, db_ready, monkeypatch):
        """with_price кладёт квартальный блок внутрь price_forecast."""
        _patch_edgar(monkeypatch, _apple_facts_growing_eps())
        monkeypatch.setattr(
            SecForecastService, "_fetch_price_growth", lambda self, ticker: 0.12
        )
        svc = SecForecastService(redis_client=None)
        out = svc.with_price(svc.get_forecast_core("AAPL"), 232.0)
        assert "quarterly" in out["price_forecast"]  # None допустим при нехватке кварталов


# ══════════════════════════════════════════════════════════════════════
#  7b. Трендовый режим, ошибка прогноза, интервалы
# ══════════════════════════════════════════════════════════════════════
class TestTrendRegime:
    def test_growth(self):
        r = trend_regime([100.0, 110.0, 121.0])
        assert r["regime"] == "growth"
        assert r["direction"] == "up"
        assert r["delta"] == pytest.approx(0.1)

    def test_acceleration(self):
        # 5% → 15%: рост ускоряется
        r = trend_regime([100.0, 105.0, 120.75])
        assert r["regime"] == "acceleration"

    def test_deceleration(self):
        # 15% → 5%: рост замедляется
        r = trend_regime([100.0, 115.0, 120.75])
        assert r["regime"] == "deceleration"

    def test_decline_and_accelerating_decline(self):
        r = trend_regime([120.0, 108.0, 97.2])
        assert r["regime"] == "decline"
        assert r["direction"] == "down"
        # падение ускоряется: -10% → -20%
        r2 = trend_regime([120.0, 108.0, 86.4])
        assert r2["regime"] == "acceleration"

    def test_reversal(self):
        # рост → падение: разворот вниз
        r = trend_regime([100.0, 110.0, 99.0])
        assert r["regime"] == "reversal"
        assert r["direction"] == "down"
        # падение → рост: разворот вверх
        r2 = trend_regime([100.0, 90.0, 99.0])
        assert r2["regime"] == "reversal"
        assert r2["direction"] == "up"

    def test_stabilization(self):
        r = trend_regime([100.0, 101.0, 100.5])
        assert r["regime"] == "stabilization"

    def test_anomaly_flag(self):
        r = trend_regime([100.0, 110.0, 200.0], anomaly_detected=True)
        assert r["regime"] == "anomaly"

    def test_unknown_short(self):
        r = trend_regime([100.0, 110.0])
        assert r["regime"] == "unknown"


class TestForecastErrorAndInterval:
    def test_error_on_linear_series_small(self):
        """Идеальный линейный ряд → ошибка near zero (LinReg ловит тренд)."""
        err = forecast_error([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0], window=3)
        assert err is not None
        assert err["std"] < 5.0

    def test_error_none_short_history(self):
        assert forecast_error([100.0, 110.0, 120.0], window=5) is None

    def test_interval_math(self):
        band = confidence_interval(100.0, 10.0, horizon=1)
        assert band["lower"] == pytest.approx(100.0 - 1.28 * 10.0)
        assert band["upper"] == pytest.approx(100.0 + 1.28 * 10.0)
        # с горизонтом растёт: ±1.28·σ·√h
        band3 = confidence_interval(100.0, 10.0, horizon=3)
        assert band3["upper"] == pytest.approx(100.0 + 1.28 * 10.0 * 3 ** 0.5)

    def test_scenarios_math(self):
        sc = scenario_forecast(100.0, 10.0, horizon=1)
        assert sc["base"] == 100.0
        assert sc["pessimistic"] == pytest.approx(90.0)
        assert sc["optimistic"] == pytest.approx(110.0)


# ══════════════════════════════════════════════════════════════════════
#  7c. Режимы калькулятора (треугольник сценариев)
# ══════════════════════════════════════════════════════════════════════
class TestValuationModes:
    def test_price_pe_to_earnings(self):
        out = calculate_valuation({
            "mode": "price_pe_to_earnings", "price": 100.0, "pe": 20.0, "shares": 1e9,
        })
        assert out["solved"] == "earnings"
        assert out["eps"] == pytest.approx(5.0)
        assert out["earnings"] == pytest.approx(5e9)  # 100·1e9/20

    def test_earnings_pe_to_price(self):
        out = calculate_valuation({
            "mode": "earnings_pe_to_price", "earnings": 5e9, "pe": 20.0, "shares": 1e9,
        })
        assert out["solved"] == "price"
        assert out["eps"] == pytest.approx(5.0)
        assert out["price"] == pytest.approx(100.0)  # (5e9/1e9)·20

    def test_price_earnings_to_pe(self):
        out = calculate_valuation({
            "mode": "price_earnings_to_pe", "price": 100.0, "earnings": 5e9, "shares": 1e9,
        })
        assert out["solved"] == "pe"
        assert out["pe"] == pytest.approx(20.0)  # 100·1e9/5e9

    def test_auto_mode_backward_compat(self):
        out = calculate_valuation({"price": 100.0, "pe": 20.0})
        assert out["mode"] == "auto"
        assert out["solved"] is None
        assert out["eps"] == pytest.approx(5.0)

    def test_request_mode_validation(self):
        ValuationRequest(mode="price_pe_to_earnings", price=100.0, pe=20.0, shares=1e9)
        ValuationRequest(mode="earnings_pe_to_price", earnings=5e9, pe=20.0, shares=1e9)
        ValuationRequest(mode="price_earnings_to_pe", price=100.0, earnings=5e9, shares=1e9)
        # без shares — 422
        with pytest.raises(ValueError):
            ValuationRequest(mode="price_pe_to_earnings", price=100.0, pe=20.0)
        # не хватает входа для режима
        with pytest.raises(ValueError):
            ValuationRequest(mode="price_pe_to_earnings", price=100.0, shares=1e9)


# ══════════════════════════════════════════════════════════════════════
#  8. API
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def client(monkeypatch):
    """TestClient с изоляцией от реального Redis."""
    import gex.adapters.cache.result_cache as rc_module

    monkeypatch.setattr(rc_module, "get_redis", lambda: None)

    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()

    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)


def _register_and_subscribe(client, email: str = "secf@test.local") -> str:
    r = client.post("/auth/register", json={
        "email": email, "password": "pass1234",
        "accept_terms": True, "telegram_username": "@secfbot",
    })
    assert r.status_code == 201
    token = r.json()["access_token"]
    db = SessionLocal()
    db.query(User).filter(User.email == email).update({
        "subscription_status": "BASIC",
        "subscription_activated_at": datetime.now(UTC),
        "subscription_expires_at": datetime.now(UTC) + timedelta(days=30),
    })
    db.commit()
    db.close()
    return token


class TestForecastApi:
    def test_forecast_ok(self, client, monkeypatch):
        token = _register_and_subscribe(client)
        _patch_edgar(monkeypatch, _apple_facts_growing_eps())
        monkeypatch.setattr(
            SecForecastService, "fetch_price", lambda self, ticker: 232.0
        )
        monkeypatch.setattr(
            SecForecastService, "_fetch_price_growth", lambda self, ticker: 0.12
        )

        r = client.get("/companies/AAPL/forecast", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["price"]["value"] == 232.0
        assert data["parameters"]["window"] == 5
        assert data["parameters"]["last_report_weight"] == 0.6  # базово 60%
        assert "revenue" in data["metrics"] and "eps" in data["metrics"]
        assert data["metrics"]["revenue"]["history"][-1]["value"] == pytest.approx(391_035_000_000)
        assert data["metrics"]["eps"]["forecast"]["combined"]["h1"] is not None
        # Характер движения + ошибка/интервалы/сценарии прогноза
        assert data["metrics"]["revenue"]["regime"]["regime"] in (
            "growth", "decline", "acceleration", "deceleration", "reversal", "stabilization", "unknown",
        )
        fc = data["metrics"]["eps"]["forecast"]
        assert "error" in fc and "interval" in fc and "scenarios" in fc
        # История короткая (5 точек < window+2) → error может быть None;
        # при наличии — интервал и сценарии консистентны
        if fc["error"]:
            assert fc["error"]["std"] > 0
            assert fc["interval"]["h1"]["lower"] < fc["interval"]["h1"]["upper"]
            assert fc["scenarios"]["h1"]["base"] == fc["combined"]["h1"]
        assert data["peg"]["peg"] > 0
        assert data["valuation"]["pe"] == pytest.approx(232.0 / 6.08)

    def test_forecast_custom_params(self, client, monkeypatch, apple_core):
        token = _register_and_subscribe(client, email="secf2@test.local")
        _patch_edgar(monkeypatch, apple_core)
        monkeypatch.setattr(SecForecastService, "fetch_price", lambda self, ticker: 232.0)
        monkeypatch.setattr(
            SecForecastService, "_fetch_price_growth", lambda self, ticker: 0.12
        )

        r = client.get(
            "/companies/AAPL/forecast?window=3&alpha=0.3&anomaly_threshold=0.4&horizon_years=3&last_report_weight=0.8",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["parameters"]["window"] == 3
        assert data["parameters"]["alpha"] == 0.3
        assert data["parameters"]["anomaly_threshold"] == 0.4
        assert data["parameters"]["horizon_years"] == 3
        assert data["parameters"]["last_report_weight"] == 0.8

    def test_forecast_last_weight_bad_422(self, client, monkeypatch, apple_core):
        """last_report_weight < 0.6 — 422 (диапазон 60..100%)."""
        token = _register_and_subscribe(client, email="secf5@test.local")
        _patch_edgar(monkeypatch, apple_core)
        r = client.get(
            "/companies/AAPL/forecast?last_report_weight=0.5",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422

    def test_forecast_unknown_ticker_404(self, client, monkeypatch):
        token = _register_and_subscribe(client, email="secf3@test.local")
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: {"AAPL": "0000320193"},
        )
        r = client.get("/companies/ZZZZ/forecast", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 404

    def test_forecast_bad_params_422(self, client, monkeypatch, apple_core):
        token = _register_and_subscribe(client, email="secf4@test.local")
        _patch_edgar(monkeypatch, apple_core)
        r = client.get(
            "/companies/AAPL/forecast?window=1&alpha=1.5&anomaly_threshold=0.1",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422


class TestValuationApi:
    def test_calculator_ok(self, client):
        token = _register_and_subscribe(client, email="secv@test.local")
        r = client.post(
            "/valuation/calculator",
            json={"price": 232.0, "pe": 30.0, "growth": 0.1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["eps"] == pytest.approx(232.0 / 30.0)
        assert data["peg"] == pytest.approx(30.0 / 10.0)

    def test_calculator_insufficient_422(self, client):
        token = _register_and_subscribe(client, email="secv2@test.local")
        r = client.post(
            "/valuation/calculator",
            json={"price": 232.0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422

    def test_calculator_mode_ok(self, client):
        token = _register_and_subscribe(client, email="secv3@test.local")
        r = client.post(
            "/valuation/calculator",
            json={"mode": "price_pe_to_earnings", "price": 232.0, "pe": 30.0, "shares": 1e9},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "price_pe_to_earnings"
        assert data["solved"] == "earnings"
        assert data["earnings"] == pytest.approx(232.0 * 1e9 / 30.0)

    def test_calculator_mode_missing_shares_422(self, client):
        token = _register_and_subscribe(client, email="secv4@test.local")
        r = client.post(
            "/valuation/calculator",
            json={"mode": "price_pe_to_earnings", "price": 232.0, "pe": 30.0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422

    def test_calculator_public_no_auth(self, client):
        """Калькулятор — публичная ручка (страница доступна без авторизации)."""
        r = client.post("/valuation/calculator", json={"price": 100.0, "pe": 20.0})
        assert r.status_code == 200
        data = r.json()
        assert data["eps"] == pytest.approx(5.0)

    def test_forecast_public_no_auth(self, client, monkeypatch):
        """Forecast — публичная ручка (страница описания метрик без авторизации)."""
        _patch_edgar(monkeypatch, _apple_facts_growing_eps())
        monkeypatch.setattr(SecForecastService, "fetch_price", lambda self, ticker: 232.0)
        monkeypatch.setattr(
            SecForecastService, "_fetch_price_growth", lambda self, ticker: 0.12
        )
        r = client.get("/companies/AAPL/forecast")
        assert r.status_code == 200
        assert r.json()["ticker"] == "AAPL"
