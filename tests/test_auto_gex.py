"""Unit-тесты AUTO-режима GEX Details (design doc §5 / §9).

Без сети: синтетические опционные цепочки + монkeypatch ``ExtendedGEXAnalyzer._fetch``.
Проверяем:
  * правило разреженности по именованным порогам (8 / ±10% / 5) и по числу
    экспираций (< AUTO_MIN_EXPIRIES ⇒ low_expiries);
  * объединение цепочек: union без двойного счёта, приоритет первичной строки;
  * отображение fallback-источника;
  * analyze_auto: happy / эскалация / частичный сбой / отсутствие fallback (крипта);
  * покрытие (AutoCoverage) и обратную совместимость схемы (+ расширение Literal source);
  * дизъюнктность AUTO-ключа кэша.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gex.domain.data_loader import OptionSnapshot
from gex.application.extended import ExtendedGEXAnalyzer
from gex.application.auto_scope import (
    AUTO_MAX_DAYS,
    AUTO_MAX_EXPIRIES,
    AUTO_MIN_EXPIRIES,
    AUTO_MIN_STRIKES,
    REASON_LOW_EXPIRIES,
    AutoCoverage,
    StrikeLite,
    count_expiries,
    detect_sparse,
    fallback_source_for,
    merge_chains,
)
from gex.schemas.extended_schemas import extended_report_to_schema


# ====================================================================== #
#  Синтетические цепочки
# ====================================================================== #
def _healthy_snapshot(
    spot: float = 100.0, symbol: str = "AAPL", n_bands: int = 11,
    expiry_days: tuple[float, ...] = (7.0, 30.0, 60.0),
) -> OptionSnapshot:
    """Полноценный профиль: есть и Call Wall, и Put Wall (не разрежен).

    OI коллов растёт к верхним страйкам, OI путов — к нижним, поэтому знак
    Net GEX меняется по страйкам и обе стены конечны. По умолчанию цепочка
    содержит **3 экспирации** (``≥ AUTO_MIN_EXPIRIES``), т.е. профиль богат и по
    страйкам, и по экспирациям — правило ``low_expiries`` молчит.
    """
    lo, hi = 80.0, 120.0
    strikes = np.linspace(lo, hi, n_bands)
    rows = []
    for days in expiry_days:
        T = days / 365.0
        for k in strikes:
            diff = float(k) - spot
            oi_c = max(100.0, 1000.0 + diff * 40.0)
            oi_p = max(100.0, 1000.0 - diff * 40.0)
            iv = 0.2 * (1.0 + 2.0 * ((float(k) - spot) / spot) ** 2)
            rows.append({"strike": float(k), "type": "C", "oi": oi_c, "iv": iv, "T": T})
            rows.append({"strike": float(k), "type": "P", "oi": oi_p, "iv": iv, "T": T})
    chain = pd.DataFrame(rows)
    return OptionSnapshot(symbol=symbol, spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)


def _thin_snapshot(spot: float = 100.0, symbol: str = "AAPL") -> OptionSnapshot:
    """Разреженный профиль: 3 страйка, нулевой OI (срабатывают low_strikes/total_oi_zero)."""
    T = 30.0 / 365.0
    rows = []
    for k in (99.0, 100.0, 101.0):
        rows.append({"strike": k, "type": "C", "oi": 0.0, "iv": 0.2, "T": T})
        rows.append({"strike": k, "type": "P", "oi": 0.0, "iv": 0.2, "T": T})
    chain = pd.DataFrame(rows)
    return OptionSnapshot(symbol=symbol, spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)


def _lite(strikes) -> list[StrikeLite]:
    return [
        StrikeLite(strike=float(s.strike), oi_call=float(s.oi_call),
                   oi_put=float(s.oi_put), gex_net=float(s.gex_net))
        for s in strikes
    ]


# ====================================================================== #
#  1. detect_sparse — правило по именованным порогам
# ====================================================================== #
class TestDetectSparse:

    def test_healthy_profile_not_sparse(self):
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_healthy_snapshot())
        reasons = detect_sparse(_lite(rep.per_strike), rep.spot, rep.call_wall, rep.put_wall)
        assert reasons == [], f"здоровый профиль ошибочно разрежен: {reasons}"

    def test_no_data(self):
        reasons = detect_sparse([], 100.0, 100.0, 90.0)
        assert "no_data" in reasons

    def test_total_oi_zero(self):
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_thin_snapshot())
        reasons = detect_sparse(_lite(rep.per_strike), rep.spot, 105.0, 95.0)
        assert "total_oi_zero" in reasons

    def test_low_strikes_threshold_boundary(self):
        # 7 страйков (< 8) → low_strikes; 8 → нет.
        def strikes(n):
            return [StrikeLite(strike=100.0 + i, oi_call=10.0, oi_put=10.0, gex_net=1.0)
                    for i in range(n)]
        assert "low_strikes" in detect_sparse(strikes(AUTO_MIN_STRIKES - 1), 100.0, 105.0, 95.0)
        assert "low_strikes" not in detect_sparse(strikes(AUTO_MIN_STRIKES), 100.0, 105.0, 95.0)

    def test_low_atm_strikes(self):
        # 10 страйков, но все далеко от spot (|k-spot|/spot > 10%) → low_atm_strikes.
        far = [StrikeLite(strike=100.0 + 25.0 + i, oi_call=10.0, oi_put=10.0, gex_net=1.0)
               for i in range(10)]
        assert "low_atm_strikes" in detect_sparse(far, 100.0, 200.0, 150.0)

    def test_spot_non_positive_triggers_low_atm(self):
        st = [StrikeLite(strike=100.0, oi_call=1.0, oi_put=1.0, gex_net=0.0)] * 10
        assert "low_atm_strikes" in detect_sparse(st, 0.0, 100.0, 90.0)

    def test_missing_walls_none_nan_and_zero(self):
        st = [StrikeLite(strike=100.0 + i, oi_call=10.0, oi_put=10.0, gex_net=1.0)
              for i in range(10)]
        r_none = detect_sparse(st, 100.0, None, None)
        assert "missing_call_wall" in r_none and "missing_put_wall" in r_none
        r_nan = detect_sparse(st, 100.0, float("nan"), float("nan"))
        assert "missing_call_wall" in r_nan and "missing_put_wall" in r_nan
        r_zero = detect_sparse(st, 100.0, 0.0, 0.0)
        assert "missing_call_wall" in r_zero and "missing_put_wall" in r_zero


# ====================================================================== #
#  2. merge_chains — union без двойного счёта, приоритет первичной строки
# ====================================================================== #
class TestMergeChains:

    def _pri(self):
        T = 30.0 / 365.0
        return pd.DataFrame([
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.20, "T": T},
            {"strike": 100.0, "type": "P", "oi": 20.0, "iv": 0.20, "T": T},
        ])

    def _fb(self):
        T = 30.0 / 365.0
        return pd.DataFrame([
            # коллизия с primary (тот же страйк/тип/экспирация) → должна победить primary
            {"strike": 100.0, "type": "C", "oi": 999.0, "iv": 0.55, "T": T},
            # уникальная строка → добавляется
            {"strike": 105.0, "type": "C", "oi": 5.0, "iv": 0.30, "T": T},
        ])

    def test_union_no_double_count(self):
        merged = merge_chains(self._pri(), self._fb())
        assert len(merged) == 3
        # Никакого суммирования перекрывающейся строки: 100C один раз.
        assert int((merged["strike"] == 100.0).sum()) == 2  # 100C + 100P
        assert float(merged["oi"].sum()) == 10.0 + 20.0 + 5.0

    def test_primary_row_wins_whole_row(self):
        merged = merge_chains(self._pri(), self._fb())
        row = merged[(merged["strike"] == 100.0) & (merged["type"] == "C")].iloc[0]
        assert float(row["oi"]) == 10.0     # не 999, не 1009
        assert float(row["iv"]) == 0.20     # вся строка первичная, без смешивания полей

    def test_fallback_unique_row_added(self):
        merged = merge_chains(self._pri(), self._fb())
        assert 105.0 in set(merged["strike"])

    def test_expiry_bucket_separates_different_expiries(self):
        T1, T2 = 30.0 / 365.0, 60.0 / 365.0
        pri = pd.DataFrame([{"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.2, "T": T1}])
        fb = pd.DataFrame([{"strike": 100.0, "type": "C", "oi": 7.0, "iv": 0.2, "T": T2}])
        merged = merge_chains(pri, fb)
        # разные экспирации → две строки (не коллизия)
        assert len(merged) == 2


# ====================================================================== #
#  3. fallback_source_for
# ====================================================================== #
class TestFallbackMapping:

    @pytest.mark.parametrize("primary,expected", [
        ("webull", "yfinance"),
        ("yfinance", "webull"),
        ("crypto", None),
        ("futures", None),
        ("stock", None),
        ("moex_iss", None),
        ("", None),
    ])
    def test_mapping(self, primary, expected):
        assert fallback_source_for(primary, "AAPL") == expected


# ====================================================================== #
#  4. count_expiries
# ====================================================================== #
class TestCountExpiries:

    def test_counts_distinct_t_days(self):
        T = pd.DataFrame({"T": [1 / 365, 1 / 365, 30 / 365, 30 / 365, 60 / 365]})
        assert count_expiries(T) == 3

    def test_empty(self):
        assert count_expiries(None) == 0
        assert count_expiries(pd.DataFrame(columns=["T"])) == 0


# ====================================================================== #
#  5. analyze_auto
# ====================================================================== #
class TestAnalyzeAuto:

    def _patch_fetch(self, analyzer, primary_snap, primary_src,
                     fallback_snap=None, fallback_src="yfinance",
                     fallback_force="yfinance", fallback_raises=False):
        """Подменить ``_fetch``: запрос с ``force_source==fallback_force`` отдаёт резерв."""
        def fake_fetch(ticker, max_expiries, force_source=None, max_days=None):
            if fallback_force is not None and force_source == fallback_force:
                if fallback_raises:
                    raise RuntimeError("fallback provider down")
                return fallback_snap, fallback_src, 100, 0.0
            return primary_snap, primary_src, 100, 0.0
        analyzer._fetch = fake_fetch  # type: ignore[assignment]

    def test_happy_path_no_escalation(self):
        an = ExtendedGEXAnalyzer()
        self._patch_fetch(an, _healthy_snapshot(), "webull")
        rep = an.analyze_auto("AAPL")
        cov = rep.coverage
        assert cov is not None
        assert cov.mode == "auto"
        assert cov.resolved_days == AUTO_MAX_DAYS
        assert cov.resolved_expiries == AUTO_MAX_EXPIRIES
        assert cov.primary_source == "webull"
        assert cov.sources_used == ["webull"]
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.partial is False
        assert cov.sparse is False
        assert cov.strike_count == len(rep.per_strike)
        assert cov.total_oi > 0
        # профиль не разрежен ⇒ эскалации нет; в цепочке ровно 3 экспирации
        assert cov.expirations_merged == 3

    def test_escalation_merges_fallback(self):
        an = ExtendedGEXAnalyzer()
        pri = _thin_snapshot()
        fb = _healthy_snapshot()
        self._patch_fetch(an, pri, "webull", fallback_snap=fb, fallback_src="yfinance")
        rep = an.analyze_auto("AAPL")
        cov = rep.coverage
        assert cov.escalated is True
        assert cov.fallback_used is True
        assert cov.partial is False
        assert cov.sources_used == ["webull", "yfinance"]
        # объединённый профиль содержит страйки обоих источников (без двойного счёта:
        # thin-строки 99/100/101 отсутствуют в healthy-наборе, значит добавляются уникально)
        assert cov.strike_count >= len(fb.chain["strike"].unique())
        # total_oi согласован с финальным профилем
        assert math.isclose(cov.total_oi, sum(s.oi_call + s.oi_put for s in rep.per_strike))

    def test_partial_when_fallback_fails(self):
        an = ExtendedGEXAnalyzer()
        self._patch_fetch(an, _thin_snapshot(), "webull", fallback_raises=True)
        rep = an.analyze_auto("AAPL")
        cov = rep.coverage
        assert cov.partial is True
        assert cov.fallback_used is False
        assert cov.escalated is True           # попытка была
        assert cov.sources_used == ["webull"]  # резерв не внёс строк
        assert cov.sparse is True              # первичный профиль остаётся разреженным
        # результат НЕ ошибка: отчёт построен
        assert len(rep.per_strike) > 0

    def test_crypto_no_fallback(self):
        an = ExtendedGEXAnalyzer()
        self._patch_fetch(an, _thin_snapshot(symbol="BTC"), "crypto")
        rep = an.analyze_auto("BTC")
        cov = rep.coverage
        assert cov.primary_source == "crypto"
        assert cov.sources_used == ["crypto"]
        assert cov.escalated is False
        assert cov.fallback_used is False

    def test_pinned_source_uses_other_as_fallback(self):
        an = ExtendedGEXAnalyzer()
        # user pinned yfinance + AUTO → primary=yfinance, fallback=webull
        self._patch_fetch(an, _thin_snapshot(), "yfinance",
                          fallback_snap=_healthy_snapshot(), fallback_src="webull",
                          fallback_force="webull")
        rep = an.analyze_auto("AAPL", source="yfinance")
        cov = rep.coverage
        assert cov.primary_source == "yfinance"
        assert cov.sources_used == ["yfinance", "webull"]
        assert cov.fallback_used is True

    def test_aggregated_source_rejected(self):
        an = ExtendedGEXAnalyzer()
        with pytest.raises(ValueError):
            an.analyze_auto("AAPL", source="aggregated")


# ====================================================================== #
#  6. Схема: покрытие + обратная совместимость + расширение Literal
# ====================================================================== #
class TestCoverageSchema:

    def test_manual_report_has_coverage(self):
        """Ручной режим ТОЖЕ отдаёт метаданные охвата (аудит 2026-09-17).

        Раньше ``coverage`` был ``None`` вне AUTO, из-за чего качество выборки
        было не видно в режиме по умолчанию: профиль из одной экспирации и
        профиль из восьми выглядели одинаково. Теперь блок считается всегда,
        с ``mode="manual"`` и без полей эскалации.
        """
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_healthy_snapshot())
        assert rep.coverage is not None
        assert rep.coverage.mode == "manual"
        assert rep.coverage.strike_count == len(rep.per_strike)
        assert rep.coverage.expirations_merged >= 1
        # Признаки AUTO-эскалации в ручном режиме не выставляются.
        assert rep.coverage.escalated is False
        assert rep.coverage.fallback_used is False
        out = extended_report_to_schema(rep, days=30.0)
        assert out.auto is not None
        assert out.auto.mode == "manual"

    def test_auto_report_maps_coverage(self):
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_healthy_snapshot())
        rep.coverage = AutoCoverage(
            mode="auto", resolved_days=90.0, resolved_expiries=20,
            sources_used=["webull", "yfinance"], primary_source="webull",
            fallback_used=True, escalated=True, partial=False,
            expirations_merged=3, strike_count=11, total_oi=12345.0,
            sparse=False, sparse_reasons=[], elapsed_ms=42,
        )
        out = extended_report_to_schema(rep, days=90.0)
        assert out.auto is not None
        assert out.auto.mode == "auto"
        assert out.auto.sources_used == ["webull", "yfinance"]
        assert out.auto.resolved_expiries == 20
        assert out.auto.fallback_used is True

    def test_coverage_schema_carries_additive_fields(self):
        """Аддитивные Phase-4 поля (strike_min/max, nearest_expiry_days) доходят до схемы."""
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_healthy_snapshot())
        rep.coverage = AutoCoverage(
            mode="auto", resolved_days=90.0, resolved_expiries=20,
            sources_used=["webull"], primary_source="webull",
            expirations_merged=3, strike_count=11, total_oi=12345.0,
            strike_min=80.0, strike_max=120.0, nearest_expiry_days=7.0,
        )
        out = extended_report_to_schema(rep, days=90.0)
        assert out.auto is not None
        assert out.auto.strike_min == 80.0
        assert out.auto.strike_max == 120.0
        assert out.auto.nearest_expiry_days == 7.0

    def test_coverage_schema_fields_absent_default_none(self):
        """Без новых полей (старый датакласс) → None, обратная совместимость."""
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_healthy_snapshot())
        rep.coverage = AutoCoverage(
            mode="auto", resolved_days=90.0, resolved_expiries=20,
            sources_used=["webull"], primary_source="webull",
            expirations_merged=3, strike_count=11, total_oi=12345.0,
        )
        out = extended_report_to_schema(rep, days=90.0)
        assert out.auto is not None
        assert out.auto.strike_min is None
        assert out.auto.strike_max is None
        assert out.auto.nearest_expiry_days is None

    def test_source_literal_accepts_yfinance(self):
        """§6: расширение Literal — yfinance больше не роняет валидацию (латентный 500)."""
        an = ExtendedGEXAnalyzer()
        rep = an.analyze("AAPL", snapshot=_healthy_snapshot())
        rep.source = "yfinance"
        out = extended_report_to_schema(rep)
        assert out.source == "yfinance"


# ====================================================================== #
#  7. Дизъюнктность AUTO-ключа кэша
# ====================================================================== #
class TestCacheKey:

    def test_auto_key_disjoint_from_manual(self):
        from gex.adapters.cache.redis_client import cache_key

        k_auto = cache_key("res", "extgexA", "AAPL", int(AUTO_MAX_DAYS), AUTO_MAX_EXPIRIES, "auto", "-1.0-1.0")
        k_man = cache_key("res", "extgex", "AAPL", 30, 5, "auto", "-1.0-1.0")
        assert k_auto != k_man
        assert "EXTGEXA" in k_auto
        assert "EXTGEX" in k_man and "EXTGEXA" not in k_man

    def test_auto_key_varies_by_source(self):
        from gex.adapters.cache.redis_client import cache_key

        k1 = cache_key("res", "extgexA", "AAPL", 90, 20, "webull", "-1.0-1.0")
        k2 = cache_key("res", "extgexA", "AAPL", 90, 20, "yfinance", "-1.0-1.0")
        assert k1 != k2


# ====================================================================== #
#  8. detect_sparse — порог по числу экспираций (low_expiries)
# ====================================================================== #
class TestDetectSparseLowExpiries:
    """Контракт: < AUTO_MIN_EXPIRIES различимых экспираций ⇒ ``low_expiries``.

    Параметр ``expiries`` опционален: при ``None`` поведение функции не меняется
    (обратная совместимость для прежних вызовов без этого аргумента).
    """

    def _rich(self) -> list[StrikeLite]:
        """Богатый по всем прочим признакам профиль (10 страйков в полосе ATM, обе стены)."""
        return [
            StrikeLite(strike=95.0 + i, oi_call=10.0, oi_put=10.0, gex_net=1.0)
            for i in range(10)
        ]

    def test_threshold_constant(self):
        assert AUTO_MIN_EXPIRIES == 3

    def test_below_threshold_is_flagged(self):
        reasons = detect_sparse(self._rich(), 100.0, 105.0, 95.0, expiries=AUTO_MIN_EXPIRIES - 1)
        assert REASON_LOW_EXPIRIES in reasons

    def test_two_expiries_is_flagged(self):
        assert "low_expiries" in detect_sparse(self._rich(), 100.0, 105.0, 95.0, expiries=2)

    def test_zero_expiries_is_flagged(self):
        assert "low_expiries" in detect_sparse(self._rich(), 100.0, 105.0, 95.0, expiries=0)

    def test_at_threshold_is_not_flagged(self):
        reasons = detect_sparse(self._rich(), 100.0, 105.0, 95.0, expiries=AUTO_MIN_EXPIRIES)
        assert reasons == []

    def test_omitted_param_is_backward_compatible(self):
        # Без параметра — байт-в-байт как до изменения правила.
        assert detect_sparse(self._rich(), 100.0, 105.0, 95.0) == []

    def test_reason_is_appended_last(self):
        # missing_call_wall + мало экспираций → стабильный порядок, low_expiries в конце.
        reasons = detect_sparse(self._rich(), 100.0, None, 95.0, expiries=1)
        assert "missing_call_wall" in reasons
        assert reasons[-1] == REASON_LOW_EXPIRIES


# ====================================================================== #
#  9. analyze_auto — эскалация по low_expiries (первичный источник: мало экспираций)
# ====================================================================== #
class TestAnalyzeAutoLowExpiries:
    """Первичный источник вернул < 3 экспираций, но профиль иначе непустой.

    Такое покрытие обязано считаться недостаточным и запускать ровно одну
    эскалацию — иначе AUTO не выполняет обещание «набрать данные из других
    периодов» (именно этот разрыв дал live-E2E на webull-пути).
    """

    def _patch(self, analyzer, primary, fallback, fallback_raises: bool = False):
        """Подменить ``_fetch`` и записать все вызовы (ticker, max_expiries, force_source)."""
        calls: list[tuple] = []

        def fake_fetch(ticker, max_expiries, force_source=None, max_days=None):
            calls.append((ticker, max_expiries, force_source, max_days))
            if force_source == "yfinance":
                if fallback_raises:
                    raise RuntimeError("fallback provider down")
                return fallback, "yfinance", 100, 0.0
            return primary, "webull", 100, 0.0

        analyzer._fetch = fake_fetch  # type: ignore[assignment]
        return calls

    def test_undercovered_primary_escalates_and_union_has_more_buckets(self):
        an = ExtendedGEXAnalyzer()
        pri = _healthy_snapshot(expiry_days=(30.0,))             # 1 экспирация → low_expiries
        fb = _healthy_snapshot(expiry_days=(7.0, 30.0, 60.0))    # 3 экспирации
        calls = self._patch(an, pri, fb)

        rep = an.analyze_auto("AAPL")
        cov = rep.coverage

        assert len(calls) == 2                       # ровно один дополнительный фетч
        assert calls[0][2] is None                   # первичная загрузка: авто-резолв
        assert calls[1][2] == "yfinance"             # эскалация: форсированный fallback
        assert count_expiries(pri.chain) == 1
        assert cov.escalated is True
        assert cov.fallback_used is True
        assert cov.partial is False
        assert cov.sources_used == ["webull", "yfinance"]
        # объединённая цепочка несёт строго больше экспирационных бакетов
        assert cov.expirations_merged == 3
        assert cov.expirations_merged > count_expiries(pri.chain)
        # после успешного объединения профиль уже не разрежен
        assert cov.sparse is False
        assert REASON_LOW_EXPIRIES not in cov.sparse_reasons

    def test_low_expiries_alone_drives_escalation(self):
        # Первичный профиль богат страйками/OI/стенами — срабатывает только low_expiries.
        an = ExtendedGEXAnalyzer()
        pri = _healthy_snapshot(expiry_days=(30.0,))
        fb = _healthy_snapshot(expiry_days=(7.0, 30.0, 60.0))
        self._patch(an, pri, fb)
        rep = an.analyze_auto("AAPL")
        assert rep.coverage.escalated is True
        assert rep.coverage.fallback_used is True

    def test_fallback_failure_on_low_expiries_is_partial(self):
        an = ExtendedGEXAnalyzer()
        pri = _healthy_snapshot(expiry_days=(30.0,))
        calls = self._patch(an, pri, None, fallback_raises=True)

        rep = an.analyze_auto("AAPL")               # не должно бросить исключение
        cov = rep.coverage

        assert len(calls) == 2
        assert cov.partial is True
        assert cov.fallback_used is False
        assert cov.escalated is True                 # попытка была
        assert cov.sources_used == ["webull"]        # fallback не внёс строк
        assert REASON_LOW_EXPIRIES in cov.sparse_reasons
        assert len(rep.per_strike) > 0               # первичный результат возвращён
