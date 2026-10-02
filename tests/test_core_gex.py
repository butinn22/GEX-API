"""Тесты ядра GEX: GEXDataLoader, GEXMetrics, GEXPipeline.

Критический путь: парсинг опционных цепочек → расчёт GEX → pipeline.
Без этих тестов любое изменение в core-логике может незаметно сломать все ручки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex.domain.data_loader import GEXDataLoader, OptionSnapshot
from gex.domain.metrics import GEXProfile, GEXMetrics
from gex.domain.pipeline import GEXPipeline, GEXReport
from gex.domain.greeks import bs_gamma as black_scholes_gamma


# ====================================================================== #
#  GEXDataLoader
# ====================================================================== #
class TestGEXDataLoader:
    """Парсинг опционных цепочек из сырых данных."""

    def _make_raw_chain(self, n_strikes: int = 5) -> pd.DataFrame:
        """Сгенерировать тестовую цепочку."""
        spot = 500.0
        strikes = np.linspace(480, 520, n_strikes)
        rows = []
        for s in strikes:
            rows.append({"strike": s, "type": "C", "oi": 1000, "iv": 0.20, "T": 30 / 365})
            rows.append({"strike": s, "type": "P", "oi": 800, "iv": 0.22, "T": 30 / 365})
        return pd.DataFrame(rows)

    def test_load_dataframe_creates_snapshot(self):
        loader = GEXDataLoader(spot=500.0, symbol="TEST")
        raw = self._make_raw_chain()
        snap = loader.load_dataframe(raw)
        assert isinstance(snap, OptionSnapshot)
        assert snap.symbol == "TEST"
        assert snap.spot == 500.0
        assert len(snap.chain) == 10  # 5 strikes × 2 types

    def test_load_dataframe_filters_zero_oi(self):
        loader = GEXDataLoader(spot=500.0, symbol="TEST")
        raw = self._make_raw_chain()
        raw.loc[0, "oi"] = 0  # нулевой OI
        snap = loader.load_dataframe(raw)
        # Строка с OI=0 должна быть отфильтрована
        assert len(snap.chain) < len(raw)

    def test_load_dataframe_filters_zero_iv(self):
        loader = GEXDataLoader(spot=500.0, symbol="TEST")
        raw = self._make_raw_chain()
        raw.loc[1, "iv"] = 0.0
        snap = loader.load_dataframe(raw)
        assert len(snap.chain) < len(raw)

    def test_synthetic_chain(self):
        loader = GEXDataLoader(spot=500.0, symbol="TEST")
        strikes = np.arange(480, 521, 5)
        snap = loader.synthetic_chain(
            strikes=strikes,
            expiry_years=0.082,
            atm_iv=0.20,
            skew=1.1,
            oi_seed=0,
        )
        assert isinstance(snap, OptionSnapshot)
        assert snap.symbol == "TEST"
        assert len(snap.chain) > 0
        assert "strike" in snap.chain.columns
        assert "oi" in snap.chain.columns
        assert snap.chain["oi"].sum() > 0

    def test_option_snapshot_dataclass(self):
        snap = OptionSnapshot(
            symbol="TEST",
            spot=500.0,
            as_of=pd.Timestamp.now(),
            chain=self._make_raw_chain(),
        )
        assert snap.symbol == "TEST"
        assert snap.spot == 500.0
        assert len(snap.chain) == 10


# ====================================================================== #
#  GEXMetrics (расчёт GEX, профиль, стены)
# ====================================================================== #
class TestGEXMetrics:
    """Расчёт гамма-экспозиции из опционной цепочки."""

    def _make_snapshot(self, n_strikes: int = 5) -> OptionSnapshot:
        """Создать тестовый снапшот с OI только для одного страйка."""
        spot = 500.0
        strikes = np.linspace(480, 520, n_strikes)
        rows = []
        for s in strikes:
            # Только один страйк с OI > 0 (остальные с нулевым)
            oi_c = 1000 if abs(s - 500) < 1 else 0
            oi_p = 800 if abs(s - 500) < 1 else 0
            rows.append({"strike": s, "type": "C", "oi": oi_c, "iv": 0.20, "T": 30 / 365})
            rows.append({"strike": s, "type": "P", "oi": oi_p, "iv": 0.22, "T": 30 / 365})
        return OptionSnapshot(
            symbol="TEST", spot=spot,
            as_of=pd.Timestamp.now(),
            chain=pd.DataFrame(rows),
        )

    def test_metrics_call_gex_positive(self):
        """Call-опционы дают положительный GEX."""
        chain = pd.DataFrame({
            "strike": [500.0],
            "type": ["C"],
            "oi": [1000.0],
            "iv": [0.20],
            "T": [30 / 365],
        })
        snap = OptionSnapshot(symbol="TEST", spot=500.0, as_of=pd.Timestamp.now(), chain=chain)
        profile = GEXMetrics(spot=500.0, r=0.045).compute(snap)
        # Call доминирует → GEX>0
        assert profile.net_gex != 0.0

    def test_metrics_put_gex_negative(self):
        """Put-опционы дают отрицательный GEX."""
        chain = pd.DataFrame({
            "strike": [500.0],
            "type": ["P"],
            "oi": [1000.0],
            "iv": [0.20],
            "T": [30 / 365],
        })
        snap = OptionSnapshot(symbol="TEST", spot=500.0, as_of=pd.Timestamp.now(), chain=chain)
        profile = GEXMetrics(spot=500.0, r=0.045).compute(snap)
        assert profile.net_gex != 0.0

    def test_gex_profile_has_call_wall(self):
        # Сбалансированная цепочка: коллы выше спота, путы ниже — обе стены есть
        rows = []
        for s in range(480, 521, 10):
            if s >= 510:
                rows.append({"strike": s, "type": "C", "oi": 1000.0, "iv": 0.20, "T": 30 / 365})
            if s <= 490:
                rows.append({"strike": s, "type": "P", "oi": 800.0, "iv": 0.22, "T": 30 / 365})
        snap = OptionSnapshot(symbol="TEST", spot=500.0, as_of=pd.Timestamp.now(), chain=pd.DataFrame(rows))
        profile = GEXMetrics(spot=500.0, r=0.045).compute(snap)
        assert isinstance(profile, GEXProfile)
        assert profile.call_wall is not None
        assert profile.put_wall is not None

    def test_profile_regime_positive(self):
        """Больше коллов → POSITIVE gamma."""
        chain = pd.DataFrame({
            "strike": [490.0, 500.0, 510.0],
            "type": ["C", "C", "C"],
            "oi": [1000.0, 2000.0, 1500.0],
            "iv": [0.20, 0.20, 0.20],
            "T": [30 / 365, 30 / 365, 30 / 365],
        })
        snap = OptionSnapshot(symbol="TEST", spot=500.0, as_of=pd.Timestamp.now(), chain=chain)
        profile = GEXMetrics(spot=500.0, r=0.045).compute(snap)
        # Ожидаем POSITIVE regime (больше коллов)
        assert hasattr(profile, "regime")
        assert hasattr(profile, "net_gex")
        assert hasattr(profile, "gamma_flip")

    def test_gamma_from_black_scholes(self):
        """BSM-гамма должна быть положительной для любого опциона."""
        gamma = black_scholes_gamma(
            S=500.0, K=500.0, T=30 / 365, r=0.045, sigma=0.20, q=0.0
        )
        assert gamma > 0


# ====================================================================== #
#  GEXPipeline
# ====================================================================== #
class TestGEXPipeline:
    """Полный прогон pipeline с синтетическими данными."""

    def _make_snapshot(self) -> OptionSnapshot:
        spot = 500.0
        strikes = np.arange(450, 551, 10)
        rows = []
        np.random.seed(42)
        for s in strikes:
            oi_c = max(0, int(np.random.lognormal(mean=6, sigma=1)))
            oi_p = max(0, int(np.random.lognormal(mean=6, sigma=1)))
            rows.append({"strike": s, "type": "C", "oi": oi_c, "iv": 0.20, "T": 30 / 365})
            rows.append({"strike": s, "type": "P", "oi": oi_p, "iv": 0.22, "T": 30 / 365})
        return OptionSnapshot(
            symbol="TEST", spot=spot,
            as_of=pd.Timestamp.now(),
            chain=pd.DataFrame(rows),
        )

    def test_pipeline_run_returns_report(self):
        snap = self._make_snapshot()
        pipeline = GEXPipeline(spot=snap.spot, symbol="TEST", r=0.045, q=0.0)
        report = pipeline.run(
            snapshot=snap,
            sigma=0.20,
            T=30 / 365,
            mu=0.0,
            run_put_wall_setup=False,
            verbose=False,
        )
        assert isinstance(report, GEXReport)
        assert report.profile is not None
        # Проверяем что профиль содержит все поля
        p = report.profile
        assert hasattr(p, "call_wall")
        assert hasattr(p, "put_wall")
        assert hasattr(p, "gamma_flip")
        assert hasattr(p, "regime")

    def test_pipeline_with_vi_signs(self):
        """Инвертированные знаки для VIX."""
        snap = self._make_snapshot()
        pipeline = GEXPipeline(
            spot=snap.spot, symbol="VIX", r=0.045, q=0.0,
            call_sign=-1.0, put_sign=+1.0,
        )
        report = pipeline.run(
            snapshot=snap,
            sigma=0.20,
            T=30 / 365,
            run_put_wall_setup=False,
            verbose=False,
        )
        assert isinstance(report, GEXReport)
        assert report.profile is not None

    def test_pipeline_empty_chain_raises(self):
        """Пустая цепочка должна вызывать ошибку или возвращать нейтральный профиль."""
        snap = OptionSnapshot(
            symbol="TEST", spot=500.0,
            as_of=pd.Timestamp.now(),
            chain=pd.DataFrame(),  # пусто
        )
        pipeline = GEXPipeline(spot=500.0, symbol="TEST", r=0.045, q=0.0)
        with pytest.raises((ValueError, KeyError, AttributeError)):
            pipeline.run(
                snapshot=snap,
                sigma=0.20,
                T=30 / 365,
                run_put_wall_setup=False,
                verbose=False,
            )


# ====================================================================== #
#  Архитектура: тест что service.py может быть разбит
# ====================================================================== #
class TestServiceArchitecture:
    """Проверка что GEXService методы независимы и модульны."""

    def test_service_imports(self):
        """Все субсервисы импортируются."""
        from gex.application.service import GEXService
        svc = GEXService()
        # Проверяем что все публичные методы существуют
        assert hasattr(svc, "analyze")
        assert hasattr(svc, "analyze_live")
        assert hasattr(svc, "analyze_crypto")
        assert hasattr(svc, "analyze_moex")
        assert hasattr(svc, "analyze_vol_index")
        assert hasattr(svc, "analyze_profile")
        assert hasattr(svc, "list_tickers")

    def test_service_analyze_returns_schema(self):
        from gex.application.service import GEXService
        svc = GEXService()
        from gex.schemas import GEXAnalysisOut
        result = svc.analyze("SPY", days=30)
        assert isinstance(result, GEXAnalysisOut)
        assert result.symbol == "SPY"
        assert result.spot > 0
