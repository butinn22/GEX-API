"""Интеграционные тесты: fetcher-ы, signal-ы и endpoint-ы.

Тестируем: сериализацию OptionSnapshot, схему ключей Redis,
расчёт направлений через direction.py, контракты Pydantic-схем.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gex.domain.data_loader import OptionSnapshot, GEXDataLoader
from gex.schemas import (
    GEXAnalysisOut,
    GEXProfileOut,
    ChainIn,
    OptionRowIn,
    OHLCVOut,
    OHLCVBarOut,
)
from gex.schemas.extended_schemas import ExtendedGEXAnalysisOut


def _sample_chain_df(n_strikes: int = 5) -> pd.DataFrame:
    """Создать тестовую цепочку."""
    strikes = np.linspace(480, 520, n_strikes)
    rows = []
    for s in strikes:
        rows.append({"strike": s, "type": "C", "oi": 1000, "iv": 0.20, "T": 30 / 365})
        rows.append({"strike": s, "type": "P", "oi": 800, "iv": 0.22, "T": 30 / 365})
    return pd.DataFrame(rows)


# ====================================================================== #
#  Pydantic schemas — контракты API
# ====================================================================== #
class TestSchemas:
    """Все Pydantic-схемы правильно сериализуются/десериализуются."""

    def test_option_row_in(self):
        row = OptionRowIn(strike=500.0, type="C", oi=1000.0, iv=0.20, T=0.08)
        assert row.strike == 500.0

    def test_option_row_invalid_type(self):
        with pytest.raises(ValidationError):
            OptionRowIn(strike=500.0, type="X", oi=1000.0, iv=0.20)

    def test_chain_in(self):
        chain = ChainIn(
            spot=500.0,
            per_contract=100,
            chain=[
                OptionRowIn(strike=500.0, type="C", oi=1000.0, iv=0.20, T=0.08),
            ],
        )
        assert chain.spot == 500.0
        assert len(chain.chain) == 1

    def test_gex_analysis_out_serialization(self):
        """GEXAnalysisOut → JSON и обратно."""
        from gex.application.service import GEXService
        svc = GEXService()
        result = svc.analyze("SPY", days=30)
        data = result.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["symbol"] == "SPY"
        assert isinstance(data["spot"], float)
        assert data["direction"] in ("BULLISH", "BEARISH", "NEUTRAL")
        # Обратно
        restored = GEXAnalysisOut.model_validate(data)
        assert restored.symbol == "SPY"

    def test_gex_profile_out(self):
        from gex.application.service import GEXService
        svc = GEXService()
        result = svc.analyze_profile("SPY", days=30)
        data = result.model_dump(mode="json")
        assert "call_wall" in data or "net_gex" in data
        assert isinstance(data, dict)

    def test_ohlcv_out(self):
        bars = [OHLCVBarOut(t="2026-01-01T00:00:00Z", o=100.0, h=105.0, l=99.0, c=104.0, v=1000.0)]
        out = OHLCVOut(symbol="SPY", asset_type="stock", timeframe="1d", spot=104.0, bars=bars)
        data = out.model_dump(mode="json")
        assert len(data["bars"]) == 1
        assert data["bars"][0]["o"] == 100.0

    def test_extended_gex_out(self):
        """ExtendedGEXAnalysisOut — минимальные поля (без сети, snapshot)."""
        from gex.application.extended import ExtendedGEXAnalyzer
        from gex.schemas.extended_schemas import extended_report_to_schema

        snap = OptionSnapshot(
            symbol="TEST", spot=500.0,
            as_of=pd.Timestamp.now("UTC"),
            chain=_sample_chain_df(),
        )
        analyzer = ExtendedGEXAnalyzer()
        report = analyzer.analyze("TEST", days=30, snapshot=snap)
        out = extended_report_to_schema(report, days=30.0)
        data = out.model_dump(mode="json")
        assert isinstance(data["symbol"], str)
        assert "zero_gamma" in data or "power_zones" in data


# ====================================================================== #
#  OptionSnapshot — сериализация через Redis
# ====================================================================== #
class TestOptionSnapshot:
    def test_pickle_roundtrip(self):
        """OptionSnapshot должен корректно сериализоваться/десериализоваться."""
        import pickle
        snap = OptionSnapshot(
            symbol="TEST", spot=500.0,
            as_of=pd.Timestamp.now(),
            chain=_sample_chain_df(),
        )
        pickled = pickle.dumps(snap)
        restored = pickle.loads(pickled)
        assert restored.symbol == "TEST"
        assert restored.spot == 500.0
        pd.testing.assert_frame_equal(restored.chain, snap.chain)

    def test_zlib_pickle_roundtrip(self):
        """OptionSnapshot через zlib+pickle (как в redis_client)."""
        import pickle
        import zlib
        from gex.adapters.cache.redis_client import serialize_value, deserialize_value
        snap = OptionSnapshot(
            symbol="TEST", spot=500.0,
            as_of=pd.Timestamp.now(),
            chain=_sample_chain_df(),
        )
        data = serialize_value(snap)
        restored = deserialize_value(data)
        assert restored.symbol == "TEST"
        assert restored.spot == 500.0


# ====================================================================== #
#  Direction (логит-модель направления)
# ====================================================================== #
class TestDirectionIntegration:
    """Проверка что direction.py работает с реальными данными GEXService."""

    def test_direction_from_service_returns_valid(self):
        from gex.application.service import GEXService
        svc = GEXService()
        result = svc.analyze("SPY", days=30)
        assert result.direction in ("BULLISH", "BEARISH", "NEUTRAL")
        assert 0 <= result.p_up <= 1.0
        assert 0 <= result.p_down <= 1.0
        # p_up + p_down ≈ 1 (с учётом нейтральной зоны)
        assert abs(result.p_up + result.p_down - 1.0) < 0.15
        assert 0 <= result.confidence <= 100.0


# ====================================================================== #
#  GEXService — фасад
# ====================================================================== #
class TestGEXServiceIntegration:
    def test_list_tickers(self):
        from gex.application.service import GEXService
        svc = GEXService()
        tickers = svc.list_tickers()
        assert "SPY" in tickers
        assert "BTC" in tickers or "SPX" in tickers
        assert isinstance(tickers, list)

    def test_analyze_all_default_tickers(self):
        """Все базовые тикеры дают валидный анализ без исключений."""
        from gex.application.service import GEXService
        svc = GEXService()
        for ticker in ["SPY", "QQQ", "IWM", "DIA", "SPX"]:
            result = svc.analyze(ticker, days=30)
            assert result.direction in ("BULLISH", "BEARISH", "NEUTRAL")
            assert result.spot > 0

    def test_profile_all_default_tickers(self):
        from gex.application.service import GEXService
        svc = GEXService()
        for ticker in ["SPY", "QQQ"]:
            result = svc.analyze_profile(ticker, days=30)
            assert hasattr(result, "call_wall") or hasattr(result, "net_gex")
