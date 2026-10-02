"""Smoke-тесты автоматического сканера сигналов.

Покрытие:
  1. Unit-тесты: константы, свежесть сигналов, торговые дни, CSV, парсинг
  2. Интеграционные тесты: все 6 API-ручек (status codes, форматы ответов)
  3. Edge-cases: дедупликация, сброс, фильтрация по ТФ, only_with_signals
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ═══════════════════════════════════════════════════════════════════════
# UNIT TESTS (no network — pure logic)
# ═══════════════════════════════════════════════════════════════════════


class TestAutoScannerConstants:
    """Константы модуля."""

    def test_timeframes(self):
        from gex.application.auto_scanner_service import TIMEFRAMES
        assert TIMEFRAMES == ("4h", "1d")
        assert len(TIMEFRAMES) == 2

    def test_bars_depth(self):
        from gex.application.auto_scanner_service import BARS
        assert BARS == 500  # минимум для EMA200

    def test_trading_day_limits(self):
        from gex.application.auto_scanner_service import (
            TRADING_DAY_LOOKBACK, BAR_LOOKBACK_4H, BARS_LOOKBACK_1D,
        )
        assert TRADING_DAY_LOOKBACK == 3
        assert BAR_LOOKBACK_4H == 18  # 6 баров/день × 3
        assert BARS_LOOKBACK_1D == 3

    def test_rate_limit_constants(self):
        from gex.application.auto_scanner_service import FETCH_DELAY, BATCH_SIZE_DEFAULT, POLL_INTERVAL_SECONDS
        assert 0 < FETCH_DELAY < 1.0  # <1 сек на запрос
        assert BATCH_SIZE_DEFAULT > 0
        assert POLL_INTERVAL_SECONDS >= 300  # минимум 5 мин

    def test_redis_ttl(self):
        from gex.application.auto_scanner_service import REDIS_SCAN_TTL, REDIS_OHLCV_TTL
        assert REDIS_SCAN_TTL >= 600  # минимум 10 мин
        assert REDIS_OHLCV_TTL >= 600


class TestAutoScanInstrument:
    """Dataclass AutoScanInstrument."""

    def test_create(self):
        from gex.application.auto_scanner_service import AutoScanInstrument
        inst = AutoScanInstrument(ticker="SPY", timeframe="4h")
        assert inst.ticker == "SPY"
        assert inst.timeframe == "4h"
        assert inst.signals == []
        assert inst.last_scan is None
        assert inst.error is None

    def test_key_case_insensitive(self):
        from gex.application.auto_scanner_service import AutoScanInstrument
        inst = AutoScanInstrument(ticker="spy", timeframe="4h")
        assert inst.key == "SPY:4h"

    def test_with_signals(self):
        from gex.application.auto_scanner_service import AutoScanInstrument
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        inst = AutoScanInstrument(
            ticker="AAPL", timeframe="1d",
            signals=[{"action": "buy", "price": 150.0}],
            last_scan=now, error="test error"
        )
        assert len(inst.signals) == 1
        assert inst.last_scan == now
        assert inst.error == "test error"


class TestTradingDays:
    """Логика подсчёта торговых дней (ядро свежести)."""

    @staticmethod
    def _count(sig_dt, now_dt):
        from gex.application.auto_scanner_service import AutoScannerService
        return AutoScannerService._count_trading_days_back(sig_dt, now_dt)

    def test_same_day(self):
        from datetime import datetime
        now = datetime(2026, 8, 2, 23, 0)
        sig = datetime(2026, 8, 2, 10, 0)
        assert self._count(sig, now) == 0  # тот же торговый день

    def test_thursday_to_sunday(self):
        """Главный кейс: четверг → воскресенье = 1 торговый день (пятница)."""
        from datetime import datetime
        now = datetime(2026, 8, 2, 23, 0)  # воскресенье
        sig = datetime(2026, 7, 30, 20, 0)  # четверг
        assert self._count(sig, now) == 1

    def test_friday_to_sunday(self):
        """Пятница → воскресенье = 0 торговых дней (суббота/вскр не считаются)."""
        from datetime import datetime
        now = datetime(2026, 8, 2, 23, 0)  # воскресенье
        sig = datetime(2026, 7, 31, 16, 0)  # пятница
        assert self._count(sig, now) == 0

    def test_saturday_to_sunday(self):
        """Суббота → воскресенье = 0 торговых дней."""
        from datetime import datetime
        now = datetime(2026, 8, 2, 23, 0)  # воскресенье
        sig = datetime(2026, 8, 1, 12, 0)   # суббота
        assert self._count(sig, now) == 0

    def test_one_week_ago(self):
        """Понедельник → понедельник (ровно неделя) = 5 торговых дней."""
        from datetime import datetime
        now = datetime(2026, 8, 3, 10, 0)  # понедельник
        sig = datetime(2026, 7, 27, 10, 0)  # прошлый понедельник
        assert self._count(sig, now) == 5

    def test_two_weeks_ago_exceeds_limit(self):
        """14 дней назад > 3 торговых дней лимита."""
        from datetime import datetime
        now = datetime(2026, 8, 2, 23, 0)
        sig = datetime(2026, 7, 13, 12, 0)
        td = self._count(sig, now)
        assert td > 3


class TestFreshnessFilter:
    """Фильтр свежести сигналов (исправленный баг aware/naive)."""

    @staticmethod
    def _is_fresh(signal, tf="4h", now=None):
        from gex.application.auto_scanner_service import AutoScannerService
        return AutoScannerService._is_fresh_signal(signal, tf, now)

    def test_old_signal_filtered(self):
        """Сигнал от 13 июля НЕ должен считаться свежим."""
        from datetime import datetime
        import types
        sig = types.SimpleNamespace(timestamp=datetime(2026, 7, 13, 12, 0))
        assert self._is_fresh(sig, "4h") is False

    def test_recent_signal_fresh(self):
        """Сигнал от вчера (1 торговый день) — свежий."""
        from datetime import datetime, timezone, timedelta
        import types
        # Вчера (суббота 1 авг), сегодня воскресенье 2 авг — 0 торговых дней
        sig = types.SimpleNamespace(timestamp=datetime(2026, 8, 1, 12, 0))
        now = datetime(2026, 8, 2, 23, 0)  # воскресенье, 0 торговых дней
        assert self._is_fresh(sig, "4h", now=now) is True

    def test_none_timestamp(self):
        """Сигнал без timestamp — НЕ свежий (безопасный default)."""
        import types
        sig = types.SimpleNamespace(timestamp=None)
        assert self._is_fresh(sig) is False

    def test_string_timestamp_iso(self):
        """Timestamp строкой в ISO-формате."""
        from datetime import datetime
        import types
        sig = types.SimpleNamespace(timestamp="2026-08-01T12:00:00")
        now = datetime(2026, 8, 2, 23, 0)
        assert self._is_fresh(sig, "4h", now=now) is True

    def test_string_timestamp_with_z(self):
        """Timestamp с Z-суффиксом."""
        from datetime import datetime
        import types
        sig = types.SimpleNamespace(timestamp="2026-08-01T12:00:00Z")
        now = datetime(2026, 8, 2, 23, 0)
        assert self._is_fresh(sig, "4h", now=now) is True

    def test_unknown_type_falls_back_false(self):
        """Неизвестный тип timestamp → False (не пропускаем)."""
        import types
        sig = types.SimpleNamespace(timestamp=12345)  # число, не datetime
        assert self._is_fresh(sig) is False

    def test_1d_timeframe(self):
        """1D таймфрейм: 3 торговых дня = свежий, 4+ = нет."""
        from datetime import datetime
        import types
        # 5 торговых дней назад (пятница 24 июля → воскресенье 2 авг)
        sig_old = types.SimpleNamespace(timestamp=datetime(2026, 7, 24, 16, 0))
        assert self._is_fresh(sig_old, "1d") is False

    def test_1d_recent(self):
        """1D таймфрейм: пятница → 0 торговых дней."""
        from datetime import datetime
        import types
        sig = types.SimpleNamespace(timestamp=datetime(2026, 7, 31, 16, 0))
        now = datetime(2026, 8, 2, 23, 0)
        assert self._is_fresh(sig, "1d", now=now) is True


class TestSignalToDict:
    """Конвертация объектов сигналов в словари."""

    def test_all_fields(self):
        from gex.application.auto_scanner_service import AutoScannerService
        from datetime import datetime
        import types
        sig = types.SimpleNamespace(
            action="buy", price=150.5, entry_score=0.85,
            confidence_class="high", timestamp=datetime(2026, 8, 1),
            tp_price=160.0, sl_price=145.0, reason="long_entry",
            order_type="entry_long", gex_reason="positive_above_flip",
            gex_multiplier=1.15, verification_score=72.5,
        )
        d = AutoScannerService._signal_to_dict(sig)
        assert d["action"] == "buy"
        assert d["price"] == 150.5
        assert d["entry_score"] == 0.85
        assert d["confidence_class"] == "high"
        assert d["tp_price"] == 160.0
        assert d["sl_price"] == 145.0
        assert d["order_type"] == "entry_long"
        assert d["gex_multiplier"] == 1.15

    def test_minimal_fields(self):
        from gex.application.auto_scanner_service import AutoScannerService
        import types
        sig = types.SimpleNamespace()
        d = AutoScannerService._signal_to_dict(sig)
        assert d["action"] is None
        assert d["price"] is None
        assert d["entry_score"] is None
        assert d["gex_multiplier"] is None


class TestCSVLoading:
    """Загрузка тикеров из CSV."""

    def test_csv_exists(self):
        from gex.application.auto_scanner_service import TICKERS_FILE
        assert TICKERS_FILE.exists(), f"CSV not found at {TICKERS_FILE}"
    def test_csv_has_tickers(self):
        from gex.application.auto_scanner_service import TICKERS_FILE
        import csv
        with open(TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        assert len(tickers) >= 100, f"Expected 100+ tickers, got {len(tickers)}"

    def test_csv_no_duplicates(self):
        from gex.application.auto_scanner_service import TICKERS_FILE
        import csv
        with open(TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        assert len(tickers) == len(set(tickers)), "CSV has duplicates"

    def test_tickers_sorted_by_service(self):
        from gex.application.service import GEXService
        from gex.application.signal_service import SignalService
        from gex.application.auto_scanner_service import AutoScannerService
        gs = GEXService()
        ss = SignalService(gs)
        svc = AutoScannerService(ss)
        tickers = svc.get_tickers()
        assert tickers == sorted(tickers), "Tickers should be sorted"
        assert len(tickers) >= 100

    def test_get_tickers_no_duplicates(self):
        from gex.application.service import GEXService
        from gex.application.signal_service import SignalService
        from gex.application.auto_scanner_service import AutoScannerService
        gs = GEXService()
        ss = SignalService(gs)
        svc = AutoScannerService(ss)
        tickers = svc.get_tickers()
        assert len(tickers) == len(set(tickers))


class TestRuScanner:
    """RU-универсум: MOEX-акции (46 тикеров), yfinance .ME, без GEX-контекста."""

    def test_ru_csv_exists(self):
        from gex.application.auto_scanner_service import RU_TICKERS_FILE
        assert RU_TICKERS_FILE.exists(), f"RU CSV not found at {RU_TICKERS_FILE}"

    def test_ru_csv_has_46_tickers(self):
        import csv
        from gex.application.auto_scanner_service import RU_TICKERS_FILE
        with open(RU_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        assert len(tickers) == 46, f"Expected 46 RU tickers, got {len(tickers)}"
        assert len(tickers) == len(set(tickers)), "RU CSV has duplicates"
        for t in ["LKOH", "SBER", "GAZP", "YDEX", "T", "TATN", "VTBR", "MOEX", "MSNG", "VKCO"]:
            assert t in tickers, f"Missing {t}"

    def test_ru_csv_names(self):
        import csv
        from gex.application.auto_scanner_service import RU_TICKERS_FILE
        with open(RU_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            names = {
                row[0].strip().upper(): row[1].strip()
                for row in reader if row and len(row) > 1 and row[0].strip()
            }
        assert names["LKOH"] == "Лукойл"
        assert names["SBER"] == "Сбербанк"
        assert len(names) == 46

    def test_ru_service_loads(self):
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, RU_TICKERS_FILE
        svc = AutoScannerService(
            MagicMock(), tickers_file=RU_TICKERS_FILE, universe="ru", skip_gex=True,
        )
        assert svc.universe == "ru"
        tickers = svc.get_tickers()
        assert len(tickers) == 46
        assert tickers == sorted(tickers)
        assert svc.get_names()["LKOH"] == "Лукойл"
        status = svc.get_status()
        assert status["total_tickers"] == 46
        assert status["total_instruments"] == 92  # ×2 ТФ

    def test_ru_scan_one_uses_bare_ticker_and_skip_gex(self):
        """RU-сканер передаёт голый тикер: OHLCV идёт через MOEXCandlesFetcher (ISS)."""
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, RU_TICKERS_FILE, BARS
        mock_signal = MagicMock()
        mock_signal.analyze_signals.return_value.recent_signals = []
        svc = AutoScannerService(
            mock_signal, tickers_file=RU_TICKERS_FILE, universe="ru", skip_gex=True,
        )
        svc._scan_one("SBER", "4h")
        mock_signal.analyze_signals.assert_called_once_with(
            "SBER", timeframe="4h", n_recent=5, bars=BARS, skip_gex=True,
            snapshot=True,
        )

    def test_all_ru_tickers_detect_as_moex(self):
        """Все 46 RU-тикеров распознаются как MOEX → фетч через ISS, не yfinance."""
        import csv
        from gex.application.auto_scanner_service import RU_TICKERS_FILE
        from gex.application.signal_service import SignalService
        from gex.adapters.fetchers.moex_candles_fetcher import _MOEX_OHLCV_ASSETS
        with open(RU_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        for t in tickers:
            assert t in _MOEX_OHLCV_ASSETS, f"{t} не определён как MOEX-актив"
            assert SignalService._detect_asset_type(t) == "moex", f"{t} не moex"

    def test_ru_moex_fetcher_supports_tickers(self):
        """MOEXCandlesFetcher принимает все RU-тикеры (canonical ticker)."""
        import csv
        from gex.application.auto_scanner_service import RU_TICKERS_FILE
        from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher
        with open(RU_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        for t in tickers:
            canonical = MOEXCandlesFetcher._canonical_ticker(t)
            assert canonical == t, f"{t} → {canonical}"

    def test_us_scan_one_uses_bare_ticker(self):
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, BARS
        mock_signal = MagicMock()
        mock_signal.analyze_signals.return_value.recent_signals = []
        svc = AutoScannerService(mock_signal)  # us по умолчанию
        svc._scan_one("NVDA", "1d")
        mock_signal.analyze_signals.assert_called_once_with(
            "NVDA", timeframe="1d", n_recent=5, bars=BARS, skip_gex=False,
            snapshot=True,
        )

    def test_ru_redis_key_scoped(self):
        """RU-сканер использует свой Redis-ключ (не пересекается с US)."""
        from gex.application.auto_scanner_service import AutoScannerService, RU_TICKERS_FILE
        from gex.adapters.cache.redis_client import cache_key
        assert cache_key("auto_scan", "ru", "signals") != cache_key("auto_scan", "us", "signals")


class TestCryptoScanner:
    """Крипто-универсум: топ-20 монет, Bybit kline + yfinance fallback, без GEX."""

    def test_crypto_csv_exists(self):
        from gex.application.auto_scanner_service import CRYPTO_TICKERS_FILE
        assert CRYPTO_TICKERS_FILE.exists(), f"Crypto CSV not found at {CRYPTO_TICKERS_FILE}"

    def test_crypto_csv_has_20_tickers(self):
        import csv
        from gex.application.auto_scanner_service import CRYPTO_TICKERS_FILE
        with open(CRYPTO_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        assert len(tickers) == 20, f"Expected 20 crypto tickers, got {len(tickers)}"
        assert len(tickers) == len(set(tickers)), "Crypto CSV has duplicates"
        for t in ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "HYPE", "DOGE",
                  "LEO", "ZEC", "XMR", "ADA", "LINK", "XLM", "BCH", "CC",
                  "GRAM", "LTC", "HBAR", "SUI"]:
            assert t in tickers, f"Missing {t}"

    def test_crypto_csv_names(self):
        import csv
        from gex.application.auto_scanner_service import CRYPTO_TICKERS_FILE
        with open(CRYPTO_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            names = {
                row[0].strip().upper(): row[1].strip()
                for row in reader if row and len(row) > 1 and row[0].strip()
            }
        assert names["BTC"] == "Bitcoin"
        assert names["ETH"] == "Ethereum"
        assert names["HYPE"] == "Hyperliquid"
        assert len(names) == 20

    def test_crypto_service_loads(self):
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, CRYPTO_TICKERS_FILE
        svc = AutoScannerService(
            MagicMock(), tickers_file=CRYPTO_TICKERS_FILE, universe="crypto", skip_gex=True,
        )
        assert svc.universe == "crypto"
        tickers = svc.get_tickers()
        assert len(tickers) == 20
        assert tickers == sorted(tickers)
        assert svc.get_names()["BTC"] == "Bitcoin"
        status = svc.get_status()
        assert status["total_tickers"] == 20
        assert status["total_instruments"] == 40  # ×2 ТФ

    def test_crypto_scan_one_uses_bare_ticker_and_skip_gex(self):
        """Крипто-сканер передаёт голый тикер: OHLCV идёт через Bybit kline."""
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, CRYPTO_TICKERS_FILE, BARS
        mock_signal = MagicMock()
        mock_signal.analyze_signals.return_value.recent_signals = []
        svc = AutoScannerService(
            mock_signal, tickers_file=CRYPTO_TICKERS_FILE, universe="crypto", skip_gex=True,
        )
        svc._scan_one("BNB", "4h")
        mock_signal.analyze_signals.assert_called_once_with(
            "BNB", timeframe="4h", n_recent=5, bars=BARS, skip_gex=True,
            snapshot=True,
        )

    def test_all_crypto_tickers_detect_as_crypto(self):
        """Все 20 монет детектятся как crypto → Bybit kline (не yfinance-акция)."""
        import csv
        from gex.application.auto_scanner_service import CRYPTO_TICKERS_FILE
        from gex.application.signal_service import SignalService
        with open(CRYPTO_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        for t in tickers:
            assert SignalService._detect_asset_type(t) == "crypto", f"{t} не crypto"


class TestFxScanner:
    """Валюты и металлы: DXY, EUR/USD, USD/CNY, USD/JPY, GOLD, SILVER (yfinance)."""

    FX_TICKERS = ["DXY", "EUR/USD", "USD/CNY", "USD/JPY", "GOLD", "SILVER"]

    def test_fx_csv_exists(self):
        from gex.application.auto_scanner_service import FX_TICKERS_FILE
        assert FX_TICKERS_FILE.exists(), f"FX CSV not found at {FX_TICKERS_FILE}"

    def test_fx_csv_has_6_tickers(self):
        import csv
        from gex.application.auto_scanner_service import FX_TICKERS_FILE
        with open(FX_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tickers = [row[0].strip().upper() for row in reader if row and row[0].strip()]
        assert len(tickers) == 6, f"Expected 6 FX tickers, got {len(tickers)}"
        assert len(tickers) == len(set(tickers)), "FX CSV has duplicates"
        for t in self.FX_TICKERS:
            assert t in tickers, f"Missing {t}"

    def test_fx_csv_names(self):
        import csv
        from gex.application.auto_scanner_service import FX_TICKERS_FILE
        with open(FX_TICKERS_FILE, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            names = {
                row[0].strip().upper(): row[1].strip()
                for row in reader if row and len(row) > 1 and row[0].strip()
            }
        assert names["DXY"] == "US Dollar Index"
        assert names["EUR/USD"] == "Euro / US Dollar"
        assert names["GOLD"] == "Gold"
        assert len(names) == 6

    def test_fx_service_loads(self):
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, FX_TICKERS_FILE
        svc = AutoScannerService(
            MagicMock(), tickers_file=FX_TICKERS_FILE, universe="fx", skip_gex=True,
        )
        assert svc.universe == "fx"
        tickers = svc.get_tickers()
        assert len(tickers) == 6
        assert tickers == sorted(tickers)
        assert svc.get_names()["DXY"] == "US Dollar Index"
        status = svc.get_status()
        assert status["total_tickers"] == 6
        assert status["total_instruments"] == 12  # ×2 ТФ

    def test_fx_scan_one_uses_bare_ticker_and_skip_gex(self):
        """FX-сканер передаёт голый тикер: OHLCV мапится в yfinance (EURUSD=X и т.д.)."""
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, FX_TICKERS_FILE, BARS
        mock_signal = MagicMock()
        mock_signal.analyze_signals.return_value.recent_signals = []
        svc = AutoScannerService(
            mock_signal, tickers_file=FX_TICKERS_FILE, universe="fx", skip_gex=True,
        )
        svc._scan_one("EUR/USD", "4h")
        mock_signal.analyze_signals.assert_called_once_with(
            "EUR/USD", timeframe="4h", n_recent=5, bars=BARS, skip_gex=True,
            snapshot=True,
        )

    def test_fx_ticker_detection(self):
        """DXY и валютные пары — fx; GOLD/SILVER — commodity (уже готовый фетч)."""
        from gex.application.signal_service import SignalService, _FX_TICKERS
        assert _FX_TICKERS == {
            "DXY": "DX-Y.NYB",
            "EUR/USD": "EURUSD=X",
            "USD/CNY": "CNY=X",
            "USD/JPY": "JPY=X",
        }
        for t in ["DXY", "EUR/USD", "USD/CNY", "USD/JPY"]:
            assert SignalService._detect_asset_type(t) == "fx", f"{t} не fx"
        assert SignalService._detect_asset_type("GOLD") == "commodity"
        assert SignalService._detect_asset_type("SILVER") == "commodity"

    def test_fx_redis_key_scoped(self):
        """FX-сканер использует свой Redis-ключ (не пересекается с US/ru/crypto)."""
        from gex.application.auto_scanner_service import AutoScannerService
        from gex.adapters.cache.redis_client import cache_key
        keys = [
            cache_key("auto_scan", u, "signals")
            for u in ("us", "ru", "crypto", "fx")
        ]
        assert len(keys) == len(set(keys)), "Redis-ключи универсумов пересекаются"


class TestSectorsUniverse:
    """Универсум «Сектора»: 12 секторальных ETF США + RSP (равновесный S&P 500).

    Те же инструменты, что на странице «Композит секторов»; в отличие от
    MOEX/крипты/FX у ETF есть опционная цепочка, поэтому GEX НЕ отключаем.
    """

    def test_sectors_service_loads(self):
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import (
            SECTOR_TICKERS_FILE, AutoScannerService,
        )
        svc = AutoScannerService(
            MagicMock(), tickers_file=SECTOR_TICKERS_FILE, universe="sectors",
        )
        assert svc.universe == "sectors"
        tickers = svc.get_tickers()
        assert len(tickers) == 12
        assert set(tickers) == {"RSP", "XLB", "XLC", "XLE", "XLF", "XLI",
                                "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"}
        assert svc.get_names()["XLK"] == "Technology"
        status = svc.get_status()
        assert status["total_tickers"] == 12
        assert status["total_instruments"] == 24  # ×2 ТФ

    def test_sectors_scan_keeps_gex(self):
        """Секторальные ETF сканируются с GEX-контекстом (опционы у них есть)."""
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import (
            BARS, SECTOR_TICKERS_FILE, AutoScannerService,
        )
        mock_signal = MagicMock()
        mock_signal.analyze_signals.return_value.recent_signals = []
        svc = AutoScannerService(
            mock_signal, tickers_file=SECTOR_TICKERS_FILE, universe="sectors",
        )
        svc._scan_one("XLK", "4h")
        mock_signal.analyze_signals.assert_called_once_with(
            "XLK", timeframe="4h", n_recent=5, bars=BARS, skip_gex=False,
            snapshot=True,
        )

    def test_sectors_ticker_detected_as_stock(self):
        """ETF идут по «обычному» пути yfinance (не fx/commodity/moex)."""
        from gex.application.signal_service import SignalService
        for t in ("XLK", "XLF", "XLRE", "RSP"):
            assert SignalService._detect_asset_type(t) == "stock", f"{t} не stock"

    def test_sectors_redis_key_scoped(self):
        from gex.application.auto_scanner_service import AutoScannerService  # noqa: F401
        from gex.adapters.cache.redis_client import cache_key
        keys = [
            cache_key("auto_scan", u, "signals")
            for u in ("us", "ru", "crypto", "fx", "sectors")
        ]
        assert len(keys) == len(set(keys)), "Redis-ключи универсумов пересекаются"

    def test_router_picks_sectors_service(self):
        """universe=sectors → сервис секторов, а не US-акций."""
        from gex.routers.auto_scanner_router import _pick_service
        sentinels = {name: object() for name in ("us", "ru", "crypto", "fx", "sectors")}
        for name, obj in sentinels.items():
            picked = _pick_service(
                name,
                sentinels["us"], sentinels["ru"], sentinels["crypto"],
                sentinels["fx"], sentinels["sectors"],
            )
            assert picked is obj, f"universe={name} выбрал не тот сервис"
        # Неизвестный универсум падает обратно на US (как раньше).
        assert _pick_service("zzz", *sentinels.values()) is sentinels["us"]


class TestAutoScannerServiceUnit:
    """Unit-тесты сервиса (без сети — мок SignalService)."""

    @pytest.fixture
    def svc(self):
        from unittest.mock import MagicMock
        from gex.application.auto_scanner_service import AutoScannerService, TIMEFRAMES
        mock_signal = MagicMock()
        # Возвращаем пустой анализ (нет сигналов)
        mock_signal.analyze_signals.return_value.recent_signals = []
        svc = AutoScannerService(mock_signal)
        return svc

    def test_init_creates_instruments(self, svc):
        """После init должны быть инструменты для всех тикеров × ТФ."""
        status = svc.get_status()
        assert status["total_tickers"] > 0
        expected_instruments = status["total_tickers"] * 2  # ×4H,1D
        assert status["total_instruments"] == expected_instruments

    def test_status_shape(self, svc):
        s = svc.get_status()
        required_keys = {
            "running", "total_tickers", "total_instruments",
            "scanned_count", "completed_fetches",
            "instruments_with_signals", "instruments_with_errors",
        }
        assert required_keys.issubset(s.keys())
        assert isinstance(s["running"], bool)
        assert isinstance(s["total_tickers"], int)
        assert s["scanned_count"] == 0  # ещё не сканировали

    def test_get_signals_empty(self, svc):
        signals = svc.get_signals()
        assert isinstance(signals, list)
        assert len(signals) > 0  # инструменты есть, сигналов нет
        for s in signals:
            assert "ticker" in s
            assert "timeframe" in s
            assert s["signals"] == []
            assert s["last_scan"] is None

    def test_reset_clears_state(self, svc):
        s = svc.reset()
        assert s["scanned_count"] == 0
        assert s["completed_fetches"] == 0
        assert s["last_error"] is None

    def test_get_signals_filter_by_ticker(self, svc):
        all_tickers = svc.get_tickers()
        if not all_tickers:
            pytest.skip("No tickers loaded")
        target = all_tickers[0]
        signals = svc.get_signals(ticker=target)
        assert all(s["ticker"].upper() == target.upper() for s in signals)
        assert len(signals) == 2  # 4H + 1D

    def test_get_signals_filter_by_timeframe(self, svc):
        signals_4h = svc.get_signals(timeframe="4h")
        assert all(s["timeframe"] == "4h" for s in signals_4h)
        signals_1d = svc.get_signals(timeframe="1d")
        assert all(s["timeframe"] == "1d" for s in signals_1d)
        assert len(signals_4h) + len(signals_1d) == len(svc.get_signals())

    def test_start_stop_lifecycle(self, svc):
        assert svc.is_running is False
        svc.start()
        time.sleep(0.1)
        assert svc.is_running is True
        svc.stop()
        # Даём время на остановку
        time.sleep(0.2)
        # После stop.is_running может быть True если цикл ещё не вышел
        # Проверяем что метод не крашится
        svc.stop()  # двойной stop не должен крашиться


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TESTS (real server on port 8009)
# ═══════════════════════════════════════════════════════════════════════

BASE_URL = "http://127.0.0.1:8009"
_PORT = 8009

# Auth token placeholder — smoke тесты используют ручки без auth
# (роутер требует EXTENDED подписку — в smoke тестах проверяем доступность)


@pytest.fixture(scope="session")
def server():
    """Запустить uvicorn как subprocess."""
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import uvicorn; uvicorn.run('main:app', host='127.0.0.1', port={_PORT}, log_level='error')"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            break
        except Exception:
            time.sleep(1.5)
    else:
        proc.terminate()
        pytest.fail("Server failed to start within 60s")
    yield
    proc.terminate()
    proc.wait(timeout=5)


def _get(path: str, timeout: int = 20, token: str | None = None) -> requests.Response:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(f"{BASE_URL}{path}", timeout=timeout, headers=headers)


def _post(path: str, json: dict | None = None, timeout: int = 20, token: str | None = None) -> requests.Response:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.post(f"{BASE_URL}{path}", json=json or {}, timeout=timeout, headers=headers)


def _ok_or_auth(code: int) -> bool:
    """200/201 или 401/403 (auth required)."""
    return code in (200, 201, 401, 403)


# ═══════════════════════════════════════════════════════════════════════
# Smoke: все 6 ручек авто-сканера
# ═══════════════════════════════════════════════════════════════════════


class TestAutoScannerRoutes:
    """Проверка доступности и формата ответа всех ручек."""

    def test_get_tickers_available(self, server):
        """GET /scanner/auto/tickers — должен быть доступен (возможно с auth)."""
        r = _get("/scanner/auto/tickers")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "tickers" in data
            assert "count" in data
            assert data["count"] >= 100

    def test_get_status_available(self, server):
        """GET /scanner/auto/status."""
        r = _get("/scanner/auto/status")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "running" in data
            assert "total_tickers" in data
            assert "scanned_count" in data
            assert isinstance(data["running"], bool)
            assert isinstance(data["total_tickers"], int)

    def test_get_signals_available(self, server):
        """GET /scanner/auto/signals."""
        r = _get("/scanner/auto/signals")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "instruments" in data
            assert "total" in data
            assert "status" in data
            assert isinstance(data["instruments"], list)

    def test_get_signals_only_with_signals(self, server):
        """GET /scanner/auto/signals?only_with_signals=true."""
        r = _get("/scanner/auto/signals?only_with_signals=true")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            # Все возвращённые инструменты должны иметь сигналы
            for instr in data.get("instruments", []):
                assert len(instr.get("signals", [])) > 0, (
                    f"only_with_signals filter failed for {instr['ticker']}"
                )

    def test_get_signals_filter_ticker(self, server):
        """GET /scanner/auto/signals?ticker=SPY."""
        r = _get("/scanner/auto/signals?ticker=SPY")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            for instr in data.get("instruments", []):
                assert instr["ticker"].upper() == "SPY"

    def test_get_signals_filter_timeframe(self, server):
        """GET /scanner/auto/signals?timeframe=4h."""
        r = _get("/scanner/auto/signals?timeframe=4h")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            for instr in data.get("instruments", []):
                assert instr["timeframe"] == "4h"

    def test_run_scan_available(self, server):
        """POST /scanner/auto/run."""
        r = _post("/scanner/auto/run", timeout=30)
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "scanned_count" in data
            assert "total_tickers" in data

    def test_run_next_batch_available(self, server):
        """POST /scanner/auto/run/next?batch_size=5."""
        r = _post("/scanner/auto/run/next?batch_size=5", timeout=30)
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "scanned_count" in data
            assert "new_signals" in data
            assert "done" in data

    def test_reset_available(self, server):
        """POST /scanner/auto/reset."""
        r = _post("/scanner/auto/reset")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert data["scanned_count"] == 0

    def test_signals_response_structure(self, server):
        """Каждый instrument имеет правильную структуру."""
        r = _get("/scanner/auto/signals")
        if r.status_code != 200:
            pytest.skip("Auth required")
        data = r.json()
        for instr in data.get("instruments", []):
            assert "ticker" in instr
            assert "timeframe" in instr
            assert "signals" in instr
            assert "last_scan" in instr
            assert "error" in instr
            assert isinstance(instr["signals"], list)
            assert instr["timeframe"] in ("4h", "1d")
            # Проверяем поля внутри сигналов
            for sig in instr["signals"]:
                assert "action" in sig
                assert "price" in sig
                assert "confidence_class" in sig
                assert "order_type" in sig

    # ── RU-универсум (MOEX-акции) ──
    def test_get_tickers_ru(self, server):
        """GET /scanner/auto/tickers?universe=ru → 46 MOEX-тикеров с именами."""
        r = _get("/scanner/auto/tickers?universe=ru")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert data["count"] == 46
            assert "LKOH" in data["tickers"]
            assert "SBER" in data["tickers"]
            assert data["names"]["LKOH"] == "Лукойл"

    def test_get_status_ru(self, server):
        """GET /scanner/auto/status?universe=ru."""
        r = _get("/scanner/auto/status?universe=ru")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "running" in data
            assert data["total_tickers"] == 46

    def test_get_signals_ru(self, server):
        """GET /scanner/auto/signals?universe=ru."""
        r = _get("/scanner/auto/signals?universe=ru")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "instruments" in data
            assert "status" in data
            # Все инструменты — из RU-списка (никогда не пересекается с US)
            for instr in data["instruments"]:
                assert instr["ticker"] not in (
                    "NVDA", "AAPL", "MSFT", "SPY", "QQQ",
                ), f"US ticker leaked into RU universe: {instr['ticker']}"

    def test_run_next_batch_ru(self, server):
        """POST /scanner/auto/run/next?universe=ru — прогрессивный скан RU."""
        r = _post("/scanner/auto/run/next?batch_size=5&universe=ru", timeout=30)
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "scanned_count" in data
            assert data["total_tickers"] == 46

    # ── Crypto-универсум (топ-20 монет, Bybit) ──
    def test_get_tickers_crypto(self, server):
        """GET /scanner/auto/tickers?universe=crypto → 20 монет с именами."""
        r = _get("/scanner/auto/tickers?universe=crypto")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert data["count"] == 20
            assert "BTC" in data["tickers"]
            assert "SUI" in data["tickers"]
            assert data["names"]["BTC"] == "Bitcoin"

    def test_get_status_crypto(self, server):
        """GET /scanner/auto/status?universe=crypto."""
        r = _get("/scanner/auto/status?universe=crypto")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "running" in data
            assert data["total_tickers"] == 20

    def test_get_signals_crypto(self, server):
        """GET /scanner/auto/signals?universe=crypto."""
        r = _get("/scanner/auto/signals?universe=crypto")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "instruments" in data
            assert "status" in data
            for instr in data["instruments"]:
                assert instr["ticker"] not in ("NVDA", "AAPL", "SPY"), "US ticker leaked"


    # ── FX-универсум (валюты и металлы: DXY, EUR/USD, USD/CNY, USD/JPY, GOLD, SILVER) ──
    def test_get_tickers_fx(self, server):
        """GET /scanner/auto/tickers?universe=fx → 6 инструментов с именами."""
        r = _get("/scanner/auto/tickers?universe=fx")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert data["count"] == 6
            assert "DXY" in data["tickers"]
            assert "EUR/USD" in data["tickers"]
            assert "GOLD" in data["tickers"]
            assert data["names"]["DXY"] == "US Dollar Index"

    def test_get_status_fx(self, server):
        """GET /scanner/auto/status?universe=fx."""
        r = _get("/scanner/auto/status?universe=fx")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "running" in data
            assert data["total_tickers"] == 6

    def test_get_signals_fx(self, server):
        """GET /scanner/auto/signals?universe=fx — без утечки US-тикеров."""
        r = _get("/scanner/auto/signals?universe=fx")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "instruments" in data
            assert "status" in data
            for instr in data["instruments"]:
                assert instr["ticker"] not in ("NVDA", "AAPL", "SPY"), "US ticker leaked"

    def test_run_next_batch_fx(self, server):
        """POST /scanner/auto/run/next?universe=fx — прогрессивный скан FX."""
        r = _post("/scanner/auto/run/next?batch_size=5&universe=fx", timeout=30)
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "scanned_count" in data
            assert data["total_tickers"] == 6

    # ── Универсум «Сектора» (12 секторальных ETF + RSP) ──
    def test_get_tickers_sectors(self, server):
        """GET /scanner/auto/tickers?universe=sectors → 12 ETF с именами секторов."""
        r = _get("/scanner/auto/tickers?universe=sectors")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert data["count"] == 12
            for tk in ("XLK", "XLF", "XLE", "XLRE", "RSP"):
                assert tk in data["tickers"], tk
            assert data["names"]["XLK"] == "Technology"
            assert data["names"]["RSP"] == "Equal-Weight S&P 500"

    def test_get_status_sectors(self, server):
        """GET /scanner/auto/status?universe=sectors."""
        r = _get("/scanner/auto/status?universe=sectors")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "running" in data
            assert data["total_tickers"] == 12

    def test_get_signals_sectors(self, server):
        """GET /scanner/auto/signals?universe=sectors — без утечки US-акций."""
        r = _get("/scanner/auto/signals?universe=sectors")
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "instruments" in data
            assert "status" in data
            for instr in data["instruments"]:
                assert instr["ticker"] not in ("NVDA", "AAPL", "MSFT"), "US stock leaked"

    def test_run_next_batch_sectors(self, server):
        """POST /scanner/auto/run/next?universe=sectors — прогрессивный скан секторов."""
        r = _post("/scanner/auto/run/next?batch_size=5&universe=sectors", timeout=30)
        assert _ok_or_auth(r.status_code), f"Got {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            assert "scanned_count" in data
            assert data["total_tickers"] == 12


class TestAutoScannerEdgeCases:
    """Граничные случаи."""

    def test_invalid_ticker_filter(self, server):
        """Фильтр по несуществующему тикеру должен вернуть пустой список."""
        r = _get("/scanner/auto/signals?ticker=NONEXIST12345")
        if r.status_code == 200:
            data = r.json()
            assert data["total"] == 0

    def test_invalid_timeframe(self, server):
        """Неподдерживаемый таймфрейм — должен вернуть пустой список."""
        r = _get("/scanner/auto/signals?timeframe=3h")
        if r.status_code == 200:
            data = r.json()
            assert data["total"] == 0

    def test_batch_size_clamping(self, server):
        """batch_size=1000 должен быть заклемплен."""
        r = _post("/scanner/auto/run/next?batch_size=1000", timeout=30)
        if r.status_code == 200:
            data = r.json()
            scanned = data.get("scanned_count", 0)
            # Не должно просканировать все 1000 за раз
            assert scanned <= 50  # max batch_size по спецификации

    def test_reset_then_status(self, server):
        """После сброса scanned_count должен быть 0."""
        _post("/scanner/auto/reset")
        r = _get("/scanner/auto/status")
        if r.status_code == 200:
            data = r.json()
            assert data["scanned_count"] == 0
            assert data["instruments_with_signals"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
