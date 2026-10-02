"""Сервис торговых сигналов: EMA Multi-Filter стратегия + GEX + verification.

Оркестрирует полный конвейер формирования торгового сигнала:
  1. Загрузка OHLCV (Bybit V5 для крипты с fallback на yfinance; yfinance для акций);
  2. Расчёт features через :class:`gex.trading_algorithm.EMAFilterTrendStrategy`;
  3. GEX-контекст (один вызов, с кэшем): regime, gamma_flip, net_gex, z_score;
  4. Backtest по истории — извлечение последних N entry-сигналов;
  5. Текущий сигнал на последнем баре с GEX-фильтром + verification.

Источники данных
----------------
* **US акции**: :class:`gex.ta_fetcher.TATimeframesFetcher` (yfinance) — переиспользуется.
* **Крипта (BTC/ETH/SOL/XRP/DOGE)**: публичный Bybit V5 ``/v5/market/kline``
  (консистентно с GEX-источником), с авто-fallback на yfinance (``BTC-USD``).
* **MOEX-фьючерсы (RTS/MIX/CNY/Si)**: :class:`gex.moex_candles_fetcher.
  MOEXCandlesFetcher` (ISS candles FORTS front-month); GEX-контекст — через
  :meth:`gex.service.GEXService.analyze_moex` (тот же ISS-источник опционов).

GEX-контекст — снимок «сейчас»
------------------------------
GEX-профиль рассчитывается по текущей опционной цепочке (один раз, с кэшем на
``gex_cache_ttl`` секунд). Для исторических баров backtest мы НЕ реконструируем
исторический GEX (это невозможно без исторических цепочек) — множитель честно
помечается как рассчитанный на *текущем* профиле.

Исключения
----------
MOEX-инструменты (RTS/MIX/CNY/Si) поддерживаются: OHLCV — через ISS candles
FORTS, GEX-контекст — через :meth:`analyze_moex`. Если свежая опционная цепочка
MOEX недоступна (внебиржевое время / сбой ISS), сигнал собирается без GEX
(``gex_context=null``), как для акций без цепочки.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
from gex.adapters.providers.bybit import fetch_ohlcv as fetch_bybit_ohlcv
from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher, _MOEX_OHLCV_ASSETS
from gex.application.service import GEXService
from gex.schemas import (
    CurrentSignalOut,
    GEXContextOut,
    PositionOut,
    SignalAnalysisOut,
    SignalRecordOut,
    TrendRegimeOut,
)
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher
from gex.strategy.trading_algorithm import (
    EMAFilterTrendStrategy,
    GEXContext,
    SignalAction,
    StrategySettings,
    TradingSignal,
)
from gex.domain.verification import verify_from_dataframe
from gex.adapters.providers.catalog import CANONICAL_TIMEFRAMES

logger = logging.getLogger(__name__)


# MOEX-инструменты — поддерживаются: OHLCV через ISS candles, GEX через analyze_moex.
# Множество переиспользуется для детекта asset_type='moex'.
_MOEX_ASSETS = _MOEX_OHLCV_ASSETS

# Маппинг таймфреймов API → «человекочитаемые» метки для OHLCV.
# yfinance: 1h/1d нативно, 2h/4h через ресемплинг. Bybit kline: D/240/60.
_SUPPORTED_TIMEFRAMES = CANONICAL_TIMEFRAMES

# Криптовалюты без опционной цепочки (только spot-сканирование): детектятся
# как "crypto" → OHLCV через Bybit kline с fallback на yfinance (*-USD).
_CRYPTO_SPOT_TICKERS: frozenset[str] = frozenset({
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "HYPE", "DOGE", "LEO", "ZEC",
    "XMR", "ADA", "LINK", "XLM", "BCH", "CC", "GRAM", "LTC", "HBAR", "SUI",
})

# yfinance-тикер для крипты (fallback): BTC → BTC-USD.
_YF_CRYPTO_TICKER = {coin: f"{coin}-USD" for coin in _CRYPTO_ASSETS}

# Валюты и металлы (валютные пары + индекс доллара) — yfinance-тикеры.
# DXY — индекс доллара (NYB), пары — canonical yfinance FX (EURUSD=X и т.д.).
# GOLD/SILVER детектятся как commodity (GC=F/SI=F) — здесь не нужны.
_FX_TICKERS: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "USD/CNY": "CNY=X",
    "USD/JPY": "JPY=X",
}

# Глубина истории по умолчанию (баров) для backtest.
_DEFAULT_BARS = 750

#: Кулдаун добавления к позиции, баров (зеркало ``_can_add`` из стратегии).
_ADD_COOLDOWN_BARS = 10
#: Доля объёма добавления к позиции (зеркало ``quantity_fraction=0.1`` в evaluate()).
_ADD_QTY_FRACTION = 0.1

#: Колонки снапшота для переигровки машины позиции на чтении (персональные
#: настройки трейлинга без повторного фетча истории): OHLC + сигнальные
#: колонки + минимум для записи сигнала (score/TP/SL/GEX-контекст не входит).
_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "high", "low", "close",
    "long_entry_signal", "short_entry_signal",
    "long_exit_signal", "short_exit_signal",
    "combined_long_add", "combined_short_add",
    "atr",
    "long_entry_a", "long_entry_b", "short_entry_a", "short_entry_b",
    "can_enter_long", "can_enter_short",
    "novelsrc", "ema10", "ema200",
    "my_vwap_state", "my_vwap_state_1", "my_vwap_state_5",
    "adline", "adl50", "adl_macd", "tp_f", "trend_coefficient",
)


class SignalService:
    """Фасад формирования торговых сигналов с GEX-интеграцией.

    Parameters
    ----------
    gex_service : GEXService
        Сервис GEX-анализа (методы ``analyze_live`` / ``analyze_crypto``).
    ta_fetcher : TATimeframesFetcher
        Источник OHLCV для US-акций. По умолчанию создаётся стандартный.
    settings : StrategySettings
        Настройки EMA Multi-Filter стратегии.
    gex_days : float
        Горизонт GEX-анализа по умолчанию, дней.
    gex_cache_ttl : float
        Время жизни кэша GEX-контекста, секунд (повторные вызовы в окне —
        без нового запроса к yfinance/Bybit).
    """

    def __init__(
        self,
        gex_service: GEXService,
        ta_fetcher: Optional[TATimeframesFetcher] = None,
        settings: Optional[StrategySettings] = None,
        gex_days: float = 30.0,
        gex_cache_ttl: float = 300.0,
    ):
        self.gex_service = gex_service
        self.ta_fetcher = ta_fetcher or create_timeframes_fetcher()
        self.strategy = EMAFilterTrendStrategy(settings=settings)
        self.gex_days = float(gex_days)
        self.gex_cache_ttl = float(gex_cache_ttl)
        # Кэш GEX-контекста: {(ticker, asset_type): (GEXContext, timestamp)}.
        self._gex_cache: dict[tuple[str, str], tuple[GEXContext | None, float]] = {}

    # ================================================================== #
    #  Публичный API
    # ================================================================== #
    def analyze_signals(
        self,
        ticker: str,
        timeframe: str = "1d",
        n_recent: int = 5,
        gex_days: Optional[float] = None,
        bars: int = _DEFAULT_BARS,
        skip_gex: bool = False,
        trailing_pct: Optional[float] = None,
        reverse: bool = False,
        snapshot: bool = False,
    ) -> SignalAnalysisOut:
        """Полный анализ сигналов по тикеру.

        Parameters
        ----------
        ticker : str
            Тикер (US-акция: ``SPY``/``AAPL``/...; крипта: ``BTC``/``ETH``/...).
        timeframe : str
            Один из ``1h``/``2h``/``4h``/``1d`` (по умолчанию ``1d``).
        n_recent : int
            Сколько последних сигналов извлечь из истории.
        gex_days : float, optional
            Горизонт GEX-анализа, дней (по умолчанию из настроек сервиса).
        bars : int
            Глубина истории OHLCV для backtest.
        skip_gex : bool
            Пропустить GEX-контекст (для активов без опционной цепочки,
            например MOEX-акции через yfinance) — экономит время на запрос.
        trailing_pct : float, optional
            Трейлинг-стоп, % (персональная настройка): выход при откате от
            экстремума цены с момента входа. ``None``/0 — выключен.
        reverse : bool
            Принудительный разворот: свежий противоположный вход закрывает
            позицию и сразу открывает обратную.
        snapshot : bool
            Приложить к ответу компактный снапшот истории (для переигровки
            машины позиции на чтении с персональными настройками).

        Raises
        ------
        ValueError
            Неподдерживаемый тикер/таймфрейм, или MOEX-актив, или данные не получены.
        RuntimeError
            При сетевых ошибках источников данных.
        """
        ticker = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in _SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Неподдерживаемый таймфрейм '{timeframe}'. Доступно: {', '.join(_SUPPORTED_TIMEFRAMES)}."
            )

        asset_type = self._detect_asset_type(ticker)
        logger.info("Signal analyze: ticker=%s type=%s tf=%s", ticker, asset_type, tf)

        # --- 1. OHLCV ---
        ohlcv = self._fetch_ohlcv(ticker, tf, bars, asset_type)
        if ohlcv is None or len(ohlcv) < 60:
            n = 0 if ohlcv is None else len(ohlcv)
            raise ValueError(
                f"Недостаточно истории OHLCV для '{ticker}' ({n} баров, нужно ≥60)."
            )

        # --- 2. Features стратегии ---
        features = self.strategy.calculate(ohlcv)
        if features is None or features.empty:
            raise ValueError(f"Не удалось рассчитать features для '{ticker}'.")

        # --- 3. GEX-контекст (1 вызов, с кэшем; пропускается при skip_gex) ---
        gex_ctx = None if skip_gex else self._get_gex_context(ticker, asset_type, gex_days)

        # --- 4. Backtest: последние N сигналов из истории (машина позиции) ---
        recent_records, position = self._extract_recent_signals(
            features, ohlcv, gex_ctx, n_recent,
            trailing_pct=trailing_pct, reverse=reverse,
        )

        # --- 5. Текущий сигнал на последнем баре ---
        current = self._build_current_signal(features, ohlcv, gex_ctx)

        # --- 6. Режим рынка (тренд/флэт на 200 барах, ATR+BBW+z) ---
        # Аддитивно: не влияет ни на features, ни на сигналы. Метрики
        # слайдер-независимы — персональный порог применяется на чтении.
        regime = self._trend_regime(ohlcv)

        spot = float(features["close"].iloc[-1])
        gex_out = self._gex_context_to_schema(gex_ctx)

        return SignalAnalysisOut(
            symbol=ticker,
            asset_type=asset_type,
            timeframe=tf,
            generated_at=datetime.now(timezone.utc),
            spot=spot,
            bars_analyzed=len(features),
            gex_context=gex_out,
            current_signal=current,
            recent_signals=recent_records,
            position=self._position_to_schema(position, features),
            regime=regime,
            snapshot=self._build_snapshot(features) if snapshot else None,
        )

    # ================================================================== #
    #  Режим рынка: тренд / флэт на 200 барах
    # ================================================================== #
    def _trend_regime(self, ohlcv: pd.DataFrame) -> Optional[TrendRegimeOut]:
        """Вердикт детектора тренда/флэта по OHLCV (None — если посчитать нельзя).

        Никогда не бросает: детектор — вспомогательный слой, его отказ не должен
        ломать основной анализ сигналов.
        """
        try:
            data = self.strategy.trend_regime(ohlcv)
        except Exception:  # noqa: BLE001 — вспомогательный слой
            logger.debug("trend_regime: ошибка расчёта режима", exc_info=True)
            return None
        if not data:
            return None
        try:
            return TrendRegimeOut(**data)
        except (TypeError, ValueError):
            logger.debug("trend_regime: не удалось сериализовать вердикт", exc_info=True)
            return None

    # ================================================================== #
    #  Определение типа актива и фетч OHLCV
    # ================================================================== #
    @staticmethod
    def _detect_asset_type(ticker: str) -> str:
        if ticker in _MOEX_OHLCV_ASSETS:
            return "moex"
        if ticker in _CRYPTO_ASSETS or ticker in _CRYPTO_SPOT_TICKERS:
            return "crypto"
        if ticker in _FX_TICKERS:
            return "fx"
        from gex.commodity_assets import COMMODITY_ASSETS
        if ticker in COMMODITY_ASSETS:
            return "commodity"
        return "stock"

    def _fetch_ohlcv(
        self, ticker: str, tf: str, bars: int, asset_type: str
    ) -> Optional[pd.DataFrame]:
        """Получить OHLCV для тикера.

        US-акции: TATimeframesFetcher (yfinance, уже с ресемплингом 2h/4h).
        Крипта: Bybit kline → fallback на yfinance (BTC-USD).
        MOEX: MOEXCandlesFetcher (ISS FORTS front-month, ресемплинг 2h/4h).
        """
        if asset_type == "moex":
            return self._fetch_moex_ohlcv(ticker, tf)

        if asset_type == "commodity":
            # Commodity: fetch via yfinance futures symbol
            from gex.commodity_assets import COMMODITY_ASSETS
            yf_sym = COMMODITY_ASSETS[ticker]["yf_symbol"]
            return self._fetch_stock_ohlcv(yf_sym, tf)

        if asset_type == "fx":
            # Валюты/индекс доллара: yfinance FX-тикер (EURUSD=X, DX-Y.NYB, ...)
            yf_sym = _FX_TICKERS.get(ticker, ticker)
            return self._fetch_stock_ohlcv(yf_sym, tf)

        if asset_type == "stock":
            return self._fetch_stock_ohlcv(ticker, tf)

        # Крипта: Bybit primary, yfinance fallback.
        try:
            df = _fetch_bybit_klines(ticker, tf, limit=bars)
            if df is not None and len(df) > 0:
                logger.info("Bybit kline %s %s: %d баров", ticker, tf, len(df))
                return df
        except Exception as exc:  # noqa: BLE001 — fallback ниже
            logger.warning("Bybit kline %s упал: %s — fallback на yfinance", ticker, exc)

        # Fallback: yfinance (BTC-USD). Используем тот же fetcher, что для акций,
        # но с yfinance-тикером крипты.
        yf_ticker = _YF_CRYPTO_TICKER.get(ticker, f"{ticker}-USD")
        logger.info("Fallback на yfinance %s для крипты %s", yf_ticker, ticker)
        return self._fetch_stock_ohlcv(yf_ticker, tf)

    def _fetch_stock_ohlcv(self, ticker: str, tf: str) -> Optional[pd.DataFrame]:
        """OHLCV через TATimeframesFetcher (yfinance) для заданного таймфрейма."""
        try:
            tfs = self.ta_fetcher.fetch(ticker)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Ошибка получения OHLCV для '{ticker}': {exc}") from exc

        df = tfs.get(tf)
        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных по таймфрейму '{tf}' для '{ticker}'.")
        return df

    def _fetch_moex_ohlcv(self, ticker: str, tf: str) -> Optional[pd.DataFrame]:
        """OHLCV через MOEXCandlesFetcher (ISS FORTS front-month) для TF.

        Фетчер грузит все 4 TF сразу (front-month резолвится один раз); здесь
        берём нужный TF. Ресемплинг 2h/4h уже выполнен внутри fetcher'а.
        Альтернативного источника OHLCV для MOEX нет — при ошибке ISS падаем
        с RuntimeError, как и при полном отказе Bybit+yfinance для крипты.
        """
        try:
            fetcher = MOEXCandlesFetcher()
            tfs = fetcher.fetch(ticker)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Ошибка получения OHLCV MOEX для '{ticker}': {exc}") from exc

        df = tfs.get(tf)
        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных по таймфрейму '{tf}' для '{ticker}'.")
        return df

    # ================================================================== #
    #  GEX-контекст (с кэшем)
    # ================================================================== #
    def _get_gex_context(
        self, ticker: str, asset_type: str, gex_days: Optional[float]
    ) -> Optional[GEXContext]:
        """Получить GEX-контекст с кэшированием на ``gex_cache_ttl`` секунд."""
        days = float(gex_days) if gex_days is not None else self.gex_days
        key = (ticker, asset_type)
        cached = self._gex_cache.get(key)
        now = time.time()
        if cached is not None and (now - cached[1]) < self.gex_cache_ttl:
            return cached[0]

        ctx = self._compute_gex_context(ticker, asset_type, days)
        self._gex_cache[key] = (ctx, now)
        return ctx

    def _compute_gex_context(
        self, ticker: str, asset_type: str, days: float
    ) -> Optional[GEXContext]:
        """Рассчитать GEX-контекст (один запрос к источнику опционов)."""
        try:
            if asset_type == "crypto":
                analysis = self.gex_service.analyze_crypto(ticker, days=days)
            elif asset_type == "moex":
                analysis = self.gex_service.analyze_moex(ticker, days=days)
            elif asset_type == "commodity":
                analysis = self.gex_service.analyze_commodity(ticker, days=days)
            else:
                analysis = self.gex_service.analyze_live(ticker, days=days)
        except (ValueError, RuntimeError) as exc:
            logger.warning("GEX-анализ для %s недоступен: %s — сигналы без GEX", ticker, exc)
            return None
        except Exception as exc:  # noqa: BLE001 — GEX не должен ронять сигналы
            logger.warning("GEX-анализ для %s упал: %s — сигналы без GEX", ticker, exc)
            return None

        return GEXContext.from_gex_analysis(analysis)

    # ================================================================== #
    #  Backtest: извлечение последних N сигналов
    # ================================================================== #
    def _extract_recent_signals(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
        gex_ctx: Optional[GEXContext],
        n_recent: int,
        trailing_pct: Optional[float] = None,
        reverse: bool = False,
        verify: bool = True,
    ) -> tuple[list[SignalRecordOut], dict]:
        """Последние N сигналов с очерёдностью по машине позиции.

        Обходит историю машиной состояний :meth:`_simulate_position_events`
        (зеркало ``strategy.evaluate()``): выход и добавление возможны ТОЛЬКО
        при открытой позиции, без позиции генерируются только входы. Сигналы
        «выход/добавление без входа» не создаются в принципе — раньше это была
        stateless-детекция краёв, из-за которой каждая пара крестов
        ``novelsrc_ema3``/``ema77`` давала «LONG EXIT»/«SHORT EXIT» без позиции.

        Returns
        -------
        (records, position)
            ``records`` — последние N сигналов, самые свежие первыми;
            ``position`` — итоговое состояние машины
            (``side`` / ``avg_price`` / ``since_idx``).
        """
        if features.empty or n_recent <= 0:
            return [], {"side": "flat", "avg_price": None, "since_idx": None}

        events, position = self._simulate_position_events(
            features, trailing_pct=trailing_pct, reverse=reverse,
        )
        records = [
            self._row_to_signal_record(features, idx, gex_ctx, order_type,
                                       reason=reason, verify=verify)
            for idx, order_type, reason in events
        ]
        # Последние N, в обратном порядке (самые свежие первыми).
        return records[-n_recent:][::-1], position

    @staticmethod
    def _simulate_position_events(
        features: pd.DataFrame,
        trailing_pct: Optional[float] = None,
        reverse: bool = False,
    ) -> tuple[list[tuple[int, str, str]], dict]:
        """Пройти бары машиной позиции — зеркало ``EMAFilterTrendStrategy.evaluate()``.

        Правила (``flat`` — как в evaluate; при позиции приоритет: разворот →
        трейлинг-стоп → добавление → логический выход):

        * ``flat``: ``long_entry_signal`` → LONG (проверяется первым), иначе
          ``short_entry_signal`` → SHORT; повторный вход при открытой позиции
          игнорируется (для доборов — отдельные колонки);
        * ``long``:
            1. ``reverse`` — СВЕЖИЙ край противоположного входа
               (``short_entry``: False→True) закрывает позицию принудительно
               (reason ``reverse_signal``) и сразу открывает SHORT;
            2. ``trailing_pct`` — трейлинг-стоп (reason ``trailing_stop``):
               пик = максимум ``high`` с бара входа включительно; выход, когда
               ``low <= пик * (1 - pct/100)``;
            3. ``combined_long_add`` (кулдаун ``_ADD_COOLDOWN_BARS``);
            4. ``long_exit_signal`` → FLAT.
        * ``short`` — зеркально (впадина по ``low``, разворот от ``long_entry``).

        Средняя цена позиции: вход — 1.0 объёма, добавления — по
        ``_ADD_QTY_FRACTION`` (как ``quantity_fraction`` в evaluate).

        Returns
        -------
        (events, position)
            ``events`` — ``(индекс_бара, order_type, reason)`` в хронологическом
            порядке; ``position`` — ``{"side", "avg_price", "since_idx"}``.
        """

        def _bool_col(name: str) -> np.ndarray:
            s = features.get(name)
            if s is None:
                return np.zeros(len(features), dtype=bool)
            return s.fillna(False).astype(bool).to_numpy()

        def _float_col(name: str) -> np.ndarray:
            s = features.get(name)
            if s is None:
                return np.full(len(features), np.nan)
            return s.to_numpy(dtype=float)

        def _edge(a: np.ndarray) -> np.ndarray:
            e = np.zeros(len(a), dtype=bool)
            e[1:] = a[1:] & ~a[:-1]
            return e

        long_entry = _bool_col("long_entry_signal")
        short_entry = _bool_col("short_entry_signal")
        long_exit = _bool_col("long_exit_signal")
        short_exit = _bool_col("short_exit_signal")
        long_add = _bool_col("combined_long_add")
        short_add = _bool_col("combined_short_add")
        high = _float_col("high")
        low = _float_col("low")
        close = _float_col("close")
        long_entry_edge = _edge(long_entry)
        short_entry_edge = _edge(short_entry)

        pct = float(trailing_pct) if trailing_pct else 0.0
        use_trail = pct > 0.0

        events: list[tuple[int, str, str]] = []
        side = "flat"
        qty = 0.0
        cost = 0.0
        since_idx: Optional[int] = None
        last_add: Optional[int] = None
        #: Экстремум цены с бара входа (пик для long / впадина для short),
        #: для трейлинг-стопа; ``None`` — стоп выключен или позиции нет.
        extreme: Optional[float] = None

        for i in range(len(features)):
            if side == "flat":
                if long_entry[i]:
                    events.append((i, "entry_long", "long_entry"))
                    side, qty, cost, since_idx, last_add = "long", 1.0, close[i], i, None
                    extreme = high[i] if use_trail else None
                elif short_entry[i]:
                    events.append((i, "entry_short", "short_entry"))
                    side, qty, cost, since_idx, last_add = "short", 1.0, close[i], i, None
                    extreme = low[i] if use_trail else None
            elif side == "long":
                # 1. Принудительный разворот по свежему противоположному входу.
                if reverse and short_entry_edge[i]:
                    events.append((i, "exit_long", "reverse_signal"))
                    events.append((i, "entry_short", "short_entry"))
                    side, qty, cost, since_idx, last_add = "short", 1.0, close[i], i, None
                    extreme = low[i] if use_trail else None
                    continue
                # 2. Трейлинг-стоп: пик включает текущий high, проверяем low.
                if use_trail:
                    extreme = max(extreme if extreme is not None else high[i], high[i])
                    if low[i] <= extreme * (1.0 - pct / 100.0):
                        events.append((i, "exit_long", "trailing_stop"))
                        side, qty, cost, since_idx, last_add = "flat", 0.0, 0.0, None, None
                        extreme = None
                        continue
                # 3. Добавление (кулдаун), 4. логический выход по колонкам.
                if long_add[i] and (last_add is None or i - last_add >= _ADD_COOLDOWN_BARS):
                    events.append((i, "add_long", "long_add"))
                    qty += _ADD_QTY_FRACTION
                    cost += _ADD_QTY_FRACTION * close[i]
                    last_add = i
                elif long_exit[i]:
                    events.append((i, "exit_long", "long_exit"))
                    side, qty, cost, since_idx, last_add = "flat", 0.0, 0.0, None, None
                    extreme = None
            else:  # short
                if reverse and long_entry_edge[i]:
                    events.append((i, "exit_short", "reverse_signal"))
                    events.append((i, "entry_long", "long_entry"))
                    side, qty, cost, since_idx, last_add = "long", 1.0, close[i], i, None
                    extreme = high[i] if use_trail else None
                    continue
                if use_trail:
                    extreme = min(extreme if extreme is not None else low[i], low[i])
                    if high[i] >= extreme * (1.0 + pct / 100.0):
                        events.append((i, "exit_short", "trailing_stop"))
                        side, qty, cost, since_idx, last_add = "flat", 0.0, 0.0, None, None
                        extreme = None
                        continue
                if short_add[i] and (last_add is None or i - last_add >= _ADD_COOLDOWN_BARS):
                    events.append((i, "add_short", "short_add"))
                    qty += _ADD_QTY_FRACTION
                    cost += _ADD_QTY_FRACTION * close[i]
                    last_add = i
                elif short_exit[i]:
                    events.append((i, "exit_short", "short_exit"))
                    side, qty, cost, since_idx, last_add = "flat", 0.0, 0.0, None, None
                    extreme = None

        position = {
            "side": side,
            "avg_price": (cost / qty) if qty > 0 else None,
            "since_idx": since_idx,
        }
        return events, position

    @staticmethod
    def _position_to_schema(position: Optional[dict], features: pd.DataFrame) -> PositionOut:
        """Словарь машины позиции → схема (``since`` — время бара входа)."""
        pos = position or {}
        side = str(pos.get("side") or "flat").lower()
        if side not in ("flat", "long", "short"):
            side = "flat"
        avg = pos.get("avg_price")
        since: Optional[datetime] = None
        since_idx = pos.get("since_idx")
        if since_idx is not None:
            try:
                since = _index_to_datetime(features.index, int(since_idx))
            except Exception:  # noqa: BLE001 — время открытия не критично
                since = None
        return PositionOut(
            side=side,
            avg_price=round(float(avg), 6) if avg is not None else None,
            since=since,
        )

    @staticmethod
    def _build_snapshot(features: pd.DataFrame) -> dict:
        """Компактный снапшот истории для переигровки машины позиции.

        Сохраняются только колонки из ``_SNAPSHOT_COLUMNS`` (OHLC + сигнальные
        + минимум для сборки записи); снапшот живёт в памяти сканера и в Redis
        не сериализуется. ``index`` — метки времени баров.
        """
        arrays: dict[str, np.ndarray] = {}
        for col in _SNAPSHOT_COLUMNS:
            s = features.get(col)
            if s is not None:
                arrays[col] = s.to_numpy()
        return {"index": features.index.to_numpy(), "arrays": arrays}

    @staticmethod
    def position_to_payload(position: Optional[dict], index: Any) -> dict:
        """Позиция машины → JSON-словарь (``side`` / ``avg_price`` / ``since``)."""
        pos = position or {}
        side = str(pos.get("side") or "flat").lower()
        if side not in ("flat", "long", "short"):
            side = "flat"
        avg = pos.get("avg_price")
        avg_val: Optional[float] = None
        try:
            avg_val = float(avg) if avg is not None else None
        except (TypeError, ValueError):
            avg_val = None
        since: Optional[str] = None
        since_idx = pos.get("since_idx")
        if since_idx is not None:
            try:
                since = _index_to_datetime(index, int(since_idx)).isoformat()
            except Exception:  # noqa: BLE001 — время открытия не критично
                since = None
        return {
            "side": side,
            "avg_price": round(avg_val, 6) if avg_val is not None else None,
            "since": since,
        }

    def _row_to_signal_record(
        self,
        features: pd.DataFrame,
        idx: int,
        gex_ctx: Optional[GEXContext],
        order_type: str,
        reason: Optional[str] = None,
        verify: bool = True,
    ) -> SignalRecordOut:
        """Превратить строку features с сигналом в SignalRecordOut.

        ``order_type`` задаёт тип: entry_long/entry_short/exit_long/exit_short/
        add_long/add_short. ``reason`` — уточнение (trailing_stop/reverse_signal);
        ``None`` → стандартный код по order_type. ``verify=False`` — не гонять
        verification-движок (переигровка снапшота: колонок недостаточно).
        """
        row = features.iloc[idx]

        # BUY: entry_long, exit_short, add_long
        # SELL: entry_short, exit_long, add_short
        if order_type in ("entry_long", "exit_short", "add_long"):
            action = SignalAction.BUY
            direction = "long"
        else:
            action = SignalAction.SELL
            direction = "short"

        reason_map = {
            "entry_long": "long_entry",
            "entry_short": "short_entry",
            "exit_long": "long_exit",
            "exit_short": "short_exit",
            "add_long": "long_add",
            "add_short": "short_add",
        }
        reason = reason or reason_map.get(order_type, order_type)
        price = float(row["close"])
        ts = _index_to_datetime(features.index, idx)

        entry_score = self.strategy.score_entry_quality(row, direction)
        atr = self.strategy._safe_float(row.get("atr"))
        tp = self.strategy.take_profit_price(price, direction, atr_value=atr)
        sl = self.strategy.trailing_stop_price(price, direction, atr_value=atr)

        # GEX-множитель на текущем профиле (для исторического бара — прокси).
        gex_mult, gex_reason = self._gex_signal_contribution(action, price, gex_ctx)

        # Verification на подвыборке до этого бара (включая).
        v_score = self._verify_at(features, idx, direction) if verify else None

        cc = self.strategy.confidence_class(
            entry_score, gex_mult, v_score, self.strategy.settings.verification_threshold
        )

        return SignalRecordOut(
            timestamp=ts,
            action=action.value,
            reason=reason,
            order_type=order_type,
            price=price,
            entry_score=round(entry_score, 3),
            gex_multiplier=round(gex_mult, 3),
            gex_reason=gex_reason,
            verification_score=round(v_score, 1) if v_score is not None else None,
            confidence_class=cc,
            atr=round(atr, 4) if atr is not None else None,
            tp_price=round(tp, 4),
            sl_price=round(sl, 4) if sl is not None else None,
        )

    # ================================================================== #
    #  Текущий сигнал
    # ================================================================== #
    def _build_current_signal(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
        gex_ctx: Optional[GEXContext],
    ) -> CurrentSignalOut:
        """Сигнал на последнем баре с полным конвейером (GEX + verification)."""
        state = None  # Анализ «с нуля»: без учёта открытой позиции.
        raw_signal = self.strategy.evaluate(features, state=state)

        latest = features.iloc[-1]
        price = float(latest["close"])

        # Направление для скоринга/TP/SL.
        if raw_signal.action == SignalAction.BUY:
            direction = "long"
        elif raw_signal.action == SignalAction.SELL:
            direction = "short"
        else:
            # HOLD — направление берём из последнего потенциального сигнала (нейтрально).
            direction = "long" if bool(latest.get("long_entry_signal", False)) else "short"

        order_type = raw_signal.metadata.get("order_type", "hold")
        reason = raw_signal.reason

        # Скор входа: только для реальных entry-сигналов; для HOLD — 0.
        entry_score = (
            self.strategy.score_entry_quality(latest, direction)
            if raw_signal.action in (SignalAction.BUY, SignalAction.SELL)
            and order_type in ("entry_long", "entry_short")
            else 0.0
        )

        atr = self.strategy._safe_float(latest.get("atr"))
        tp = self.strategy.take_profit_price(price, direction, atr_value=atr) if entry_score > 0 else None
        sl = self.strategy.trailing_stop_price(price, direction, atr_value=atr) if entry_score > 0 else None

        # GEX-множитель.
        gex_mult, gex_reason = self._gex_signal_contribution(raw_signal.action, price, gex_ctx)

        # Verification на всей истории (текущий снимок).
        v_score = None
        v_regime = None
        if raw_signal.action in (SignalAction.BUY, SignalAction.SELL):
            action_str = "buy" if raw_signal.action == SignalAction.BUY else "sell"
            v = self._verify_at(features, len(features) - 1, direction)
            v_score = v
            # Регим достаём из verification-движка отдельно (лёгкий повтор).
            v_regime = self._verification_regime(features)

        cc = self.strategy.confidence_class(
            entry_score, gex_mult, v_score, self.strategy.settings.verification_threshold
        )
        trend_coef = self.strategy._safe_float(latest.get("trend_coefficient"))

        # --- TA-контекст для подробного описания ---
        rsi_val = self.strategy._safe_float(latest.get("rsi_close"))
        ema_alignment = self._ema_alignment_label(latest)
        vwap_position = self._vwap_position_label(latest)
        trend_dir = self.strategy._safe_float(latest.get("trend_direction"))
        is_flat = bool(latest.get("is_flat_zone", False))
        spot_vs_flip = self._spot_vs_level_label(price, gex_ctx.gamma_flip if gex_ctx else None)
        # Risk/reward = (TP - entry) / (entry - SL) для long, зеркало для short.
        rr_ratio = self._rr_ratio(tp, sl, price, direction)

        return CurrentSignalOut(
            action=raw_signal.action.value,
            reason=reason,
            order_type=order_type,
            price=price,
            entry_score=round(entry_score, 3),
            gex_multiplier=round(gex_mult, 3),
            gex_reason=gex_reason,
            verification_score=round(v_score, 1) if v_score is not None else None,
            verification_regime=v_regime,
            confidence_class=cc,
            atr=round(atr, 4) if atr is not None else None,
            tp_price=round(tp, 4) if tp is not None else None,
            sl_price=round(sl, 4) if sl is not None else None,
            trend_coefficient=round(trend_coef, 3) if trend_coef is not None else None,
            rsi=round(rsi_val, 1) if rsi_val is not None else None,
            ema_alignment=ema_alignment,
            vwap_position=vwap_position,
            trend_direction=trend_dir if trend_dir is not None else None,
            is_flat=is_flat,
            spot_vs_gamma_flip=spot_vs_flip,
            rr_ratio=round(rr_ratio, 2) if rr_ratio is not None else None,
        )

    # ================================================================== #
    #  TA-метки для описания сигнала
    # ================================================================== #
    @staticmethod
    def _ema_alignment_label(row: Any) -> Optional[str]:
        """Выстраивание EMA fast/mid/slow: bullish / bearish / mixed."""
        if bool(row.get("ema_bull_alignment", False)):
            return "bullish"
        if bool(row.get("ema_bear_alignment", False)):
            return "bearish"
        return "mixed"

    @staticmethod
    def _vwap_position_label(row: Any) -> Optional[str]:
        """Позиция novelsrс относительно адаптивного VWAP: above / below / at."""
        ns = float(row.get("novelsrc", 0.0))
        vwap = float(row.get("my_vwap_state", 0.0))
        if vwap <= 0:
            return None
        # Порог «at» — в пределах 0.15% от VWAP.
        if abs(ns - vwap) / vwap < 0.0015:
            return "at"
        return "above" if ns > vwap else "below"

    @staticmethod
    def _spot_vs_level_label(spot: float, level: Optional[float]) -> Optional[str]:
        """Позиция спота относительно уровня (Gamma Flip): above / below / at."""
        if level is None or level <= 0:
            return None
        if abs(spot - level) / level < 0.0015:
            return "at"
        return "above" if spot > level else "below"

    @staticmethod
    def _rr_ratio(tp: Optional[float], sl: Optional[float], entry: float, direction: str) -> Optional[float]:
        """Risk/reward: reward/risk на основе TP/SL и направления."""
        if tp is None or sl is None or entry <= 0:
            return None
        if direction == "long":
            reward = tp - entry
            risk = entry - sl
        else:
            reward = entry - tp
            risk = sl - entry
        if risk <= 0:
            return None
        return reward / risk

    # ================================================================== #
    #  Вспомогательные: GEX-вклад, verification
    # ================================================================== #
    def _gex_signal_contribution(
        self,
        action: SignalAction,
        price: float,
        gex_ctx: Optional[GEXContext],
    ) -> tuple[float, str]:
        """GEX-множитель и причина для сигнала (обёртка над apply_gex_filter)."""
        sig = TradingSignal(
            action=action,
            reason="quality_check",
            metadata={"order_type": "entry_long" if action == SignalAction.BUY else "entry_short",
                      "close_price": price},
        )
        _, mult, reason = self.strategy.apply_gex_filter(sig, gex_ctx)
        return mult, reason

    def _verify_at(
        self, features: pd.DataFrame, idx: int, direction: str
    ) -> Optional[float]:
        """Запустить verification на подвыборке features[:idx+1].

        Возвращает score (0..100) или None при ошибке/недостатке данных.
        Берём последние ~300 баров до idx (verification нужна ≥50).
        """
        if idx < 50:
            return None
        start = max(0, idx - 300)
        sub = features.iloc[start:idx + 1].copy()
        if len(sub) < 50:
            return None

        order_type = "entry_long" if direction == "long" else "entry_short"
        action = "buy" if direction == "long" else "sell"
        try:
            result = verify_from_dataframe(
                signal_action=action,
                order_type=order_type,
                timeframe="1d",
                dataframe=sub,
            )
        except Exception as exc:  # noqa: BLE001 — verification не должна ронять анализ
            logger.debug("verification упала на idx=%d: %s", idx, exc)
            return None
        if result is None:
            return None
        return float(result.get("score"))

    def _verification_regime(self, features: pd.DataFrame) -> Optional[str]:
        """Извлечь market_regime из verification (один вызов на последнем баре)."""
        sub = features.iloc[-300:].copy() if len(features) > 300 else features.copy()
        if len(sub) < 50:
            return None
        try:
            result = verify_from_dataframe(
                signal_action="buy",
                order_type="entry_long",
                timeframe="1d",
                dataframe=sub,
            )
        except Exception:  # noqa: BLE001
            return None
        if result is None:
            return None
        breakdown = result.get("breakdown") or {}
        return breakdown.get("market_regime")

    @staticmethod
    def _gex_context_to_schema(ctx: Optional[GEXContext]) -> Optional[GEXContextOut]:
        if ctx is None:
            return None
        return GEXContextOut(
            regime=ctx.regime,
            gamma_flip=ctx.gamma_flip,
            net_gex=ctx.net_gex,
            z_score=ctx.z_score,
            call_wall=ctx.call_wall,
            put_wall=ctx.put_wall,
            direction=ctx.direction,
            confidence=ctx.confidence,
        )


# ====================================================================== #
#  Bybit V5 kline fetcher (публичный, без ключа)
# ====================================================================== #
def _fetch_bybit_klines(
    coin: str, timeframe: str, limit: int = 750
) -> Optional[pd.DataFrame]:
    """Получить OHLCV криптовалюты через публичный Bybit V5 API.

    Endpoint: ``GET https://api.bybit.com/v5/market/kline``
    Параметры: ``category=spot, symbol={coin}USDT, interval={tf}, limit={n}``.

    Возвращает DataFrame с колонками ``Open/High/Low/Close/Volume`` (совместимый
    с :class:`TATimeframesFetcher`), отсортированный по времени.

    Raises
    ------
    RuntimeError
        При сетевых ошибках или некорректном ответе API.

    Реализация — :func:`gex.adapters.providers.bybit.fetch_ohlcv` (единый транспорт,
    ретраи и таймауты вместо локального ``requests.get``).
    """
    return fetch_bybit_ohlcv(coin, timeframe, limit)


# ====================================================================== #
#  Вспомогательные функции
# ====================================================================== #
#: order_type, которым для читаемости нужен предшествующий вход в окне.
_NEEDS_ENTRY_CONTEXT: frozenset[str] = frozenset(
    {"exit_long", "exit_short", "add_long", "add_short"}
)
#: order_type-«входы» (якорь очерёдности).
_ENTRY_ORDER_TYPES: frozenset[str] = frozenset({"entry_long", "entry_short"})


def with_entry_context(all_signals: list, shown: list) -> list:
    """Дополнить список показа предшествующим входом (якорь очерёдности).

    ``all_signals`` — полный список ``recent_signals`` (новые первыми, БЕЗ
    фильтра свежести), ``shown`` — отфильтрованное по свежести окно. Если
    самый старый из ``shown`` — выход/добавление, добавляем в конец ближайший
    более старый вход (добавления пропускаем) — чтобы на странице не осталось
    «выхода без входа перед ним» из-за границы окна свежести. Если входа в
    доступной истории нет — список не меняется.
    """
    if not shown:
        return shown
    last = shown[-1]
    if getattr(last, "order_type", None) not in _NEEDS_ENTRY_CONTEXT:
        return shown
    pos: Optional[int] = None
    for n, s in enumerate(all_signals):
        if s is last:
            pos = n
            break
    if pos is None:  # запасной путь: совпадение по (timestamp, order_type)
        ts_last = getattr(last, "timestamp", None)
        ot_last = getattr(last, "order_type", None)
        for n, s in enumerate(all_signals):
            if (getattr(s, "timestamp", None) == ts_last
                    and getattr(s, "order_type", None) == ot_last):
                pos = n
                break
    if pos is None:
        return shown
    for cand in all_signals[pos + 1:]:
        ot = getattr(cand, "order_type", None)
        if ot in _ENTRY_ORDER_TYPES:
            return [*shown, cand]
        if ot not in ("add_long", "add_short"):
            break  # между выходом и входом не может быть другого сигнала
    return shown


def position_to_dict(analysis: Any) -> Optional[dict]:
    """Позиция из ответа :meth:`SignalService.analyze_signals` → JSON-словарь.

    Для сканеров (``AutoScannerService`` / ``SignalScannerService``). Устойчив
    к mock-объектам в тестах: при отсутствующем/невалидном поле возвращает
    ``None`` (не string-приведение произвольного объекта).
    """
    pos = getattr(analysis, "position", None)
    if pos is None:
        return None
    if isinstance(pos, dict):
        side = pos.get("side")
        avg = pos.get("avg_price")
        since = pos.get("since")
    else:
        side = getattr(pos, "side", None)
        avg = getattr(pos, "avg_price", None)
        since = getattr(pos, "since", None)

    side = str(side).lower() if side is not None else ""
    if side not in ("flat", "long", "short"):
        return None
    try:
        avg_val: Optional[float] = float(avg) if avg is not None else None
    except (TypeError, ValueError):
        avg_val = None
    if isinstance(since, datetime):
        since_str: Optional[str] = since.isoformat()
    elif isinstance(since, str):
        since_str = since
    else:
        since_str = None
    return {
        "side": side,
        "avg_price": round(avg_val, 6) if avg_val is not None else None,
        "since": since_str,
    }


def _index_to_datetime(index: Any, idx: int) -> datetime:
    """Безопасно извлечь datetime из индекса DataFrame по позиции."""
    try:
        val = index[idx]
        ts = pd.Timestamp(val)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()
    except Exception:  # noqa: BLE001 — fallback на «сейчас»
        return datetime.now(timezone.utc)
