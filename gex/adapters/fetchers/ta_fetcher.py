"""Загрузка истории цен для технического анализа через Yahoo Finance (yfinance).

Модуль получает OHLCV-историю для четырёх таймфреймов: ``1h``, ``2h``, ``4h``,
``1d``. Поскольку yfinance **не поддерживает** интервалы ``2h`` и ``4h``
нативно (доступны только ``1h`` и ``1d`` + кратные), промежуточные таймфреймы
получаются ресемплингом из часовых данных:

  * ``1h`` → ``yf.history(interval="1h")`` (нативно);
  * ``2h`` → ``df_1h.resample("2h")`` с OHLC-агрегацией;
  * ``4h`` → ``df_1h.resample("4h")`` с OHLC-агрегацией;
  * ``1d`` → ``yf.history(interval="1d")`` (нативно).

Кэш закрытых баров (FIFO)
-------------------------
С 2026-09-23 история больше не перекачивается на каждый вызов. Серии баров живут в
:class:`gex.adapters.cache.bar_store.BarStore` (память процесса + Redis), по ``max_bars``
(по умолчанию 500) последних баров на таймфрейм:

* повторный вызов читает бары из кэша, **не обращаясь к провайдеру**, пока не наступит
  расчётное время закрытия следующего бара (``BarStore.is_due``);
* когда новый бар закрылся, к провайдеру уходит **только хвост** — маленькое окно после
  последнего известного бара, — и новые бары дописываются в конец серии;
* при первом обращении серия строится из ``intraday_period``/``daily_period`` (бутстрап)
  и сразу ужимается до FIFO-лимита.

Агрегация OHLC корректна по канону биржевых данных::

    Open="first", High="max", Low="min", Close="last", Volume="sum"

Пустые корзины (например, ночные часы для акций) удаляются, чтобы не
порождать NaN-бары.

Пример::

    fetcher = TATimeframesFetcher()
    tfs = fetcher.fetch("AAPL")      # dict: "1h","2h","4h","1d" -> DataFrame
    df4h = tfs["4h"]
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from gex.adapters.cache.bar_store import DEFAULT_MAX_BARS, BarStore, bars_to_frame, frame_to_bars
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
from gex.adapters.cache.keys import PROVIDER_YFINANCE, ohlcv_key, spot_key
from gex.adapters.transport.yf_transport import get_shared_yf_transport
from gex.adapters.providers.catalog import CANONICAL_TIMEFRAMES, RESAMPLED_TIMEFRAMES

# yfinance ticker mapping for non-standard symbols
_YF_TICKER_MAP: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "ES": "ES=F",
    "NQ": "NQ=F",
    # Крипта: yfinance требует суффикс "-USD" (BTC → BTC-USD).
    # Список совпадает с крипто-юниверсумом приложения (Bybit spot).
    **{coin: f"{coin}-USD" for coin in _CRYPTO_ASSETS},
}

logger = logging.getLogger(__name__)


# Канон правила агрегации OHLCV при ресемплинге.
_OHLCV_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}

# Доступные таймфреймы и порядок (от младшего к старшему).
TIMEFRAMES: tuple[str, ...] = CANONICAL_TIMEFRAMES

#: Производные таймфреймы в каноническом порядке (объявлены в каталоге:
#: ``RESAMPLED_TIMEFRAMES`` — источник знания «2h/4h выводятся из 1h»).
_DERIVED_ORDER: tuple[str, ...] = tuple(tf for tf in TIMEFRAMES if tf in RESAMPLED_TIMEFRAMES)

#: Перекрытие инкрементальной догрузки (дней перед последним известным баром).
_INCREMENT_BACKOFF_DAYS = 3
_DAILY_BACKOFF_DAYS = 40

class TATimeframesFetcher:
    """Получение OHLCV-истории по четырём таймфреймам через yfinance.

    Parameters
    ----------
    intraday_period : str
        Глубина **бутстрапа** часовой серии (передаётся в
        ``yf.history(period=...)`` при первом обращении к тикеру). По умолчанию
        ``"1y"`` — хватает, чтобы построить 500 баров 4h (625 при ~2500 часовых);
        максимум yfinance для часового интервала — ``"730d"``, но он больше не
        нужен: инкрементальные догрузки добирают только новые бары.
    daily_period : str
        Глубина бутстрапа дневной серии. По умолчанию ``"2y"`` (≈500 баров).
    max_bars : int
        FIFO-лимит серии в :class:`~gex.adapters.cache.bar_store.BarStore`
        (по умолчанию 500 — достаточно всем страницам, см. docs/TA-CANDLE-FLOW.md).
    """

    def __init__(
        self,
        intraday_period: str = "1y",
        daily_period: str = "2y",
        redis_client: Optional[RedisClient] = None,
        max_bars: int = DEFAULT_MAX_BARS,
    ):
        self.intraday_period = intraday_period
        self.daily_period = daily_period
        self._redis = redis_client
        # FIFO-кэш баров (память + Redis) — единый источник серий для всех ТФ.
        self._bars = BarStore(redis_client, max_bars=max_bars)

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def fetch(self, ticker: str) -> dict[str, pd.DataFrame]:
        """OHLCV для всех таймфреймов: серии закрытых баров из FIFO-кэша + догрузка.

        Возврат прежнего контракта (``{"1h","2h","4h","1d"} -> DataFrame``), но источник
        другой: бары живут в :class:`BarStore` (500 последних на ТФ). Провайдер опрашивается
        только когда по расписанию закрылся следующий бар, и только за **хвостом** серии.

        Raises
        ------
        ValueError
            Если тикер не найден или данные пустые.
        RuntimeError
            При сетевых/внутренних ошибках yfinance.
        """
        ticker_str = ticker.strip().upper()
        yf_ticker = _YF_TICKER_MAP.get(ticker_str, ticker_str)
        logger.info("TA fetch: ticker=%s", ticker_str)

        sessions = self._sessions_for(ticker_str)
        store = self._bars

        # ── Часовая семья (1h/2h/4h) ──
        if store.is_due(ticker_str, "1h", sessions=sessions):
            hourly = self._refresh_hourly_family(ticker_str, yf_ticker)
        else:
            hourly = self._family_from_store(ticker_str)
        if "1h" not in hourly:
            raise ValueError(
                f"Не удалось получить часовую историю для '{ticker_str}'. "
                "Убедитесь, что тикер корректный."
            )

        # ── Дневная серия ──
        if store.is_due(ticker_str, "1d", sessions=sessions):
            daily = self._refresh_daily(ticker_str, yf_ticker)
        else:
            daily = self._frame_from_store(ticker_str, "1d")
        if daily is None:
            raise ValueError(f"Не удалось получить дневную историю для '{ticker_str}'.")

        result = {**hourly, "1d": daily}
        logger.info(
            "  1h: %d баров | 2h: %d | 4h: %d | 1d: %d (FIFO %d)",
            len(result["1h"]), len(result.get("2h", [])), len(result.get("4h", [])),
            len(result["1d"]), store.max_bars,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Серии баров: чтение из FIFO-кэша и инкрементальная догрузка
    # ------------------------------------------------------------------ #
    def _sessions_for(self, ticker_str: str) -> Optional[tuple[str, ...]]:
        """Торговые сессии тикера для расписания проверок (``None`` — круглосуточно).

        Крипта торгуется 24/7 (следующий бар закрывается ровно через длину ТФ); акции,
        ETF, фьючерсы и валюты аппроксимируются сессией США — вне сессии провайдер не
        опрашивается, и это ровно то, что отличает «обновляем, когда бар закрылся» от
        «дёргаем по таймеру».
        """
        return None if ticker_str in _CRYPTO_ASSETS else ("us",)

    def _frame_from_store(self, ticker_str: str, tf: str) -> Optional[pd.DataFrame]:
        entry = self._bars.get(ticker_str, tf)
        if entry is None or not entry.bars:
            return None
        return entry.to_frame()

    def _family_from_store(self, ticker_str: str) -> dict[str, pd.DataFrame]:
        """Кадры 1h/2h/4h из кэша (только если все три серии на месте)."""
        frames = {tf: self._frame_from_store(ticker_str, tf) for tf in ("1h", *_DERIVED_ORDER)}
        return {tf: df for tf, df in frames.items() if df is not None}

    def _refresh_hourly_family(self, ticker_str: str, yf_ticker: str) -> dict[str, pd.DataFrame]:
        """Догрузить часовую семью одним запросом провайдера и вернуть кадры 1h/2h/4h.

        Бутстрап (серии нет или неполная) — глубокая часовая история (``intraday_period``);
        тёплый путь — только хвост после последнего известного бара с перекрытием.
        Один запрос на всю семью: 2h/4h — ресемплинг той же часовой серии.
        """
        store = self._bars
        sessions = self._sessions_for(ticker_str)
        existing = {tf: store.get(ticker_str, tf) for tf in ("1h", *_DERIVED_ORDER)}

        if any(entry is None for entry in existing.values()):
            df = self._safe_history(yf_ticker, interval="1h", period=self.intraday_period)
        else:
            last_ts = min(entry.last_ts for entry in existing.values())
            start = datetime.fromtimestamp(last_ts, tz=timezone.utc) - timedelta(
                days=_INCREMENT_BACKOFF_DAYS
            )
            df = self._safe_history(yf_ticker, interval="1h", start=start.isoformat())

        if df is None or df.empty:
            store.mark_checked(ticker_str, "1h")
            raise ValueError(
                f"Не удалось получить часовую историю для '{ticker_str}'. "
                "Убедитесь, что тикер корректный."
            )
        df = self._normalize_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

        frames = {"1h": df}
        for tf in _DERIVED_ORDER:
            frames[tf] = self._resample_ohlcv(df, tf)

        for tf, frame in frames.items():
            store.update(
                ticker_str, tf, frame_to_bars(frame),
                source="yfinance" if tf == "1h" else "derived",
            )
            frames[tf] = bars_to_frame(store.get(ticker_str, tf).bars)  # ровно FIFO-хвост

        # Расписание следующей проверки: от последнего бара (у производных — от их хвоста).
        for tf in _DERIVED_ORDER:
            if store.get(ticker_str, tf) is None:
                store.mark_checked(ticker_str, tf)
        _ = sessions
        return frames

    def _refresh_daily(self, ticker_str: str, yf_ticker: str) -> Optional[pd.DataFrame]:
        """Догрузить дневную серию: бутстрап ``daily_period`` или хвост после последнего бара."""
        store = self._bars
        existing = store.get(ticker_str, "1d")
        if existing is None:
            df = self._safe_history(yf_ticker, interval="1d", period=self.daily_period)
        else:
            start = datetime.fromtimestamp(existing.last_ts, tz=timezone.utc) - timedelta(
                days=_DAILY_BACKOFF_DAYS
            )
            df = self._safe_history(yf_ticker, interval="1d", start=start.isoformat())

        if df is None or df.empty:
            store.mark_checked(ticker_str, "1d")
            raise ValueError(f"Не удалось получить дневную историю для '{ticker_str}'.")
        df = self._normalize_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        store.update(ticker_str, "1d", frame_to_bars(df))
        return self._frame_from_store(ticker_str, "1d")

    def fetch_timeframe(self, ticker: str, timeframe: str) -> pd.DataFrame:
        """Свечи ОДНОГО таймфрейма для графиков.

        Источники по приоритету:
          1. серия закрытых баров из FIFO-кэша (пишут ``fetch()``/этот метод —
             консистентность «график = аналитика» сохраняется: ключ один на всех);
          2. снапшоты прежней схемы ``gex:ohlcv:{T}:all``/``gex:ohlcv:{T}:{tf}``
             (переходный период — ключи истекают сами);
          3. живая догрузка по тем же правилам (бутстрап/хвост) + запись в FIFO-кэш.

        Raises
        ------
        ValueError
            Неподдерживаемый таймфрейм или нет данных.
        """
        ticker_str = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(
                f"Неподдерживаемый таймфрейм '{timeframe}'. Доступно: {list(TIMEFRAMES)}."
            )
        yf_ticker = _YF_TICKER_MAP.get(ticker_str, ticker_str)

        # 1) FIFO-кэш закрытых баров: догрузка хвоста, только если подошёл срок.
        sessions = self._sessions_for(ticker_str)
        store = self._bars
        if tf == "1d":
            df = self._refresh_daily(ticker_str, yf_ticker) if store.is_due(
                ticker_str, "1d", sessions=sessions
            ) else self._frame_from_store(ticker_str, "1d")
            if df is not None:
                return df
        else:
            if store.is_due(ticker_str, tf, sessions=sessions):
                frames = self._refresh_hourly_family(ticker_str, yf_ticker)
            else:
                frames = self._family_from_store(ticker_str)
            df = frames.get(tf)
            if df is not None:
                return df

        # 2) Переходный период: значения прежней схемы (пикл/pickle), пока не истекут.
        legacy = self._read_legacy_slice(ticker_str, tf)
        if legacy is not None:
            return legacy

        # 3) Пусто везде — догрузка по обычным правилам (бутстрап, если кэша не было).
        if tf == "1d":
            df = self._refresh_daily(ticker_str, yf_ticker)
        else:
            df = self._refresh_hourly_family(ticker_str, yf_ticker).get(tf)
        if df is None or df.empty:
            raise ValueError(f"Не удалось получить историю для '{ticker_str}' на '{tf}'.")
        return df

    def _read_legacy_slice(self, ticker_str: str, tf: str) -> Optional[pd.DataFrame]:
        """Значения прежней схемы (``gex:ohlcv:*``) — только чтение на переходный период."""
        redis = self._redis
        if redis is None or not redis.connected:
            return None
        try:
            from gex.adapters.cache.redis_client import deserialize_value
        except ImportError:
            return None
        for kind in ("all", tf):
            try:
                raw = redis.get(ohlcv_key(ticker_str, kind, provider=PROVIDER_YFINANCE))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Legacy slice GET %s %s: %s", ticker_str, kind, exc)
                continue
            if raw is None:
                continue
            try:
                value = deserialize_value(raw)
                if kind == "all" and isinstance(value, dict):
                    df = value.get(tf)
                else:
                    df = value
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    logger.debug("Legacy slice HIT %s %s", ticker_str, tf)
                    return df
            except Exception:  # noqa: BLE001 — битый переходный ключ не роняет запрос
                logger.debug("Legacy slice corrupt for %s %s — skip", ticker_str, tf)
        return None

    def fetch_spot(self, ticker: str) -> float:
        """Получить только текущую цену базиса (для enrich ответа)."""
        ticker_str = ticker.strip().upper()
        yf_ticker = _YF_TICKER_MAP.get(ticker_str, ticker_str)

        # ── Redis cache check ──
        if self._redis is not None and self._redis.connected:
            key = spot_key(ticker_str, provider=PROVIDER_YFINANCE)
            cached_data = self._redis.get(key)
            if cached_data is not None:
                try:
                    spot = float(cached_data.decode("utf-8"))
                    if spot > 0:
                        logger.debug("Spot CACHE HIT for %s: %.2f", ticker_str, spot)
                        return spot
                except (ValueError, UnicodeDecodeError, TypeError):
                    pass

        spot = self._get_spot(yf_ticker)
        if spot is None or spot <= 0:
            raise ValueError(
                f"Не удалось получить текущую цену для '{ticker_str}'."
            )

        # ── Сохраняем в Redis (Spot TTL короче — 300s) ──
        if self._redis is not None and self._redis.connected:
            key = spot_key(ticker_str, provider=PROVIDER_YFINANCE)
            self._redis.set(key, spot, ex=300)

        return spot

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы
    # ------------------------------------------------------------------ #
    def _safe_history(
        self,
        yf_ticker: str,
        interval: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Безопасный вызов ``history`` с перехватом ошибок и ограничением по времени.

        ``period`` — бутстрап («последние N дней»); ``start`` — инкрементальная догрузка
        («всё после момента X»). Одновременно задаётся не больше одного: yfinance не
        принимает timeout, поэтому вызов идёт через
        :mod:`gex.adapters.transport.yf_transport` — зависший источник не держит поток.
        """
        kwargs: dict = {"interval": interval, "auto_adjust": False}
        if period:
            kwargs["period"] = period
        if start:
            kwargs["start"] = start
        try:
            return get_shared_yf_transport().history(yf_ticker, **kwargs)
        except Exception as exc:
            logger.warning("history(%s, %s) упал: %s", interval, period or start, exc)
            return None

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Привести столбцы yfinance к каноничному виду Open/High/Low/Close/Volume.

        yfinance при ``auto_adjust=False`` отдаёт также ``Adj Close`` и
        мульти-индекс колонок (при нескольких тикерах) — убираем лишнее.
        """
        # На случай мульти-уровневых колонок берём верхний уровень
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        present = [c for c in cols if c in df.columns]
        out = df[present].copy()
        # Гарантируем числовые типы
        for c in present:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    @staticmethod
    def _resample_ohlcv(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Агрегировать часовые OHLCV до ``rule`` ('2h' / '4h').

        Использует канон биржевой агрегации; пустые корзины (ночь/выходные)
        удаляются. ``label="left"`` — временная метка корзины = её начало
        (как у биржевых баров).
        """
        out = df_1h.resample(rule, label="left", closed="left").agg(_OHLCV_AGG)
        # Удаляем корзины без данных (NaN в OHLC)
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        # Volume мог стать NaN в корзинах без объёма — заполняем нулём
        if "Volume" in out.columns:
            out["Volume"] = out["Volume"].fillna(0.0)
        return out

    @staticmethod
    def _get_spot(yf_ticker: str) -> Optional[float]:
        """Текущая цена из нескольких источников (как в YFOptionsFetcher).

        Каждый источник — отдельный сетевой вызов, и каждый ограничен дедлайном:
        ``fast_info`` обычно быстрый, ``info`` — самый медленный, поэтому он второй.
        """
        yt = get_shared_yf_transport().ticker(yf_ticker)
        try:
            spot = yt.fast_info.get("lastPrice")
            if spot is not None and spot > 0:
                return float(spot)
        except (KeyError, AttributeError, TypeError):
            pass
        except Exception as exc:
            logger.debug("fast_info(%s) недоступен: %s", yf_ticker, exc)
        try:
            info = yt.info
            if info and "regularMarketPrice" in info:
                spot = info["regularMarketPrice"]
                if spot is not None and spot > 0:
                    return float(spot)
        except (KeyError, AttributeError, TypeError):
            pass
        except Exception as exc:
            logger.debug("info(%s) недоступен: %s", yf_ticker, exc)
        try:
            hist = yt.history(period="1d")
            if hist is not None and not hist.empty:
                spot = float(hist["Close"].iloc[-1])
                if spot > 0:
                    return spot
        except Exception as exc:
            logger.debug("history(%s) для spot недоступен: %s", yf_ticker, exc)
        return None
