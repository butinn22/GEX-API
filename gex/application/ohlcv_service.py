"""Сервис получения OHLCV-свечей для отрисовки графиков на фронтенде.

Единый источник OHLCV, **идентичный** тому, что используют TA/GEX/trendline/
macd/signal-сервисы для расчётов. Это гарантирует, что график на фронте
консистентен с анализом бэкенда.

Поддерживаемые инструменты
--------------------------
* **US-акции/ETF** (SPY, AAPL, NVDA, …) — :class:`gex.ta_fetcher.TATimeframesFetcher`
  (yfinance, с ресемплингом 2h/4h из 1h);
* **Крипта** (BTC, ETH, SOL, XRP, DOGE) — публичный Bybit V5
  ``/v5/market/kline`` с авто-fallback на yfinance (``BTC-USD``);
* **MOEX-фьючерсы** (RTS/MIX/CNY/Si) — :class:`gex.moex_candles_fetcher.
  MOEXCandlesFetcher` (ISS candles FORTS front-month по OI, с ресемплингом
  2h/4h из 1h и MSK→UTC).

Канонический паттерн (детект asset_type + маршрутизация фетчера) повторяет
:class:`gex.trendline_service.TrendlineService._fetch_all`.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
from gex.adapters.providers.bybit import fetch_ohlcv as fetch_bybit_ohlcv
from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher, _MOEX_OHLCV_ASSETS
from gex.commodity_assets import COMMODITY_ASSETS
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher, TIMEFRAMES
from gex.adapters.cache.redis_client import get_redis
from gex.ports.cache_keys import PROVIDER_BYBIT, PROVIDER_YFINANCE, ohlcv_key
from gex.schemas import OHLCVBarOut, OHLCVOut

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Публичные хелперы (переиспользуются сервисами вместо приватных копий)
# ====================================================================== #
def detect_asset_type(ticker: str) -> str:
    """Определить тип актива по членству в _CRYPTO_ASSETS / _MOEX_OHLCV_ASSETS.

    Та же логика, что в trendline/macd/signal/ta-сервисах: крипта и MOEX
    детектятся по явному множеству (регистронезависимо), иначе — акция.
    """
    t = ticker.strip().upper()
    if t in _CRYPTO_ASSETS:
        return "crypto"
    if t in _MOEX_OHLCV_ASSETS:
        return "moex"
    if t in COMMODITY_ASSETS:
        return "commodity"
    return "stock"


# Карта Bybit-интервалов и сам запрос свечей вынесены в gex/adapters/providers/bybit.py:
# раньше они копировались в пять модулей и расходились.
_YF_CRYPTO_TICKER = {coin: f"{coin}-USD" for coin in _CRYPTO_ASSETS}

# yfinance ticker mapping for non-standard symbols
_YF_TICKER_MAP: dict[str, str] = {
    "DXY": "DX-Y.NYB",
}


def fetch_bybit_klines(
    coin: str, timeframe: str, limit: int = 1000
) -> Optional[pd.DataFrame]:
    """Получить OHLCV криптовалюты через публичный Bybit V5 API.

    Endpoint: ``GET https://api.bybit.com/v5/market/kline``
    Параметры: ``category=spot, symbol={coin}USDT, interval={tf}, limit={n}``.

    Возвращает DataFrame с колонками ``Open/High/Low/Close/Volume`` и
    DatetimeIndex (tz=UTC), отсортированный по времени. ``None``, если данных
    нет. Поднимает ``RuntimeError`` при сетевых ошибках (для fallback).

    Тело функции живёт в :mod:`gex.adapters.providers.bybit`; здесь оставлена
    публичная точка входа, потому что её импортирует ``orchestrator/adapters/bybit_adapter.py``.
    """
    return fetch_bybit_ohlcv(coin, timeframe, limit)


# Экземпляр фетчера акций (переиспользуется между вызовами).
_stock_fetcher = TATimeframesFetcher()
# Экземпляр фетчера MOEX-фьючерсов (переиспользуется; кэш class-level).
_moex_fetcher = MOEXCandlesFetcher()


def fetch_all_timeframes(
    ticker: str, asset_type: str
) -> tuple[dict[str, pd.DataFrame], float]:
    """Получить OHLCV по всем 4 TF + текущий спот.

    Для акций — :class:`TATimeframesFetcher` (yfinance, с ресемплингом).
    Для крипты — Bybit kline по каждому TF с fallback на yfinance.
    Для MOEX — :class:`MOEXCandlesFetcher` (ISS FORTS front-month).

    Та же логика, что ``TrendlineService._fetch_all`` (переиспользование
    канонического паттерна без дублирования кода).
    """
    if asset_type == "stock":
        yf_ticker = _YF_TICKER_MAP.get(ticker, ticker)
        tfs = _stock_fetcher.fetch(yf_ticker)
        spot = _stock_fetcher.fetch_spot(yf_ticker)
        return tfs, spot

    if asset_type == "moex":
        tfs = _moex_fetcher.fetch(ticker)
        spot = _moex_fetcher.fetch_spot(ticker)
        return tfs, spot

    if asset_type == "commodity":
        from gex.adapters.fetchers.commodity_fetcher import CommodityFetcher
        yf_sym = COMMODITY_ASSETS[ticker]["yf_symbol"]
        tfs = _stock_fetcher.fetch(yf_sym)
        spot = _stock_fetcher.fetch_spot(yf_sym)
        return tfs, spot

    # Крипта: Bybit primary, yfinance fallback.
    tfs: dict[str, pd.DataFrame] = {}
    spot: Optional[float] = None
    for tf in TIMEFRAMES:
        df = None
        try:
            df = fetch_bybit_klines(ticker, tf, limit=1000)
        except Exception as exc:  # noqa: BLE001 — fallback ниже
            logger.warning("Bybit kline %s %s упал: %s", ticker, tf, exc)
        if df is None or len(df) == 0:
            yf_ticker = _YF_CRYPTO_TICKER.get(ticker, f"{ticker}-USD")
            logger.info("Fallback на yfinance %s для крипты %s [%s]", yf_ticker, ticker, tf)
            try:
                tfs_yf = _stock_fetcher.fetch(yf_ticker)
                df = tfs_yf.get(tf)
            except Exception as exc:  # noqa: BLE001
                logger.warning("yfinance fallback %s %s упал: %s", yf_ticker, tf, exc)
                df = None
        if df is not None and len(df) > 0:
            tfs[tf] = df

    if not tfs:
        raise RuntimeError(
            f"Не удалось получить OHLCV для крипты '{ticker}' (Bybit + yfinance)."
        )

    # Спот: последняя цена закрытия старшего доступного TF.
    for tf in reversed(TIMEFRAMES):
        df = tfs.get(tf)
        if df is not None and len(df) > 0:
            spot = float(df["Close"].iloc[-1])
            break
    if spot is None or spot <= 0:
        raise ValueError(f"Не удалось определить спот для '{ticker}'.")
    return tfs, spot


def _stock_fetcher_with_cache(redis, redis_ok: bool) -> TATimeframesFetcher:
    """Фетчер с кэшем в Redis, если он доступен (иначе — без кэша).

    Живёт в модульной области, а не внутри ``fetch_ohlcv``: после вынесения
    :func:`_crypto_ohlcv` (итер. 25) вызов остался внутри неё, а определение — снаружи, и
    **крипто-ветка падала с NameError** ровно тогда, когда нужна была больше всего: Bybit
    не отдал бары, а yfinance-фолбэк вместо ответа поднимал исключение наружу (try/except
    там нет). Redis передаётся параметром — вне ``fetch_ohlcv`` замыкания уже нет.
    """
    return TATimeframesFetcher(redis_client=redis if redis_ok else None)


def _crypto_ohlcv(ticker: str, tf: str, redis, redis_ok: bool) -> Optional[pd.DataFrame]:
    """Свечи крипты: срез-кэш по провайдерам → Bybit → yfinance-fallback.

    Вынесено из :func:`fetch_ohlcv` при итер. 25: логика про **происхождение** данных,
    и читать её удобнее отдельно от разбора ответа.

    Ключ среза строится по провайдеру. До итер. 25 Bybit-свечи и свечи из
    yfinance-fallback писались в ОДИН ключ, поэтому читатель получал бары неизвестного
    происхождения (у площадок разные границы интервалов и объёмы), а первый записавший
    источник определял ответ для всех последующих. Теперь запись идёт в ключ того
    источника, который реально отдал бары.
    """
    df: Optional[pd.DataFrame] = None
    source: Optional[str] = None

    if redis_ok:
        for prov in (PROVIDER_BYBIT, PROVIDER_YFINANCE):
            raw = redis.get(ohlcv_key(ticker, tf, provider=prov))
            if raw is None:
                continue
            try:
                from gex.adapters.cache.redis_client import deserialize_value
                cand = deserialize_value(raw)
            except Exception as exc:  # noqa: BLE001 — битый срез одного источника
                # Не глушим: битый кэш одного провайдера — повод проверить второй,
                # но в логе причина остаётся (иначе деградация кэша невидима).
                logger.warning("OHLCV slice %s %s ← %s не читается: %s", ticker, tf, prov, exc)
                continue
            if isinstance(cand, pd.DataFrame) and len(cand) > 0:
                logger.debug("OHLCV slice CACHE HIT %s %s ← %s", ticker, tf, prov)
                return cand

    source = PROVIDER_BYBIT
    try:
        df = fetch_bybit_klines(ticker, tf, limit=1000)
    except Exception as exc:  # noqa: BLE001 — fallback ниже
        logger.warning("Bybit kline %s %s упал: %s", ticker, tf, exc)
        df = None
    if df is None or len(df) == 0:
        yf_ticker = _YF_CRYPTO_TICKER.get(ticker, f"{ticker}-USD")
        logger.info("Fallback на yfinance %s для крипты %s [%s]", yf_ticker, ticker, tf)
        df = _stock_fetcher_with_cache(redis, redis_ok).fetch_timeframe(yf_ticker, tf)
        source = PROVIDER_YFINANCE

    # Пишем срез — повторные запросы графика не ходят в live-API.
    if redis_ok and df is not None and len(df) > 0:
        try:
            redis.set(ohlcv_key(ticker, tf, provider=source), df, ex=300)
        except Exception:
            pass
    return df


# ====================================================================== #
#  Главный публичный метод: один таймфрейм → OHLCVOut (для ручки /ohlcv)
# ====================================================================== #
def fetch_ohlcv(ticker: str, timeframe: str, limit: int = 200) -> OHLCVOut:
    """Получить OHLCV по ОДНОМУ таймфрейму и собрать API-схему.

    Phase-A (2026-09-03): графический путь больше НЕ грузит все 4 таймфрейма
    и не ходит в live-API на каждый запрос:

      * акции/commodity — :meth:`TATimeframesFetcher.fetch_timeframe`:
        сначала канонический снапшот ``gex:ohlcv:{T}:all`` (пишут TA-сервисы
        и prefetch — консистентность «график = аналитика»), затем срез
        ``gex:ohlcv:{T}:{tf}``, и только при промахе — минимальный live-fetch
        одного сырого интервала (+ресемплинг 2h/4h из 1h);
      * крипта — Bybit V5 одним интервалом (без цикла по всем TF) + срез-кэш
        в Redis (TTL 300), fallback на yfinance;
      * MOEX — как раньше (MOEXCandlesFetcher с class-level кэшем).

    Parameters
    ----------
    ticker : str
        Тикер (US-акция/ETF, крипта из _CRYPTO_ASSETS или MOEX/commodity).
    timeframe : str
        Один из ``1h, 2h, 4h, 1d``.
    limit : int
        Сколько последних свечей вернуть (обрезается с конца). 1..1000.

    Returns
    -------
    OHLCVOut
        Свечи для графика + метаданные (asset_type, spot).

    Raises
    ------
    ValueError
        Неподдерживаемый таймфрейм или тикер не найден.
    RuntimeError
        Сетевые ошибки источника данных.
    """
    ticker_clean = ticker.strip().upper()
    tf = timeframe.strip().lower()
    if tf not in TIMEFRAMES:
        raise ValueError(
            f"Неподдерживаемый таймфрейм '{timeframe}'. Доступно: {list(TIMEFRAMES)}."
        )

    asset_type = detect_asset_type(ticker_clean)
    limit = max(1, min(int(limit), 1000))
    redis = get_redis()
    redis_ok = redis is not None and redis.connected


    df: Optional[pd.DataFrame] = None

    if asset_type == "stock":
        df = _stock_fetcher_with_cache(redis, redis_ok).fetch_timeframe(ticker_clean, tf)
    elif asset_type == "commodity":
        yf_sym = COMMODITY_ASSETS[ticker_clean]["yf_symbol"]
        df = _stock_fetcher_with_cache(redis, redis_ok).fetch_timeframe(yf_sym, tf)
    elif asset_type == "moex":
        # MOEX-фетчер держит class-level кэш — поведение не меняем.
        tfs = _moex_fetcher.fetch(ticker_clean)
        df = tfs.get(tf)
    else:
        df = _crypto_ohlcv(ticker_clean, tf, redis, redis_ok)

    if df is None or len(df) == 0:
        raise RuntimeError(
            f"Нет OHLCV для '{ticker_clean}' на таймфрейме '{tf}'."
        )

    # Обрезаем до последних `limit` свечей (свежие).
    if len(df) > limit:
        df = df.iloc[-limit:]

    bars = []
    for ts, row in df.iterrows():
        # ts — tz-aware Timestamp; ISO-строка для JSON.
        try:
            iso = ts.isoformat()
        except Exception:  # noqa: BLE001
            iso = str(ts)
        bars.append(OHLCVBarOut(
            t=iso,
            o=float(row["Open"]),
            h=float(row["High"]),
            l=float(row["Low"]),
            c=float(row["Close"]),
            v=float(row.get("Volume", 0.0) or 0.0),
        ))

    # Спот из последнего Close запрошенного TF (точнее для графика, чем общий).
    last_close = float(df["Close"].iloc[-1])
    spot = last_close if last_close > 0 else 0.0

    return OHLCVOut(
        symbol=ticker_clean,
        asset_type=asset_type,  # type: ignore[arg-type]
        timeframe=tf,
        spot=spot,
        bars=bars,
    )
