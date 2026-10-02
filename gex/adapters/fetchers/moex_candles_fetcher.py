"""Загрузка OHLCV-истории MOEX через ISS для теханализа.

Параллельный (дополнительный) fetcher OHLCV — рядом с существующими
:data:`gex.ta_fetcher.TATimeframesFetcher` (yfinance, US-акции) и
:func:`gex.ohlcv_service.fetch_bybit_klines` (крипта). Питает модули
технического анализа (``gex.ta``), трендовых линий (``gex.trendlines``),
MACD-тренда (``gex.macd_trend``) и торговых сигналов
(``gex.signal_service``) для MOEX-инструментов.

Контракт строго совпадает с :class:`gex.ta_fetcher.TATimeframesFetcher`::

    fetch(ticker) -> dict[str, pd.DataFrame]
        {"1h": df, "2h": df, "4h": df, "1d": df}

каждый DataFrame имеет колонки ``Open, High, Low, Close, Volume`` и
DatetimeIndex (tz=UTC), отсортированный по времени. Поэтому движки TA
остаются asset-agnostic.

Поддерживаемые инструменты и их источники
-----------------------------------------
Разные семейства MOEX-инструментов живут на разных engine/market/board и
требуют разной резолвции тикера:

* **Валюты** (``USDRUB``, ``CNYRUB``) — бессрочные фьючерсы ``USDRUBF`` /
  ``CNYRUBF`` на FORTS (board RFUD, ``LASTTRADEDATE=2100`` — не экспирируются).
  Это **текущий курс** в рублях (~78 для USDRUB, ~11.6 для CNYRUB), с реальным
  объёмом. Альтернатива ``SiU6``/``CRU6`` (помесячные фьючерсы) давала цену в
  пунктах и дальние/неликвидные контракты — отвергнута по требованию.
* **Индексы** (``RTS``, ``MIX``) — **ближайший по экспирации** фьючерс FORTS
  (минимальная ``LASTTRADEDATE`` в будущем), напр. ``RIU6``/``MXU6``. По
  требованию — именно ближайший контракт, а не max-OI.
* **Акции** (``SBER``, ``GAZP``, ``LKOH``, …) — сам **спот** на board TQBR
  (engine=stock, market=shares), **не** фьючерс. Цена = рыночная цена акции.

ISS candles endpoint
--------------------
::

    https://iss.moex.com/iss/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json
        ?iss.meta=off&iss.only=candles&interval={code}&from={YYYY-MM-DD}&iss.reverse=true&start={offset}

ISS interval codes: ``1``=1min, ``10``=10min, ``60``=1h, ``24``=1d.
Интервалов 2h/4h/15/30/120 мин ISS **не** имеет → ресемплинг из 1h по биржевому
канону (как :class:`TATimeframesFetcher`).

ВАЖНО — пагинация ISS
~~~~~~~~~~~~~~~~~~~~~
ISS отдаёт **максимум 500 свечей на запрос** и по умолчанию — **старые первыми**.
Запрос 1h за 90 дней без ``iss.reverse`` возвращал апрель-май вместо свежих
баров (баг v1 фетчера — давал «странные» цены вроде MIX=260000 при текущей
~200000). Решение: ``iss.reverse=true`` (свежие первыми) + пагинация параметром
``start`` (0, 500, 1000, …), пока не наберётся нужная глубина или не кончатся
данные. Так последний бар всегда = сегодня, а не N*500 часов назад.

Часовые пояса
~~~~~~~~~~~~~
ISS отдаёт ``begin``/``end`` в **MSK (UTC+3)**. Переводим в UTC (−3 часа),
чтобы индекс был tz-aware UTC — консистентно с Bybit/yfinance. Критично для
ресемплинга 2h/4h: корзины строятся по единой оси UTC.

Пример::

    fetcher = MOEXCandlesFetcher()
    tfs = fetcher.fetch("RTS")       # {"1h","2h","4h","1d"} -> DataFrame
    tfs = fetcher.fetch("SBER")      # акции через TQBR
    spot = fetcher.fetch_spot("MIX") # последний Close 1h
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd

# Переиспользуем канонический список MOEX-активов из опционного fetcher'а —
# RTS/MIX/CNY/SI (для опционов). Здесь он расширяется валютами и акциями.
from gex.adapters.fetchers.moex_fetcher import _INSTRUMENTS
from gex.adapters.providers.catalog import CANONICAL_TIMEFRAMES, MOEX_ISS_INTERVAL_CODE

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Типы инструментов MOEX
# ====================================================================== #
InstrumentKind = Literal["currency_perp", "index_future", "stock"]


# ====================================================================== #
#  Конфиг инструментов: тикер → (kind, engine, market, board, secid-resolver)
# ====================================================================== #
# Каждый инструмент описан конфигом:
#   kind   — семейство (определяет стратегию резолва secid);
#   engine/market/board — где живёт candles endpoint;
#   secid  — либо фиксированный тикер (валюты/акции), либо префикс фьючерса
#            (для index_future резолвится до ближайшей экспирации динамически).
#
# Почему так:
#   * Валюты: USDRUBF/CNYRUBF — перманентные (LTD=2100), фиксированный secid,
#     цена = прямой рублёвый курс, есть реальный объём. SiU6/CRU6 (помесячные)
#     давали цену в пунктах и были отвергнуты.
#   * Индексы RTS/MIX: ближайший по экспирации фьючерс (min LASTTRADEDATE > now).
#   * Акции: сам спот (TQBR), не фьючерс.
_MOEX_INSTRUMENTS: dict[str, dict] = {
    # --- Валюты: перманентные фьючерсы FORTS (текущий курс, реальный объём) ---
    "USDRUB": {"kind": "currency_perp", "engine": "futures", "market": "forts",
               "board": "RFUD", "secid": "USDRUBF"},
    "CNYRUB": {"kind": "currency_perp", "engine": "futures", "market": "forts",
               "board": "RFUD", "secid": "CNYRUBF"},
    # --- Индексы: ближайший фьючерс (резолв по min LASTTRADEDATE в будущем) ---
    # secid здесь — префикс тикера (RI/MX); активная серия подставляется позже.
    "RTS": {"kind": "index_future", "engine": "futures", "market": "forts",
            "board": "RFUD", "prefix": "RI", "assetcode": "RTS"},
    "MIX": {"kind": "index_future", "engine": "futures", "market": "forts",
            "board": "RFUD", "prefix": "MX", "assetcode": "MIX"},
    # --- Акции: спот на TQBR (фиксированный secid = тикер) ---
    "SBER": {"kind": "stock", "engine": "stock", "market": "shares",
             "board": "TQBR", "secid": "SBER"},
    "GAZP": {"kind": "stock", "engine": "stock", "market": "shares",
             "board": "TQBR", "secid": "GAZP"},
    "LKOH": {"kind": "stock", "engine": "stock", "market": "shares",
             "board": "TQBR", "secid": "LKOH"},
    "GMKN": {"kind": "stock", "engine": "stock", "market": "shares",
             "board": "TQBR", "secid": "GMKN"},
    "ROSN": {"kind": "stock", "engine": "stock", "market": "shares",
             "board": "TQBR", "secid": "ROSN"},
    "YDEX": {"kind": "stock", "engine": "stock", "market": "shares",
             "board": "TQBR", "secid": "YDEX"},
}

# Остальные российские акции (спот TQBR) — полный список автосканера RU
# (префы SBERP/SNGSP/TATNP/TRNFP, расписки X5/OZON/HEAD/VKCO/RUAL и др.).
# Все проверены против ISS candles endpoint (board TQBR, 1d).
_MOEX_STOCKS: tuple[str, ...] = (
    "AFLT", "AFKS", "ALRS", "BSPB", "CBOM", "CHMF", "CNRU", "DOMRF",
    "ENPG", "FLOT", "HEAD", "IRAO", "LENT", "MAGN", "MDMG", "MOEX",
    "MSNG", "MTSS", "NLMK", "NVTK", "OZON", "PHOR", "PLZL", "POSI",
    "RAGR", "RENI", "RTKM", "RUAL", "SBERP", "SNGS", "SNGSP", "SVCB",
    "T", "TATN", "TATNP", "TRNFP", "UGLD", "VKCO", "VTBR", "X5",
)
_MOEX_INSTRUMENTS.update({
    code: {"kind": "stock", "engine": "stock", "market": "shares",
           "board": "TQBR", "secid": code}
    for code in _MOEX_STOCKS
})

# Алиасы: внешние коды (как их ждет GEX-конвенция) → канонический тикер.
# "SI"/"CNY" — legacy из опционного fetcher'а (SiU6/CRU6); теперь маппятся на
# валютные перпы USDRUBF/CNYRUBF по требованию. "USDRUB"/"CNYRUB" — основной ввод.
_MOEX_ALIASES: dict[str, str] = {
    "SI": "USDRUB",      # Si (USDRUB futures) → перп USDRUBF
    "CNY": "CNYRUB",     # CNY (CNYRUB futures) → перп CNYRUBF
}

# Множество всех поддерживаемых MOEX-тикеров (канонические + алиасы) для
# быстрого детекта asset_type == "moex" в сервисах. Регистронезависимо.
_MOEX_OHLCV_ASSETS: set[str] = (
    set(_MOEX_INSTRUMENTS.keys()) | set(_MOEX_ALIASES.keys())
)


# ISS candles endpoint (board явно, т.к. акции на TQBR, фьючерсы на RFUD).
_ISS_CANDLES_URL = (
    "https://iss.moex.com/iss/engines/{engine}/markets/{market}/boards/{board}"
    "/securities/{secid}/candles.json"
)
# ISS securities endpoint FORTS (для резолва ближайшего фьючерса по ASSETCODE).
_ISS_FORTS_SEC_URL = (
    "https://iss.moex.com/iss/engines/futures/markets/forts/securities.json"
    "?iss.meta=off&iss.only=securities"
)

# ISS interval code по таймфрейму (единственный источник — каталог): 2h/4h
# обслуживаются часовым запросом и ресемплингом.
_ISS_INTERVAL: dict[str, int] = dict(MOEX_ISS_INTERVAL_CODE)

# Доступные таймфреймы — канонический словарь приложения.
TIMEFRAMES: tuple[str, ...] = CANONICAL_TIMEFRAMES

# Глубина истории (в днях) по TF для параметра `from`. Достаточно для EMA200
# после ресемплинга (1h→4h): 120д × ~14ч/день ≈ 1700 часовых → ~420 баров 4h.
_PERIOD_DAYS: dict[str, int] = {
    "1h": 120,
    "1d": 365 * 3,
}

# ISS отдаёт максимум 500 свечей на запрос. Для 1h за 120 дней нужно ~6 страниц.
# Лимит страниц — страховка от бесконечного цикла при странном ответе ISS.
_ISS_PAGE_SIZE = 500
_ISS_MAX_PAGES = 20

# Канон биржевой агрегации OHLCV при ресемплинге 2h/4h (как в ta_fetcher).
_OHLCV_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}

# Срок жизни кэшей ISS, секунды (10 минут — дольше 5 мин: на нестабильном
# канале к iss.moex.com повторные запросы в пределах TTL не долбят ISS).
_ISS_CACHE_TTL_SECONDS = 10 * 60

# MSK = UTC+3. Сдвиг для перевода `begin` свечи в UTC (консистентность с Bybit).
_MSK_OFFSET = timezone(timedelta(hours=3))


class MOEXCandlesFetcher:
    """Получение OHLCV-истории MOEX через ISS (валюты/индексы/акции).

    Parameters
    ----------
    timeout : float
        Таймаут HTTP-запроса к ISS, секунды.
    max_bars : int
        Ограничение числа возвращаемых свечей на TF (берутся последние/свежие).
        ``0`` = без ограничения (всё, что отдал ISS в окне глубины истории).
    """

    # --- Кэши class-level (общие для всех экземпляров) --- #
    _cache_lock = threading.Lock()
    # Ближайший фьючерс: {assetcode -> (monotonic_ts, secid)} (только для index_future).
    _cache_frontmonth: dict[str, tuple[float, str]] = {}
    # Свечи: {(ticker, tf) -> (monotonic_ts, DataFrame)}.
    _cache_candles: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}

    def __init__(self, timeout: float = 30.0, max_bars: int = 0):
        if timeout <= 0:
            raise ValueError("timeout должен быть > 0")
        if max_bars < 0:
            raise ValueError("max_bars должен быть >= 0")
        self.timeout = float(timeout)
        self.max_bars = int(max_bars)

    # ------------------------------------------------------------------ #
    #  Главный API
    # ------------------------------------------------------------------ #
    def fetch(self, ticker: str) -> dict[str, pd.DataFrame]:
        """Polling: получить OHLCV для всех таймфреймов MOEX-инструмента.

        Parameters
        ----------
        ticker : str
            Тикер: ``RTS``, ``MIX`` (ближайший фьючерс); ``USDRUB``/``CNYRUB``
            (или алиасы ``SI``/``CNY`` — перманентные фьючерсы USDRUBF/CNYRUBF);
            ``SBER``/``GAZP``/... (акция, спот TQBR). Регистронезависимо.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{"1h": df, "2h": df, "4h": df, "1d": df}`` (1d может отсутствовать
            при пустой истории). Колонки ``Open, High, Low, Close, Volume``,
            DatetimeIndex (tz=UTC), сортировка по времени.

        Raises
        ------
        ValueError
            Инструмент не поддерживается, или ISS вернул пустую историю 1h.
        RuntimeError
            При сетевых ошибках ISS.
        """
        canonical = self._canonical_ticker(ticker)
        logger.info("MOEX candles fetch: ticker=%s -> %s", ticker, canonical)

        # --- 1. 1h (нативно из ISS, основа для ресемплинга 2h/4h) ---
        df_1h = self._fetch_tf(canonical, "1h")
        if df_1h is None or df_1h.empty:
            raise ValueError(
                f"Не удалось получить часовую историю ISS для '{ticker}'."
            )

        # --- 2. 1d (нативно) ---
        df_1d = self._fetch_tf(canonical, "1d")
        if df_1d is None or df_1d.empty:
            logger.warning("MOEX %s: пустая дневная история ISS — 1d пропущен",
                           canonical)

        # --- 3. Ресемплинг 2h и 4h из 1h ---
        df_2h = self._resample_ohlcv(df_1h, "2h")
        df_4h = self._resample_ohlcv(df_1h, "4h")
        logger.info("  1h: %d | 2h: %d | 4h: %d | 1d: %s баров",
                    len(df_1h), len(df_2h), len(df_4h),
                    len(df_1d) if df_1d is not None else 0)

        out: dict[str, pd.DataFrame] = {"1h": df_1h, "2h": df_2h, "4h": df_4h}
        if df_1d is not None and not df_1d.empty:
            out["1d"] = df_1d
        return out

    def fetch_daily(self, ticker: str) -> Optional[pd.DataFrame]:
        """Только дневные свечи (1d) для тикера — без дорогого 1h.

        Используется там, где нужна история для волатильности/ATR
        (напр. /gexcone), а не полный набор таймфреймов.
        """
        canonical = self._canonical_ticker(ticker)
        df = self._fetch_tf(canonical, "1d")
        return df.copy() if df is not None else None

    def fetch_spot(self, ticker: str) -> float:
        """Текущая цена инструмента (последний Close 1h).

        Берётся последний часовой Close — для валют это прямой рублёвый курс
        (~78 USDRUB, ~11.6 CNYRUB), для индексов — цена ближайшего фьючерса,
        для акций — рыночная цена.
        """
        canonical = self._canonical_ticker(ticker)
        df_1h = self._fetch_tf(canonical, "1h")
        if df_1h is None or df_1h.empty:
            raise ValueError(
                f"Не удалось получить спот (часовую историю) для '{ticker}'."
            )
        spot = float(df_1h["Close"].iloc[-1])
        if spot <= 0:
            raise ValueError(f"Некорректный спот ({spot}) для '{ticker}'.")
        return spot

    # ------------------------------------------------------------------ #
    #  Загрузка одного TF (с кэшированием per-(ticker,tf))
    # ------------------------------------------------------------------ #
    def _fetch_tf(self, canonical: str, tf: str) -> Optional[pd.DataFrame]:
        """Получить OHLCV одного TF (1h или 1d) с кэшированием.

        Сначала резолвится secid по типу инструмента (для index_future —
        ближайший фьючерс, кэш 5 мин), затем тянутся свечи (тоже кэш 5 мин)
        через ISS с ``iss.reverse=true`` + пагинацией — свежие N баров.
        """
        cfg = _MOEX_INSTRUMENTS[canonical]
        try:
            secid = self._resolve_secid(canonical, cfg)
        except (ValueError, RuntimeError) as exc:
            logger.warning("MOEX %s: secid не резолвится: %s", canonical, exc)
            return None

        key = (canonical, tf)
        cached = self._cache_candles.get(key)
        if cached is not None:
            cached_at, payload = cached
            if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                return payload.copy()

        with self._cache_lock:
            cached = self._cache_candles.get(key)
            if cached is not None:
                cached_at, payload = cached
                if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                    return payload.copy()
            try:
                df = self._fetch_candles_network(
                    cfg["engine"], cfg["market"], cfg["board"], secid, tf,
                )
            except RuntimeError as exc:
                logger.warning("MOEX candles %s %s упал: %s", secid, tf, exc)
                return None
            if df is None or df.empty:
                return None
            MOEXCandlesFetcher._cache_candles[key] = (time.monotonic(), df)
            return df.copy()

    # ------------------------------------------------------------------ #
    #  Резолв secid по типу инструмента
    # ------------------------------------------------------------------ #
    def _resolve_secid(self, canonical: str, cfg: dict) -> str:
        """Вернуть secid для candles endpoint.

        * currency_perp / stock — фиксированный secid из конфига.
        * index_future — ближайший по экспирации фьючерс (min LASTTRADEDATE в
          будущем среди контрактов данного ASSETCODE), с кэшем 5 мин.
        """
        if cfg["kind"] in ("currency_perp", "stock"):
            return cfg["secid"]

        # index_future — резолв ближайшего фьючерса.
        assetcode = cfg["assetcode"]
        key = assetcode.upper()
        cached = self._cache_frontmonth.get(key)
        if cached is not None:
            cached_at, secid = cached
            if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                return secid

        with self._cache_lock:
            cached = self._cache_frontmonth.get(key)
            if cached is not None:
                cached_at, secid = cached
                if time.monotonic() - cached_at < _ISS_CACHE_TTL_SECONDS:
                    return secid
            secid = self._resolve_nearest_future(assetcode)
            MOEXCandlesFetcher._cache_frontmonth[key] = (time.monotonic(), secid)
            return secid

    @staticmethod
    def _resolve_nearest_future(assetcode: str) -> str:
        """Ближайший по экспирации фьючерс FORTS (min LASTTRADEDATE > сегодня).

        По требованию — именно ближайший контракт, а не max-OI: ближний
        остаётся ликвидным до последних дней, и его цена точнее отражает
        «текущий» рынок. ISS securities.json фильтруется по ASSETCODE; среди
        контрактов с LASTTRADEDATE в будущем берётся минимальная дата.
        """
        import requests

        try:
            resp = requests.get(
                _ISS_FORTS_SEC_URL, timeout=30.0,
                headers={"User-Agent": "gex-app/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ошибка запроса securities FORTS: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Некорректный JSON securities FORTS: {exc}") from exc

        sec = data.get("securities", {})
        cols = sec.get("columns", [])
        rows = sec.get("data", [])
        idx = {c: i for i, c in enumerate(cols)}

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        asset_upper = assetcode.upper()
        # (LASTTRADEDATE, SECID) для контрактов с будущей экспирацией.
        candidates: list[tuple[str, str]] = []
        for row in rows:
            if not row:
                continue
            rec = dict(zip(cols, row))
            if str(rec.get("ASSETCODE", "")).upper() != asset_upper:
                continue
            secid = rec.get("SECID")
            ltd = rec.get("LASTTRADEDATE")
            if not secid or not ltd:
                continue
            if str(ltd) < today:
                continue  # просроченный контракт
            candidates.append((str(ltd), str(secid)))

        if not candidates:
            raise ValueError(
                f"Нет активных фьючерсов FORTS для ASSETCODE='{assetcode}'."
            )
        # Ближайшая экспирация = min LASTTRADEDATE.
        candidates.sort(key=lambda x: x[0])
        secid = candidates[0][1]
        logger.info("MOEX nearest-future %s -> %s (LTD=%s)",
                    assetcode, secid, candidates[0][0])
        return secid

    # ------------------------------------------------------------------ #
    #  Сетевой запрос свечей с пагинацией (свежие первыми)
    # ------------------------------------------------------------------ #
    def _fetch_candles_network(
        self, engine: str, market: str, board: str, secid: str, tf: str,
    ) -> Optional[pd.DataFrame]:
        """Живой запрос candles.json для (secid, tf) → DataFrame OHLCV (UTC).

        ISS отдаёт ≤500 свечей на запрос и по умолчанию старые первыми. Поэтому:
          * ``iss.reverse=true`` — свежие первыми (последний бар = сегодня);
          * пагинация параметром ``start`` (0, 500, 1000, …), пока не наберётся
            глубина ``_PERIOD_DAYS`` или не кончатся данные.
        Все страницы склеиваются, дедуплицируются по индексу, сортируются.
        Возвращает ``None`` при пустом ответе.
        """
        import requests

        interval = _ISS_INTERVAL.get(tf)
        if interval is None:
            raise ValueError(f"Нативный ISS-интервал для tf='{tf}' не определён")

        days = _PERIOD_DAYS.get(tf, 120)
        from_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        url = _ISS_CANDLES_URL.format(
            engine=engine, market=market, board=board, secid=secid,
        )

        frames: list[pd.DataFrame] = []
        for page in range(_ISS_MAX_PAGES):
            params = {
                "iss.meta": "off",
                "iss.only": "candles",
                "interval": interval,
                "from": from_str,
                "iss.reverse": "true",   # свежие первыми (критично!)
                "start": page * _ISS_PAGE_SIZE,
            }
            data = None
            # ISS с этого хоста периодически рвёт соединение (ConnectTimeout) —
            # один повтор каждой страницы заметно снижает «случайные» 404/502
            # по MOEX-акциям. Сетевая ошибка на странице >0 и так не роняет
            # запрос (частичная история), фатальна только страница 0.
            for attempt in range(2):
                try:
                    resp = requests.get(
                        url, params=params, timeout=self.timeout,
                        headers={"User-Agent": "gex-app/1.0"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except requests.RequestException as exc:
                    if attempt == 0:
                        logger.warning(
                            "MOEX candles %s %s page %d: сетевой сбой (%s) — повтор…",
                            secid, tf, page, exc,
                        )
                        time.sleep(0.6)
                        continue
                    if frames:
                        # Уже есть данные с прошлых страниц — не роняем, отдаём что есть.
                        logger.warning(
                            "MOEX candles %s %s page %d упал: %s (использую %d страниц)",
                            secid, tf, page, exc, page,
                        )
                        break
                    raise RuntimeError(
                        f"Ошибка запроса candles ISS {secid} {tf}: {exc}"
                    ) from exc
                except ValueError as exc:
                    raise RuntimeError(
                        f"Некорректный JSON candles ISS {secid} {tf}: {exc}"
                    ) from exc
            if data is None:
                break  # исчерпали повторы и уже собрали frames выше

            c = data.get("candles", {})
            rows = c.get("data", [])
            if not rows:
                break  # данных больше нет

            df_page = self._parse_candles_rows(c.get("columns", []), rows)
            if df_page is not None and not df_page.empty:
                frames.append(df_page)

            # Если страница неполная — это последняя (данных больше нет).
            if len(rows) < _ISS_PAGE_SIZE:
                break

        if not frames:
            return None

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if self.max_bars > 0 and len(df) > self.max_bars:
            df = df.iloc[-self.max_bars:]
        return df

    @staticmethod
    def _parse_candles_rows(
        cols: list[str], rows: list[list],
    ) -> Optional[pd.DataFrame]:
        """Распарсить rows candles-ответа ISS → DataFrame (tz=UTC).

        Колонки ISS: ``open, close, high, low, value, volume, begin, end``
        (порядок может варьироваться — ключаемся по именам). ``value`` на FORTS
        всегда 0, объём берём из ``volume``. ``begin`` (MSK) → tz-aware UTC.
        """
        ci = {name: i for i, name in enumerate(cols)}
        records = []
        for r in rows:
            try:
                open_ = float(r[ci["open"]])
                close = float(r[ci["close"]])
                high = float(r[ci["high"]])
                low = float(r[ci["low"]])
                volume = r[ci["volume"]]
                volume = float(volume) if volume is not None else 0.0
                begin_str = r[ci["begin"]]
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            ts = _parse_msk_to_utc(begin_str)
            if ts is None:
                continue
            records.append({
                "ts": ts,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            })
        if not records:
            return None
        return pd.DataFrame(records).set_index("ts")

    # ------------------------------------------------------------------ #
    #  Ресемплинг 2h/4h из 1h (биржевой канон)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resample_ohlcv(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Агрегировать часовые OHLCV до ``rule`` ('2h' / '4h').

        Канон как в :meth:`TATimeframesFetcher._resample_ohlcv`; пустые корзины
        (ночь/клиринг) удаляются. ``label="left"`` — метка корзины = её начало.
        """
        out = df_1h.resample(rule, label="left", closed="left").agg(_OHLCV_AGG)
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        if "Volume" in out.columns:
            out["Volume"] = out["Volume"].fillna(0.0)
        return out

    # ------------------------------------------------------------------ #
    #  Нормализация тикера
    # ------------------------------------------------------------------ #
    @staticmethod
    def _canonical_ticker(ticker: str) -> str:
        """Привести внешний тикер к каноническому ключу в _MOEX_INSTRUMENTS.

        Принимает алиасы (``SI``→``USDRUB``, ``CNY``→``CNYRUB``) и любой регистр.
        Возвращает ключ конфига (напр. ``USDRUB``, ``RTS``, ``SBER``) или
        поднимает ``ValueError`` для неподдерживаемых.
        """
        code = ticker.strip().upper()
        if code in _MOEX_ALIASES:
            code = _MOEX_ALIASES[code]
        if code in _MOEX_INSTRUMENTS:
            return code
        raise ValueError(
            f"Неподдерживаемый MOEX-инструмент '{ticker}'. "
            f"Доступны: {sorted(_MOEX_OHLCV_ASSETS)}."
        )


# ====================================================================== #
#  Вспомогательные функции
# ====================================================================== #
def _parse_msk_to_utc(begin_str: object) -> Optional[pd.Timestamp]:
    """Перевести ISS ``begin`` ('YYYY-MM-DD HH:MM:SS', MSK) → tz-aware UTC.

    Возвращает ``None`` при неразборчивой дате. MSK = UTC+3.
    """
    if not begin_str:
        return None
    try:
        ts = pd.Timestamp(str(begin_str))
    except (ValueError, TypeError):
        return None
    if not isinstance(ts, pd.Timestamp) or ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(_MSK_OFFSET)
    return ts.tz_convert("UTC")
