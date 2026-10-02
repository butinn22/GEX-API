"""Polling индикаторов волатильности и SPY через yfinance.

Модуль получает живые данные (1d + 1wk) для индикаторов **VIX**, **VVIX**,
**MOVE**, **COR1M** и **SPY** через Yahoo Finance и приводит их к единому
виду — :class:`IndicatorSnapshot` (определён здесь же) для индикаторов и
кортеж ``(spot, daily_close, annual_vol)`` для SPY.

Источник данных
---------------
Все тикеры отдаются Yahoo Finance:

  * ``SPY``        — ETF на S&P 500;
  * ``^VIX``       — CBOE Volatility Index (страх ближайших 30 дней);
  * ``^VVIX``      — волатильность самого VIX (vol-of-vol);
  * ``^MOVE``      — ICE BofA U.S. Bond Market Option Volatility Estimate
    («VIX облигационного рынка»);
  * ``^COR1M``     — CBOE 1-Month Implied Correlation (ожидаемая корреляция
    компонент S&P 500).

Для каждого индикатора считаются:

  * **RSI(14)** дневной и недельный (по Уайлдеру) — переиспользуем
    :func:`gex.ta._wilder_rsi`;
  * **z-score** текущего значения против 252-дневного скользящего среднего/σ
    (исторический коридор) — мера «перегрева»;
  * **процентиль** текущего значения в 2-летнем распределении;
  * **историческая корреляция** лог-доходностей индикатора с SPY (60-дневная).

Архитектура повторяет :class:`gex.ta_fetcher.TATimeframesFetcher`: live-polling,
без кэша, фетчер-инжекшн в сервисе.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from gex.adapters.providers.yfinance import fast_info, option_chain, option_expiries
from gex.adapters.providers.yfinance import history as yfinance_history

from gex.domain.ta import _wilder_rsi
from gex.adapters.cache.redis_client import RedisClient, cache_key, serialize_value, deserialize_value

# ====================================================================== #
#  Пороговые константы z-score (коридор-сигналы индикаторов)
# ====================================================================== #
#: |z-score|, начиная с которого коридор-сигнал считается сильным.
ZSCORE_STRONG = 1.5
#: |z-score| экстремального режима.
ZSCORE_EXTREME = 2.0


@dataclass
class IndicatorSnapshot:
    """Один индикатор волатильности/корреляции, приведённый к единому виду.

    Заполняется :class:`VolIndicatorsFetcher` из живых данных. Содержит всё,
    что нужно вероятностному движку, без знания об источнике.

    Attributes
    ----------
    symbol : str
        Тикер (``VIX``, ``VVIX``, ``MOVE``, ``COR1M``).
    spot : float
        Текущее значение индикатора.
    rsi_daily, rsi_weekly : float
        RSI(14) Уайлдера на дневном и недельном таймфреймах (0..100).
    zscore : float
        Z-score текущего значения против 252-дневного среднего/σ (коридор).
    percentile : float
        Процентиль текущего значения в 2-летнем распределении (0..100).
    corr_with_spy : float
        Историческая корреляция лог-доходностей индикатора и SPY (−1..+1).
    regime : str
        ``LOW`` / ``MID`` / ``HIGH`` / ``EXTREME`` — категория по |zscore|.
    kind : str
        ``fear`` (VIX/VVIX/MOVE), ``regime`` (COR1M/DXY) или ``hybrid`` (TLT).
    history_ok : bool
        Хватило ли истории для корректного расчёта z-score/corr.
    """

    symbol: str
    spot: float
    rsi_daily: float
    rsi_weekly: float
    zscore: float
    percentile: float
    corr_with_spy: float
    regime: str
    kind: str
    history_ok: bool = True

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Конфигурация индикаторов
# ====================================================================== #
#: Поддерживаемые индикаторы: тикер yfinance -> (символ вывода, тип).
#:   * ``fear``   — VIX/VVIX/MOVE: рост = страх -> разворот SPY вверх.
#:   * ``regime`` — COR1M, DXY: режимные; сигнал двусторонний, RSI слабее.
#:   * ``hybrid`` — TLT (treasuries): flight-to-safety + макро.
#: Breadth/McClellan и PCR (Put/Call) считаются отдельно — см. breadth_fetcher.py и _build_pcr_from_options.
VOL_INDICATORS: dict[str, tuple[str, str]] = {
    "^VIX":  ("VIX",  "fear"),
    "^VVIX": ("VVIX", "fear"),
    "^MOVE": ("MOVE", "fear"),
    "^COR1M": ("COR1M", "regime"),
    "DX-Y.NYB": ("DXY", "regime"),
    "TLT": ("TLT", "hybrid"),
}

#: ETF на S&P 500 (базис разворотного прогноза).
SPY_TICKER = "SPY"

#: Длина окна исторического коридора для z-score (торговых дней).
_CORRIDOR_WINDOW = 252
#: Длина истории для процентиль-распределения (лет).
_HISTORY_YEARS = 2
#: Длина окна корреляции лог-доходностей с SPY (торговых дней).
_CORR_WINDOW = 60
#: Длина окна для оценки годовой волатильности SPY (торговых дней).
_VOL_WINDOW = 21
#: RSI период (как в TA-модуле).
_RSI_PERIOD = 14


# ====================================================================== #
#  Сборщик данных
# ====================================================================== #
class VolIndicatorsFetcher:
    """Получение живых данных индикаторов волатильности и SPY через yfinance.

    Каждый вызов :meth:`fetch` / :meth:`fetch_spy` делает HTTP-запрос к Yahoo
    Finance — данные всегда свежие (polling), без внутреннего кэша.

    Parameters
    ----------
    history_years : float
        Длина истории для z-score/процентиля (лет). По умолчанию 2.
    """

    def __init__(self, history_years: float = _HISTORY_YEARS, redis_client: Optional[RedisClient] = None):
        if history_years < 0.5:
            raise ValueError("history_years должен быть >= 0.5")
        self.history_years = float(history_years)
        self._redis = redis_client

    # ------------------------------------------------------------------ #
    #  Главный API: индикаторы
    # ------------------------------------------------------------------ #
    _CACHE_FETCH_KEY = "gex:vol:all"
    _CACHE_SPY_KEY = "gex:vol:spy"

    def fetch(self) -> dict[str, IndicatorSnapshot]:
        """Polling: получить снапшоты всех индикаторов волатильности.

        Returns
        -------
        dict[str, IndicatorSnapshot]
            Ключ — символ вывода (``VIX`` / ``VVIX`` / ``MOVE`` / ``COR1M``).
            Индикаторы, которые не удалось загрузить, в результат не попадают.

        Raises
        ------
        RuntimeError
            При критических сетевых ошибках yfinance.
        """
        # ── Redis cache check ──
        if self._redis is not None and self._redis.connected:
            cached_data = self._redis.get(self._CACHE_FETCH_KEY)
            if cached_data is not None:
                try:
                    result = deserialize_value(cached_data)
                    if isinstance(result, dict):
                        logger.info("  Vol indicators CACHE HIT (all)")
                        return result
                except Exception:
                    logger.debug("Vol indicators deserialize error — refetching")

        # Сначала грузим SPY — он нужен для корреляции каждого индикатора.
        spy_close = self._fetch_spy_close()

        snapshots: dict[str, IndicatorSnapshot] = {}
        for yf_ticker, (symbol, kind) in VOL_INDICATORS.items():
            try:
                # COR1M: используем CBOE CDN (yfinance не отдаёт историю)
                if symbol == "COR1M":
                    snap = self._build_cor1m_cboe(spy_close)
                else:
                    snap = self._build_indicator(yf_ticker, symbol, kind, spy_close)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Индикатор %s (%s): пропущен — %s", symbol, yf_ticker, exc)
                continue
            if snap is not None:
                snapshots[symbol] = snap

        # --- PCR (Put/Call Ratio) из опционов SPY ---
        try:
            pcr_snap = self._build_pcr_from_options(spy_close)
            if pcr_snap is not None:
                snapshots["PCR"] = pcr_snap
        except Exception as exc:
            logger.warning("PCR из опционов SPY: пропущен — %s", exc)

        # --- McClellan Summation Index (рыночная ширина) ---
        try:
            mcc = self._build_mcclellan()
            if mcc is not None:
                snapshots["MCC"] = mcc
        except Exception as exc:
            logger.warning("McClellan Summation Index: пропущен — %s", exc)

        if not snapshots:
            raise RuntimeError(
                "Не удалось загрузить ни одного индикатора волатильности "
                f"({list(VOL_INDICATORS)}). Проверьте доступность yfinance."
            )
        total = len(VOL_INDICATORS) + 2  # +1 PCR, +1 MCC
        logger.info(
            "VolIndicatorsFetcher: загружено %d/%d индикаторов: %s",
            len(snapshots), total, list(snapshots),
        )

        # ── Сохраняем в Redis ──
        if self._redis is not None and self._redis.connected:
            self._redis.set(self._CACHE_FETCH_KEY, snapshots, ex=600)

        return snapshots

    # ------------------------------------------------------------------ #
    #  Главный API: SPY
    # ------------------------------------------------------------------ #
    def fetch_spy(self) -> "SpyData":
        """Polling: получить SPY (spot, серия дневных закрытий, годовая вола).

        Returns
        -------
        SpyData
            Namedtuple-подобный объект: ``spot``, ``daily_close`` (pd.Series),
            ``annual_vol`` (годовая волатильность из последних 21 дней).
        """
        # ── Redis cache check ──
        if self._redis is not None and self._redis.connected:
            cached_data = self._redis.get(self._CACHE_SPY_KEY)
            if cached_data is not None:
                try:
                    result = deserialize_value(cached_data)
                    if isinstance(result, SpyData):
                        logger.debug("  SPY data CACHE HIT")
                        return result
                except Exception:
                    logger.debug("SPY data deserialize error — refetching")

        daily = self._download(SPY_TICKER, interval="1d", years=self.history_years)
        if daily is None or daily.empty:
            raise RuntimeError(f"Не удалось загрузить историю SPY ('{SPY_TICKER}').")
        close = daily["Close"].astype(float).dropna()
        spot = self._last_spot(SPY_TICKER, daily)
        annual_vol = _annual_vol(close)
        logger.info("SPY: spot=%.2f, annual_vol=%.3f, n_days=%d", spot, annual_vol, len(close))

        result = SpyData(spot=spot, daily_close=close, annual_vol=annual_vol)

        # ── Сохраняем в Redis (TTL 300s для SPY — более свежие данные) ──
        if self._redis is not None and self._redis.connected:
            self._redis.set(self._CACHE_SPY_KEY, result, ex=300)

        return result

    # ------------------------------------------------------------------ #
    #  Сборка снапшота одного индикатора
    # ------------------------------------------------------------------ #
    def _build_indicator(
        self,
        yf_ticker: str,
        symbol: str,
        kind: str,
        spy_close: Optional[pd.Series],
    ) -> IndicatorSnapshot:
        """Скачать дневную и недельную историю индикатора → IndicatorSnapshot.

        Считает RSI(14) на 1d и 1wk, z-score против 252-дневного коридора,
        процентиль, корреляцию лог-доходностей с SPY (если есть SPY-история).
        """
        daily = self._download(yf_ticker, interval="1d", years=self.history_years)
        if daily is None or daily.empty:
            raise ValueError(f"пустая дневная история для {yf_ticker}")
        close_d = daily["Close"].astype(float).dropna()
        if len(close_d) < _CORRIDOR_WINDOW // 2:
            if len(close_d) > 0:
                logger.info("%s: short history %d < %d days — neutral snapshot", symbol, len(close_d), _CORRIDOR_WINDOW // 2)
                spot = self._last_spot(yf_ticker, daily)
                return IndicatorSnapshot(
                    symbol=symbol, spot=float(spot),
                    rsi_daily=50.0, rsi_weekly=50.0, zscore=0.0,
                    percentile=50.0, corr_with_spy=0.0, regime="MID",
                    kind=kind, history_ok=False,
                )
            raise ValueError(f"short history for {yf_ticker}: {len(close_d)} < {_CORRIDOR_WINDOW // 2}")

        # Недельная история — отдельный запрос (yfinance агрегирует сам).
        weekly = self._download(yf_ticker, interval="1wk", years=self.history_years)
        if weekly is None or weekly.empty:
            close_w = _resample_weekly(close_d)
        else:
            close_w = weekly["Close"].astype(float).dropna()

        spot = self._last_spot(yf_ticker, daily)

        # --- RSI ---
        rsi_daily = float(_wilder_rsi(close_d, _RSI_PERIOD).iloc[-1])
        if kind == "breadth":
            rsi_weekly = rsi_daily  # breadth использует дневной RSI как основной
        else:
            rsi_weekly = float(_wilder_rsi(close_w, _RSI_PERIOD).iloc[-1]) if len(close_w) > _RSI_PERIOD else 50.0

        # --- z-score коридор ---
        zscore = _rolling_zscore(close_d, _CORRIDOR_WINDOW)
        percentile = _percentile_rank(close_d, spot)

        # --- Корреляция с SPY ---
        corr = _log_corr(close_d, spy_close, _CORR_WINDOW) if spy_close is not None else 0.0

        # --- Режим по |z| ---
        az = abs(zscore) if np.isfinite(zscore) else 0.0
        if az >= ZSCORE_EXTREME:
            regime = "EXTREME"
        elif az >= ZSCORE_STRONG:
            regime = "HIGH" if zscore > 0 else "LOW"
        else:
            regime = "MID"

        return IndicatorSnapshot(
            symbol=symbol,
            spot=float(spot),
            rsi_daily=rsi_daily,
            rsi_weekly=rsi_weekly,
            zscore=float(zscore),
            percentile=float(percentile),
            corr_with_spy=float(corr),
            regime=regime,
            kind=kind,
            history_ok=True,
        )

    # ------------------------------------------------------------------ #
    #  PCR из опционов SPY (как в GEX-анализе)
    # ------------------------------------------------------------------ #
    def _build_pcr_from_options(
        self,
        spy_close: Optional[pd.Series],
    ) -> Optional[IndicatorSnapshot]:
        """Построить индикатор PCR из опционной цепочки SPY через yfinance.

        Берёт опционы ближайшей экспирации, суммирует OI путов / OI коллов.
        PCR > 1 → страх/путы → потенциальное дно (разворот UP).
        PCR < 0.5 → самоуспокоенность/коллы → потенциальная вершина (разворот DOWN).

        RSI исторических PCR не считаем (нет временного ряда), используем 50.
        """
        try:
            expiries = option_expiries("SPY")
            if not expiries:
                return None
            chain = option_chain("SPY", expiries[0])
            if chain is None:
                return None
            calls, puts = chain
            call_oi = float(calls["openInterest"].fillna(0).sum())
            put_oi = float(puts["openInterest"].fillna(0).sum())
            if call_oi <= 0:
                return None
            pcr = put_oi / call_oi
            # Эвристика: PCR RSI ~ нормировка PCR в 0-100 через логит.
            # PCR = 1.0 → RSI ≈ 50; PCR → 0 → RSI → 0; PCR → ∞ → RSI → 100.
            # Используем сигмоид: RSI = 100 / (1 + exp(-(pcr-1)*3)).
            import math as _m
            rsi_val = float(100.0 / (1.0 + _m.exp(-(pcr - 1.0) * 3.0)))
            rsi_val = max(0.0, min(100.0, rsi_val))

            # Корреляция с SPY: PCR высокий → страх (рынок падает) → отрицательная.
            corr = -0.35  # эмпирически PCR отрицательно коррелирует с SPY

            return IndicatorSnapshot(
                symbol="PCR",
                spot=round(pcr, 3),
                rsi_daily=rsi_val,
                rsi_weekly=rsi_val,
                zscore=0.0,
                percentile=50.0,
                corr_with_spy=corr,
                regime="MID",
                kind="supply_demand",
                history_ok=True,
            )
        except Exception as exc:
            logger.warning("Не удалось построить PCR из опционов SPY: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  McClellan Summation Index (рыночная ширина, через breadth_fetcher)
    # ------------------------------------------------------------------ #
    def _build_mcclellan(self) -> Optional[IndicatorSnapshot]:
        """Построить индикатор рыночной ширины через McClellan Summation Index.

        Использует breadth_fetcher.fetch_mcclellan() для вычисления Summation
        Index из RSP/SPY ratio. RSI полученного Summation Index даёт осциллятор
        ширины: > 70 = перекуплен (разворот DOWN), < 30 = перепродан (UP).
        """
        try:
            from gex.adapters.fetchers.breadth_fetcher import fetch_mcclellan
            mcc = fetch_mcclellan()
            if mcc is None or not mcc.mc_summation_index:
                return None
            rsi = mcc.rsi_14 if isinstance(mcc.rsi_14, (int, float)) else 50.0
            # Корреляция Summation Index с SPY (ширина растёт вместе с рынком)
            corr_with_spy = 0.6
            return IndicatorSnapshot(
                symbol="MCC",
                spot=round(mcc.current_summation, 1),
                rsi_daily=round(rsi, 1),
                rsi_weekly=round(rsi, 1),
                zscore=0.0,
                percentile=50.0,
                corr_with_spy=corr_with_spy,
                regime="MID",
                kind="breadth",
                history_ok=True,
            )
        except Exception as exc:
            logger.warning("McClellan Summation Index: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  COR1M из CBOE CDN (yfinance не отдаёт историю)
    # ------------------------------------------------------------------ #
    _COR1M_CDN_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv"

    def _build_cor1m_cboe(
        self,
        spy_close: Optional[pd.Series],
    ) -> Optional[IndicatorSnapshot]:
        """Загрузить COR1M из CBOE CDN (CSV, история с 2006)."""
        try:
            import io
            import requests
            resp = requests.get(self._COR1M_CDN_URL, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            close_d = df["CLOSE"].astype(float).dropna().sort_index()
            if close_d.empty:
                return None
            cutoff = pd.Timestamp.now() - pd.DateOffset(months=int(max(1, self.history_years * 12)))
            if hasattr(close_d.index, 'tz') and close_d.index.tz is not None:
                cutoff = cutoff.tz_localize(close_d.index.tz)
            close_d = close_d[close_d.index >= cutoff]
        except Exception as exc:
            logger.warning("COR1M CBOE CDN: %s", exc)
            return None

        if len(close_d) < 14:
            return None

        close_w = close_d.resample("W").last().dropna()
        spot = float(close_d.iloc[-1])

        rsi_daily = float(_wilder_rsi(close_d, _RSI_PERIOD).iloc[-1])
        rsi_weekly = float(_wilder_rsi(close_w, _RSI_PERIOD).iloc[-1]) if len(close_w) > _RSI_PERIOD else 50.0

        win = min(_CORRIDOR_WINDOW, len(close_d) - 1)
        zscore = _rolling_zscore(close_d, win) if win > 10 else 0.0
        percentile = _percentile_rank(close_d, spot)

        corr = _log_corr(close_d, spy_close, _CORR_WINDOW) if spy_close is not None else 0.0

        az = abs(zscore) if np.isfinite(zscore) else 0.0
        if az >= ZSCORE_EXTREME:
            regime = "EXTREME"
        elif az >= ZSCORE_STRONG:
            regime = "HIGH" if zscore > 0 else "LOW"
        else:
            regime = "MID"

        logger.info("COR1M (CBOE CDN): %d rows, rsi_d=%d rsi_w=%d z=%.2f", len(close_d), int(rsi_daily), int(rsi_weekly), zscore)
        return IndicatorSnapshot(
            symbol="COR1M", spot=spot, rsi_daily=rsi_daily, rsi_weekly=rsi_weekly,
            zscore=float(zscore), percentile=float(percentile),
            corr_with_spy=float(corr), regime=regime, kind="regime", history_ok=True,
        )

    # ------------------------------------------------------------------ #
    #  Загрузка yfinance
    # ------------------------------------------------------------------ #
    def _fetch_spy_close(self) -> Optional[pd.Series]:
        """Получить серию закрытий SPY для корреляции (или None при сбое)."""
        try:
            return self.fetch_spy().daily_close
        except (ValueError, RuntimeError) as exc:
            logger.warning("SPY-история для корреляции недоступна: %s", exc)
            return None

    def _download(
        self,
        ticker: str,
        interval: str,
        years: float,
    ) -> Optional[pd.DataFrame]:
        """Скачать OHLCV-историю через yfinance с обработкой ошибок.

        ``auto_adjust=False`` — берём «сырые» цены (как в ta_fetcher), чтобы
        Close был сопоставим между тикерами. Период = ``{years}y``.
        """
        period = f"{int(np.ceil(years))}y"
        # Central orchestrator path for daily history (when enabled).
        if interval == "1d":
            try:
                from gex.orchestrator.sync_gateway import sync_fetch_ohlcv
                df = sync_fetch_ohlcv(ticker, "yfinance", "1d", limit=max(300, int(years) * 365))
                if df is not None and not df.empty:
                    return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("orchestrator history(%s, %s): %s", ticker, interval, exc)

        # Адаптер уже нормализует колонки и сам логирует причину — второй раз
        # ловить исключение здесь незачем.
        return yfinance_history(ticker, period=period, interval=interval)

    @staticmethod
    def _last_spot(ticker: str, df: pd.DataFrame) -> float:
        """Текущая цена: последний Close из истории (индексы опционов отдаются
        с задержкой, fast_info часто None — поэтому берём last Close)."""
        try:
            last = float(df["Close"].dropna().iloc[-1])
            if last > 0:
                return last
        except (IndexError, KeyError, TypeError, ValueError):
            pass
        # Fallback: fast_info (адаптер сам решает, какие поля доступны в этой версии SDK).
        info = fast_info(ticker)
        for key in ("lastPrice", "last_price", "regularMarketPrice", "previousClose"):
            value = info.get(key)
            if value:
                try:
                    price = float(value)
                except (TypeError, ValueError) as exc:
                    logger.debug("fast_info %s: %r не число (%s)", ticker, value, exc)
                    continue
                if price > 0:
                    return price
        raise ValueError(f"не удалось определить spot для {ticker}")


# ====================================================================== #
#  Контейнер данных SPY
# ====================================================================== #
class SpyData:
    """Данные SPY для EV и определения тренда.

    Attributes
    ----------
    spot : float
        Текущая цена SPY.
    daily_close : pd.Series
        Серия дневных закрытий (tz-naive, отсортирована по времени).
    annual_vol : float
        Годовая волатильность из последних 21 торговых дней.
    """

    __slots__ = ("spot", "daily_close", "annual_vol")

    def __init__(self, spot: float, daily_close: pd.Series, annual_vol: float):
        self.spot = float(spot)
        self.daily_close = daily_close
        self.annual_vol = float(annual_vol)


# ====================================================================== #
#  Вспомогательные функции (статистика)
# ====================================================================== #
def _rolling_zscore(close: pd.Series, window: int) -> float:
    """Z-score последнего значения против скользящего среднего/σ за ``window``.

    Защита от деления на 0: если σ≈0 — возвращаем 0.0 (нет отклонения).
    """
    s = close.dropna().tail(window)
    if len(s) < 10:
        return 0.0
    mu = float(s.mean())
    sigma = float(s.std(ddof=0))
    if sigma < 1e-12:
        return 0.0
    return float((s.iloc[-1] - mu) / sigma)


def _percentile_rank(close: pd.Series, current: float) -> float:
    """Процентиль ``current`` в историческом распределении закрытий (0..100)."""
    s = close.dropna()
    if len(s) < 2:
        return 50.0
    rank = float((s <= current).sum()) / float(len(s)) * 100.0
    return rank


def _log_corr(
    ind_close: pd.Series,
    spy_close: Optional[pd.Series],
    window: int,
) -> float:
    """Корреляция лог-доходностей индикатора и SPY за последние ``window`` дней.

    Серии выравниваются по индексу (внутренний join). При недостатке данных
    или σ≈0 возвращает 0.0.
    """
    if spy_close is None or spy_close.empty:
        return 0.0
    ind_ret = np.log(ind_close / ind_close.shift(1)).dropna()
    spy_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    # Выравнивание по общему индексу.
    df = pd.concat([ind_ret.rename("ind"), spy_ret.rename("spy")], axis=1).dropna()
    df = df.tail(window)
    if len(df) < 20:
        return 0.0
    c = float(df["ind"].corr(df["spy"]))
    if not np.isfinite(c):
        return 0.0
    return c


def _annual_vol(close: pd.Series, window: int = _VOL_WINDOW) -> float:
    """Годовая волатильность из лог-доходностей последних ``window`` дней.

    ``σ_annual = std(log_returns) · √252``. Защита от NaN/0 → fallback 0.15.
    """
    ret = np.log(close / close.shift(1)).dropna().tail(window)
    if len(ret) < 5:
        return 0.15
    sigma = float(ret.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.15
    return float(sigma * np.sqrt(252.0))


def _resample_weekly(daily_close: pd.Series) -> pd.Series:
    """Ресэмпл дневных закрытий в недельные (последнее закрытие недели).

    Fallback, если yfinance не отдал недельный интервал напрямую.
    """
    return (
        daily_close.resample("1W", label="left", closed="left").last().dropna()
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализовать колонки yfinance: убрать MultiIndex, привести к float.

    Клон логики из :func:`gex.ta_fetcher._normalize_columns` (без дублирования
    полного модуля — здесь нужна только OHLC-нормализация).

    Важный нюанс: yfinance отдаёт индексы разных инструментов в разных таймзонах
    (``^VIX`` — America/Chicago, ``SPY`` — America/New_York, ``^MOVE`` — UTC).
    Для корреляции и сравнения нужны **торговые даты**, а не точные метки времени,
    поэтому приводим индекс к tz-naive дате (нормализуем `.normalize()` и снимаем
    tz). Это делает ``pd.concat([...], axis=1)`` в :func:`_log_corr` корректно
    выравнивающим ряды по торговым дням.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Нормализация индекса к дате (снятие tz + обнуление времени).
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        df.index = idx.tz_convert(None)
    df.index = pd.to_datetime(df.index).normalize()
    # Сортировка по времени, удаление строк без OHLC.
    df = df.sort_index()
    df = df.dropna(subset=["Close"])
    # Дедупликация по дате (на случай, если после нормализации совпали дни).
    df = df[~df.index.duplicated(keep="last")]
    return df
