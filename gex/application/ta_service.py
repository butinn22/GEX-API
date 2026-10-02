"""Сервис технического анализа: оркестрация фетчер → расчёты → консенсус → схема.

Параллелен :class:`gex.service.GEXService`, но проще: TA всегда работает на
свежих данных (live-polling), поэтому отдельного репозитория нет — каждый
вызов делает запрос к Yahoo Finance, как live-ручки GEX.

Консенсус по таймфреймам
------------------------
Старшие таймфреймы тяжелее младших (тренд на 1d важнее шума на 1h). Веса::

    1h : 0.10
    2h : 0.20
    4h : 0.30
    1d : 0.40   (нормированы к 1.0)

Консенсус-тренд определяется взвешенным голосованием направлений с учётом
силы тренда; вероятность разворота — взвешенное среднее покадровых.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from gex.domain.ta import (
    analyze_timeframe,
    build_timeframe_confirmations,
    apply_confirmation,
)
from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher, _MOEX_OHLCV_ASSETS
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher, TIMEFRAMES
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher
from gex.domain.ta_visualization import build_ta_summary, render_ta_telegram_html
from gex.schemas import TAAnalysisOut, TASummaryOut, TimeframeOut, timeframe_to_schema
from gex.adapters.cache.redis_client import RedisClient, get_redis

logger = logging.getLogger(__name__)

# Веса таймфреймов в консенсусе (старшие тяжелее).
_TF_WEIGHTS: dict[str, float] = {
    "1h": 0.10,
    "2h": 0.20,
    "4h": 0.30,
    "1d": 0.40,
}

# Горизонт прогноза разворота в барах по таймфрейму.
_HORIZON_BARS: dict[str, int] = {
    "1h": 24,   # ~1 торговый день
    "2h": 12,   # ~1 торговый день
    "4h": 6,    # ~1 торговый день
    "1d": 20,   # ~1 торговый месяц
}


class TAService:
    """Фасад технического анализа: данные → индикаторы/тренд/вероятность → схема.

    Parameters
    ----------
    fetcher : TATimeframesFetcher
        Источник OHLCV. По умолчанию создаётся стандартный.
    n_paths : int
        Число путей Монте-Карло в оценке вероятности разворота.
    """

    def __init__(
        self,
        fetcher: Optional[TATimeframesFetcher] = None,
        n_paths: int = 10_000,
        redis_client: Optional[RedisClient] = None,
    ):
        redis = redis_client or get_redis()
        if fetcher is not None:
            self.fetcher = fetcher
        else:
            self.fetcher = create_timeframes_fetcher(redis_client=redis)
        self._redis = redis
        self.n_paths = int(n_paths)

    # ------------------------------------------------------------------ #
    #  Маршрутизация фетчера по типу актива
    # ------------------------------------------------------------------ #
    def _fetch_tfs_spot(self, ticker: str) -> "tuple[dict, float]":
        """OHLCV по всем TF + спот, с маршрутизацией по типу актива.

        MOEX-фьючерсы (RTS/MIX/CNY/Si) — через :class:`MOEXCandlesFetcher`
        (ISS FORTS front-month); всё прочее — стандартный :class:`self.fetcher`
        (yfinance). Контракт возврата идентичен, поэтому движки TA не меняются.
        """
        if ticker.strip().upper() in _MOEX_OHLCV_ASSETS:
            fetcher = MOEXCandlesFetcher()
            return fetcher.fetch(ticker), fetcher.fetch_spot(ticker)
        tfs = self.fetcher.fetch(ticker)
        spot = self.fetcher.fetch_spot(ticker)
        return tfs, spot

    # ------------------------------------------------------------------ #
    #  Полный анализ по всем таймфреймам
    # ------------------------------------------------------------------ #
    def analyze(self, ticker: str, n_paths: int | None = None) -> TAAnalysisOut:
        """Polling + TA по 4 таймфреймам → TAAnalysisOut с консенсусом.

        Сначала по каждому TF считается базовый анализ (индикаторы, тренд,
        вероятность разворота). Затем строится **multi-timeframe подтверждение**:
        направление и сила тренда старшего TF уточняются младшими TF
        (:func:`gex.ta.build_timeframe_confirmations`). Консенсус считается по
        уже подтверждённым направлениям/силам.

        Parameters
        ----------
        ticker : str
        n_paths : int or None
            Число путей Монте-Карло. Если None — используется self.n_paths.

        Raises
        ------
        ValueError
            Данные не получены / тикер не найден.
        RuntimeError
            Сетевые ошибки yfinance.
        """
        ticker = ticker.strip().upper()
        _n_paths = n_paths if n_paths is not None else self.n_paths
        logger.info("TA analyze: ticker=%s", ticker)

        tfs, spot = self._fetch_tfs_spot(ticker)  # dict tf -> DataFrame + spot

        # --- 1. Базовый покадровый анализ ---
        raw_analyses = []
        for tf in TIMEFRAMES:
            df = tfs.get(tf)
            if df is None or len(df) == 0:
                continue
            raw_analyses.append(
                analyze_timeframe(
                    df,
                    timeframe=tf,
                    horizon_bars=_HORIZON_BARS.get(tf, 20),
                    n_paths=_n_paths,
                )
            )

        if not raw_analyses:
            raise ValueError(
                f"Не удалось получить данные ни по одному таймфрейму для '{ticker}'."
            )

        # --- 2. Multi-timeframe подтверждение (младшие TF → старший) ---
        confirmations = build_timeframe_confirmations(raw_analyses)
        analyses = apply_confirmation(raw_analyses, confirmations)

        # --- 3. Сериализация + консенсус по подтверждённым трендам ---
        timeframe_outs: list[TimeframeOut] = []
        dir_scores = {"BULLISH": 0.0, "BEARISH": 0.0, "RANGE": 0.0}
        p_rev_weighted = 0.0
        w_sum = 0.0
        for a in analyses:
            out = timeframe_to_schema(a)
            timeframe_outs.append(out)
            w = _TF_WEIGHTS.get(a.timeframe, 0.0)
            dir_scores[out.trend.direction] += (out.trend.strength / 100.0) * w
            p_rev_weighted += out.reversal.p_reversal * w
            w_sum += w

        # Нормируем (веса могут не покрыть все TF, если чего-то нет)
        if w_sum <= 0:
            w_sum = 1.0
        consensus_trend = max(dir_scores, key=dir_scores.get)  # noqa: E731 — имя для читаемости
        consensus_p_reversal = p_rev_weighted / w_sum

        # --- 4. Summarize-блок + Telegram HTML ---
        summary_dict = build_ta_summary(
            symbol=ticker,
            spot=spot,
            analyses=analyses,
            consensus_trend=consensus_trend,
            consensus_p_reversal=consensus_p_reversal,
            weights=_TF_WEIGHTS,
        )
        telegram_html = render_ta_telegram_html(
            symbol=ticker,
            spot=spot,
            analyses=analyses,
            consensus_trend=consensus_trend,
            consensus_p_reversal=consensus_p_reversal,
            weights=_TF_WEIGHTS,
        )
        summary = TASummaryOut(**summary_dict, telegram_html_message=telegram_html)

        return TAAnalysisOut(
            symbol=ticker,
            spot=spot,
            generated_at=datetime.now(timezone.utc),
            timeframes=timeframe_outs,
            consensus_trend=consensus_trend,
            consensus_p_reversal=consensus_p_reversal,
            weights=_TF_WEIGHTS,
            summarize=summary,
        )

    # ------------------------------------------------------------------ #
    #  Анализ одного таймфрейма
    # ------------------------------------------------------------------ #
    def analyze_timeframe(self, ticker: str, timeframe: str, n_paths: int | None = None) -> TimeframeOut:
        """Polling + TA по одному таймфрейму.

        Parameters
        ----------
        ticker : str
        timeframe : str
            Один из ``1h``/``2h``/``4h``/``1d``.
        n_paths : int or None
            Число путей Монте-Карло. Если None — используется self.n_paths.

        Raises
        ------
        ValueError
            Неверный таймфрейм или данные не получены.
        """
        ticker = ticker.strip().upper()
        _n_paths = n_paths if n_paths is not None else self.n_paths
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(
                f"Неверный таймфрейм '{timeframe}'. Доступно: {', '.join(TIMEFRAMES)}."
            )

        tfs, _spot = self._fetch_tfs_spot(ticker)
        df = tfs.get(tf)
        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных по таймфрейму '{tf}' для '{ticker}'.")

        analysis = analyze_timeframe(
            df,
            timeframe=tf,
            horizon_bars=_HORIZON_BARS.get(tf, 20),
            n_paths=_n_paths,
        )
        return timeframe_to_schema(analysis)
