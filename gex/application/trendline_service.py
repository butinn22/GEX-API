"""Сервис анализа трендовых линий: фетчер → линии/фракталы → консенсус → схема.

Параллелен :class:`gex.ta_service.TAService`, но использует движок трендовых
линий из :mod:`gex.trendlines` (порт Pinescript v5 «Trend lines Andreu»).

Поддерживаемые инструменты
--------------------------
* **US-акции/ETF** (SPY, AAPL, NVDA, …) — OHLCV через
  :class:`gex.ta_fetcher.TATimeframesFetcher` (yfinance, уже с ресемплингом
  2h/4h из 1h);
* **Крипта** (BTC, ETH, SOL, XRP, DOGE) — OHLCV через публичный Bybit V5
  ``/v5/market/kline`` (консистентно с GEX-источником), с авто-fallback на
  yfinance (``BTC-USD``);
* **MOEX-фьючерсы** (RTS/MIX/CNY/Si) — OHLCV через
  :class:`gex.moex_candles_fetcher.MOEXCandlesFetcher` (ISS candles FORTS
  front-month по OI, с ресемплингом 2h/4h из 1h).

Консенсус по таймфреймам
------------------------
Старшие таймфреймы тяжелее младших (тренд на 1d важнее шума на 1h). Веса::

    1h : 0.10
    2h : 0.20
    4h : 0.30
    1d : 0.40   (нормированы к 1.0)

Консенсус-тренд считается взвешенным голосованием объединённых
(``combined_trend``) направлений по каждому TF с учётом силы.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
from gex.adapters.providers.bybit import fetch_ohlcv as fetch_bybit_ohlcv
from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher, _MOEX_OHLCV_ASSETS
from gex.domain.trendlines import analyze_trendlines, TrendlineAnalysis
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher, TIMEFRAMES
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher
from gex.schemas import TrendlineAnalysisOut, trendline_to_schema

logger = logging.getLogger(__name__)

# Веса таймфреймов в консенсусе (старшие тяжелее) — как в ta_service.
_TF_WEIGHTS: dict[str, float] = {
    "1h": 0.10,
    "2h": 0.20,
    "4h": 0.30,
    "1d": 0.40,
}

# Карта Bybit-интервалов и запрос свечей — gex/adapters/providers/bybit.py.
_YF_CRYPTO_TICKER = {coin: f"{coin}-USD" for coin in _CRYPTO_ASSETS}


class TrendlineService:
    """Фасад анализа трендовых линий: данные → линии/фракталы → консенсус → схема.

    Parameters
    ----------
    fetcher : TATimeframesFetcher
        Источник OHLCV для акций (yfinance). По умолчанию создаётся стандартный.
    resolution : int
        Окно поиска экстремумов (Pine ``x1``), по умолчанию 6.
    history_bars : int
        Глубина истории для построения линий (Pine ``history_bars``), 300.
    max_support_lines, max_resistance_lines : int
        Лимит линий каждого типа, 5/5 (как в оригинальном индикаторе).
    pivot_left, pivot_right : int
        Окно фрактальной логики HH/HL/LH/LL, 5/5.
    """

    def __init__(
        self,
        fetcher: Optional[TATimeframesFetcher] = None,
        resolution: int = 6,
        history_bars: int = 300,
        max_support_lines: int = 5,
        max_resistance_lines: int = 5,
        pivot_left: int = 5,
        pivot_right: int = 5,
    ):
        self.fetcher = fetcher or create_timeframes_fetcher()
        self.resolution = int(resolution)
        self.history_bars = int(history_bars)
        self.max_support_lines = int(max_support_lines)
        self.max_resistance_lines = int(max_resistance_lines)
        self.pivot_left = int(pivot_left)
        self.pivot_right = int(pivot_right)

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        ticker: str,
        resolution: int | None = None,
        history_bars: int | None = None,
        max_support_lines: int | None = None,
        max_resistance_lines: int | None = None,
        pivot_left: int | None = None,
        pivot_right: int | None = None,
    ) -> TrendlineAnalysisOut:
        """Polling + анализ трендовых линий по 4 таймфреймам → TrendlineAnalysisOut.

        Авто-определение типа актива: крипта (BTC/ETH/SOL/XRP/DOGE) → Bybit,
        иначе — акция через yfinance.

        Parameters
        ----------
        ticker : str
        resolution, history_bars, max_support_lines, max_resistance_lines,
        pivot_left, pivot_right : int or None
            Параметры анализа. Если None — используются self-значения по умолчанию.

        Raises
        ------
        ValueError
            Данные не получены / тикер не найден / слишком короткая история.
        RuntimeError
            Сетевые ошибки источника OHLCV.
        """
        _res = resolution if resolution is not None else self.resolution
        _hist = history_bars if history_bars is not None else self.history_bars
        _msl = max_support_lines if max_support_lines is not None else self.max_support_lines
        _mrl = max_resistance_lines if max_resistance_lines is not None else self.max_resistance_lines
        _pl = pivot_left if pivot_left is not None else self.pivot_left
        _pr = pivot_right if pivot_right is not None else self.pivot_right

        ticker = ticker.strip().upper()
        asset_type = self._detect_asset_type(ticker)
        logger.info("Trendline analyze: ticker=%s, asset_type=%s", ticker, asset_type)

        # --- 1. OHLCV по всем таймфреймам ---
        tfs, spot = self._fetch_all(ticker, asset_type)

        # --- 2. Анализ по каждому TF ---
        analyses: list[TrendlineAnalysis] = []
        for tf in TIMEFRAMES:
            df = tfs.get(tf)
            if df is None or len(df) == 0:
                continue
            try:
                a = analyze_trendlines(
                    df,
                    timeframe=tf,
                    resolution=_res,
                    history_bars=_hist,
                    max_support_lines=_msl,
                    max_resistance_lines=_mrl,
                    pivot_left=_pl,
                    pivot_right=_pr,
                )
                analyses.append(a)
            except ValueError as exc:
                logger.warning("  %s: слишком короткая история (%s)", tf, exc)

        if not analyses:
            raise ValueError(
                f"Не удалось проанализировать ни один таймфрейм для '{ticker}' "
                "(недостаточно баров)."
            )

        # --- 3. Консенсус ---
        consensus_trend = self._consensus(analyses)

        # --- 4. Summarize-блок + HTML ---
        summary_dict = _build_trendline_summary(
            symbol=ticker,
            spot=spot,
            analyses=analyses,
            consensus_trend=consensus_trend,
            weights=_TF_WEIGHTS,
        )

        # --- 5. Сериализация ---
        out = trendline_to_schema(
            analyses,
            symbol=ticker,
            asset_type=asset_type,
            spot=spot,
            generated_at=datetime.now(timezone.utc),
            consensus_trend=consensus_trend,
            weights=_TF_WEIGHTS,
            summarize=summary_dict,
        )
        return out

    def analyze_timeframe(
        self,
        ticker: str,
        timeframe: str,
        resolution: int | None = None,
        history_bars: int | None = None,
        max_support_lines: int | None = None,
        max_resistance_lines: int | None = None,
        pivot_left: int | None = None,
        pivot_right: int | None = None,
    ) -> TrendlineAnalysisOut:
        """Анализ трендовых линий по **одному** таймфрейму (1h/2h/4h/1d).

        Лёгкий эндпоинт — без сводного консенсуса (возвращается один TF).

        Parameters
        ----------
        ticker : str
        timeframe : str
        resolution, history_bars, max_support_lines, max_resistance_lines,
        pivot_left, pivot_right : int or None
            Параметры анализа. Если None — используются self-значения.
        """
        _res = resolution if resolution is not None else self.resolution
        _hist = history_bars if history_bars is not None else self.history_bars
        _msl = max_support_lines if max_support_lines is not None else self.max_support_lines
        _mrl = max_resistance_lines if max_resistance_lines is not None else self.max_resistance_lines
        _pl = pivot_left if pivot_left is not None else self.pivot_left
        _pr = pivot_right if pivot_right is not None else self.pivot_right

        ticker = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(
                f"Неверный таймфрейм '{timeframe}'. Доступно: {', '.join(TIMEFRAMES)}."
            )
        asset_type = self._detect_asset_type(ticker)
        tfs, spot = self._fetch_all(ticker, asset_type)
        df = tfs.get(tf)
        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных по таймфрейму '{tf}' для '{ticker}'.")

        a = analyze_trendlines(
            df,
            timeframe=tf,
            resolution=_res,
            history_bars=_hist,
            max_support_lines=_msl,
            max_resistance_lines=_mrl,
            pivot_left=_pl,
            pivot_right=_pr,
        )
        summary_dict = _build_trendline_summary(
            symbol=ticker, spot=spot, analyses=[a],
            consensus_trend=a.combined_trend, weights={tf: 1.0},
        )
        return trendline_to_schema(
            [a],
            symbol=ticker,
            asset_type=asset_type,
            spot=spot,
            generated_at=datetime.now(timezone.utc),
            consensus_trend=a.combined_trend,
            weights={tf: 1.0},
            summarize=summary_dict,
        )

    # ------------------------------------------------------------------ #
    #  Фетч OHLCV (акции + крипта)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_asset_type(ticker: str) -> str:
        if ticker in _MOEX_OHLCV_ASSETS:
            return "moex"
        if ticker in _CRYPTO_ASSETS:
            return "crypto"
        from gex.commodity_assets import COMMODITY_ASSETS
        if ticker in COMMODITY_ASSETS:
            return "commodity"
        return "stock"

    def _fetch_all(
        self, ticker: str, asset_type: str
    ) -> tuple[dict[str, pd.DataFrame], float]:
        """Получить OHLCV по всем TF + текущий спот.

        Для акций — :class:`TATimeframesFetcher` (yfinance, с ресемплингом).
        Для крипты — Bybit kline по каждому TF с fallback на yfinance.
        Для MOEX — :class:`MOEXCandlesFetcher` (ISS FORTS front-month).
        """
        if asset_type == "stock":
            tfs = self.fetcher.fetch(ticker)
            spot = self.fetcher.fetch_spot(ticker)
            return tfs, spot

        if asset_type == "moex":
            fetcher = MOEXCandlesFetcher()
            tfs = fetcher.fetch(ticker)
            spot = fetcher.fetch_spot(ticker)
            return tfs, spot

        if asset_type == "commodity":
            from gex.commodity_assets import COMMODITY_ASSETS
            yf_sym = COMMODITY_ASSETS[ticker]["yf_symbol"]
            tfs = self.fetcher.fetch(yf_sym)
            spot = self.fetcher.fetch_spot(yf_sym)
            return tfs, spot

        # Крипта: Bybit primary, yfinance fallback.
        tfs: dict[str, pd.DataFrame] = {}
        spot: Optional[float] = None
        for tf in TIMEFRAMES:
            df = None
            try:
                df = _fetch_bybit_klines(ticker, tf, limit=1000)
            except Exception as exc:  # noqa: BLE001 — fallback ниже
                logger.warning("Bybit kline %s %s упал: %s", ticker, tf, exc)
            if df is None or len(df) == 0:
                # Fallback на yfinance.
                yf_ticker = _YF_CRYPTO_TICKER.get(ticker, f"{ticker}-USD")
                logger.info("Fallback на yfinance %s для крипты %s [%s]", yf_ticker, ticker, tf)
                try:
                    tfs_yf = self.fetcher.fetch(yf_ticker)
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

    # ------------------------------------------------------------------ #
    #  Консенсус по таймфреймам
    # ------------------------------------------------------------------ #
    @staticmethod
    def _consensus(analyses: list[TrendlineAnalysis]) -> str:
        """Взвешенное голосование по ``combined_trend`` каждого TF."""
        scores = {"BULLISH": 0.0, "BEARISH": 0.0, "RANGE": 0.0}
        for a in analyses:
            w = _TF_WEIGHTS.get(a.timeframe, 0.0)
            scores[a.combined_trend] += (a.combined_strength / 100.0) * w
        # RANGE не должен «побеждать» за счёт веса — берём max из всех.
        return max(scores, key=scores.get)


# ====================================================================== #
#  Bybit V5 kline fetcher (публичный, без ключа) — приватная копия
# ====================================================================== #
def _fetch_bybit_klines(
    coin: str, timeframe: str, limit: int = 1000
) -> Optional[pd.DataFrame]:
    """Получить OHLCV криптовалюты через публичный Bybit V5 API.

    Endpoint: ``GET https://api.bybit.com/v5/market/kline``
    Параметры: ``category=spot, symbol={coin}USDT, interval={tf}, limit={n}``.

    Возвращает DataFrame с колонками ``Open/High/Low/Close/Volume``,
    отсортированный по времени. Поднимает ``RuntimeError`` при сетевых ошибках
    или некорректном ответе (чтобы вызывающий код мог сделать fallback).

    Реализация — :func:`gex.adapters.providers.bybit.fetch_ohlcv`.
    """
    return fetch_bybit_ohlcv(coin, timeframe, limit)


# ====================================================================== #
#  Summarize-блок (для JSON + Telegram HTML)
# ====================================================================== #
_TREND_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "RANGE": "🟡"}
_TREND_LABEL_RU = {
    "BULLISH": "Восходящий",
    "BEARISH": "Нисходящий",
    "RANGE": "Боковик",
}


def _build_trendline_summary(
    *,
    symbol: str,
    spot: float,
    analyses: list[TrendlineAnalysis],
    consensus_trend: str,
    weights: dict[str, float],
) -> dict:
    """Собрать summarize-блок + HTML-сообщение для Telegram.

    Возвращает словарь с полями :class:`TrendlineSummaryOut` + готовым
    ``telegram_html_message``.
    """
    by_tf: dict[str, dict] = {}
    for a in analyses:
        by_tf[a.timeframe] = {
            "line_trend": a.trend_direction,
            "line_strength": round(a.trend_strength, 1),
            "line_angle_deg": round(a.line_angle_deg, 2),
            "fractal_trend": a.fractal_trend,
            "fractal_strength": round(a.fractal_strength, 1),
            "combined_trend": a.combined_trend,
            "combined_strength": round(a.combined_strength, 1),
            "n_support": len(a.support_lines),
            "n_resistance": len(a.resistance_lines),
        }

    # Ближайшие уровни по старшему доступному TF (обычно 1d).
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    fractal_summary: dict = {}
    for a in reversed(analyses):  # старший TF последним в списке TIMEFRAMES
        if a.strongest_support is not None:
            nearest_support = a.strongest_support.current_price
        if a.strongest_resistance is not None:
            nearest_resistance = a.strongest_resistance.current_price
        fractal_summary = {
            "timeframe": a.timeframe,
            "higher_highs": a.fractals.higher_highs,
            "higher_lows": a.fractals.higher_lows,
            "lower_highs": a.fractals.lower_highs,
            "lower_lows": a.fractals.lower_lows,
            "n_swing_highs": len(a.fractals.swing_highs),
            "n_swing_lows": len(a.fractals.swing_lows),
            "last_high": (a.fractals.last_high.price if a.fractals.last_high else None),
            "last_low": (a.fractals.last_low.price if a.fractals.last_low else None),
            "fractal_trend": a.fractal_trend,
        }
        break

    consensus_strength = _consensus_strength(analyses, weights, consensus_trend)
    verdict = _build_verdict(
        symbol, consensus_trend, consensus_strength, nearest_support, nearest_resistance
    )
    html = _render_trendline_telegram_html(
        symbol=symbol,
        spot=spot,
        analyses=analyses,
        consensus_trend=consensus_trend,
        consensus_strength=consensus_strength,
        weights=weights,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
    )

    return {
        "consensus_trend": consensus_trend,
        "consensus_strength": round(consensus_strength, 1),
        "by_timeframe": by_tf,
        "nearest_support": round(nearest_support, 4) if nearest_support else None,
        "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance else None,
        "fractal_summary": fractal_summary,
        "verdict": verdict,
        "telegram_html_message": html,
    }


def _consensus_strength(
    analyses: list[TrendlineAnalysis], weights: dict[str, float], consensus_trend: str
) -> float:
    """Взвешенная средняя сила тренда среди TF, согласных с консенсусом."""
    num = 0.0
    den = 0.0
    for a in analyses:
        w = weights.get(a.timeframe, 0.0)
        if a.combined_trend == consensus_trend:
            num += a.combined_strength * w
            den += w
    if den <= 0:
        return 0.0
    return num / den


def _build_verdict(
    symbol: str,
    trend: str,
    strength: float,
    support: Optional[float],
    resistance: Optional[float],
) -> str:
    """Текстовый вердикт с рекомендацией (для JSON и шапки Telegram)."""
    label = _TREND_LABEL_RU.get(trend, trend)
    parts = [f"{symbol}: тренд {label.lower()} (сила {strength:.0f}/100)."]
    if trend == "BULLISH":
        parts.append(
            "Линии поддержки направлены вверх, фракталы формируют HH/HL — "
            "рассматривать long-идеи на откате к поддержке."
        )
    elif trend == "BEARISH":
        parts.append(
            "Линии сопротивления направлены вниз, фракталы формируют LH/LL — "
            "рассматривать short-идеи на откате к сопротивлению."
        )
    else:
        parts.append(
            "Направление линий и фракталы не согласованы — диапазон/флэт, "
            "торговля от границ канала."
        )
    if support and resistance:
        parts.append(f"Зона: поддержка {support:.2f} / сопротивление {resistance:.2f}.")
    return " ".join(parts)


def _fmt_price(price: Optional[float]) -> str:
    if price is None:
        return "—"
    return f"{price:.2f}".rstrip("0").rstrip(".")


def _strength_bar(strength: float, width: int = 10) -> str:
    filled = max(0, min(width, int(round(strength / 100.0 * width))))
    return "█" * filled + "░" * (width - filled)


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_trendline_telegram_html(
    *,
    symbol: str,
    spot: float,
    analyses: list[TrendlineAnalysis],
    consensus_trend: str,
    consensus_strength: float,
    weights: dict[str, float],
    nearest_support: Optional[float],
    nearest_resistance: Optional[float],
) -> str:
    """HTML-сообщение для Telegram (parse_mode=HTML).

    Структура: шапка (консенсус + спот), покадровая таблица (линии + фракталы),
    ближайшие уровни, вердикт. Без эмодзи.
    """
    arrow = "▲" if consensus_trend == "BULLISH" else ("▼" if consensus_trend == "BEARISH" else "◆")
    lines: list[str] = [
        f"<b>{_esc(symbol)} · Трендовые линии + фракталы</b>",
        f"Спот: {_fmt_price(spot)} · Консенсус: {arrow} "
        f"{_esc(_TREND_LABEL_RU.get(consensus_trend, consensus_trend))} "
        f"<code>{_strength_bar(consensus_strength)}</code> {_esc(round(consensus_strength))}/100",
        "",
        "<b>По таймфреймам</b>",
    ]

    for a in analyses:
        w = weights.get(a.timeframe, 0.0)
        comb_arrow = "▲" if a.combined_trend == "BULLISH" else ("▼" if a.combined_trend == "BEARISH" else "◆")
        lin_arrow = "▲" if a.trend_direction == "BULLISH" else ("▼" if a.trend_direction == "BEARISH" else "◆")
        fr_arrow = "▲" if a.fractal_trend == "BULLISH" else ("▼" if a.fractal_trend == "BEARISH" else "◆")
        lines.append(
            f"  [{_esc(a.timeframe)}] · вклад {_esc(int(w*100))}% · {comb_arrow} "
            f"{_esc(_TREND_LABEL_RU.get(a.combined_trend, a.combined_trend))} "
            f"({_esc(round(a.combined_strength))}/100)"
        )
        lines.append(
            f"    линии {lin_arrow} угол {_esc(round(a.line_angle_deg, 1))}° "
            f"({_esc(round(a.trend_strength))}) · фракталы {fr_arrow} "
            f"({_esc(round(a.fractal_strength))})"
        )
        fr = a.fractals
        lines.append(
            f"    HH {_esc(fr.higher_highs)} · HL {_esc(fr.higher_lows)} · "
            f"LH {_esc(fr.lower_highs)} · LL {_esc(fr.lower_lows)} · "
            f"S/R: {_esc(len(a.support_lines))}/{_esc(len(a.resistance_lines))}"
        )

    lines.append("")
    lines.append(
        f"Поддержка: {_fmt_price(nearest_support)} · "
        f"Сопротивление: {_fmt_price(nearest_resistance)}"
    )

    verdict = _build_verdict(
        symbol, consensus_trend, consensus_strength,
        nearest_support, nearest_resistance,
    )
    lines.append("")
    lines.append(_esc(verdict))
    return "\n".join(lines)
