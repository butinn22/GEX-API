"""Сервис тренда по MACD: фетчер OHLCV → MACD → analyzer → консенсус → схема.

Параллелен :class:`gex.ta_service.TAService` и :class:`gex.trendline_service.
TrendlineService`, но использует движок из :mod:`gex.macd_trend`: определение
направления и силы тренда на основе индикатора MACD (линии MACD и Signal) с
вероятностным (перцентиль + Марков) и теоретико-игровым (Multiplicative
Weights Update + conviction по спреду) слоем поверх базовой геометрии.

Поддерживаемые инструменты
--------------------------
* **US-акции/ETF** (SPY, AAPL, NVDA, …) — OHLCV через
  :class:`gex.ta_fetcher.TATimeframesFetcher` (yfinance, с ресемплингом
  2h/4h из 1h);
* **Крипта** (BTC, ETH, SOL, XRP, DOGE) — OHLCV через публичный Bybit V5
  ``/v5/market/kline`` (консистентно с GEX-источником), с авто-fallback на
  yfinance (``BTC-USD``);
* **MOEX-фьючерсы** (RTS/MIX/CNY/Si) — OHLCV через
  :class:`gex.moex_candles_fetcher.MOEXCandlesFetcher` (ISS candles FORTS
  front-month по OI, с ресемплингом 2h/4h из 1h).

Консенсус по таймфреймам
------------------------
Старшие таймфреймы тяжелее младших (как в ``ta_service``)::

    1h : 0.10
    2h : 0.20
    4h : 0.30
    1d : 0.40   (нормированы к 1.0)

Консенсус считается взвешенным голосованием итогового ``final_trend_score``
каждого TF (знак задаёт направление, модуль — силу).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
from gex.adapters.providers.bybit import fetch_ohlcv as fetch_bybit_ohlcv
from gex.domain.macd_trend import (
    AnalyzerConfig,
    BarResult,
    Quadrant,
    analyze_history,
    compute_macd,
)
from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher, _MOEX_OHLCV_ASSETS
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher, TIMEFRAMES
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher
from gex.schemas import MacdTrendAnalysisOut, macd_trend_to_schema

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


class MacdTrendService:
    """Фасад MACD-тренда: данные → MACD → analyzer → консенсус → схема.

    Parameters
    ----------
    fetcher : TATimeframesFetcher, optional
        Источник OHLCV для акций (yfinance). По умолчанию создаётся стандартный.
    fast, slow, signal : int
        Параметры MACD (12/26/9) — используются для расчёта линий из Close.
    """

    def __init__(
        self,
        fetcher: Optional[TATimeframesFetcher] = None,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        self.fetcher = fetcher or create_timeframes_fetcher()
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def analyze(self, ticker: str, config: Optional[AnalyzerConfig] = None) -> MacdTrendAnalysisOut:
        """Polling + MACD-тренд по 4 таймфреймам → MacdTrendAnalysisOut.

        Авто-определение типа актива: крипта (BTC/ETH/SOL/XRP/DOGE) → Bybit,
        иначе — акция через yfinance.

        Raises
        ------
        ValueError
            Данные не получены / тикер не найден / слишком короткая история.
        RuntimeError
            Сетевые ошибки источника OHLCV.
        """
        ticker = ticker.strip().upper()
        asset_type = self._detect_asset_type(ticker)
        config = config or AnalyzerConfig()
        logger.info("MACD trend analyze: ticker=%s, asset_type=%s", ticker, asset_type)

        # --- 1. OHLCV по всем таймфреймам ---
        tfs, spot = self._fetch_all(ticker, asset_type)

        # --- 2. Анализ по каждому TF ---
        analyses: list[tuple[str, pd.DataFrame, BarResult]] = []
        for tf in TIMEFRAMES:
            df = tfs.get(tf)
            if df is None or len(df) == 0:
                continue
            try:
                last_bar = self._analyze_df(df, tf, config)
            except ValueError as exc:
                logger.warning("  %s: слишком короткая история (%s)", tf, exc)
                continue
            analyses.append((tf, df, last_bar))

        if not analyses:
            raise ValueError(
                f"Не удалось проанализировать ни один таймфрейм для '{ticker}' "
                "(недостаточно баров)."
            )

        # --- 3. Консенсус ---
        consensus = self._consensus(analyses)

        # --- 4. Summarize-блок + HTML ---
        summary_dict = _build_macd_summary(
            symbol=ticker,
            spot=spot,
            analyses=analyses,
            consensus_trend=consensus,
            weights=_TF_WEIGHTS,
            config=config,
        )

        # --- 5. Сериализация ---
        return macd_trend_to_schema(
            analyses,
            symbol=ticker,
            asset_type=asset_type,
            spot=spot,
            generated_at=datetime.now(timezone.utc),
            consensus_trend=consensus,
            weights=_TF_WEIGHTS,
            summarize=summary_dict,
            config=config,
        )

    def analyze_timeframe(
        self, ticker: str, timeframe: str, config: Optional[AnalyzerConfig] = None
    ) -> MacdTrendAnalysisOut:
        """MACD-тренд по **одному** таймфрейму (1h/2h/4h/1d).

        Лёгкий эндпоинт — без сводного консенсуса (возвращается один TF).
        """
        ticker = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(
                f"Неверный таймфрейм '{timeframe}'. Доступно: {', '.join(TIMEFRAMES)}."
            )
        config = config or AnalyzerConfig()
        asset_type = self._detect_asset_type(ticker)
        tfs, spot = self._fetch_all(ticker, asset_type)
        df = tfs.get(tf)
        if df is None or len(df) == 0:
            raise ValueError(f"Нет данных по таймфрейму '{tf}' для '{ticker}'.")

        last_bar = self._analyze_df(df, tf, config)
        summary_dict = _build_macd_summary(
            symbol=ticker, spot=spot, analyses=[(tf, df, last_bar)],
            consensus_trend=_trend_from_score(last_bar.final_trend_score),
            weights={tf: 1.0}, config=config,
        )
        return macd_trend_to_schema(
            [(tf, df, last_bar)],
            symbol=ticker,
            asset_type=asset_type,
            spot=spot,
            generated_at=datetime.now(timezone.utc),
            consensus_trend=_trend_from_score(last_bar.final_trend_score),
            weights={tf: 1.0},
            summarize=summary_dict,
            config=config,
        )

    # ------------------------------------------------------------------ #
    #  Core analysis of one DataFrame
    # ------------------------------------------------------------------ #
    def _analyze_df(
        self, df: pd.DataFrame, tf: str, config: AnalyzerConfig
    ) -> BarResult:
        """Compute MACD from Close, run analyze_history, return last BarResult.

        OHLC (high/low/close) is passed through for ``atr`` normalization.
        """
        if df is None or len(df) == 0 or "Close" not in df.columns:
            raise ValueError("Empty DataFrame / no Close column.")

        close = df["Close"].astype(float)
        if len(close) < self.slow + self.signal:
            raise ValueError(
                f"Слишком короткая история ({len(close)} баров) для MACD"
                f"({self.fast}/{self.slow}/{self.signal})."
            )

        macd_line, signal_line = compute_macd(
            close, fast=self.fast, slow=self.slow, signal=self.signal
        )
        # Drop the warm-up region (NaN-free from the start for adjust=False EMA,
        # but the slow EMA still needs `slow` bars to be meaningful).
        start = self.slow
        macd_line = macd_line.iloc[start:].reset_index(drop=True)
        signal_line = signal_line.iloc[start:].reset_index(drop=True)
        close_c = close.iloc[start:].reset_index(drop=True)
        high_c = (
            df["High"].astype(float).iloc[start:].reset_index(drop=True)
            if "High" in df.columns else None
        )
        low_c = (
            df["Low"].astype(float).iloc[start:].reset_index(drop=True)
            if "Low" in df.columns else None
        )

        df_res = analyze_history(
            macd_line, signal_line, config,
            close=close_c, high=high_c, low=low_c,
        )
        last = df_res.iloc[-1].to_dict()
        return _row_to_bar_result(last)

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
        (Логика идентична ``TrendlineService._fetch_all``.)
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
    def _consensus(
        analyses: list[tuple[str, pd.DataFrame, BarResult]]
    ) -> str:
        """Взвешенное голосование по ``final_trend_score`` каждого TF."""
        score = 0.0
        for tf, _df, bar in analyses:
            w = _TF_WEIGHTS.get(tf, 0.0)
            f = bar.final_trend_score
            if f is None:
                continue
            score += float(f) * w
        if score > 1e-9:
            return "BULLISH"
        if score < -1e-9:
            return "BEARISH"
        return "RANGE"


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
#  Helpers
# ====================================================================== #
def _trend_from_score(final_trend_score: Optional[float]) -> str:
    """Map a final_trend_score sign → BULLISH/BEARISH/RANGE."""
    if final_trend_score is None:
        return "RANGE"
    if final_trend_score > 1e-9:
        return "BULLISH"
    if final_trend_score < -1e-9:
        return "BEARISH"
    return "RANGE"


def _row_to_bar_result(row: dict) -> BarResult:
    """Reconstruct a :class:`BarResult` from an analyze_history DataFrame row.

    Pandas may turn ``None`` into ``NaN`` (object/float dtype coercion) when the
    batch result is assembled into a DataFrame; here we coerce ``NaN``/non-string
    values back to ``None`` so downstream Pydantic Literals stay valid.
    """
    quadrant = row.get("quadrant")
    quadrant_enum: Optional[Quadrant] = None
    if isinstance(quadrant, str):
        try:
            quadrant_enum = Quadrant(quadrant)
        except ValueError:
            quadrant_enum = None
    return BarResult(
        avg_value=_safe_float(row.get("avg_value")),
        position=_safe_str(row.get("position"), ("above", "below")),
        zero_cross=_safe_str(row.get("zero_cross"), ("bull_cross", "bear_cross")),
        norm_slope=_safe_float(row.get("norm_slope")),
        angle_degrees=_safe_float(row.get("angle_degrees")),
        quadrant=quadrant_enum,
        strength_instant=_safe_float(row.get("strength_instant")),
        strength_percentile=_safe_float(row.get("strength_percentile")),
        markov_next_state_probs=_safe_probs(row.get("markov_next_state_probs")),
        composite_score=_safe_float(row.get("composite_score")),
        conviction_multiplier=_safe_float(row.get("conviction_multiplier")),
        final_trend_score=_safe_float(row.get("final_trend_score")),
        kelly_fraction=_safe_float(row.get("kelly_fraction")),
    )


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(xf):
        return None
    return xf


def _safe_str(x, allowed: tuple[str, ...]) -> Optional[str]:
    """Coerce a DataFrame cell to one of ``allowed`` strings, else ``None``.

    Guards against pandas turning a Python ``None`` into ``NaN`` for object
    columns (which would then break Pydantic ``Literal`` validation).
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x if x in allowed else None
    # NaN / float fallback
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return None


def _safe_probs(x) -> dict[str, float]:
    if not isinstance(x, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in x.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


# ====================================================================== #
#  Summarize-блок (для JSON + Telegram HTML)
# ====================================================================== #
_TREND_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "RANGE": "🟡"}
_TREND_LABEL_RU = {
    "BULLISH": "Восходящий",
    "BEARISH": "Нисходящий",
    "RANGE": "Боковик",
}

_QUADRANT_LABEL_RU = {
    "BULLISH_STRENGTHENING": "Бычий, усиливается",
    "BULLISH_WEAKENING": "Бычий, ослабевает",
    "BEARISH_STRENGTHENING": "Медвежий, усиливается",
    "BEARISH_WEAKENING": "Медвежий, ослабевает",
    "FLAT": "Боковик",
}


def _build_macd_summary(
    *,
    symbol: str,
    spot: float,
    analyses: list[tuple[str, pd.DataFrame, BarResult]],
    consensus_trend: str,
    weights: dict[str, float],
    config: AnalyzerConfig,
) -> dict:
    """Собрать summarize-блок + HTML-сообщение для Telegram."""
    by_tf: dict[str, dict] = {}
    for tf, df, bar in analyses:
        by_tf[tf] = {
            "quadrant": bar.quadrant.value if bar.quadrant else None,
            "position": bar.position,
            "zero_cross": bar.zero_cross,
            "angle_degrees": _rnd(bar.angle_degrees, 2),
            "strength_instant": _rnd(bar.strength_instant, 3),
            "strength_percentile": _rnd(bar.strength_percentile, 1),
            "composite_score": _rnd(bar.composite_score, 3),
            "conviction_multiplier": _rnd(bar.conviction_multiplier, 3),
            "final_trend_score": _rnd(bar.final_trend_score, 3),
            "kelly_fraction": _rnd(bar.kelly_fraction, 3),
            "trend": _trend_from_score(bar.final_trend_score),
            "n_bars": int(len(df)),
        }

    consensus_strength = _consensus_strength(analyses, weights, consensus_trend)
    verdict = _build_verdict(symbol, consensus_trend, consensus_strength, by_tf, config)
    html = _render_macd_telegram_html(
        symbol=symbol,
        spot=spot,
        analyses=analyses,
        consensus_trend=consensus_trend,
        consensus_strength=consensus_strength,
        weights=weights,
        config=config,
    )

    return {
        "consensus_trend": consensus_trend,
        "consensus_strength": round(consensus_strength, 1),
        "by_timeframe": by_tf,
        "verdict": verdict,
        "telegram_html_message": html,
    }


def _consensus_strength(
    analyses: list[tuple[str, pd.DataFrame, BarResult]],
    weights: dict[str, float],
    consensus_trend: str,
) -> float:
    """Взвешенное среднее |final_trend_score| среди TF, согласных с консенсусом."""
    num = 0.0
    den = 0.0
    for tf, _df, bar in analyses:
        w = weights.get(tf, 0.0)
        if bar.final_trend_score is None:
            continue
        if _trend_from_score(bar.final_trend_score) == consensus_trend:
            num += abs(bar.final_trend_score) * 100.0 * w
            den += w
    if den <= 0:
        return 0.0
    return num / den


def _rnd(x: Optional[float], ndigits: int) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), ndigits)


def _build_verdict(
    symbol: str,
    trend: str,
    strength: float,
    by_tf: dict[str, dict],
    config: AnalyzerConfig,
) -> str:
    """Текстовый вердикт с рекомендацией (для JSON и шапки Telegram)."""
    label = _TREND_LABEL_RU.get(trend, trend)
    parts = [f"{symbol}: MACD-тренд {label.lower()} (сила {strength:.0f}/100)."]
    # Самый старший доступный TF задаёт детали.
    senior = None
    for tf in ("1d", "4h", "2h", "1h"):
        if tf in by_tf:
            senior = by_tf[tf]
            break
    q = (senior or {}).get("quadrant")
    ang = (senior or {}).get("angle_degrees")
    if trend == "BULLISH":
        parts.append(
            "MACD выше сигнальной/ноля, угол положительный — "
            "рассматривать long-идеи на откате."
        )
    elif trend == "BEARISH":
        parts.append(
            "MACD ниже сигнальной/ноля, угол отрицательный — "
            "рассматривать short-идеи на откате."
        )
    else:
        parts.append(
            "Угол в «плоской» зоне, MACD и Signal близки — "
            "боковик/равновесие, торговля от границ."
        )
    if q:
        parts.append(f"Старший TF: режим {q}" + (f", угол {ang}°." if ang is not None else "."))
    if config.kelly_enabled:
        k = (senior or {}).get("kelly_fraction")
        if k is not None:
            parts.append(
                f"Kelly fraction = {k} (исследовательский компонент, НЕ фин. рекомендация)."
            )
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


def _render_macd_telegram_html(
    *,
    symbol: str,
    spot: float,
    analyses: list[tuple[str, pd.DataFrame, BarResult]],
    consensus_trend: str,
    consensus_strength: float,
    weights: dict[str, float],
    config: AnalyzerConfig,
) -> str:
    """HTML-сообщение для Telegram (parse_mode=HTML).

    Структура: шапка (консенсус + спот), покадровая таблица (квадрант, угол,
    сила, conviction, final score), вердикт. Без эмодзи и служебных параметров.
    """
    arrow = "▲" if consensus_trend == "BULLISH" else ("▼" if consensus_trend == "BEARISH" else "◆")
    lines: list[str] = [
        f"<b>{_esc(symbol)} · Тренд по MACD</b>",
        f"Спот: {_fmt_price(spot)} · Консенсус: {arrow} "
        f"{_esc(_TREND_LABEL_RU.get(consensus_trend, consensus_trend))} "
        f"<code>{_strength_bar(consensus_strength)}</code> "
        f"{_esc(round(consensus_strength))}/100",
        "",
        "<b>По таймфреймам</b>",
    ]

    for tf, _df, bar in analyses:
        w = weights.get(tf, 0.0)
        q_str = bar.quadrant.value if bar.quadrant else None
        q_ru = _QUADRANT_LABEL_RU.get(q_str, q_str or "—")
        trend_tf = _trend_from_score(bar.final_trend_score)
        tf_arrow = "▲" if trend_tf == "BULLISH" else ("▼" if trend_tf == "BEARISH" else "◆")
        ang = "—" if bar.angle_degrees is None else f"{bar.angle_degrees:+.1f}°"
        si = "—" if bar.strength_instant is None else f"{bar.strength_instant:.2f}"
        sp = "—" if bar.strength_percentile is None else f"{bar.strength_percentile:.0f}"
        cv = "—" if bar.conviction_multiplier is None else f"{bar.conviction_multiplier:.2f}"
        fs = "—" if bar.final_trend_score is None else f"{bar.final_trend_score:+.2f}"
        lines.append(
            f"  [{_esc(tf)}] · вклад {_esc(int(w*100))}% · {tf_arrow} "
            f"{_esc(_TREND_LABEL_RU.get(trend_tf, trend_tf))} · final {fs}"
        )
        lines.append(
            f"    {_esc(q_ru)} · угол {ang} · сила {si} "
            f"(перц. {sp}) · conviction {cv}"
        )

    verdict = _build_verdict(
        symbol, consensus_trend, consensus_strength,
        {tf: {} for tf, _, _ in analyses}, config,
    )
    lines.append("")
    lines.append(_esc(verdict))
    return "\n".join(lines)
