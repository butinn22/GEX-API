"""Novel Candles — гибрид стандартных свечей и Heikin-Ashi.

Порт индикатора PineScript v5 «Novel Candles» by butinn22.
Трансформирует стандартные OHLCV в сглаженные «novel»-свечи,
рассчитывает EMA на их базе и трендовые линии через
существующий движок :func:`gex.trendlines.analyze_trendlines`.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
from gex.adapters.providers.bybit import fetch_ohlcv as fetch_bybit_ohlcv
from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher, _MOEX_OHLCV_ASSETS
from gex.domain.trendlines import analyze_trendlines, TrendlineAnalysis
from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher, TIMEFRAMES
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher
from gex.adapters.cache.redis_client import RedisClient, cached, cache_key

logger = logging.getLogger(__name__)

DEFAULT_EMA_PERIODS = (10, 20, 50, 100, 200)


class NovelCandlesService:
    """Сервис Novel Candles: трансформация OHLCV → расчёт → анализ.

    Parameters
    ----------
    redis_client : RedisClient | None
        Клиент Redis для кэширования ответов.
    """

    def __init__(self, redis_client: Optional[RedisClient] = None):
        self._redis = redis_client
        self._ta_fetcher = create_timeframes_fetcher(redis_client=redis_client)

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def fetch_and_analyze(
        self,
        ticker: str,
        timeframe: str = "1d",
        limit: int = 500,
        ema_periods: tuple[int, ...] = DEFAULT_EMA_PERIODS,
        trendline_resolution: int = 6,
        max_trendlines: int = 5,
        pivot_left: int = 5,
        pivot_right: int = 5,
        use_linreg: bool = False,
        linreg_length: int = 11,
        use_linreg_for_ema: bool = False,
        two_pole: bool = False,
        tp_length: int = 20,
        tp_damping: float = 0.9,
        tp_bands: float = 1.0,
        tp_ris_fal: int = 5,
        tp_signals: bool = False,
    ) -> dict:
        """Основной метод: получить OHLCV → novel свечи → EMA + трендовые линии.

        Returns
        -------
        dict
            Ключи: ``ticker``, ``timeframe``, ``bars``, ``emas``, ``trendlines``,
            ``two_pole`` (если включён).
        """
        ticker = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(f"Неподдерживаемый таймфрейм '{timeframe}'. Доступно: {', '.join(TIMEFRAMES)}")

        asset_type = self._detect_asset_type(ticker)
        logger.info("NovelCandles: ticker=%s, tf=%s, asset=%s, limit=%d, linreg=%s",
                     ticker, tf, asset_type, limit, use_linreg)

        # --- 1. Получить OHLCV ---
        df_raw = self._fetch_ohlcv(ticker, tf, asset_type, limit)
        if df_raw is None or len(df_raw) == 0:
            raise ValueError(f"Нет OHLCV данных для '{ticker}' [{tf}]")

        # --- 2. Novel-трансформация ---
        novel_df = self.compute_novel_candles(df_raw)

        # --- 2.5 LinReg-сглаживание (если включено) ---
        if use_linreg:
            display_df = self.compute_linreg_candles(novel_df, linreg_length)
        else:
            display_df = novel_df

        # --- 2.6 Hybridsrc для EMA (если use_linreg_for_ema) ---
        ema_src: np.ndarray | None = None
        if use_linreg and use_linreg_for_ema:
            # Нужны hybridHigh/Low для hybridsrc
            ha_df = self.compute_heikin_ashi(df_raw)
            hybrid_high, hybrid_low = self._compute_hybrid_high_low(df_raw, ha_df)
            ema_src = self.compute_hybridsrc_for_ema(
                novel_df, hybrid_high, hybrid_low, linreg_length,
            )

        # --- 3. EMA ---
        emas = self.compute_emas(novel_df, list(ema_periods), src=ema_src)

        # --- 3.5 Two-Pole Filter (если включён) ---
        two_pole_data = None
        if two_pole:
            # Источник для фильтра: hybridsrc (если linreg for ema) или ohlcnovel
            if ema_src is not None:
                tp_src = ema_src
            else:
                tp_src = novel_df[["Open", "High", "Low", "Close"]].mean(axis=1).values

            atr = self.compute_atr(df_raw, period=200)
            two_pole_data = self.compute_two_pole_filter(
                tp_src,
                length=tp_length,
                damping=tp_damping,
                atr=atr,
                bands=tp_bands,
                ris_fal=tp_ris_fal,
                signals=tp_signals,
            )

        # --- 4. Трендовые линии (на отображаемых барах) ---
        # При включённом LinReg линии считаются по сглаженным свечам (display_df),
        # иначе они «висят в воздухе» относительно показанного графика.
        trendlines = None
        if max_trendlines > 0:
            try:
                tl_df = display_df if use_linreg else novel_df
                tl = self.compute_trendlines(
                    tl_df, tf,
                    resolution=trendline_resolution,
                    max_support_lines=max_trendlines,
                    max_resistance_lines=max_trendlines,
                    pivot_left=pivot_left,
                    pivot_right=pivot_right,
                )
                trendlines = {
                    "support": [ln.as_dict() for ln in tl.support_lines],
                    "resistance": [ln.as_dict() for ln in tl.resistance_lines],
                    "combined_trend": tl.combined_trend,
                    "combined_strength": round(tl.combined_strength, 1),
                }
            except Exception as exc:
                logger.warning("NovelCandles trendline failed for %s: %s", ticker, exc)

        # --- 5. Сериализация баров ---
        bars = []
        for idx, (ts, row) in enumerate(display_df.iterrows()):
            bars.append({
                "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
            })

        result = {
            "ticker": ticker,
            "timeframe": tf,
            "asset_type": asset_type,
            "n_bars": len(bars),
            "bars": bars,
            "emas": emas,
            "trendlines": trendlines,
        }

        if two_pole_data is not None:
            result["two_pole"] = two_pole_data

        return result

    # ------------------------------------------------------------------ #
    #  LinReg-сглаживание (порт PineScript v5 ta.linreg)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_linreg(series: np.ndarray, length: int) -> np.ndarray:
        """Линейная регрессия — аналог PineScript ``ta.linreg(src, length, 0)``.

        Для каждого бара i:
        - Берём окно ``max(0, i-length+1)..i``
        - Fit y = a*x + b
        - Результат: значение линии в последней точке окна ``b + a*(len-1)``
        """
        n = len(series)
        result = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - length + 1)
            window = series[start:i + 1]
            wlen = len(window)
            if wlen < 2:
                result[i] = series[i]
                continue
            # polyfit degree=1 → [slope, intercept]
            x = np.arange(wlen, dtype=np.float64)
            slope, intercept = np.polyfit(x, window, 1)
            result[i] = intercept + slope * (wlen - 1)
        return result

    @staticmethod
    def compute_linreg_candles(novel_df: pd.DataFrame, linreg_length: int) -> pd.DataFrame:
        """Применить LinReg-сглаживание к Novel OHLC."""
        df = novel_df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = NovelCandlesService._apply_linreg(df[col].values, linreg_length)
        return df

    @staticmethod
    def _compute_hybrid_high_low(df_std: pd.DataFrame, df_ha: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Гибридные High/Low: max(stdH, haH), min(stdL, haL)."""
        hybrid_high = np.maximum(df_std["High"].values, df_ha["High"].values)
        hybrid_low = np.minimum(df_std["Low"].values, df_ha["Low"].values)
        return hybrid_high, hybrid_low

    @staticmethod
    def compute_hybridsrc_for_ema(
        novel_df: pd.DataFrame,
        hybrid_high: np.ndarray,
        hybrid_low: np.ndarray,
        linreg_length: int,
    ) -> np.ndarray:
        """Вычислить hybridsrc = avg(8 linreg-серий) для EMA.

        Серии (как в PineScript):
        - lr_line_close = linreg(novelsrc)  [novelsrc ≡ novelClose]
        - lr_line_open  = lr_line_close shifted by 1
        - lr_hybrid_high/low = linreg(hybridHigh/Low)
        - lr_open/high/low/close = linreg(novel OHLC)
        """
        novel_open = novel_df["Open"].values.astype(np.float64)
        novel_high = novel_df["High"].values.astype(np.float64)
        novel_low = novel_df["Low"].values.astype(np.float64)
        novel_close = novel_df["Close"].values.astype(np.float64)
        novelsrc = novel_close.copy()  # в Novel-алгоритме novelClose ≡ novelsrc

        _lr = NovelCandlesService._apply_linreg

        lr_line_close = _lr(novelsrc, linreg_length)
        lr_line_open = np.roll(lr_line_close, 1)
        lr_line_open[0] = lr_line_close[0]

        lr_hybrid_high = _lr(hybrid_high.astype(np.float64), linreg_length)
        lr_hybrid_low = _lr(hybrid_low.astype(np.float64), linreg_length)

        lr_open = _lr(novel_open, linreg_length)
        lr_close = _lr(novel_close, linreg_length)
        lr_high = _lr(novel_high, linreg_length)
        lr_low = _lr(novel_low, linreg_length)

        # Усреднение 8 серий
        hybridsrc = np.mean([
            lr_line_close, lr_line_open,
            lr_hybrid_high, lr_hybrid_low,
            lr_open, lr_close, lr_high, lr_low,
        ], axis=0)
        return hybridsrc

    # ------------------------------------------------------------------ #
    #  Алгоритм Novel Candles (порт PineScript v5)
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """Вычислить Heikin-Ashi OHLC из стандартных свечей.

        haClose = (O + H + L + C) / 4
        haOpen  = (prev_haOpen + prev_haClose) / 2
        haHigh  = max(H, haOpen, haClose)
        haLow   = min(L, haOpen, haClose)
        """
        n = len(df)
        ha_close = np.zeros(n)
        ha_open = np.zeros(n)
        ha_high = np.zeros(n)
        ha_low = np.zeros(n)

        o = df["Open"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        l = df["Low"].values.astype(np.float64)
        c = df["Close"].values.astype(np.float64)

        for i in range(n):
            ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
            if i == 0:
                ha_open[i] = (o[i] + c[i]) / 2.0
            else:
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
            ha_high[i] = max(h[i], ha_open[i], ha_close[i])
            ha_low[i] = min(l[i], ha_open[i], ha_close[i])

        return pd.DataFrame({
            "Open": ha_open,
            "High": ha_high,
            "Low": ha_low,
            "Close": ha_close,
        }, index=df.index)

    @staticmethod
    def compute_novel_candles(df: pd.DataFrame) -> pd.DataFrame:
        """Трансформировать стандартные OHLCV в Novel Candles.

        Полный порт PineScript v5 алгоритма (см. spec.md).
        """
        if df.empty:
            raise ValueError("DataFrame пуст — не из чего строить свечи")
        if len(df) < 2:
            # 1 бар: novelOpen = novelClose = novelsrc
            o = float(df["Open"].iloc[0])
            h = float(df["High"].iloc[0])
            l = float(df["Low"].iloc[0])
            c = float(df["Close"].iloc[0])
            hlcc4 = (h + l + c + c) / 4.0
            ha_close = (o + h + l + c) / 4.0
            ha_open = (o + c) / 2.0
            ha_high = max(h, ha_open, ha_close)
            ha_low = min(l, ha_open, ha_close)
            novelsrc = hlcc4  # упрощённо
            return pd.DataFrame({
                "Open": [novelsrc], "High": [novelsrc], "Low": [novelsrc], "Close": [novelsrc],
            }, index=df.index)

        # --- Heikin-Ashi ---
        ha = NovelCandlesService.compute_heikin_ashi(df)
        ha_o = ha["Open"].values
        ha_c = ha["Close"].values
        ha_h = ha["High"].values
        ha_l = ha["Low"].values

        std_o = df["Open"].values.astype(np.float64)
        std_h = df["High"].values.astype(np.float64)
        std_l = df["Low"].values.astype(np.float64)
        std_c = df["Close"].values.astype(np.float64)

        n = len(df)
        novel_open = np.zeros(n)
        novel_high = np.zeros(n)
        novel_low = np.zeros(n)
        novel_close = np.zeros(n)

        for i in range(n):
            # Гибридные экстремумы
            hybrid_high = max(std_h[i], ha_h[i])
            hybrid_low = min(std_l[i], ha_l[i])

            # medianTop / medianBottom
            median_top = (max(std_o[i], std_c[i]) + max(ha_o[i], ha_c[i])) / 2.0
            median_bottom = (min(std_o[i], std_c[i]) + min(ha_o[i], ha_c[i])) / 2.0

            hybrid_open = (std_o[i] + ha_o[i]) / 2.0
            hybrid_close = (std_c[i] + ha_c[i]) / 2.0

            candle_top = max(hybrid_open, hybrid_close)
            candle_bottom = min(hybrid_open, hybrid_close)
            avg_candle = (candle_bottom + candle_top) / 2.0

            # sourceformas
            if std_o[i] > std_c[i]:
                sourceformas = (std_o[i] + std_l[i]) / 2.0
            else:
                sourceformas = (std_c[i] + std_h[i]) / 2.0

            hlcc4 = (std_h[i] + std_l[i] + std_c[i] + std_c[i]) / 4.0
            novelsrc = (hlcc4 + avg_candle + sourceformas) / 3.0

            novel_close[i] = novelsrc
            if i == 0:
                novel_open[i] = novelsrc
            else:
                novel_open[i] = novel_close[i - 1]

            novel_high[i] = max(hybrid_high, novel_open[i], novel_close[i])
            novel_low[i] = min(hybrid_low, novel_open[i], novel_close[i])

        result = pd.DataFrame({
            "Open": novel_open,
            "High": novel_high,
            "Low": novel_low,
            "Close": novel_close,
        }, index=df.index)

        return result

    @staticmethod
    def compute_emas(
        novel_df: pd.DataFrame,
        periods: list[int],
        src: np.ndarray | None = None,
    ) -> dict[str, list[Optional[float]]]:
        """Вычислить EMA на ``ohlcnovel = avg(O,H,L,C)`` или переданном ``src``.

        Returns
        -------
        dict
            ``{"ema10": [val, ...], "ema20": [...], ...}``.
            Значения — float или None (где недостаточно истории).
        """
        if novel_df.empty:
            return {}

        if src is not None:
            ema_input = pd.Series(src, index=novel_df.index)
        else:
            ema_input = novel_df[["Open", "High", "Low", "Close"]].mean(axis=1)

        result: dict[str, list[Optional[float]]] = {}

        for period in periods:
            ema_series = ema_input.ewm(span=period, adjust=False).mean()
            key = f"ema{period}"
            vals = []
            for v in ema_series.values:
                if pd.isna(v):
                    vals.append(None)
                else:
                    vals.append(round(float(v), 4))
            result[key] = vals

        return result

    # ------------------------------------------------------------------ #
    #  Two-Pole Filter + Signals
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 200) -> np.ndarray:
        """Вычислить ATR (Average True Range) для смещения сигналов."""
        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        close = df["Close"].values.astype(np.float64)
        n = len(df)

        tr = np.zeros(n)
        for i in range(n):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )

        # EMA-smoothing of TR (RMA in PineScript)
        atr = np.zeros(n)
        alpha = 1.0 / period
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return atr

    @staticmethod
    def compute_two_pole_filter(
        src: np.ndarray,
        length: int = 20,
        damping: float = 0.9,
        atr: np.ndarray | None = None,
        bands: float = 1.0,
        ris_fal: int = 5,
        signals: bool = False,
    ) -> dict:
        """Two-Pole Filter второго порядка с сигналами rising/falling.

        Порт PineScript v5 two_pole_filter.

        Returns
        -------
        dict
            ``tp_f``: значения фильтра,
            ``rising``: счётчики последовательных rising,
            ``falling``: счётчики falling,
            ``rising_positions``: индексы сигналов rising (crossover),
            ``falling_positions``: индексы сигналов falling,
            ``signal_up``: [{index, price}, ...],
            ``signal_dn``: [{index, price}, ...]
        """
        n = len(src)
        omega = 2.0 * np.pi / length
        alpha = damping * omega
        beta = omega ** 2

        f1 = np.zeros(n)
        f2 = np.zeros(n)

        for i in range(n):
            if i == 0:
                f1[i] = src[i]
                f2[i] = src[i]
            else:
                f1[i] = f1[i - 1] + alpha * (src[i] - f1[i - 1])
                f2[i] = f2[i - 1] + beta * (f1[i] - f2[i - 1])

        tp_f = f2

        # Rising / Falling: tp_f > tp_f[2] / tp_f < tp_f[2]
        rising = np.zeros(n, dtype=int)
        falling = np.zeros(n, dtype=int)

        for i in range(2, n):
            if tp_f[i] > tp_f[i - 2]:
                rising[i] = rising[i - 1] + 1
                falling[i] = 0
            elif tp_f[i] < tp_f[i - 2]:
                rising[i] = 0
                falling[i] = falling[i - 1] + 1
            else:
                rising[i] = 0
                falling[i] = 0

        # ATR offset для позиционирования сигналов
        atr_offset = 0.0
        if atr is not None and len(atr) == n:
            atr_offset = atr[-1] * bands if len(atr) > 0 else 0.0

        signal_up = []
        signal_dn = []

        if signals:
            for i in range(1, n):
                # crossover rising above ris_fal
                if rising[i] >= ris_fal and rising[i - 1] < ris_fal:
                    signal_up.append({
                        "index": i - 1,
                        "price": float(tp_f[i - 1] - atr_offset),
                    })
                # crossover falling above ris_fal
                if falling[i] >= ris_fal and falling[i - 1] < ris_fal:
                    signal_dn.append({
                        "index": i - 1,
                        "price": float(tp_f[i - 1] + atr_offset),
                    })

        # Mark bars where rising/falling ≥ ris_fal (для отрисовки квадратов)
        rising_positions = []
        falling_positions = []
        for i in range(n):
            if rising[i] >= ris_fal:
                rising_positions.append({"index": i, "price": float(tp_f[i] - atr_offset)})
            if falling[i] >= ris_fal:
                falling_positions.append({"index": i, "price": float(tp_f[i] + atr_offset)})

        return {
            "tp_f": [float(v) for v in tp_f],
            "rising": rising.tolist(),
            "falling": falling.tolist(),
            "rising_positions": rising_positions,
            "falling_positions": falling_positions,
            "signal_up": signal_up,
            "signal_dn": signal_dn,
        }

    @staticmethod
    def compute_trendlines(
        novel_df: pd.DataFrame,
        timeframe: str,
        resolution: int = 6,
        history_bars: int = 300,
        max_support_lines: int = 5,
        max_resistance_lines: int = 5,
        pivot_left: int = 5,
        pivot_right: int = 5,
    ) -> TrendlineAnalysis:
        """Запустить анализ трендовых линий на novel-свечах."""
        return analyze_trendlines(
            novel_df,
            timeframe=timeframe,
            resolution=resolution,
            history_bars=history_bars,
            max_support_lines=max_support_lines,
            max_resistance_lines=max_resistance_lines,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
        )

    # ------------------------------------------------------------------ #
    #  Определение типа актива
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_asset_type(ticker: str) -> str:
        if ticker in _MOEX_OHLCV_ASSETS:
            return "moex"
        if ticker in _CRYPTO_ASSETS:
            return "crypto"
        try:
            from gex.commodity_assets import COMMODITY_ASSETS
            if ticker in COMMODITY_ASSETS:
                return "commodity"
        except ImportError:
            pass
        return "stock"

    # ------------------------------------------------------------------ #
    #  Фетч OHLCV с роутингом по типам
    # ------------------------------------------------------------------ #
    def _fetch_ohlcv(
        self, ticker: str, timeframe: str, asset_type: str, limit: int,
    ) -> Optional[pd.DataFrame]:
        """Получить OHLCV для заданного тикера и таймфрейма.

        Роутинг:
        - stock → TATimeframesFetcher (yfinance)
        - crypto → Bybit kline с fallback на yfinance
        - moex → MOEXCandlesFetcher
        - commodity → TATimeframesFetcher с yf_symbol
        """
        if asset_type == "stock":
            tfs = self._ta_fetcher.fetch(ticker)
            return tfs.get(timeframe)

        if asset_type == "moex":
            fetcher = MOEXCandlesFetcher()
            tfs = fetcher.fetch(ticker)
            return tfs.get(timeframe)

        if asset_type == "commodity":
            from gex.commodity_assets import COMMODITY_ASSETS
            yf_sym = COMMODITY_ASSETS[ticker]["yf_symbol"]
            tfs = self._ta_fetcher.fetch(yf_sym)
            return tfs.get(timeframe)

        # crypto: Bybit → fallback yfinance
        df = self._fetch_bybit_kline(ticker, timeframe, limit)
        if df is not None and len(df) > 0:
            return df
        # fallback
        yf_ticker = f"{ticker}-USD"
        logger.info("NovelCandles crypto fallback: yfinance %s [%s]", yf_ticker, timeframe)
        tfs = self._ta_fetcher.fetch(yf_ticker)
        return tfs.get(timeframe)

    @staticmethod
    def _fetch_bybit_kline(coin: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Bybit V5 kline (публичный).

        Единственная разница с остальными точками: здесь сбой **не** пробрасывается наружу,
        а превращается в ``None`` — вызывающий сам уходит в fallback на yfinance. Это
        поведение сохранено намеренно; сам запрос теперь общий
        (:func:`gex.adapters.providers.bybit.fetch_ohlcv`).
        """
        try:
            return fetch_bybit_ohlcv(coin, timeframe, limit)
        except Exception as exc:  # noqa: BLE001 — fallback на yfinance у вызывающего
            logger.debug("Bybit kline failed for %s: %s", coin, exc)
            return None
