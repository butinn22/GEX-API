"""Sector breadth analysis: blended Tail Mean / Cubic Mean across US sector ETFs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher
from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher

logger = logging.getLogger(__name__)

# 12 секторов США + RSP (равновесный SPY)
SECTOR_TICKERS = [
    "XLK", "XLF", "XLY", "XLP", "XLE",
    "XLV", "XLI", "XLU", "XLB", "XLRE",
    "XLC", "RSP",
]

# Параметры расчёта по умолчанию
ROLLING_WINDOW = 20        # дней для скользящих статистик


@dataclass
class SectorBreadthReport:
    """Отчёт по широте секторов."""

    composite_series: pd.Series = field(default_factory=pd.Series)
    per_sector: dict[str, dict] = field(default_factory=dict)
    trend_analysis: dict = field(default_factory=dict)
    macd_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    dates: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    current: Optional[float] = None
    n_days: int = 0


class SectorBreadthService:
    """Анализ широты рынка через blended Tail Mean / Cubic Mean."""

    def __init__(self, fetcher: Optional[TATimeframesFetcher] = None):
        self.fetcher = fetcher or create_timeframes_fetcher()

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #
    def analyze(self, days: int = 500) -> SectorBreadthReport:
        """Полный анализ: среднее нормализованных Close 12 секторов + MACD ADL.

        Простое среднее 12 секторов США (XLK, XLF, ..., RSP), каждый
        нормализован к 100. От этого ряда считается MACD ADL-based.
        """
        closes = self._fetch_all_sectors()
        if not closes:
            raise ValueError("Не удалось загрузить данные ни по одному сектору")

        # Выравнивание по общим датам
        df = pd.DataFrame(closes)
        df = df.dropna()

        # Нормализуем каждый сектор к индексу 100
        df_norm = df.div(df.iloc[0]) * 100

        # Среднее всех секторов = композит (отклонение от 100)
        composite = df_norm.mean(axis=1) - 100

        # Данные по каждому сектору (текущие значения)
        per_sector = {}
        for ticker in df.columns:
            s = df[ticker]
            chg = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) > 1 else 0
            per_sector[ticker] = {
                "close": round(float(s.iloc[-1]), 2) if len(s) > 0 else None,
                "change_pct": round(float(chg), 2),
            }

        # Трендовый анализ
        trend = self._simple_trend_analysis(composite)

        # EMA20 (для отрисовки на графике)
        ema20 = composite.ewm(span=20, adjust=False).mean()

        # MACD на базе ADL от композита
        macd_df = self._macd_on_adl(composite)

        # Отчёт (200 точек)
        display = composite.tail(min(500, len(composite)))

        return SectorBreadthReport(
            composite_series=composite,
            per_sector=per_sector,
            trend_analysis=trend,
            macd_analysis=macd_df.tail(500) if macd_df is not None else pd.DataFrame(),
            dates=[str(d.date()) for d in display.index],
            values=[round(float(v), 4) for v in display.values],
            current=round(float(display.iloc[-1]), 4) if len(display) > 0 else None,
            n_days=len(display),
        )

    # ------------------------------------------------------------------ #
    #  Fetch all sector ETFs (daily)
    # ------------------------------------------------------------------ #
    def _fetch_all_sectors(self) -> dict[str, pd.Series]:
        """Загрузить дневные Close для всех секторов."""
        closes = {}
        for ticker in SECTOR_TICKERS:
            try:
                tfs = self.fetcher.fetch(ticker)
                df = tfs.get("1d")
                if df is not None and not df.empty:
                    s = df["Close"].astype(float).dropna()
                    s.name = ticker
                    closes[ticker] = s
                    logger.debug("Loaded %s: %d rows", ticker, len(s))
                else:
                    logger.warning("No daily data for %s", ticker)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", ticker, exc)
        return closes

    # ------------------------------------------------------------------ #
    #  Tail / Cubic Mean (удалены — не используются)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------ #
    #  Simple trend analysis
    # ------------------------------------------------------------------ #
    @staticmethod
    def _simple_trend_analysis(series: pd.Series) -> dict:
        """Простой трендовый анализ: наклон, HH/HL, сила."""
        if len(series) < 20:
            return {"trend": "NEUTRAL", "strength": 0, "slope": 0}

        last_20 = series.tail(20)
        x = np.arange(len(last_20))
        slope, _ = np.polyfit(x, last_20.values, 1)

        # HH/HL
        recent_high = series.tail(10).max()
        recent_low = series.tail(10).min()
        prev_high = series.shift(10).tail(10).max()
        prev_low = series.shift(10).tail(10).min()

        direction = "BULLISH" if slope > 0 else "BEARISH"
        strength = min(abs(slope) * 1000, 100)

        return {
            "trend": direction,
            "strength": round(strength, 1),
            "slope": round(float(slope), 6),
            "recent_high": round(float(recent_high), 4),
            "recent_low": round(float(recent_low), 4),
            "higher_high": bool(recent_high > prev_high) if not np.isnan(prev_high) else None,
            "higher_low": bool(recent_low > prev_low) if not np.isnan(prev_low) else None,
        }

    # ------------------------------------------------------------------ #
    #  ADL-based MACD (полный порт Pine Script v5)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _macd_on_adl(series: pd.Series) -> pd.DataFrame:
        """MACD на основе ADL — полный порт Pine Script v5.

        Цепочка вычислений (Pine → Python):
          1. sym = avg(novelsrc, AvgCandle, sourceformas, AvgCandle)
          2. diff → cumsum(sqrt) → EMA2 → adline
          3. ADL-RSI bands: ubb, lbb, lm
          4. ADL-BB/linreg: basisbb, devbb, upperbb, lowerbb
          5. ad = avg(basisbb, lm)
          6. adl50 = 4-term avg(sma50, ema50, ema(avg(lowerbb,lbb)), ema(avg(upperbb,ubb)))
          7. MACD = ad - adl50; Signal = sma(macd, 9); tl = avg(linreg, rma)
        """
        if len(series) < 50:
            return pd.DataFrame()

        idx = series.index
        ns = series.values
        n = len(ns)

        # ---- 1. novelsrc-подобная обработка (для sym) ----
        # Используем сам композит как novelsrc, а AvgCandle и sourceformas
        # аппроксимируем через rolling статистики
        df = pd.DataFrame({"novelsrc": ns, "close": ns}, index=idx)
        df["avg_candle"] = ns  # proxy: AvgCandle ≈ novelsrc
        df["sourceformas"] = ns  # proxy: sourceformas ≈ novelsrc

        # ---- 2. ADL chain (Pine: sym → diff → cum→EMA2) ----
        # sym = avg(novelsrc, AvgCandle, sourceformas, AvgCandle)  # 4-term, AvgCandle×2
        sym = (df["novelsrc"] + 2 * df["avg_candle"] + df["sourceformas"]) / 4
        diff = np.diff(sym.values, prepend=sym.iloc[0]) / (sym.values + 1e-12)
        sqrt_diff = np.sqrt(np.abs(diff)) * np.sign(diff)
        cum_sqrt = np.cumsum(sqrt_diff)
        adline = pd.Series(cum_sqrt, index=idx).ewm(alpha=2/3, adjust=False, min_periods=1).mean()

        # ---- 3. ADL-RSI bands (ubb/lbb/lm) ----
        length12 = 14
        ep = 2 * length12 - 1
        diff_src = adline.diff()
        auc = diff_src.clip(lower=0).ewm(span=ep, adjust=False, min_periods=1).mean()
        adc = (-diff_src).clip(lower=0).ewm(span=ep, adjust=False, min_periods=1).mean()

        sr2 = adline
        ob, os, om = 70, 30, 50
        x11 = (length12 - 1) * (adc * ob / (100 - ob) - auc)
        ubb = pd.Series(np.where(x11 >= 0, sr2 + x11, sr2 + x11 * (100 - ob) / ob), index=idx)
        x22 = (length12 - 1) * (adc * os / (100 - os) - auc)
        lbb = pd.Series(np.where(x22 >= 0, sr2 + x22, sr2 + x22 * (100 - os) / os), index=idx)
        x3 = (length12 - 1) * (adc * om / (100 - om) - auc)
        lm = pd.Series(np.where(x3 >= 0, sr2 + x3, sr2 + x3 * (100 - om) / om), index=idx)

        # ---- 4. ADL-BB/linreg (basisbb, devbb, upperbb, lowerbb) ----
        lkbk = int(round(4.618))
        srcbb = adline.shift(lkbk)
        lengthbb = 33
        multbb = 2.618
        percentbb = 61.8 / 100

        basisbb = SectorBreadthService._linreg(srcbb.rolling(lengthbb, min_periods=1).mean(), 10)
        devbb = multbb * srcbb.rolling(lengthbb, min_periods=1).std()
        upperbb = basisbb + devbb * percentbb
        lowerbb = basisbb - devbb * percentbb

        # ---- 5. ad = avg(basisbb, lm) ----
        ad = (basisbb + lm) / 2

        # ---- 6. adl50 = 4-term avg ----
        sma50 = adline.rolling(50, min_periods=1).mean()
        ema50 = adline.ewm(span=50, adjust=False, min_periods=1).mean()
        ema_lower = ((lowerbb + lbb) / 2).ewm(span=50, adjust=False, min_periods=1).mean()
        ema_upper = ((upperbb + ubb) / 2).ewm(span=50, adjust=False, min_periods=1).mean()
        adl50 = (sma50 + ema50 + ema_lower + ema_upper) / 4

        # ---- 7. MACD (Pine: macd = ad - adl50; signal = sma(macd,9); tl = avg(linreg,rma)) ----
        macd = ad - adl50
        signal = macd.rolling(9, min_periods=1).mean()
        hist = macd - signal
        avg_ms = (macd + signal) / 2
        linreg_val = SectorBreadthService._linreg(avg_ms, 50)
        rma_val = avg_ms.ewm(alpha=1/50, adjust=False, min_periods=1).mean()
        tl = (linreg_val + rma_val) / 2

        return pd.DataFrame({
            "macd": macd, "signal": signal, "hist": hist, "tl": tl,
            "adline": adline, "ad": ad, "adl50": adl50,
            "ubb": ubb, "lbb": lbb, "lm": lm,
            "basisbb": basisbb, "upperbb": upperbb, "lowerbb": lowerbb,
        }, index=idx)

    @staticmethod
    def _linreg(series: pd.Series, length: int) -> pd.Series:
        """Линейная регрессия (offset=0) — порт Pine ta.linreg."""
        def _lr(win):
            if len(win) < 2:
                return win.iloc[-1] if len(win) > 0 else np.nan
            import numpy as _np
            x = _np.arange(len(win))
            A = _np.vstack([x, _np.ones(len(x))]).T
            slope, intercept = _np.linalg.lstsq(A, win.values, rcond=None)[0]
            return slope * (len(win) - 1) + intercept
        return series.rolling(length, min_periods=2).apply(_lr, raw=False)
