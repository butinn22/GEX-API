"""Commodity Dynamics: равновзвешенный композит товарного рынка.

Формула:
  1. Каждый товар нормализуется к 100 на начало периода.
  2. Дневное значение композита = среднее арифметическое нормализованных товаров.
     → равный вес %-изменений независимо от абсолютной цены ($2 газ vs $4000 gold).
  3. Сглаживание через широкий товарный ETF (DBC):
     smoothed[t] = 0.70 * EWCI[t] + 0.30 * DBC_norm[t]
  4. EMA20 поверх сглаженного ряда.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from gex.commodity_assets import COMMODITY_ASSETS, COMMODITY_TICKERS
from gex.adapters.cache.redis_client import RedisClient, deserialize_value
from gex.adapters.providers.yfinance import history as yfinance_history
from gex.ports.cache_keys import PROVIDER_YFINANCE, commodity_key

logger = logging.getLogger(__name__)

# Широкий товарный ETF для сглаживания
_BROAD_ETF = "DBC"  # Invesco DB Commodity Index Tracking Fund
_BLEND_ALPHA = 0.70  # 70% EWCI, 30% ETF


class CommodityDynamicsService:
    """Расчёт динамики товарного рынка.

    Parameters
    ----------
    redis_client : Optional[RedisClient]
        Клиент Redis для кэширования.
    """

    def __init__(self, redis_client: Optional[RedisClient] = None):
        self._redis = redis_client

    def compute(self, bars: int = 500) -> dict:
        """Рассчитать композит товарного рынка.

        Parameters
        ----------
        bars : int
            Глубина истории в дневных барах (10..500).

        Returns
        -------
        dict с ключами:
          * composite: { dates, values, smoothed, ema20, current }
          * per_commodity: { TICKER: { close, change_pct, label, unit } }
          * trendlines: поддержка/сопротивление по smoothed
          * meta: { fetched_at, broad_etf, blend_alpha }
        """
        bars = max(10, min(int(bars), 500))
        cache_key_str = commodity_key("dynamics", bars, provider=PROVIDER_YFINANCE)

        if self._redis is not None and self._redis.connected:
            cached = self._redis.get(cache_key_str)
            if cached is not None:
                try:
                    return deserialize_value(cached)
                except Exception:
                    pass

        logger.info("Computing commodity dynamics: %d bars", bars)

        # ── 1. Fetch OHLCV for all commodities ──
        commodity_closes: dict[str, np.ndarray] = {}
        commodity_dates = None

        for ticker in COMMODITY_TICKERS:
            cfg = COMMODITY_ASSETS[ticker]
            yf_sym = cfg["yf_symbol"]
            try:
                df = yfinance_history(yf_sym, period="2y", interval="1d")
                if df is None:
                    logger.warning("No data for %s (%s)", ticker, yf_sym)
                    continue
                closes = df["Close"].values[-bars:]
                if commodity_dates is None:
                    commodity_dates = df.index[-bars:]
                commodity_closes[ticker] = closes
            except Exception as e:
                logger.warning("Fetch failed for %s (%s): %s", ticker, yf_sym, e)

        if not commodity_closes:
            raise RuntimeError("No commodity data available")

        # ── 2. Fetch broad ETF (DBC) for smoothing ──
        etf_norm = None
        try:
            etf_df = yfinance_history(_BROAD_ETF, period="2y", interval="1d")
            if etf_df is not None:
                etf_closes = etf_df["Close"].values[-bars:]
                etf_norm = etf_closes / etf_closes[0] * 100
        except Exception as e:
            logger.warning("Broad ETF %s unavailable: %s", _BROAD_ETF, e)

        # ── 3. Equal-weight composite (EWCI) ──
        n_assets = len(commodity_closes)
        normalized = {}
        for ticker, closes in commodity_closes.items():
            closes = np.nan_to_num(closes, nan=100.0)
            if closes[0] > 0:
                normalized[ticker] = closes / closes[0] * 100
            else:
                normalized[ticker] = np.ones_like(closes) * 100

        # Average across all commodities (equal weight)
        all_norm = np.column_stack(list(normalized.values()))
        ewci = np.mean(all_norm, axis=1)

        # ── 4. Smooth with broad ETF ──
        if etf_norm is not None and len(etf_norm) == len(ewci):
            smoothed = _BLEND_ALPHA * ewci + (1.0 - _BLEND_ALPHA) * etf_norm
        else:
            smoothed = ewci

        # ── 5. EMA20 ──
        ema20 = _ema(smoothed, 20)

        # ── 6. Per-commodity change ──
        per_commodity = {}
        for ticker in COMMODITY_TICKERS:
            cfg = COMMODITY_ASSETS[ticker]
            closes = commodity_closes.get(ticker)
            if closes is not None and len(closes) >= 2:
                current = float(closes[-1])
                prev = float(closes[-2])
                change_pct = (current / prev - 1.0) * 100 if prev > 0 else 0.0
                per_commodity[ticker] = {
                    "close": None if np.isnan(current) else round(current, 4),
                    "change_pct": None if np.isnan(change_pct) else round(change_pct, 2),
                    "label": cfg["label"],
                    "unit": cfg["unit"],
                    "category": cfg["category"],
                }
            else:
                per_commodity[ticker] = {
                    "close": None,
                    "change_pct": None,
                    "label": cfg["label"],
                    "unit": cfg["unit"],
                    "category": cfg["category"],
                }

        # ── Dates ──
        dates = []
        if commodity_dates is not None:
            for d in commodity_dates:
                dates.append(
                    d.isoformat() if hasattr(d, "isoformat") else str(d)
                )

        # ── Trendlines (simple support/resistance) ──
        trend = _simple_trend(smoothed)
        # Sanitize trend values
        for k in ("support", "resistance", "strength"):
            v = trend.get(k)
            if v is not None and (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                trend[k] = None

        # ── Sanitize NaN/Inf for JSON ──
        def _safe(val):
            if val is None:
                return None
            f = float(val)
            return None if np.isnan(f) or np.isinf(f) else round(f, 4)

        result = {
            "composite": {
                "dates": dates,
                "values": [_safe(v) for v in ewci],
                "smoothed": [_safe(v) for v in smoothed],
                "ema20": [_safe(v) for v in ema20],
                "current": _safe(smoothed[-1]) if len(smoothed) > 0 else None,
            },
            "per_commodity": per_commodity,
            "trend": trend,
            "meta": {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "broad_etf": _BROAD_ETF,
                "blend_alpha": _BLEND_ALPHA,
                "n_assets": n_assets,
                "bars": bars,
            },
        }

        if self._redis is not None and self._redis.connected:
            try:
                self._redis.set(cache_key_str, result, ex=600)
            except Exception:
                pass

        return result


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _simple_trend(series: np.ndarray) -> dict:
    """Определить направление тренда и уровни поддержки/сопротивления."""
    n = len(series)
    if n < 20:
        return {"trend": "NEUTRAL", "support": None, "resistance": None,
                "strength": 0}

    recent = series[-20:]
    sma_short = float(np.mean(series[-10:]))
    sma_long = float(np.mean(recent))

    if sma_short > sma_long * 1.002:
        trend = "BULLISH"
    elif sma_short < sma_long * 0.998:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    support = float(np.min(recent))
    resistance = float(np.max(recent))
    strength = abs(sma_short - sma_long) / max(abs(sma_long), 1e-8) * 100

    return {
        "trend": trend,
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "strength": round(float(strength), 1),
    }
