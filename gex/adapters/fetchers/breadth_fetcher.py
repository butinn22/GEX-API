"""McClellan Summation Index + рыночная ширина — вычисление из данных yfinance.

Использует два источника:
  1. NYSE Composite (^NYA) + SPY для прокси-расчёта ширины.
  2. McClellan Summation Index по формуле (Том МакКлеллан):
     * McClellan Oscillator = EMA19(raw) - EMA39(raw)
       где raw = (Advances - Declines) / (Advances + Declines) * 1000
     * Summation Index = кумулятивная сумма Oscillator

Т.к. данные Advances/Declines недоступны напрямую через yfinance, строим
прокси-ширину через соотношение RSP (equal-weight S&P 500) к SPY
(cap-weight S&P 500). Рост RSP/SPY = ширина рынка растёт (бычий),
падение = ширина сужается (медвежий).

Для истинного McClellan Summation Index используем формулу на основе
прокси-raw:
  raw = daily_pct_change(RSP/SPY) * 1000
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from gex.adapters.providers.yfinance import history as yfinance_history

from gex.adapters.cache.redis_client import RedisClient, deserialize_value

logger = logging.getLogger(__name__)

# Символы для расчёта ширины
_BREADTH_PROXY_TICKER = "RSP"  # equal-weight S&P 500
_BENCHMARK_TICKER = "SPY"       # cap-weight S&P 500
_HISTORY_DAYS = 504             # ~2 года торгов

# Redis-кэш: RSP+SPY (2y) — тяжёлый фетч, кэшируем 10 минут
_CACHE_KEY = "gex:breadth:mcclellan"
_CACHE_TTL = 600
_cache = RedisClient()


@dataclass
class BreadthData:
    """Рыночная ширина и McClellan Summation Index."""

    symbol: str                              # "RSP/SPY"
    dates: list[str]                         # ISO даты
    raw_oscillator: list[float]              # McClellan raw oscillator
    mc_oscillator: list[float]               # McClellan Oscillator (EMA19-EMA39)
    mc_summation_index: list[float]          # McClellan Summation Index
    breadth_ratio: list[float]               # RSP/SPY ratio
    current_oscillator: float                # последнее значение Oscillator
    current_summation: float                 # последнее значение Summation Index
    current_ratio: float                     # последнее значение RSP/SPY
    rsi_14: float                            # RSI(14) Summation Index


def fetch_mcclellan() -> BreadthData | None:
    """Загрузить данные и вычислить McClellan Summation Index + ширину.

    Returns None при ошибке сети / отсутствии данных.
    """
    try:
        if _cache.connected:
            cached = _cache.get(_CACHE_KEY)
            if cached is not None:
                return deserialize_value(cached)
    except Exception:
        pass

    rsp = yfinance_history(_BREADTH_PROXY_TICKER, period="2y")
    spy = yfinance_history(_BENCHMARK_TICKER, period="2y")
    if rsp is None or spy is None:
        logger.warning("Не удалось загрузить RSP/SPY: yfinance не отдал данные")
        return None

    if rsp.empty or spy.empty:
        return None

    # Align индексы
    common = rsp.index.intersection(spy.index)
    if len(common) < 100:
        return None
    rsp = rsp.loc[common]
    spy = spy.loc[common]

    # RSP/SPY ratio = breadth proxy
    ratio = rsp["Close"] / spy["Close"]
    ratio = ratio.dropna()

    # McClellan-like raw oscillator на основе daily %change ratio
    daily_pct = ratio.pct_change().fillna(0) * 1000  # scaled

    # EMA19 и EMA39
    ema19 = daily_pct.ewm(span=19, adjust=False).mean()
    ema39 = daily_pct.ewm(span=39, adjust=False).mean()
    mc_osc = (ema19 - ema39).dropna()

    # McClellan Summation Index = кумулятивная сумма + сдвиг на 1000
    mc_sum = mc_osc.cumsum() + 1000

    # RSI(14) на Summation Index
    delta = mc_sum.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)

    result = BreadthData(
        symbol="RSP/SPY",
        dates=[str(d.date()) for d in mc_sum.index],
        raw_oscillator=[float(v) for v in daily_pct.values[-len(mc_sum):]],
        mc_oscillator=[float(v) for v in mc_osc.values],
        mc_summation_index=[float(v) for v in mc_sum.values],
        breadth_ratio=[float(v) for v in ratio.values[-len(mc_sum):]],
        current_oscillator=float(mc_osc.iloc[-1]),
        current_summation=float(mc_sum.iloc[-1]),
        current_ratio=float(ratio.iloc[-1]),
        rsi_14=float(rsi.iloc[-1]),
    )
    try:
        if _cache.connected:
            _cache.set(_CACHE_KEY, result, ex=_CACHE_TTL)
    except Exception:
        pass
    return result
