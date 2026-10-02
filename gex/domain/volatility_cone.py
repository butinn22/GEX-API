"""
Quarterly Volatility Cone (VWAP + RSI + Bollinger Bands) + VWAP Price Channel.

Порт Pine Script индикатора «Quarterly Volatility Cone (VWAP + RSI + BB)»
и «VWAP Price Channel» на Python (pandas/numpy). Только расчёты, без
визуализации.

Использование
-------------
>>> from gex.domain.volatility_cone import compute_volatility_cone, compute_vpc
>>> df = pd.DataFrame(...)  # OHLCV + datetime index
>>> result = compute_volatility_cone(df)
>>> result['upper_2sd']     # pandas Series
>>> vpc = compute_vpc(df, length=20)
>>> vpc['vpc_upper']        # pandas Series

Архитектура
-----------
Все вычисления — чистые функции без побочных эффектов. На вход DataFrame,
на выход DataFrame/дикт Series. Потокобезопасны.

Визуализация (НЕ реализована в этом модуле — см. DOCSTRING в конце файла).
"""

# --------------------------------------------------------------------------- #
#  Разбор файла (итерация 41)
# --------------------------------------------------------------------------- #
# Расчёт разложен: `domain/indicators/quarterly_cone.py` (дневная волатильность, границы
# кварталов, гибридный источник, сам конус с полосами σ1/σ2 и коррекциями) и
# `domain/indicators/vpc.py` (VWAP Price Channel).
#
# Здесь остаются `compute_all` (роутер зовёт именно его) и реэкспорт 20 имён, поэтому
# `from gex.domain.volatility_cone import ...` у вызывающих не меняется.
#
# Дубли закрыты: `compute_rsi_wilder` делегирует каноническому ядру
# `domain/indicators/rsi.wilder_rsi` с политиками этого модуля (`warmup='nan'`,
# `flat='hundred'`) — паритет измерен (0.0 на всей длине, совпадает и число NaN).
# `compute_heikin_ashi` перенесён дословно: у ядра `candles.heikin_ashi` seed по умолчанию
# `midpoint`, а здесь нужен сид от `open`, и подменять одно другим без измерения нельзя.
#
# Числа закреплены эталоном `tests/test_volatility_cone_golden.py` (пять прогонов: полный
# кадр, кадр с не-дефолтными параметрами, конус отдельно, конус без коррекции, VPC).

from __future__ import annotations

from typing import Optional
import pandas as pd

from gex.domain.indicators.quarterly_cone import (
    DEFAULT_BB_MULT,
    DEFAULT_CARRY_WEIGHT,
    DEFAULT_CORRECTION_PCT,
    DEFAULT_EMA_LEN,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_RSI_1SD_BOOST,
    DEFAULT_RSI_INFLUENCE,
    DEFAULT_RSI_LENGTH,
    DEFAULT_SD1_MULT,
    DEFAULT_SD2_MULT,
    DEFAULT_USE_CORRECTION,
    DEFAULT_VWAP_INFLUENCE,
    compute_daily_volatility,
    compute_heikin_ashi,
    compute_hybrid_source,
    compute_rsi_wilder,
    compute_volatility_cone,
    detect_quarters,
)
from gex.domain.indicators.vpc import (
    DEFAULT_VPC_LENGTH,
    compute_vpc,
)


__all__ = [
    "DEFAULT_BB_MULT",
    "DEFAULT_CARRY_WEIGHT",
    "DEFAULT_CORRECTION_PCT",
    "DEFAULT_EMA_LEN",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_RSI_1SD_BOOST",
    "DEFAULT_RSI_INFLUENCE",
    "DEFAULT_RSI_LENGTH",
    "DEFAULT_SD1_MULT",
    "DEFAULT_SD2_MULT",
    "DEFAULT_USE_CORRECTION",
    "DEFAULT_VPC_LENGTH",
    "DEFAULT_VWAP_INFLUENCE",
    "compute_all",
    "compute_daily_volatility",
    "compute_heikin_ashi",
    "compute_hybrid_source",
    "compute_rsi_wilder",
    "compute_volatility_cone",
    "compute_vpc",
    "detect_quarters",
]


def compute_all(
    df: pd.DataFrame,
    cone_params: Optional[dict] = None,
    vpc_length: int = DEFAULT_VPC_LENGTH,
) -> pd.DataFrame:
    """Compute Volatility Cone + VWAP Price Channel in one pass.

    Args:
        df: OHLCV DataFrame.
        cone_params: Dict of params for :func:`compute_volatility_cone`.
        vpc_length: Window length for VPC.

    Returns:
        DataFrame with all columns from both functions.
    """
    if cone_params is None:
        cone_params = {}
    result = compute_volatility_cone(df, **cone_params)
    result = compute_vpc(result, length=vpc_length)
    return result
