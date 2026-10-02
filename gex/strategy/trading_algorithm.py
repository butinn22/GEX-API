"""Combined EMA Multi-Filter + ADL/MACD/Two-Pole Filter Strategy (точка входа).

Полный порт Pine Script v5 стратегии, объединяющей две подстратегии через OR:

  **Strategy A** (EMA-based): вход по novelsrc > EMA10/20/77/200 + VWAP-контекст.
  **Strategy B** (ADL-based): вход по adline > adl50/ad/adl200 + MACD>0/Signal>0 > tl + Two-Pole Filter.
  **Combined**: combinedLongEntry = longEntryA or longEntryB (аналогично для short/exit/add).

Модуль был 1 447 строк и 46 методов в одном классе (итерация 37 разложила его по предмету,
см. :mod:`gex.strategy`). Здесь остаётся **точка входа**: имя класса, его состав из миксинов
и реэкспорт имён, которые импортируют другие модули и тесты. Ни один вызов снаружи не
изменился, а поведение зафиксировано golden-эталоном до выноса.

Порядок баз важен не для переопределения (имена не пересекаются), а для читаемости:
решение → сигналы → фичи → режим → риск → примитивы.
"""
from __future__ import annotations

from gex.strategy.decision import _DecisionMixin
from gex.strategy.features import _FeaturesMixin
from gex.strategy.indicators import (
    _HAVE_NUMBA,
    _IndicatorMixin,
    _bb,
    _cumsum_reset,
    _two_pole_filter,
    _two_pole_filter_vectorized,
)
from gex.strategy.ports import GEXContext
from gex.strategy.regime import _RegimeMixin
from gex.strategy.risk import _RiskMixin
from gex.strategy.settings import SignalAction, StrategySettings, TradingSignal, TradingState
from gex.strategy.signals import _SignalsMixin

__all__ = [
    "EMAFilterTrendStrategy",
    "GEXContext",
    "SignalAction",
    "StrategySettings",
    "TradingSignal",
    "TradingState",
    "_HAVE_NUMBA",
    "_bb",
    "_cumsum_reset",
    "_two_pole_filter",
    "_two_pole_filter_vectorized",
]


# ====================================================================== #
#  Совмещённая стратегия (Combined EMF MF + ADL)
# ====================================================================== #
class EMAFilterTrendStrategy(
    _DecisionMixin,
    _SignalsMixin,
    _FeaturesMixin,
    _RegimeMixin,
    _RiskMixin,
    _IndicatorMixin,
):
    """EMF MF + ADL STRAT (Combined Strategy A ∪ Strategy B).

    Реализация разложена по миксинам (``gex.strategy.*``); здесь — только состояние.
    """

    #: Человекочитаемое имя стратегии. Атрибут класса, а не метод, поэтому при выносе
    #: миксинов его легко потерять — на этом и споткнулся первый прогон
    #: ``tests/test_trading_algorithm.py::test_strategy_initialization``.
    name = "EMF MF + ADL STRAT (Combined)"

    def __init__(self, settings: StrategySettings | None = None):
        self.settings = settings or StrategySettings()
