"""Порт доступа к рыночным данным: свечи и спот.

Контракт описывает то, что домен/application ожидают от внешнего мира, и **ничего** о том, как это
добывается (yfinance, Bybit, MOEX ISS, кэш, шлюз). Реализации живут в ``gex/adapters/providers/**``.

Ключевое требование, зашитое в контракт: у каждой операции есть **timeframe** и **глубина истории**,
потому что кэш-ключ и лимиты провайдера зависят именно от них (аудит 05: F-03, F-09).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pandas нужен только для аннотаций — в рантайме порт от него не зависит
    import pandas as pd

__all__ = ["MarketDataPort", "OhlcvMap"]

#: Результат мульти-таймфрейм-загрузки: имя таймфрейма → DataFrame (колонки Open/High/Low/Close/Volume).
OhlcvMap = "dict[str, pd.DataFrame]"


@runtime_checkable
class MarketDataPort(Protocol):
    """Единая точка получения OHLCV для всех потребителей (TA, сканеры, конусы, бэктест)."""

    def fetch_timeframes(self, ticker: str, *, limit: int | None = None) -> OhlcvMap:
        """Свечи по нескольким таймфреймам сразу (для графиков и мульти-ТФ анализа)."""
        ...

    def fetch_single(self, ticker: str, timeframe: str, *, limit: int = 400) -> "pd.DataFrame":
        """Свечи одного таймфрейма (график/одиночный расчёт)."""
        ...

    def fetch_spot(self, ticker: str) -> float | None:
        """Последняя цена (может быть недоступна — тогда ``None``, а не исключение)."""
        ...
