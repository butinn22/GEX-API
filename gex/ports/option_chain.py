"""Порт получения опционной цепочки.

Сегодня цепочку добывают четыре разных пути (``yf_fetcher``, ``bybit_fetcher``, ``webull_fetcher``,
``moex_fetcher`` плюс роутеры напрямую — аудит 03: EC-7, 05: F-05). Порт сводит это к одному
контракту, чтобы реализацию можно было подменить на шлюз с лимитами и кэшем.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ChainSnapshot", "OptionChainPort"]


@dataclass(frozen=True)
class ChainSnapshot:
    """Снимок цепочки на момент ``as_of``.

    ``chain`` — DataFrame с колонками ``strike``, ``type`` (``"C"``/``"P"``), ``oi``, ``iv``,
    ``T`` (годы до экспирации) и, по возможности, ``expiry``; ``spot`` — цена базового актива.
    ``source`` фиксируется явно: он нужен и для кэш-ключа, и для диагностики «откуда числа».
    """

    symbol: str
    spot: float
    as_of: Any  # datetime — Any, чтобы порт не тянул datetime-типизацию в рантайм
    source: str
    chain: "pd.DataFrame"

    def __len__(self) -> int:
        return len(self.chain)


@runtime_checkable
class OptionChainPort(Protocol):
    """Источник опционной цепочки: одна реализация на провайдера, подменяемая через DI.

    Требование к реализациям: ``max_expiries`` входит в кэш-ключ и в бюджет запросов к провайдеру
    (аудит 03: EC-8 — ключ ``/gexcone`` его терял).
    """

    def fetch(self, symbol: str, *, max_expiries: int = 5) -> ChainSnapshot:
        """Получить снимок цепочки; вместо исключений — ``ChainSnapshot`` с пустой цепочкой."""
        ...
