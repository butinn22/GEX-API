"""Слой хранения опционных цепочек.

Архитектурное решение
---------------------
Бизнес-логика (pipeline/service) **не знает**, где хранятся цепочки — она зависит
от абстракции :class:`ChainRepository` (typing.Protocol). Сейчас реализован
in-memory вариант; позже добавляется :class:`SqliteChainRepository` на
SQLAlchemy/SQLite — **без изменения service-слоя**.

Потоковая модель
----------------
- **Sync-хендлеры FastAPI** выполняются в thread pool (AnyIO).
  ``threading.Lock`` корректен для этого режима — защищает от гонки между потоками.
- **Async-методы** (``aget``, ``aput``, ...) используют ``asyncio.Lock`` — готовы
  к будущей миграции хендлеров на ``async def``.
- **Не смешивать:** sync-методы нельзя вызывать из event loop (блокировка),
  async-методы нельзя вызывать из thread pool (нет event loop).

Точки расширения под ORM
~~~~~~~~~~~~~~~~~~~~~~~~
Классы :class:`OptionContractORM` / :class:`SnapshotMetaORM` внизу файла —
готовая схема таблиц (SQLAlchemy 2.0 style). Достаточно создать engine и
применить ``Base.metadata.create_all(engine)``. Маппинг ORM ↔ dataclass уже
описан в :class:`SqliteChainRepository` (пока как заглушка-ориентир).
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable
from collections.abc import Iterable

import pandas as pd

from gex.domain.data_loader import OptionSnapshot


# ====================================================================== #
#  Абстракция хранилища
# ====================================================================== #
@runtime_checkable
class ChainRepository(Protocol):
    """Контракт: загрузить/сохранить опционную цепочку по тикеру.

    Реализации: :class:`InMemoryRepository`, будущая :class:`SqliteChainRepository`.

    Sync-методы (``get``, ``put``, ...) — для thread-pool хендлеров.
    Async-методы (``aget``, ``aput``, ...) — для async-хендлеров.
    Реализация обязана предоставить хотя бы один набор.
    """

    def get(self, ticker: str) -> Optional[OptionSnapshot]:
        """Вернуть последний снапшот по тикеру или ``None`` (sync)."""
        ...

    def put(self, ticker: str, snapshot: OptionSnapshot) -> None:
        """Сохранить (или перезаписать) снапшот по тикеру (sync)."""
        ...

    def list_tickers(self) -> list[str]:
        """Список тикеров, по которым есть данные (sync)."""
        ...

    def delete(self, ticker: str) -> bool:
        """Удалить тикер. Вернуть ``True`` если что-то было удалено (sync)."""
        ...

    # ── Async variants (optional, for future async migration) ───────
    async def aget(self, ticker: str) -> Optional[OptionSnapshot]:
        """Async-версия get()."""
        ...

    async def aput(self, ticker: str, snapshot: OptionSnapshot) -> None:
        """Async-версия put()."""
        ...

    async def alist_tickers(self) -> list[str]:
        """Async-версия list_tickers()."""
        ...

    async def adelete(self, ticker: str) -> bool:
        """Async-версия delete()."""
        ...


# ====================================================================== #
#  In-memory реализация (дефолт)
# ====================================================================== #
@dataclass
class InMemoryRepository:
    """Потокобезопасная in-memory реализация для dev/тестов/демо.

    Хранит только последний снапшот на тикер (история добавляется через SQLite).

    Threading model
    --------------
    - **Sync methods** (``get``, ``put``, ...) используют ``threading.Lock``.
      Безопасны для вызова из thread pool (текущие sync-хендлеры FastAPI).
      **Не вызывать из event loop** — блокируют его.
    - **Async methods** (``aget``, ``aput``, ...) используют ``asyncio.Lock``.
      Безопасны для вызова из event loop (будущие async-хендлеры).
      **Не вызывать из thread pool** — нет event loop.
    """

    _store: dict[str, OptionSnapshot] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _alock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ── Sync API (thread-safe, для sync-хендлеров) ──────────────────

    def get(self, ticker: str) -> Optional[OptionSnapshot]:
        with self._lock:
            return self._store.get(ticker.upper())

    def put(self, ticker: str, snapshot: OptionSnapshot) -> None:
        with self._lock:
            self._store[ticker.upper()] = snapshot

    def list_tickers(self) -> list[str]:
        with self._lock:
            return sorted(self._store.keys())

    def delete(self, ticker: str) -> bool:
        with self._lock:
            return self._store.pop(ticker.upper(), None) is not None

    # ── Async API (event-loop-safe, для async-хендлеров) ───────────

    async def aget(self, ticker: str) -> Optional[OptionSnapshot]:
        async with self._alock:
            return self._store.get(ticker.upper())

    async def aput(self, ticker: str, snapshot: OptionSnapshot) -> None:
        async with self._alock:
            self._store[ticker.upper()] = snapshot

    async def alist_tickers(self) -> list[str]:
        async with self._alock:
            return sorted(self._store.keys())

    async def adelete(self, ticker: str) -> bool:
        async with self._alock:
            return self._store.pop(ticker.upper(), None) is not None


# ====================================================================== #
#  SQLite/SQLAlchemy — заготовка под ORM (раскомментировать при подключении)
# ====================================================================== #
# from sqlalchemy import String, Float, DateTime, ForeignKey, Integer
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
#
# class Base(DeclarativeBase):
#     pass
#
# class SnapshotMetaORM(Base):
#     __tablename__ = "snapshots"
#     id: Mapped[int] = mapped_column(primary_key=True)
#     ticker: Mapped[str] = mapped_column(String(16), index=True)
#     spot: Mapped[float] = mapped_column(Float)
#     as_of: Mapped[datetime] = mapped_column(DateTime, index=True)
#     contracts: Mapped[list["OptionContractORM"]] = relationship(
#         back_populates="snapshot", cascade="all, delete-orphan")
#
# class OptionContractORM(Base):
#     __tablename__ = "option_contracts"
#     id: Mapped[int] = mapped_column(primary_key=True)
#     snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"), index=True)
#     strike: Mapped[float] = mapped_column(Float)
#     type:   Mapped[str] = mapped_column(String(1))
#     oi:     Mapped[float] = mapped_column(Float)
#     iv:     Mapped[float] = mapped_column(Float)
#     T:      Mapped[float] = mapped_column(Float)
#     snapshot: Mapped["SnapshotMetaORM"] = relationship(back_populates="contracts")
#
#
# class SqliteChainRepository:
#     """Репозиторий на SQLite. Включается одной подменой в main.py.
#
#     engine = create_engine("sqlite:///gex.db")
#     Base.metadata.create_all(engine)
#     repo = SqliteChainRepository(engine)
#     """
#     def __init__(self, engine):
#         self._engine = engine
#         from sqlalchemy.orm import sessionmaker
#         self._Session = sessionmaker(bind=engine)
#
#     def get(self, ticker: str) -> Optional[OptionSnapshot]:
#         ...  # SELECT последнего snapshot_meta + JOIN option_contracts → OptionSnapshot
#
#     def put(self, ticker: str, snapshot: OptionSnapshot) -> None:
#         ...  # INSERT SnapshotMetaORM + bulk INSERT OptionContractORM
#
#     def list_tickers(self) -> list[str]:
#         ...  # SELECT DISTINCT ticker
#
#     def delete(self, ticker: str) -> bool:
#         ...  # DELETE WHERE ticker=...
