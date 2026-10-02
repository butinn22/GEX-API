"""Порты персистентности: что слой приложения ожидает от хранилища (ring: ports).

Зачем порт там, где всё работало и без него
-------------------------------------------
``application/auth/admin_stats.py`` принимает «репозитории» и вызывает у них методы —
то есть опирается на контракт, который нигде не был записан. Duck typing здесь дешевле
протокола ровно до первого расхождения: пока реализации жили рядом, в ``application/auth``,
разойтись они не могли (и, как выяснилось, репозиторий там же и **нарушал R3**, потому что
SQLAlchemy — инфраструктура). Теперь реализация — адаптер
(``gex/adapters/persistence/auth_repositories.py``), и контракт обязан быть явным: иначе
связь между слоями держится на памяти автора.

Протоколы ниже описывают **только** то, что действительно нужно use-case'ам. Это не
описание таблиц: ``UserReader`` не знает ни о SQLAlchemy, ни о схеме, ни о ``User``.
Такой контракт может выполнить и ORM-репозиторий, и кэш, и подставной объект в тесте —
что и делает ``tests/test_admin_and_sec.py`` (там ``FakeUsers``/``FakePayments``).

Проверка соответствия — в ``tests/test_admin_and_sec.py``: реализации из адаптера сверяются
с протоколами по именам методов, поэтому «добавили метод в адаптер и забыли в протоколе»
или наоборот видно сразу.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = ["PaymentReader", "UserReader"]


@runtime_checkable
class UserReader(Protocol):
    """Чтение пользователей для статистики и постраничного списка.

    ``Optional`` в возвратах — не формальность: ``by_id``/``by_email`` отвечают «нет такого»,
    и решение, что с этим делать (404, пропустить строку импорта), принимает вызывающий.
    """

    def total(self) -> int: ...

    def active_count(self) -> int: ...

    def inactive_status(self) -> str: ...

    def by_status(self) -> dict[str, int]: ...

    def by_provider(self) -> dict[str, int]: ...

    def new_since(self, since: datetime) -> int: ...

    def expiring_within(self, horizon: timedelta, *, now: Optional[datetime] = None) -> int: ...

    def registrations_by_day(self, days: int, *, now: Optional[datetime] = None) -> dict[str, int]: ...

    def by_id(self, user_id: str) -> Optional[Any]: ...

    def by_email(self, email: str) -> Optional[Any]: ...

    def all_ordered(self) -> list[Any]: ...

    def page(
        self,
        *,
        search: Optional[str] = None,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Any], int]: ...


@runtime_checkable
class PaymentReader(Protocol):
    """Чтение платежей: счётчики, разбивка по статусам и выручка.

    Оговорка про выручку живёт в реализации, но контракт её фиксирует: ``revenue``
    считает **подтверждённые** платежи, с необязательным окном по моменту подтверждения.
    """

    def total(self) -> int: ...

    def pending_count(self) -> int: ...

    def by_status(self) -> dict[str, int]: ...

    def revenue(self, *, since: Optional[datetime] = None) -> float: ...

    def by_id(self, payment_id: str) -> Optional[Any]: ...

    def all_ordered(self) -> list[Any]: ...

    def page(
        self,
        *,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Any], int]: ...
