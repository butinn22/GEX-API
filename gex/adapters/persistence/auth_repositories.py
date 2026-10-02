"""Репозитории админки: SQLAlchemy-запросы за портом (ring: adapters).

Почему адаптер, а не application
--------------------------------
Первая версия этого модуля лежала в ``gex/application/auth/`` и **нарушала R3**: репозиторий
с SQLAlchemy — это инфраструктура, а ``application/**`` не имеет права импортировать ни
``sqlalchemy``, ни ``gex.adapters``. Правило R3 и поймало нарушение (2 вхождения: ``sqlalchemy``
и ``sqlalchemy.func``). Раскладку подсказала сама архитектура: пакет
``gex/adapters/persistence/`` был объявлен ровно для «SQLAlchemy-реализаций репозиториев»
и до сих пор пустовал.

Что было до
-----------
Запросы жили прямо в обработчиках админки: ``db.query(User).count()``, пять группировок,
подсчёты по датам, суммы по платежам. Последствия: роутер нельзя было проверить без БД;
запросы повторялись (подсчёт платежей с нужным статусом — трижды с разными условиями);
и «нет SQL в роутерах» не проверялось, потому что SQL был не виден при чтении HTTP-кода.

Разделение
----------
Здесь — только запросы. Решения «что считать» живут в
:mod:`gex.application.auth.admin_stats` и проверяются юнит-тестами без БД. Контракт, на
который опирается use-case, описан протоколами в :mod:`gex.ports.persistence`, поэтому
application не знает ни о SQLAlchemy, ни об этих классах.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from gex.auth.models import SUBSCRIPTION_VALUES, SubscriptionStatus

__all__ = ["PaymentRepository", "UserRepository"]


class UserRepository:
    """Запросы по пользователям (статистика и постраничный список)."""

    def __init__(self, session: Any, *, status_values: Optional[Iterable[str]] = None) -> None:
        self._session = session
        # Список статусов — из модели, а не второй копией рядом с SQL: копия разошлась бы
        # в тот день, когда в модель добавят новый статус, и админка молча не показала бы его.
        self._status_values = tuple(status_values or SUBSCRIPTION_VALUES)

    # ------------------------------------------------------------------ #
    #  Точечные выборки
    # ------------------------------------------------------------------ #
    def by_id(self, user_id: str) -> Optional[Any]:
        """Пользователь по идентификатору или ``None``.

        Строка «выбрать по id и, если пусто, отдать 404» повторялась в роутере девять раз —
        с одинаковым телом и разными формулировками сообщения. Здесь остаётся только выборка;
        решение «404 это или 403» принимает HTTP-слой, потому что это его предмет.
        """
        if not user_id:
            return None
        user = self._user()
        return self._session.query(user).filter(user.id == user_id).first()

    def by_email(self, email: str) -> Optional[Any]:
        """Пользователь по email (без учёта регистра) или ``None``.

        Регистр не важен: email — это идентификатор для входа, и ``Ivan@Example.com`` в базе
        не должен означать «пользователя нет» при импорте или повторной отправке письма.
        """
        if not email:
            return None
        user = self._user()
        return self._session.query(user).filter(user.email.ilike(email.strip())).first()

    def all_ordered(self) -> list[Any]:
        """Все пользователи от новых к старым (выгрузка в CSV).

        Выгрузка намеренно без пагинации: это архив для админа, а не экран со страницами.
        """
        user = self._user()
        return list(self._session.query(user).order_by(user.created_at.desc()).all())

    # ------------------------------------------------------------------ #
    #  Счётчики
    # ------------------------------------------------------------------ #
    def total(self) -> int:
        return int(self._session.query(self._user()).count())

    def inactive_status(self) -> str:
        """Значение «подписка неактивна» — константа модели, а не строка рядом с SQL.

        ``SubscriptionStatus`` — это пространство имён со строками, а **не** ``Enum``:
        обращения к ``.value`` здесь нет, потому что у строки его не существует. Первая
        версия писала ``SubscriptionStatus.INACTIVE.value`` и работала лишь потому, что
        широкий ``except Exception`` подменял ``AttributeError`` на строковый литерал —
        то есть ошибка в предположении о типе была скрыта обработчиком.
        """
        return str(SubscriptionStatus.INACTIVE)

    def active_count(self) -> int:
        user = self._user()
        return int(
            self._session.query(user)
            .filter(user.subscription_status != self.inactive_status())
            .count()
        )

    def by_status(self) -> dict[str, int]:
        from sqlalchemy import func

        user = self._user()
        counts = {status: 0 for status in self._status_values}
        for status, count in (
            self._session.query(user.subscription_status, func.count(user.id))
            .group_by(user.subscription_status)
            .all()
        ):
            if status in counts:
                counts[status] = int(count)
        return counts

    def by_provider(self) -> dict[str, int]:
        from sqlalchemy import func

        user = self._user()
        out: dict[str, int] = {}
        for provider, count in (
            self._session.query(user.oauth_provider, func.count(user.id))
            .group_by(user.oauth_provider)
            .all()
        ):
            # Пользователи без OAuth регистрировались по email — так их и называем.
            out[provider or "email"] = int(count)
        return out

    def new_since(self, since: datetime) -> int:
        from sqlalchemy import func

        user = self._user()
        return int(
            self._session.query(func.count(user.id)).filter(user.created_at >= since).scalar() or 0
        )

    def expiring_within(self, horizon: timedelta, *, now: Optional[datetime] = None) -> int:
        """Сколько активных подписок истекает в ближайшие ``horizon`` (и ещё не истекли)."""
        from sqlalchemy import func

        user = self._user()
        now = now or datetime.now(timezone.utc)
        return int(
            self._session.query(func.count(user.id))
            .filter(
                user.subscription_expires_at.isnot(None),
                user.subscription_expires_at <= now + horizon,
                user.subscription_expires_at > now,
                user.subscription_status != self.inactive_status(),
            )
            .scalar()
            or 0
        )

    def registrations_by_day(self, days: int, *, now: Optional[datetime] = None) -> dict[str, int]:
        """Регистрации по датам за последние ``days`` дней (только непустые даты)."""
        from sqlalchemy import func

        from gex.adapters.persistence.database import active_dialect

        user = self._user()
        now = now or datetime.now(timezone.utc)
        if active_dialect() == "postgresql":
            day_expr = func.date(func.timezone("UTC", user.created_at))
        else:
            day_expr = func.date(user.created_at)
        rows = (
            self._session.query(day_expr.label("d"), func.count(user.id).label("cnt"))
            .filter(user.created_at >= now - timedelta(days=days))
            .group_by("d")
            .all()
        )
        return {str(day): int(count) for day, count in rows}

    # ------------------------------------------------------------------ #
    #  Постраничный список
    # ------------------------------------------------------------------ #
    def page(
        self,
        *,
        search: Optional[str] = None,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Any], int]:
        """(строки страницы, всего подходящих). Фильтры применяются до пагинации."""
        user = self._user()
        query = self._session.query(user)
        if search:
            query = query.filter(user.email.ilike(f"%{search.strip()}%"))
        if status:
            query = query.filter(user.subscription_status == status)
        total = int(query.count())
        rows = query.order_by(user.created_at.desc()).offset(offset).limit(limit).all()
        return rows, total

    @staticmethod
    def _user():
        from gex.auth.models import User

        return User


class PaymentRepository:
    """Запросы по платежам: счётчики и выручка.

    Выручка считается **только по подтверждённым** платежам: считать по всему, что лежит
    в таблице, значило бы показывать в админке деньги, которых не поступало.
    """

    #: Статусы, которые считаются «в процессе» (оплата начата, но не подтверждена).
    PENDING_STATUSES: tuple[str, ...] = ("PENDING", "PAID_CLIENT")

    #: Статус подтверждённого платежа.
    CONFIRMED = "CONFIRMED"

    def __init__(self, session: Any, *, model: Any = None) -> None:
        self._session = session
        self._model = model

    def total(self) -> int:
        from sqlalchemy import func

        model = self._model_()
        return int(self._session.query(func.count(model.id)).scalar() or 0)

    def pending_count(self) -> int:
        from sqlalchemy import func

        model = self._model_()
        return int(
            self._session.query(func.count(model.id))
            .filter(model.status.in_(self.PENDING_STATUSES))
            .scalar()
            or 0
        )

    def by_status(self) -> dict[str, int]:
        from sqlalchemy import func

        model = self._model_()
        return {
            str(status): int(count)
            for status, count in (
                self._session.query(model.status, func.count(model.id))
                .group_by(model.status)
                .all()
            )
        }

    def revenue(self, *, since: Optional[datetime] = None) -> float:
        """Сумма подтверждённых платежей (за всё время или с указанного момента)."""
        from sqlalchemy import func

        model = self._model_()
        query = self._session.query(func.coalesce(func.sum(model.amount_usd), 0.0)).filter(
            model.status == self.CONFIRMED
        )
        if since is not None:
            query = query.filter(model.confirmed_at >= since)
        return float(query.scalar() or 0.0)

    def _model_(self):
        if self._model is not None:
            return self._model
        from gex.auth.payment_models import Payment

        return Payment

    # ------------------------------------------------------------------ #
    #  Точечные выборки и список
    # ------------------------------------------------------------------ #
    def by_id(self, payment_id: str) -> Optional[Any]:
        """Платёж по идентификатору или ``None`` (импорт ищет существующую запись)."""
        if not payment_id:
            return None
        model = self._model_()
        return self._session.query(model).filter(model.id == payment_id).first()

    def all_ordered(self) -> list[Any]:
        """Все платежи от новых к старым (выгрузка без пагинации)."""
        model = self._model_()
        return list(self._session.query(model).order_by(model.created_at.desc()).all())

    def page(
        self,
        *,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Any], int]:
        """(страница платежей, всего подходящих). Фильтр применяется до пагинации.

        Как и у пользователей: ``total`` — число подходящих, иначе в пагинации появится
        страница, которая открывается пустой.
        """
        model = self._model_()
        query = self._session.query(model)
        if status:
            query = query.filter(model.status == status)
        total = int(query.count())
        rows = query.order_by(model.created_at.desc()).offset(offset).limit(limit).all()
        return list(rows), total
