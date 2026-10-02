"""Статистика админки: сборка ответа без SQL и без HTTP (ring: application).

Что здесь
---------
Логика «что считать»: окна времени, zero-fill серии регистраций, значения по умолчанию
для статусов подписки, округление выручки, курс валюты. Запросы приходят репозиториями
(:mod:`gex.application.auth.repositories`), поэтому это проверяется юнит-тестами — с
подставными репозиториями и подставными часами, без БД и без FastAPI.

Почему окна времени считаются здесь, а не в запросах
---------------------------------------------------
«Новые за 24 часа» и «истекают за 7 дней» — это продуктовые решения, а не свойства SQL.
В репозитории уходит уже готовый момент времени (``since``/``now``), и тест может задать
его явно: иначе проверка зависела бы от текущих часов и падала бы на границе суток.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

__all__ = ["AdminStats", "AdminStatsService", "REGISTRATION_DAYS"]

#: Глубина серии регистраций (дней) — то, что рисует график в админке.
REGISTRATION_DAYS = 14

#: Горизонт «истекает скоро» (дней).
EXPIRING_DAYS = 7


@dataclass
class AdminStats:
    """Собранная статистика: ровно то, что уходит в ответ API."""

    total_users: int = 0
    active_subscriptions: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    by_provider: dict[str, int] = field(default_factory=dict)
    new_users_24h: int = 0
    new_users_7d: int = 0
    expiring_7d: int = 0
    registrations_14d: list[dict] = field(default_factory=list)
    payments_total: int = 0
    payments_pending: int = 0
    payments_by_status: dict[str, int] = field(default_factory=dict)
    revenue_total_usd: float = 0.0
    revenue_30d_usd: float = 0.0
    usd_rub_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "total_users": self.total_users,
            "active_subscriptions": self.active_subscriptions,
            "by_status": dict(self.by_status),
            "by_provider": dict(self.by_provider),
            "new_users_24h": self.new_users_24h,
            "new_users_7d": self.new_users_7d,
            "expiring_7d": self.expiring_7d,
            "registrations_14d": list(self.registrations_14d),
            "payments_total": self.payments_total,
            "payments_pending": self.payments_pending,
            "payments_by_status": dict(self.payments_by_status),
            "revenue_total_usd": self.revenue_total_usd,
            "revenue_30d_usd": self.revenue_30d_usd,
            "usd_rub_rate": self.usd_rub_rate,
        }


class AdminStatsService:
    """Собирает статистику из репозиториев.

    ``usd_rub_rate`` — функция, а не значение: курс берётся у платёжного сервиса, и он
    может быть недоступен. Если он падает, статистика обязана отдаться **без** курса,
    а не упасть целиком: админка нужна именно в такие моменты.
    """

    def __init__(
        self,
        users: Any,
        payments: Any,
        *,
        usd_rub_rate: Optional[Callable[[], float]] = None,
        clock: Callable[[], datetime] = None,
        registration_days: int = REGISTRATION_DAYS,
    ) -> None:
        self._users = users
        self._payments = payments
        self._rate = usd_rub_rate
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._registration_days = max(int(registration_days), 1)

    def collect(self) -> AdminStats:
        now = self._clock()
        registrations = self._users.registrations_by_day(self._registration_days, now=now)

        return AdminStats(
            total_users=self._users.total(),
            active_subscriptions=self._users.active_count(),
            by_status=self._users.by_status(),
            by_provider=self._users.by_provider(),
            new_users_24h=self._users.new_since(now - timedelta(days=1)),
            new_users_7d=self._users.new_since(now - timedelta(days=7)),
            expiring_7d=self._users.expiring_within(timedelta(days=EXPIRING_DAYS), now=now),
            registrations_14d=self._fill_days(registrations, now),
            payments_total=self._payments.total(),
            payments_pending=self._payments.pending_count(),
            payments_by_status=self._payments.by_status(),
            revenue_total_usd=round(self._payments.revenue(), 2),
            revenue_30d_usd=round(self._payments.revenue(since=now - timedelta(days=30)), 2),
            usd_rub_rate=self._rate_value(),
        )

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _fill_days(self, counts: dict[str, int], now: datetime) -> list[dict]:
        """Ровная серия за N дней, включая дни без регистраций.

        Пустые дни обязательны: график, который «схлопывает» их, врёт о динамике —
        затишье выглядит как непрерывный рост.
        """
        out: list[dict] = []
        for i in range(self._registration_days - 1, -1, -1):
            day = (now - timedelta(days=i)).date().isoformat()
            out.append({"date": day, "count": int(counts.get(day, 0))})
        return out

    def _rate_value(self) -> float:
        if self._rate is None:
            return 0.0
        try:
            return round(float(self._rate()), 2)
        except Exception:  # noqa: BLE001 — курс недоступен: отдаём статистику без него
            return 0.0
