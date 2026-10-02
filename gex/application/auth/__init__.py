"""Слой приложения для админки: use-case'ы без SQL и HTTP (ring: application).

Репозитории здесь **не** экспортируются: их реализации — инфраструктура
(``gex/adapters/persistence/auth_repositories.py``), а application опирается на протоколы из
:mod:`gex.ports.persistence`. Экспорт реализации отсюда снова затянул бы SQLAlchemy в граф
импортов application и нарушил R3.
"""

from gex.application.auth.admin_stats import AdminStats, AdminStatsService

__all__ = ["AdminStats", "AdminStatsService"]
