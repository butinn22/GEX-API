"""UserDashboardSettings — персональные настройки дашборда (на пользователя).

Хранит JSON-блоб: включённые EMA, выбранные инструменты и кастомные
трендовые линии (до 5 на тикер), нарисованные пользователем руками.

Одна строка на пользователя (unique user_id).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base


class UserDashboardSettings(Base):
    """Персональные настройки дашборда (EMA / инструменты / линии)."""

    __tablename__ = "user_dashboard_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # {"emas": [20, 50], "instruments": ["SPY", "QQQ"], "trendlines": {"SPY": [...]}}
    data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<UserDashboardSettings(user={self.user_id})>"
