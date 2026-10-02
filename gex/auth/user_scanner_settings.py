"""UserScannerSettings — персональные настройки авто-сканера (на пользователя).

Хранит JSON-блоб: слайдеры верификации боковика (общий + отдельные для
ATR / BBW / %-движения), пороги вердикта, флаг отсечения сигналов и личный
набор отслеживаемых тикеров по каждому универсуму (us / ru / crypto / fx).

Отдельная таблица (а не блоб дашборда) — чтобы PUT настроек дашборда не
затирал настройки сканера и наоборот. Одна строка на пользователя.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base


class UserScannerSettings(Base):
    """Персональные настройки авто-сканера (слайдер флэта + свои тикеры)."""

    __tablename__ = "user_scanner_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # {"flat_slider": 0.5, "filter_signals": true, "tickers": {"us": ["AAPL"]}}
    data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<UserScannerSettings(user={self.user_id})>"
