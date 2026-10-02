"""SQLAlchemy User model — только модель, без engine/session.

Engine и сессия живут в :mod:`gex.database`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base


# ================================================================= #
#  Enum для статуса подписки
# ================================================================= #
class SubscriptionStatus:
    INACTIVE = "INACTIVE"
    BASIC = "BASIC"
    EXTENDED = "EXTENDED"
    ADMIN = "ADMIN"


SUBSCRIPTION_VALUES = [
    SubscriptionStatus.INACTIVE,
    SubscriptionStatus.BASIC,
    SubscriptionStatus.EXTENDED,
    SubscriptionStatus.ADMIN,
]

SUBSCRIPTION_ORDER = {
    SubscriptionStatus.INACTIVE: 0,
    SubscriptionStatus.BASIC: 1,
    SubscriptionStatus.EXTENDED: 2,
    SubscriptionStatus.ADMIN: 3,
}


# ================================================================= #
#  Модель User
# ================================================================= #
class User(Base):
    """Пользователь GEX Analytics.

    Хранит email (уникальный), bcrypt-хэш пароля, статус верификации
    email и уровень подписки.  Не-email-пользователи (OAuth) могут не
    иметь password_hash.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False,
    )
    password_hash: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    is_email_verified: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
    )
    is_blocked: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
    )
    telegram_chat_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None,
    )
    telegram_username: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None,
    )
    telegram_connect_token: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None,
    )
    telegram_notify: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
    )
    telegram_test_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None,
    )
    subscription_status: Mapped[str] = mapped_column(
        String(50),
        default=SubscriptionStatus.INACTIVE,
        nullable=False,
    )
    subscription_activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    subscription_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    # Дата окончания, о которой уже отправлено Telegram-уведомление
    # («подписка истекает завтра») — защита от повторных сообщений.
    subscription_expiry_notified_for: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    verification_token: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
    )
    oauth_provider: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
    )
    oauth_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User(email={self.email}, sub={self.subscription_status})>"


def _as_utc(dt: datetime | None) -> datetime | None:
    """Нормализовать datetime к timezone-aware UTC (SQLite отдаёт naive)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def subscription_is_active(user, now: datetime | None = None) -> bool:
    """Подписка пользователя действует (по статусу и сроку).

    Правила:
      * ADMIN — всегда активен (срока нет);
      * BASIC/EXTENDED — активна, пока ``subscription_expires_at`` не наступил;
        если срок не задан (None) — считаем активной (старые записи);
      * INACTIVE и прочее — неактивна.
    """
    status = getattr(user, "subscription_status", None)
    if status == SubscriptionStatus.ADMIN:
        return True
    if status not in (SubscriptionStatus.BASIC, SubscriptionStatus.EXTENDED):
        return False
    exp = _as_utc(getattr(user, "subscription_expires_at", None))
    if exp is None:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    return exp > now


# ================================================================= #
#  TelegramStart — кто нажал /start боту (username → chat_id)
# ================================================================= #
class TelegramStart(Base):
    """Запись о том, что пользователь Telegram запустил бота (/start).

    Нужна, чтобы сервис знал chat_id пользователя ещё ДО активации/подключения:
    бот может писать в чат только после того, как пользователь нажал Start.
    Ключ — публичный username (без @, в нижнем регистре): он уникален в Telegram,
    поэтому по нему можно безопасно сопоставить чат с аккаунтом на сайте.
    """

    __tablename__ = "telegram_starts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
    )
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<TelegramStart(@{self.username} -> {self.chat_id})>"
