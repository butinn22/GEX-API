"""Payment model: оплата подписки через СБП или крипту."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base


class PaymentMethod:
    SBP = "SBP"
    CRYPTO = "CRYPTO"
    BANK = "BANK"


class PaymentStatus:
    PENDING = "PENDING"                 # создан, ждёт оплаты
    PAID_CLIENT = "PAID_CLIENT"         # клиент отметил «оплатил»
    CONFIRMED = "CONFIRMED"             # админ подтвердил
    REJECTED = "REJECTED"               # админ отклонил
    EXPIRED = "EXPIRED"                 # истёк срок


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True, nullable=False,
    )
    user_email: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    plan: Mapped[str] = mapped_column(
        String(50), nullable=False,  # BASIC / EXTENDED
    )
    amount_rub: Mapped[float] = mapped_column(Float, nullable=False)
    amount_usd: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(
        String(20), nullable=False,  # SBP / CRYPTO
    )
    crypto_currency: Mapped[str | None] = mapped_column(
        String(20), nullable=True,  # USDT_TRC20 / USDT_BEP20
    )
    crypto_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(30), default=PaymentStatus.PENDING, nullable=False,
    )
    admin_note: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc) + timedelta(hours=24),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Payment(id={self.id[:8]}, {self.plan}, {self.method}, {self.status})>"


class PaymentSettings(Base):
    """Платёжные реквизиты сервиса (одна строка, id=1).

    Редактируются администратором из админ-панели (GET/PUT
    /auth/admin/payment-requisites) и отражаются на странице оплаты.
    При первом обращении строка сидится значениями из .env
    (SBP_*, CRYPTO_*, BANK_*) — дальше источник истины — БД
    (переживает пересоздание контейнеров).

    Пустая строка/значение-плейсхолдер (содержит '*') = реквизит не
    настроен → соответствующий способ оплаты скрыт на странице оплаты.
    """

    __tablename__ = "payment_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)

    # СБП
    sbp_phone: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    sbp_bank: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    sbp_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)

    # Крипта (USDT)
    crypto_usdt_trc20: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    crypto_usdt_bep20: Mapped[str] = mapped_column(String(128), default="", nullable=False)

    # Банковский счёт (перевод по реквизитам, ₽)
    bank_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    bank_bic: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    bank_account: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    bank_recipient: Mapped[str] = mapped_column(String(128), default="", nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
