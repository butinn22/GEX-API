"""Модель UserInstrument — индивидуальные инструменты сканера на пользователя."""
from __future__ import annotations

from sqlalchemy import Column, String, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gex.adapters.persistence.database import Base


class UserInstrument(Base):
    """Один инструмент в персональном сканере пользователя.

    Каждый пользователь может иметь до 10 инструментов.
    Уникальность: (user_id, ticker, timeframe).
    """

    __tablename__ = "user_instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "ticker", "timeframe", name="uq_user_instrument"),
    )

    def __repr__(self) -> str:
        return f"<UserInstrument(user={self.user_id}, {self.ticker}:{self.timeframe})>"
