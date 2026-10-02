"""SQLAlchemy-модель фундаментальных данных SEC EDGAR.

Таблица ``company_metrics`` — персистентное хранилище нормализованных строк
XBRL-показателей (потоковых: revenue/net_income/EPS/CFO/CapEx/... и
балансовых: debt/cash/equity/shares) из companyfacts SEC.

Ключ строки — (ticker, metric, end, start): одна и та же отчётность может
переподаваться, в БД попадает только последняя версия по ``filed``
(см. :mod:`gex.sec_fundamentals` — replace-семантика upsert'а).
"""
from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CompanyMetric(Base):
    """Одна строка XBRL-показателя компании за период (end → start)."""

    __tablename__ = "company_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    cik: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    #: Нормализованное имя показателя: revenue, net_income, operating_income,
    #: eps_basic, cfo, capex, dda, debt_combined, lt_debt, st_debt, cash,
    #: st_investments, equity, shares_outstanding
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    tag: Mapped[str] = mapped_column(String(128), nullable=False)
    end: Mapped[date] = mapped_column(Date, nullable=False)
    start: Mapped[date | None] = mapped_column(Date, nullable=True)
    val: Mapped[float] = mapped_column(Float, nullable=False)
    fy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fp: Mapped[str | None] = mapped_column(String(8), nullable=True)
    form: Mapped[str | None] = mapped_column(String(16), nullable=True)
    frame: Mapped[str | None] = mapped_column(String(32), nullable=True)
    filed: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    __table_args__ = (
        UniqueConstraint("ticker", "metric", "end", "start", name="uq_company_metrics_key"),
        Index("ix_company_metrics_ticker_metric_end", "ticker", "metric", "end"),
    )

    def as_dict(self) -> dict:
        """Строка → dict для расчётов (даты остаются date)."""
        return {
            "metric": self.metric,
            "tag": self.tag,
            "end": self.end,
            "start": self.start,
            "val": self.val,
            "fy": self.fy,
            "fp": self.fp,
            "form": self.form,
            "frame": self.frame,
            "filed": self.filed,
        }

    def __repr__(self) -> str:
        return f"<CompanyMetric {self.ticker} {self.metric} {self.end} = {self.val}>"
