"""Persistent cache for expensive GEX levels/values.

GEX computation (option-chain aggregation, walls, regime, direction) is expensive and should not
be repeated on every dashboard/backtest request. This module stores one current JSON state per
ticker and refreshes it at most once per ``GEX_STATE_MAX_AGE_SECONDS`` (default 1 hour).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, Session

from gex.adapters.persistence.database import Base

DEFAULT_MAX_AGE_SECONDS = 3600  # refresh at most once per hour


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GexState(Base):
    """Current GEX snapshot for one ticker."""

    __tablename__ = "gex_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    data_json: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self.data_json)


def get_fresh_gex_state(
    session: Session,
    ticker: str,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> tuple[dict[str, Any] | None, bool]:
    """Return (state_data, is_fresh). is_fresh=True if a state exists and is younger than max age."""
    state = session.query(GexState).filter(GexState.ticker == ticker.upper()).first()
    if state is None:
        return None, False
    now = _utcnow()
    updated = state.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age = (now - updated).total_seconds()
    return state.data, age < max_age_seconds


def upsert_gex_state(session: Session, ticker: str, data: dict[str, Any]) -> GexState:
    """Insert or replace the current GEX state for a ticker."""
    ticker = ticker.upper()
    state = session.query(GexState).filter(GexState.ticker == ticker).first()
    if state is None:
        state = GexState(ticker=ticker, data_json=json.dumps(data, ensure_ascii=False, default=str))
        session.add(state)
    else:
        state.data_json = json.dumps(data, ensure_ascii=False, default=str)
        state.updated_at = _utcnow()
    session.commit()
    return state


def get_or_refresh_gex_state(
    session: Session,
    ticker: str,
    compute: Callable[[], dict[str, Any]],
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> tuple[dict[str, Any], bool]:
    """Return (state_data, was_computed_now).

    If a fresh state exists it is returned without calling ``compute``. If stale/missing,
    ``compute`` is called, the result is stored in DB, and the new state is returned.
    """
    ticker = ticker.upper()
    data, fresh = get_fresh_gex_state(session, ticker, max_age_seconds)
    if fresh and data is not None:
        return data, False

    # If stale data exists and compute fails, keep the stale data rather than erroring out.
    try:
        new_data = compute()
    except Exception:
        if data is not None:
            return data, False
        raise

    upsert_gex_state(session, ticker, new_data)
    return new_data, True
