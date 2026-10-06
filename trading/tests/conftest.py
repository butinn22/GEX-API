"""Test configuration: isolate the trading tests from real secrets/storage.

Set env vars *before* any trading module import so ``trading.config.Settings``
picks up a throwaway SQLite DB and deterministic credentials.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import event

import pytest

_tmp = Path(tempfile.mkdtemp(prefix="gex_trading_test_"))
_db = _tmp / "test.db"

os.environ.setdefault("TRADING_DATABASE_URL", f"sqlite+aiosqlite:///{_db.as_posix()}")
os.environ.setdefault("TRADING_SECRET_KEY", "test-secret-key")
os.environ.setdefault("TRADING_ADMIN_USERNAME", "admin")
os.environ.setdefault("TRADING_ADMIN_PASSWORD", "admin")

# Shared SQLite file + many async sessions (live-engine background tasks,
# request sessions, fixture cleanups) → writers occasionally hit the default
# 5 s busy timeout with "database is locked", which flakes the suite. WAL
# allows readers during writes and a longer busy timeout makes writers wait
# instead of erroring. Test-infra only; production is Postgres.
import trading.adapters.persistence.database as _database

_engine = _database.configure()


@event.listens_for(_engine.sync_engine, "connect")
def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA busy_timeout=15000")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


@pytest.fixture(autouse=True)
def _clear_bar_cache():
    """Keep the process-wide OHLCV cache from leaking state between tests.

    ``load_bars`` caches bars for a few minutes by design (so re-running a
    backtest is instant); without this reset a fixture-mocked fetch in one test
    would be served to the next one.
    """
    from trading.application.backtest.portfolio import clear_bar_cache

    clear_bar_cache()
    yield
    clear_bar_cache()
