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
# Static token the /ws/client handshake must present (Round-2 WS auth).
os.environ.setdefault("TRADING_LOCAL_CLIENT_TOKEN", "test-client-token")

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


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "real_auth: exercise the real require_auth guard instead of the test "
        "auto-auth override (used by the anonymous-access / 401 assertions).",
    )


@pytest.fixture(autouse=True)
def _auto_auth_override(request):
    """Give most tests an authenticated caller.

    Round 2 moved the auth perimeter to the ``include_router`` mounts in
    ``trading.main``, so every business route now requires a bearer token. Tests
    that verify *behaviour* (not the guard itself) would otherwise all 401. We
    install a permissive FastAPI dependency override of ``require_auth`` and let
    the handful of guard tests opt out with ``@pytest.mark.real_auth`` (the
    module-wide sweep in ``test_authz_regression.py`` marks the whole module).
    """
    if request.node.get_closest_marker("real_auth") is not None:
        yield
        return
    from trading.api.deps import require_auth
    from trading.main import app

    app.dependency_overrides[require_auth] = lambda: "test-admin"
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_auth, None)


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
