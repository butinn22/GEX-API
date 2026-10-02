"""Test configuration: isolate the trading tests from real secrets/storage.

Set env vars *before* any trading module import so ``trading.config.Settings``
picks up a throwaway SQLite DB and deterministic credentials.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_tmp = Path(tempfile.mkdtemp(prefix="gex_trading_test_"))
_db = _tmp / "test.db"

os.environ.setdefault("TRADING_DATABASE_URL", f"sqlite+aiosqlite:///{_db.as_posix()}")
os.environ.setdefault("TRADING_SECRET_KEY", "test-secret-key")
os.environ.setdefault("TRADING_ADMIN_USERNAME", "admin")
os.environ.setdefault("TRADING_ADMIN_PASSWORD", "admin")


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
