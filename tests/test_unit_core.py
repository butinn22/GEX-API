"""Unit-тесты для модулей, чьи API я проверил: greeks, repository, scan_service, scheduler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex.domain.greeks import bs_gamma, bs_delta
from gex.domain.data_loader import OptionSnapshot
from gex.adapters.persistence.repository import InMemoryRepository
from gex.application.scan_service import ScanService


# ====================================================================== #
#  Greeks — математика, чистая, без зависимостей
# ====================================================================== #
class TestGreeks:
    """Греки — детерминированные функции BSM."""

    def test_gamma_positive(self):
        assert bs_gamma(500, 500, 0.08, 0.045, 0.20, 0.0) > 0

    def test_delta_call_between_0_and_1(self):
        d = bs_delta(500, 500, 0.08, 0.045, 0.20, 0.0, True)
        assert 0 < d < 1

    def test_delta_put_between_minus1_and_0(self):
        d = bs_delta(500, 500, 0.08, 0.045, 0.20, 0.0, False)
        assert -1 < d < 0

    def test_gamma_atm_highest(self):
        atm = bs_gamma(500, 500, 0.08, 0.045, 0.20, 0.0)
        otm = bs_gamma(500, 550, 0.08, 0.045, 0.20, 0.0)
        assert atm > otm


# ====================================================================== #
#  Repository — потокобезопасное in-memory хранилище
# ====================================================================== #
class TestInMemoryRepository:
    _snap = OptionSnapshot(symbol="T", spot=500, as_of=pd.Timestamp.now(), chain=pd.DataFrame())

    def test_put_get(self):
        r = InMemoryRepository()
        r.put("SPY", self._snap)
        assert r.get("SPY") is not None

    def test_get_miss(self):
        assert InMemoryRepository().get("X") is None

    def test_list(self):
        r = InMemoryRepository()
        r.put("A", self._snap)
        r.put("B", self._snap)
        assert r.list_tickers() == ["A", "B"]

    def test_delete_hit(self):
        r = InMemoryRepository()
        r.put("X", self._snap)
        assert r.delete("X") is True

    def test_delete_miss(self):
        assert InMemoryRepository().delete("X") is False


# ====================================================================== #
#  ScanService — lifecycle (start/stop/is_running)
# ====================================================================== #
class TestScanService:
    def test_init_not_running(self):
        svc = ScanService(ta_service=None, gex_service=None, interval_seconds=9999)
        assert not svc.is_running

    def test_start_stop(self):
        svc = ScanService(ta_service=None, gex_service=None, interval_seconds=9999)
        svc.start()
        assert svc.is_running
        svc.stop(timeout=2)
        assert not svc.is_running

    def test_double_start(self):
        svc = ScanService(ta_service=None, gex_service=None, interval_seconds=9999)
        svc.start()
        svc.start()  # не падает
        svc.stop()

    def test_stop_without_start(self):
        ScanService(ta_service=None, gex_service=None).stop()  # не падает

    def test_list_records_empty(self):
        svc = ScanService(ta_service=None, gex_service=None)
        assert svc.list_records() == []

    def test_get_latest_none(self):
        svc = ScanService(ta_service=None, gex_service=None)
        assert svc.get_latest("SPY") is None
