"""Тесты Finnhub: клиент profile2, нормировка масштабов, сверка shares с SEC, API."""
from __future__ import annotations

import pytest
from gex.auth.models import User
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.adapters.providers.finnhub_client import (
    FinnhubError,
    best_scale,
    get_company_profile2,
    reconcile_shares,
)
from gex.application.sec.sec_fundamentals import SecFundamentalsService  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
#  1. Нормировка масштабов
# ══════════════════════════════════════════════════════════════════════
class TestBestScale:
    def test_thousands_scale(self):
        """Finnhub в тысячах: 15_208_306 × 1e3 ≈ 15.2B — близко к SEC."""
        best = best_scale(15_208_306.0, 15_208_306_000.0)
        assert best["scale"] == 1e3
        assert best["value"] == pytest.approx(15_208_306_000.0)
        assert best["diff"] < 1e-9

    def test_billions_raw(self):
        """Finnhub уже в миллиардах (15.2) — масштаб 1e9."""
        best = best_scale(15.2, 15_200_000_000.0)
        assert best["scale"] == 1e9
        assert best["diff"] < 0.001

    def test_millions_scale(self):
        best = best_scale(1_520.8, 1_520_800_000.0)
        assert best["scale"] == 1e6
        assert best["diff"] < 0.001

    def test_invalid_input(self):
        best = best_scale(0.0, 100.0)
        assert best["value"] is None
        best2 = best_scale(100.0, 0.0)
        assert best2["value"] is None


# ══════════════════════════════════════════════════════════════════════
#  2. Сверка shares
# ══════════════════════════════════════════════════════════════════════
class TestReconcile:
    def test_match_when_close(self):
        """SEC 15.208B, Finnhub 15_208_306 (тысячи) → согласованы, источник SEC."""
        out = reconcile_shares(15_208_306_000.0, 15_208_306.0)
        assert out["source"] == "sec"
        assert out["match"] is True
        assert out["shares"] == pytest.approx(15_208_306_000.0)
        assert out["finnhub_scale"] == 1e3
        assert out["warning"] is None

    def test_mismatch_warns(self):
        """Finnhub не сходится ни в одном масштабе → warning, источник SEC."""
        out = reconcile_shares(100_000_000.0, 15_208_306.0)
        assert out["source"] == "sec"
        assert out["match"] is False
        assert out["warning"] is not None
        assert "Расхождение" in out["warning"]

    def test_sec_missing_uses_finnhub(self):
        """SEC не дал акции → берём Finnhub."""
        out = reconcile_shares(None, 15_208_306.0)
        assert out["source"] == "finnhub"
        assert out["shares"] == pytest.approx(15_208_306.0)
        assert "Finnhub" in out["warning"]

    def test_finnhub_missing_uses_sec(self):
        out = reconcile_shares(15_208_306_000.0, None)
        assert out["source"] == "sec"
        assert out["match"] is None
        assert "Finnhub" in out["warning"]

    def test_both_missing(self):
        out = reconcile_shares(None, None)
        assert out["source"] is None
        assert out["shares"] is None

    def test_tolerance_tunable(self):
        """Расхождение 10% при tolerance 5% → mismatch; при 15% → match."""
        out = reconcile_shares(100.0, 110.0, tolerance=0.05)
        assert out["match"] is False
        out2 = reconcile_shares(100.0, 110.0, tolerance=0.15)
        assert out2["match"] is True


# ══════════════════════════════════════════════════════════════════════
#  3. Клиент
# ══════════════════════════════════════════════════════════════════════
class TestClient:
    def test_profile2(self, monkeypatch):
        captured: dict = {}

        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "ticker": "AAPL",
                    "name": "Apple Inc",
                    "shareOutstanding": 15208306.0,
                    "marketCapitalization": 2584583.0,
                }

            def raise_for_status(self):
                pass

        def fake_get(url, params=None, timeout=15):
            captured["url"] = url
            captured["params"] = params
            return FakeResp()

        # Ключ задаём явно: без него `_http_get_json` падает с FinnhubError **до** выхода
        # в сеть, поэтому тест проходил лишь там, где в окружении лежал настоящий
        # FINNHUB_API_KEY (то есть был недетерминированным по среде).
        monkeypatch.setattr(
            "gex.adapters.providers.finnhub_client.settings.FINNHUB_API_KEY", "test-key"
        )
        monkeypatch.setattr("gex.adapters.providers.finnhub_client.requests.get", fake_get)
        profile = get_company_profile2("aapl")
        assert profile["ticker"] == "AAPL"
        assert captured["url"].endswith("/stock/profile2")
        assert captured["params"]["symbol"] == "AAPL"
        assert captured["params"]["token"]  # токен передан

    def test_401_raises(self, monkeypatch):
        class FakeResp:
            status_code = 401

            def raise_for_status(self):
                pass

        monkeypatch.setattr("gex.adapters.providers.finnhub_client.requests.get", lambda *a, **k: FakeResp())
        with pytest.raises(FinnhubError):
            get_company_profile2("AAPL")

    def test_404_raises(self, monkeypatch):
        class FakeResp:
            status_code = 404

            def raise_for_status(self):
                pass

        monkeypatch.setattr("gex.adapters.providers.finnhub_client.requests.get", lambda *a, **k: FakeResp())
        with pytest.raises(FinnhubError):
            get_company_profile2("ZZZZ")

    def test_no_api_key(self, monkeypatch):
        monkeypatch.setattr("gex.adapters.providers.finnhub_client.settings.FINNHUB_API_KEY", "")
        with pytest.raises(FinnhubError):
            get_company_profile2("AAPL")


# ══════════════════════════════════════════════════════════════════════
#  4. Сервис + API
# ══════════════════════════════════════════════════════════════════════
def _patch_edgar(monkeypatch, facts):
    monkeypatch.setattr(
        "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
        lambda redis=None: {"AAPL": "0000320193"},
    )
    monkeypatch.setattr("gex.application.sec.sec_fundamentals.get_company_facts", lambda cik: facts)


@pytest.fixture
def db_ready():
    recreate_tables()
    yield
    recreate_tables()


def _apple_core_facts():
    from tests.test_sec_edgar import _apple_facts

    return _apple_facts()


class TestSharesApi:
    def test_shares_ok(self, client, monkeypatch, db_ready):
        _patch_edgar(monkeypatch, _apple_core_facts())
        # Заполняем PG данными SEC (shares_outstanding = 15.42e9)
        SecFundamentalsService(redis_client=None).get_fundamentals_core("AAPL")
        # Finnhub: профиль с shareOutstanding (в тысячах: 14.6B / 1e3)
        monkeypatch.setattr(
            "gex.adapters.providers.finnhub_client.get_company_profile2",
            lambda symbol: {"ticker": symbol, "shareOutstanding": 14_600_000.0},
        )

        r = client.get("/companies/AAPL/shares")
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        # SEC дал акции (15.42e9 из apple_core) → source sec
        assert data["source"] == "sec"
        assert data["shares"] == pytest.approx(15_419_532_000.0)
        # Finnhub 14.6M × 1e3 = 14.6B — разница ~5.3% → tolerance 5% → mismatch с warning
        assert data["finnhub_scale"] == 1e3
        assert "match" in data and data["diff_pct"] is not None

    def test_shares_unknown_ticker_404(self, client, monkeypatch):
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: {"AAPL": "0000320193"},
        )
        r = client.get("/companies/ZZZZ/shares")
        assert r.status_code == 404

    def test_shares_finnhub_fallback(self, client, monkeypatch, db_ready):
        """Finnhub упал → warning, источник SEC."""
        _patch_edgar(monkeypatch, _apple_core_facts())
        # Заполняем PG данными SEC (shares_outstanding)
        SecFundamentalsService(redis_client=None).get_fundamentals_core("AAPL")

        def _boom(symbol):
            raise RuntimeError("Finnhub недоступен")

        monkeypatch.setattr("gex.adapters.providers.finnhub_client.get_company_profile2", _boom)
        r = client.get("/companies/AAPL/shares")
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "sec"
        assert data["warning"] is not None


@pytest.fixture
def client(monkeypatch):
    """TestClient с изоляцией от реального Redis."""
    import gex.adapters.cache.result_cache as rc_module

    monkeypatch.setattr(rc_module, "get_redis", lambda: None)

    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()

    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)
