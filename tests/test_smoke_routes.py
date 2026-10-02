"""Smoke-тесты всех HTTP-ручек (v2 — после router extraction).

Запускает uvicorn через subprocess. Проверяет каждую ручку на статус-код.
Покрытие: 100% роутеров (16 модулей).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gex.auth.config import settings as app_settings

BASE_URL = "http://127.0.0.1:8008"
_PORT = 8008

# Bearer-токен Master Admin, заполняется фикстурой server (см. ниже).
_TOKEN: str | None = None


def _headers() -> dict:
    return {"Authorization": f"Bearer {_TOKEN}"} if _TOKEN else {}


@pytest.fixture(scope="session")
def server():
    """Запустить uvicorn как subprocess на время всех тестов."""
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import uvicorn; uvicorn.run('main:app', host='127.0.0.1', port={_PORT}, log_level='error')"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Ждём готовности
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    else:
        proc.terminate()
        pytest.fail("Server failed to start within 30s")

    # Почти все роутеры под auth-барьерами (require_subscription/require_master_admin) —
    # логинимся мастер-админом и шлём Bearer-токен во всех запросах.
    global _TOKEN
    r = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": app_settings.MASTER_EMAIL, "password": app_settings.MASTER_PASSWORD},
        timeout=15,
    )
    if r.status_code == 200:
        _TOKEN = r.json()["access_token"]
    else:
        print(f"[smoke] WARN: master login failed: {r.status_code} {r.text}")

    yield
    proc.terminate()
    proc.wait(timeout=5)


def _get(path: str, timeout: int = 30) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", timeout=timeout, headers=_headers())


def _post(path: str, json: dict | None = None, timeout: int = 15) -> requests.Response:
    return requests.post(f"{BASE_URL}{path}", json=json or {}, timeout=timeout, headers=_headers())


def _delete(path: str, timeout: int = 10) -> requests.Response:
    return requests.delete(f"{BASE_URL}{path}", timeout=timeout, headers=_headers())


def _ok(code: int) -> bool:
    return code in (200, 201)


def _ok_or_no_data(code: int) -> bool:
    """200 (OK), 502 (no network/bad gateway — external API down)."""
    return code in (200, 502)


def _ok_or_warming(code: int) -> bool:
    """Для снапшот-страниц (/breadth, /sector/breadth, /breadth-imoex).

    200 — снапшот есть; 502 — источник недоступен; 503 — хранилище холодное
    (страница отвечает «данные готовятся» + Retry-After и неблокирующе просит
    пересчёт). 503 здесь — штатный ответ вынесенного из запроса расчёта, а не
    дефект: проверяется он отдельными тестами (test_market_pages_cache.py).
    """
    return code in (200, 502, 503)


def _ok_or_not_found(code: int) -> bool:
    """200, 404 (ticker not found), 422 (validation)."""
    return code in (200, 404, 422)


# ═══════════════════════════════════════════════════════════════════════
# META (meta_router)
# ═══════════════════════════════════════════════════════════════════════
class TestMeta:
    def test_health(self, server):
        r = _get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "tickers_loaded" in data

    def test_root_serves_frontend(self, server):
        r = _get("/")
        assert r.status_code == 200

    def test_tickers(self, server):
        r = _get("/tickers")
        assert r.status_code == 200
        assert "tickers" in r.json()


# ═══════════════════════════════════════════════════════════════════════
# GEX STATIC (gex_router)
# ═══════════════════════════════════════════════════════════════════════
class TestGEXStatic:
    def test_gex_analysis_spy(self, server):
        r = _get("/gex/SPY?days=30")
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "SPY"

    def test_gex_analysis_404(self, server):
        r = _get("/gex/NONEXISTENT")
        assert r.status_code == 404

    def test_gex_profile(self, server):
        r = _get("/gex/SPY/profile?days=30")
        assert r.status_code == 200

    def test_post_chain(self, server):
        body = {"spot": 500.0, "chain": [
            {"strike": 500, "type": "C", "oi": 1000, "iv": 0.20, "T": 0.08}
        ]}
        r = _post("/chains/TEST_SMOKE", json=body)
        assert r.status_code == 201

    def test_delete_chain(self, server):
        r = _delete("/chains/TEST_SMOKE")
        assert r.status_code in (200, 404)  # 404 if already deleted


# ═══════════════════════════════════════════════════════════════════════
# LIVE (live_router) — yfinance
# ═══════════════════════════════════════════════════════════════════════
class TestLive:
    def test_live_gex(self, server):
        r = _get("/live/gex/SPY?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)

    def test_live_gex_profile(self, server):
        r = _get("/live/gex/SPY/profile?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# MOEX (moex_router) — MOEX ISS
# ═══════════════════════════════════════════════════════════════════════
class TestMOEX:
    def test_moex_gex(self, server):
        r = _get("/moex/gex/RTS?days=30", timeout=180)
        assert _ok_or_no_data(r.status_code)

    def test_moex_gex_profile(self, server):
        r = _get("/moex/gex/RTS/profile?days=30", timeout=45)
        assert _ok_or_no_data(r.status_code)

    def test_moex_invalid_asset(self, server):
        r = _get("/moex/gex/INVALID")
        assert r.status_code in (404, 422)


# ═══════════════════════════════════════════════════════════════════════
# VOL-INDEX (vix_router) — VIX/VVIX
# ═══════════════════════════════════════════════════════════════════════
class TestVolIndex:
    def test_vix_gex(self, server):
        r = _get("/vix/gex/VIX?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)

    def test_vix_profile(self, server):
        r = _get("/vix/gex/VIX/profile?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# CRYPTO (crypto_router) — Bybit
# ═══════════════════════════════════════════════════════════════════════
class TestCrypto:
    def test_crypto_gex(self, server):
        r = _get("/crypto/gex/BTC?days=30&expiries=2", timeout=30)
        assert _ok_or_no_data(r.status_code)

    def test_crypto_profile(self, server):
        r = _get("/crypto/gex/BTC/profile?days=30&expiries=2", timeout=30)
        assert _ok_or_no_data(r.status_code)

    def test_crypto_404(self, server):
        r = _get("/crypto/gex/NONECOIN")
        assert _ok_or_not_found(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# EXTENDED GEX (extended_router)
# ═══════════════════════════════════════════════════════════════════════
class TestExtended:
    def test_extended_gex(self, server):
        r = _get("/ext/gex/SPY?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)

    def test_extended_with_hedge(self, server):
        r = _get("/ext/gex/SPY?days=30&expiries=2&hedge_pct=-2&hedge_pct=2", timeout=45)
        assert _ok_or_no_data(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# TA + OHLCV (ta_router)
# ═══════════════════════════════════════════════════════════════════════
class TestTA:
    def test_ta_analysis(self, server):
        r = _get("/ta/SPY?n_paths=200", timeout=60)
        assert _ok_or_no_data(r.status_code)

    def test_ta_timeframe(self, server):
        r = _get("/ta/SPY/1d?n_paths=200", timeout=60)
        assert _ok_or_no_data(r.status_code)

    def test_ohlcv(self, server):
        r = _get("/ohlcv/SPY?timeframe=1d&limit=10", timeout=30)
        assert _ok_or_no_data(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# COMMODITY (commodity_router) — NEW!
# ═══════════════════════════════════════════════════════════════════════
class TestCommodity:
    def test_commodity_analysis(self, server):
        r = _get("/commodity/analysis/GOLD?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)

    def test_commodity_profile(self, server):
        r = _get("/commodity/analysis/GOLD/profile?days=30&expiries=2", timeout=45)
        assert _ok_or_no_data(r.status_code)

    def test_commodity_ohlcv(self, server):
        r = _get("/commodity/ohlcv/GOLD?timeframe=1d&limit=10", timeout=30)
        assert _ok_or_no_data(r.status_code)

    def test_commodity_spot(self, server):
        r = _get("/commodity/spot/GOLD", timeout=30)
        assert _ok_or_no_data(r.status_code)

    def test_commodity_tickers(self, server):
        r = _get("/commodity/tickers")
        assert r.status_code == 200
        data = r.json()
        assert "GOLD" in data

    def test_commodity_dynamics(self, server):
        r = _get("/commodity/dynamics?bars=30", timeout=30)
        assert _ok_or_no_data(r.status_code)

    def test_commodity_unsupported(self, server):
        # PALLAD получил ETF-прокси (PALL) в аудите 2026-09-17 и теперь поддерживается;
        # NICKEL остаётся без опционов (has_options=False) — на нём проверяем отказ.
        r = _get("/commodity/analysis/NICKEL?days=30", timeout=15)
        assert r.status_code in (404, 422)


# SIGNALS (signal-scanner) — см. signal_scanner_router (/scanner/signals)
# ═══════════════════════════════════════════════════════════════════════
# TRENDLINES (trendline_router)
# ═══════════════════════════════════════════════════════════════════════
class TestTrendlines:
    def test_trendlines_all(self, server):
        r = _get("/trendlines/SPY?timeframe=all&resolution=4&history_bars=100", timeout=60)
        assert _ok_or_no_data(r.status_code)

    def test_trendlines_single_tf(self, server):
        r = _get("/trendlines/SPY?timeframe=1d&resolution=4&history_bars=100", timeout=45)
        assert _ok_or_no_data(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# MACD (macd_router)
# ═══════════════════════════════════════════════════════════════════════
class TestMACD:
    def test_macd_trend(self, server):
        r = _get("/macd/trend/SPY?timeframe=all&N=5&M=20&H=100", timeout=60)
        assert _ok_or_no_data(r.status_code)

    def test_macd_bad_params(self, server):
        r = _get("/macd/trend/SPY?normalization_method=BAD")
        assert r.status_code == 422



# ═══════════════════════════════════════════════════════════════════════
# SCANNER (scanner_router)
# ═══════════════════════════════════════════════════════════════════════
class TestScanner:
    def test_scans(self, server):
        r = _get("/scans", timeout=15)
        assert _ok(r.status_code)

    def test_scan_ticker(self, server):
        r = _get("/scans/SPY", timeout=15)
        # Может быть 404, если сканер ещё не запускался
        assert r.status_code in (200, 404)

    @pytest.mark.skip(reason="POST /scans/run — синхронный scan_all всей витрины через rate-limited внешние API (десятки минут). Эндпоинт жив: проверяется GET /scans + фоновый сканер приложения + unit-тесты ScanService (test_unit_core.py).")
    def test_run_scans(self, server):
        r = _post("/scans/run", timeout=600)
        # 200 если есть сеть, 502 если нет — оба допустимы
        assert r.status_code in (200, 502)


# ═══════════════════════════════════════════════════════════════════════
# BREADTH + SECTOR (breadth_sector_router)
# ═══════════════════════════════════════════════════════════════════════
class TestBreadthSector:
    def test_breadth(self, server):
        r = _get("/breadth", timeout=45)
        assert _ok_or_warming(r.status_code)

    def test_composite_formula(self, server):
        # Расчёт идёт синхронно и зависит от нескольких провайдеров (yfinance, CBOE CDN,
        # McClellan); каждый вызов ограничен дедлайном транспорта, но суммарный бюджет
        # честно измеряется минутами — как у sector_breadth ниже.
        r = _get("/composite-formula", timeout=180)
        assert _ok_or_no_data(r.status_code)

    def test_sector_breadth(self, server):
        r = _get("/sector/breadth", timeout=180)
        assert _ok_or_warming(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# FETCHER (fetcher_router)
# ═══════════════════════════════════════════════════════════════════════
class TestFetcher:
    def test_fetcher_status(self, server):
        r = _get("/fetcher/status")
        assert r.status_code == 200
        data = r.json()
        assert "queue_length" in data

    def test_fetcher_prewarm(self, server):
        r = _post("/fetcher/prewarm")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# NOVEL CANDLES + RSI NOVEL (novel_candles_router / rsi_novel_router)
# ═══════════════════════════════════════════════════════════════════════
class TestNovelCandles:
    def test_novel_candles(self, server):
        # Расчёт зависит от живых провайдеров; каждый вызов ограничен дедлайном,
        # но суммарный бюджет — минуты (как у sector/composite ниже).
        r = _get("/novel-candles/SPY?timeframe=1d&limit=50&no_trendlines=true", timeout=120)
        assert _ok_or_no_data(r.status_code)


class TestTaStructure:
    def test_ta_structure(self, server):
        r = _get("/ta-structure/SPY?timeframe=1d&limit=100", timeout=120)
        assert _ok_or_no_data(r.status_code)
        if r.status_code == 200:
            data = r.json()
            assert data["ticker"] == "SPY"
            assert data["n_bars"] == len(data["bars"])
            assert len(data["novel_bars"]) == data["n_bars"]
            assert data["stage"] in ("UPTREND", "DOWNTREND", "REVERSAL", "RANGE")
            assert len(data["trend_final"]) == data["n_bars"]
            assert isinstance(data["fractals"], list)
            assert isinstance(data["events"], list)
            assert isinstance(data["zigzag"], list)

    def test_ta_structure_bad_source(self, server):
        r = _get("/ta-structure/SPY?timeframe=1d&limit=100&fractal_source=bogus", timeout=30)
        assert r.status_code in (404, 422)

    def test_ta_structure_min_distance(self, server):
        r = _get("/ta-structure/SPY?timeframe=1d&limit=100&min_fractal_distance=5", timeout=120)
        assert _ok_or_no_data(r.status_code)
        if r.status_code == 200:
            data = r.json()
            assert data["stage"] in ("UPTREND", "DOWNTREND", "REVERSAL", "RANGE")
            assert len(data["trend_final"]) == data["n_bars"]

    def test_ta_structure_bad_distance(self, server):
        r = _get("/ta-structure/SPY?timeframe=1d&limit=100&min_fractal_distance=999", timeout=30)
        assert r.status_code in (404, 422)


class TestRsiNovel:
    def test_rsi_novel(self, server):
        r = _get("/rsi-novel/SPY?timeframe=1d&limit=100", timeout=45)
        assert _ok_or_no_data(r.status_code)
        if r.status_code == 200:
            data = r.json()
            assert data["ticker"] == "SPY"
            # yfinance отдаёт полную историю (limit — верхняя граница)
            assert data["n_bars"] >= 100
            assert len(data["bars"]) == data["n_bars"]
            assert "rsi_close" in data and "resistance" in data
            assert data["signal"] in ("overbought", "oversold", "neutral", None)

    def test_rsi_novel_bad_timeframe(self, server):
        r = _get("/rsi-novel/SPY?timeframe=3d&limit=50", timeout=20)
        assert r.status_code in (404, 422)

    def test_rsi_novel_custom_params(self, server):
        r = _get("/rsi-novel/SPY?timeframe=1d&limit=80&lenn=10&wicks=false&ob_level=80&os_level=20", timeout=45)
        assert _ok_or_no_data(r.status_code)


# ═══════════════════════════════════════════════════════════════════════
# AUTH (уже отдельный роутер, проверяем mount)
# ═══════════════════════════════════════════════════════════════════════
class TestAuth:
    def test_auth_login_available(self, server):
        r = _post("/auth/login", json={"email": "test@test.com", "password": "wrong"})
        assert r.status_code in (403, 401, 422)

    def test_auth_register_available(self, server):
        r = _post("/auth/register", json={"email": "x@x.com", "password": "short"})
        assert r.status_code in (422, 400, 409)
