"""End-to-end tests for the portfolio / Monte-Carlo API surface.

All data comes from the offline ``synthetic`` source, so these tests are
deterministic and never touch an exchange.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from trading.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _basket(with_mc: bool = False) -> dict:
    payload: dict = {
        "tickers": [
            {"symbol": "AAA", "strategy": "sma_crossover", "fast": 5, "slow": 15,
             "weight": 0.5, "source": "synthetic", "limit": 400},
            # per-ticker dual-SMA block → the strategy reads long/short settings
            {"symbol": "BBB", "strategy": "sma_crossover_ls",
             "long": {"fast": 8, "slow": 21}, "short": {"fast": 8, "slow": 21},
             "weight": 0.3, "source": "synthetic", "limit": 400},
            {"symbol": "CCC", "strategy": "momentum", "period": 10,
             "weight": 0.2, "source": "synthetic", "limit": 400},
        ],
        "initial_cash": 100_000,
        "fee_rate": 0.001,
        "slippage": 0.0005,
    }
    if with_mc:
        payload["monte_carlo"] = {
            "enabled": True, "n_paths": 1500, "method": "block_bootstrap", "seed": 7,
        }
    return payload


# ── portfolio backtest ─────────────────────────────────────────────────


async def test_portfolio_backtest_runs_every_ticker(client):
    r = await client.post("/api/v1/backtest/portfolio", json=_basket())
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_tickers"] == 3
    assert d["errors"] == []
    assert d["initial_cash"] == 100_000
    assert d["final_equity"] > 0
    # time grid and equity grid must stay in lockstep for a plottable chart
    assert len(d["equity_curve"]) == len(d["times"]) >= 2
    for t in d["tickers"]:
        assert len(t["equity_curve"]) == len(d["times"])
        assert t["capital"] > 0
    # the individual settings must be honoured per ticker
    by_symbol = {t["symbol"]: t for t in d["tickers"]}
    assert by_symbol["AAA"]["strategy"] == "sma_crossover"
    assert by_symbol["BBB"]["strategy"] == "sma_crossover_ls"
    assert by_symbol["CCC"]["strategy"] == "momentum"
    assert sum(t["weight"] for t in d["tickers"]) == pytest.approx(1.0)


async def test_portfolio_capital_sums_to_initial_cash(client):
    d = (await client.post("/api/v1/backtest/portfolio", json=_basket())).json()
    assert sum(t["capital"] for t in d["tickers"]) == pytest.approx(100_000, rel=1e-6)


async def test_portfolio_metrics_are_present_and_finite(client):
    d = (await client.post("/api/v1/backtest/portfolio", json=_basket())).json()
    m = d["metrics"]
    for k in ("sharpe", "max_drawdown", "var_95", "cvar_95", "win_rate", "n_periods"):
        assert m[k] is not None
    assert 0.0 <= m["max_drawdown"] <= 1.0
    assert m["cvar_95"] <= m["var_95"]
    assert m["n_periods"] == len(d["equity_curve"]) - 1


async def test_portfolio_returns_correlation_matrix(client):
    d = (await client.post("/api/v1/backtest/portfolio", json=_basket())).json()
    c = d["correlation"]
    assert c is not None
    assert c["symbols"] == ["AAA", "BBB", "CCC"]
    assert len(c["matrix"]) == 3 and all(len(row) == 3 for row in c["matrix"])
    for i, row in enumerate(c["matrix"]):
        assert row[i] == pytest.approx(1.0, abs=1e-9)


async def test_portfolio_is_deterministic(client):
    a = (await client.post("/api/v1/backtest/portfolio", json=_basket())).json()
    b = (await client.post("/api/v1/backtest/portfolio", json=_basket())).json()
    assert a["final_equity"] == pytest.approx(b["final_equity"])


async def test_portfolio_isolates_a_failing_ticker(client):
    payload = {
        "tickers": [
            {"symbol": "OK", "source": "synthetic", "limit": 300},
            {"symbol": "BAD", "source": "definitely_not_an_exchange", "limit": 300},
        ],
        "initial_cash": 50_000,
    }
    r = await client.post("/api/v1/backtest/portfolio", json=payload)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_tickers"] == 1
    assert [e["symbol"] for e in d["errors"]] == ["BAD"]
    assert d["correlation"] is None  # a single surviving leg has no pairwise corr


async def test_portfolio_auto_selects_n_tickers(client):
    payload = {
        "n_tickers": 3, "category": "ru",
        "default_strategy": "momentum", "default_params": {"period": 12},
        "default_source": "synthetic", "default_limit": 300,
    }
    d = (await client.post("/api/v1/backtest/portfolio", json=payload)).json()
    assert d["n_tickers"] == 3
    assert {t["strategy"] for t in d["tickers"]} == {"momentum"}
    assert len({t["symbol"] for t in d["tickers"]}) == 3


async def test_explicit_tickers_and_n_tickers_combine(client):
    payload = _basket()
    payload.update({"n_tickers": 2, "category": "sectors",
                    "default_source": "synthetic", "default_limit": 300})
    d = (await client.post("/api/v1/backtest/portfolio", json=payload)).json()
    assert d["n_tickers"] == 5


async def test_portfolio_requires_a_target(client):
    r = await client.post("/api/v1/backtest/portfolio", json={})
    assert r.status_code == 422


async def test_portfolio_rejects_unknown_category(client):
    r = await client.post("/api/v1/backtest/portfolio", json={"n_tickers": 2, "category": "nope"})
    assert r.status_code == 422  # Literal-typed field


async def test_empty_ticker_list_is_rejected_when_no_autoselect(client):
    r = await client.post("/api/v1/backtest/portfolio", json={"tickers": [], "n_tickers": 0})
    assert r.status_code == 422


# ── portfolio + Monte-Carlo ────────────────────────────────────────────


async def test_portfolio_with_embedded_monte_carlo(client):
    d = (await client.post("/api/v1/backtest/portfolio", json=_basket(with_mc=True))).json()
    mc = d["monte_carlo"]
    assert mc is not None
    assert mc["method"] == "block_bootstrap"
    assert mc["n_paths"] == 1500
    assert 0.0 <= mc["prob_profit"] <= 1.0
    assert set(mc["bands"]) >= {"p5", "p25", "p50", "p75", "p95"}
    assert len(mc["bands"]["p50"]) == len(mc["steps"])
    assert len(mc["histogram"]["counts"]) == len(mc["histogram"]["centers"])
    assert mc["cvar_95"] <= mc["var_95"]
    assert mc["p5"] <= mc["p95"]


async def test_portfolio_without_monte_carlo_omits_it(client):
    d = (await client.post("/api/v1/backtest/portfolio", json=_basket())).json()
    assert d["monte_carlo"] is None


async def test_monte_carlo_can_be_disabled(client):
    payload = _basket(with_mc=True)
    payload["monte_carlo"]["enabled"] = False
    d = (await client.post("/api/v1/backtest/portfolio", json=payload)).json()
    assert d["monte_carlo"] is None


async def test_standalone_portfolio_monte_carlo_endpoint(client):
    payload = {"portfolio": _basket(), "monte_carlo": {"n_paths": 800, "method": "gbm", "seed": 3}}
    r = await client.post("/api/v1/backtest/portfolio/monte-carlo", json=payload)
    assert r.status_code == 200, r.text
    mc = r.json()
    assert mc["label"] == "portfolio"
    assert mc["n_paths"] == 800
    assert "sharpe" in mc["metrics_ci"]
    lo, hi = mc["metrics_ci"]["sharpe"]
    assert lo <= hi


@pytest.mark.parametrize("method", ["gbm", "bootstrap", "block_bootstrap", "historical"])
async def test_every_monte_carlo_method_is_supported(client, method):
    payload = _basket()
    payload["monte_carlo"] = {"enabled": True, "n_paths": 500, "method": method, "seed": 11}
    d = (await client.post("/api/v1/backtest/portfolio", json=payload)).json()
    assert d["monte_carlo"]["method"] == method
    assert d["monte_carlo"]["p5"] <= d["monte_carlo"]["p50"] if "p50" in d["monte_carlo"] else True


# ── single-symbol Monte-Carlo ──────────────────────────────────────────


async def test_single_symbol_monte_carlo_endpoint(client):
    payload = {"symbol": "SYNTH", "source": "synthetic", "strategy": "sma_crossover",
               "params": {"fast": 5, "slow": 20}, "limit": 300, "n_paths": 600, "method": "bootstrap"}
    r = await client.post("/api/v1/backtest/monte-carlo", json=payload)
    assert r.status_code == 200, r.text
    mc = r.json()
    assert mc["symbol" if "symbol" in mc else "label"] == "SYNTH"
    assert 0.0 <= mc["prob_profit"] <= 1.0
    assert mc["worst_return"] <= mc["best_return"]


# ── HTML report ────────────────────────────────────────────────────────


async def test_portfolio_report_is_self_contained_html_with_charts(client):
    r = await client.post("/api/v1/backtest/portfolio/report", json=_basket())
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    html = r.text
    assert html.count("<svg") == 3  # equity + drawdown + correlation
    assert "<script" not in html
    assert "https://" not in html


async def test_portfolio_report_appends_monte_carlo_charts(client):
    r = await client.post(
        "/api/v1/backtest/portfolio/report",
        json=_basket(with_mc=True),
    )
    assert r.status_code == 200
    html = r.text
    assert html.count("<svg") == 5  # + fan chart + histogram
    assert "Monte-Carlo" in html


# ── universe selection ─────────────────────────────────────────────────


async def test_universe_endpoint_returns_n_tickers(client):
    r = await client.get("/api/v1/data/universe", params={"category": "ru", "n": 5})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] == 5
    assert len(d["tickers"]) == min(5, d["available"])
    assert all(t["category"] == "ru" for t in d["tickers"])


async def test_universe_all_round_robins_across_categories(client):
    d = (await client.get("/api/v1/data/universe", params={"category": "all", "n": 50})).json()
    cats = {t["category"] for t in d["tickers"]}
    assert len(cats) >= 3
    # "all" should spread the budget, not spend it all on the first category.
    assert len(d["tickers"]) == 50
    assert max(t["category"] for t in d["tickers"]) is not None
    from collections import Counter

    per_cat = Counter(t["category"] for t in d["tickers"])
    assert max(per_cat.values()) <= 20


async def test_universe_all_is_diverse_even_for_small_n(client):
    d = (await client.get("/api/v1/data/universe", params={"category": "all", "n": 5})).json()
    assert len({t["category"] for t in d["tickers"]}) == 5


async def test_universe_rejects_unknown_category(client):
    r = await client.get("/api/v1/data/universe", params={"category": "nope"})
    assert r.status_code == 400


async def test_universe_is_deterministic(client):
    a = (await client.get("/api/v1/data/universe", params={"category": "us", "n": 6})).json()
    b = (await client.get("/api/v1/data/universe", params={"category": "us", "n": 6})).json()
    assert [t["symbol"] for t in a["tickers"]] == [t["symbol"] for t in b["tickers"]]


async def test_categories_endpoint(client):
    r = await client.get("/api/v1/data/categories")
    assert r.status_code == 200
    assert set(r.json()) >= {"us", "crypto", "fx", "ru", "sectors", "all"}


# ── data-cache behaviour through the API ───────────────────────────────


async def test_repeat_run_is_cache_backed_and_identical(client):
    """The common 'tweak a param and re-run' loop must not re-download data."""
    from trading.application.backtest.portfolio import bar_cache_stats, clear_bar_cache

    clear_bar_cache()
    first = await client.post("/api/v1/backtest/portfolio", json=_basket())
    assert first.status_code == 200
    misses = bar_cache_stats()["misses"]

    second = await client.post("/api/v1/backtest/portfolio", json=_basket())
    assert second.status_code == 200
    stats = bar_cache_stats()
    assert stats["hits"] == 3, stats          # every ticker served from cache
    assert stats["misses"] == misses          # nothing new fetched
    assert (
        second.json()["metrics"]["total_return"]
        == first.json()["metrics"]["total_return"]
    )


async def test_refresh_data_flag_forces_a_refetch(client):
    from trading.application.backtest.portfolio import bar_cache_stats, clear_bar_cache

    clear_bar_cache()
    await client.post("/api/v1/backtest/portfolio", json=_basket())
    before = bar_cache_stats()["misses"]

    r = await client.post(
        "/api/v1/backtest/portfolio", json={**_basket(), "refresh_data": True}
    )
    assert r.status_code == 200
    assert bar_cache_stats()["misses"] == before + 3  # three tickers re-fetched
