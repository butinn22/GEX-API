"""Тесты SEC EDGAR fundamentals: клиент, нормализация, дедуп, расчёты, API.

Покрывают «грабли» из ТЗ:
1. разные XBRL-теги (включая финансовый fallback для банков);
2. дубликаты (переподача) — берём последнюю версию по ``filed``;
3. годовой (10-K) vs квартальный (10-Q), включая 3M vs 9M на одну дату end;
4. rate limit / User-Agent — на уровне клиента;
5. производные показатели: FCF, EBITDA, Operating Margin, Net Debt, ROE,
   P/E, P/S, Net Debt/EBITDA (цена из yfinance подмешивается отдельно).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from gex.auth.models import User
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.adapters.providers.sec_edgar import get_company_facts, get_ticker_to_cik
from gex.application.sec.sec_fundamentals import (
    FINANCIAL_REVENUE_TAGS,
    REVENUE_TAGS,
    SecFundamentalsService,
    _dedupe_rows,
    build_annual,
    filter_period,
    latest_balance,
    normalize_metrics,
)
from gex.adapters.persistence.sec_models import CompanyMetric


# ══════════════════════════════════════════════════════════════════════
#  Фабрики фейковых данных SEC
# ══════════════════════════════════════════════════════════════════════
def _fact_row(**overrides) -> dict:
    """Строка us-gaap факта (как в companyfacts JSON — даты ISO-строками)."""
    row = {
        "end": "2023-09-30",
        "start": "2022-10-01",
        "val": 383285000000.0,
        "fy": 2023,
        "fp": "FY",
        "form": "10-K",
        "frame": None,
        "filed": "2023-11-03",
    }
    row.update(overrides)
    return {k: v for k, v in row.items() if v is not None}


def _dated_row(**overrides) -> dict:
    """Как _fact_row, но даты уже date-объекты (вид после нормализатора)."""
    row = _fact_row(**overrides)
    for key in ("end", "start", "filed"):
        if row.get(key):
            row[key] = datetime.strptime(row[key], "%Y-%m-%d").date()
    return row


def _facts_multi(metrics: dict[str, list[dict]] | None = None) -> dict:
    """companyfacts-ответ с несколькими us-gaap тегами."""
    data = {"cik": 320193, "entityName": "APPLE INC", "facts": {"us-gaap": {}}}
    for tag, rows in (metrics or {"Revenues": [_fact_row()]}).items():
        data["facts"]["us-gaap"][tag] = {"units": {"USD": rows}}
    return data


def _apple_facts() -> dict:
    """Полный набор фактов Apple на FY2023/FY2024 (примерно реальные числа)."""
    return _facts_multi({
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _fact_row(end="2023-09-30", val=383_285_000_000, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=391_035_000_000, fy=2024, filed="2024-11-01"),
        ],
        "NetIncomeLoss": [
            _fact_row(end="2023-09-30", val=96_995_000_000, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=93_736_000_000, fy=2024, filed="2024-11-01"),
        ],
        "OperatingIncomeLoss": [
            _fact_row(end="2023-09-30", val=114_301_000_000, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=123_216_000_000, fy=2024, filed="2024-11-01"),
        ],
        "EarningsPerShareBasic": [
            _fact_row(end="2023-09-30", val=6.16, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=6.08, fy=2024, filed="2024-11-01"),
        ],
        "NetCashProvidedByUsedInOperatingActivities": [
            _fact_row(end="2023-09-30", val=110_543_000_000, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=118_254_000_000, fy=2024, filed="2024-11-01"),
        ],
        "PaymentsToAcquirePropertyPlantAndEquipment": [
            _fact_row(end="2023-09-30", val=-10_959_000_000, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=-9_447_000_000, fy=2024, filed="2024-11-01"),
        ],
        "DepreciationDepletionAndAmortization": [
            _fact_row(end="2023-09-30", val=11_519_000_000, fy=2023),
            _fact_row(end="2024-09-28", start="2023-10-01", val=11_445_000_000, fy=2024, filed="2024-11-01"),
        ],
        "LongTermDebtNoncurrent": [
            _fact_row(end="2023-09-30", val=95_287_000_000, start=None),
            _fact_row(end="2024-09-28", start="2023-10-01", val=85_750_000_000, filed="2024-11-01"),
        ],
        "LongTermDebtCurrent": [
            _fact_row(end="2023-09-30", val=9_822_000_000, start=None),
            _fact_row(end="2024-09-28", start="2023-10-01", val=9_812_000_000, filed="2024-11-01"),
        ],
        "CashAndCashEquivalentsAtCarryingValue": [
            _fact_row(end="2023-09-30", val=29_965_000_000, start=None),
            _fact_row(end="2024-09-28", start="2023-10-01", val=29_943_000_000, filed="2024-11-01"),
        ],
        "ShortTermInvestments": [
            _fact_row(end="2023-09-30", val=31_597_000_000, start=None),
            _fact_row(end="2024-09-28", start="2023-10-01", val=31_601_000_000, filed="2024-11-01"),
        ],
        "StockholdersEquity": [
            _fact_row(end="2023-09-30", val=62_146_000_000, start=None),
            _fact_row(end="2024-09-28", start="2023-10-01", val=56_950_000_000, filed="2024-11-01"),
        ],
        "CommonStockSharesOutstanding": [
            _fact_row(end="2023-09-30", val=15_550_061_000, start=None),
            _fact_row(end="2024-09-28", start="2023-10-01", val=15_419_532_000, filed="2024-11-01"),
        ],
    })


def _ticker_map_data() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "brk.b", "title": "Berkshire Hathaway"},
    }


@pytest.fixture
def fakeredis_client():
    """RedisClient на fakeredis (паттерн из test_redis_client.py)."""
    import fakeredis
    import gex.adapters.cache.redis_client as rc_module

    fake_conn = fakeredis.FakeRedis()
    client = rc_module.RedisClient.__new__(rc_module.RedisClient)
    client._host = "fake"
    client._port = 0
    client._db = 0
    client._password = None
    client._socket_timeout = 1
    client._socket_connect_timeout = 1
    client._maxmemory = "1mb"
    client._conn = fake_conn
    client._connected = True
    client._pool = None
    fake_conn.flushdb()
    return client


@pytest.fixture
def db_ready():
    """Свежая схема БД (in-memory SQLite) для сервисных тестов."""
    recreate_tables()
    yield
    recreate_tables()


# ══════════════════════════════════════════════════════════════════════
#  1. EDGAR-клиент
# ══════════════════════════════════════════════════════════════════════
class TestEdgarClient:
    def test_ticker_map_parses_and_zfills(self, monkeypatch):
        captured: dict = {}

        def fake_get(url: str, timeout: int):
            captured["url"] = url
            return _ticker_map_data()

        monkeypatch.setattr("gex.adapters.providers.sec_edgar._http_get", fake_get)
        # В полном прогоне main уже инициализировал реальный Redis — изолируемся,
        # чтобы не читать закешированный реестр вместо мока.
        monkeypatch.setattr("gex.adapters.providers.sec_edgar.get_redis", lambda: None)
        result = get_ticker_to_cik()
        assert result["AAPL"] == "0000320193"          # zfill до 10 цифр
        assert result["BRK.B"] == "0001067983"         # lowercase → uppercase
        assert captured["url"].endswith("company_tickers.json")

    def test_ticker_map_redis_cached(self, monkeypatch, fakeredis_client):
        calls = {"n": 0}

        def fake_get(url: str, timeout: int):
            calls["n"] += 1
            return _ticker_map_data()

        monkeypatch.setattr("gex.adapters.providers.sec_edgar._http_get", fake_get)
        assert get_ticker_to_cik(fakeredis_client)["AAPL"] == "0000320193"
        assert get_ticker_to_cik(fakeredis_client)["AAPL"] == "0000320193"
        assert calls["n"] == 1  # второй вызов — из Redis, без HTTP

    def test_company_facts_url_uses_zfilled_cik(self, monkeypatch):
        captured: dict = {}

        def fake_get(url: str, timeout: int):
            captured["url"] = url
            return {"facts": {"us-gaap": {}}}

        monkeypatch.setattr("gex.adapters.providers.sec_edgar._http_get", fake_get)
        # В полном прогоне main уже поднял настоящий Redis, и `get_company_facts`
        # возвращал закешированные факты, не доходя до `_http_get` — мок не срабатывал
        # и тест падал с KeyError('url'). Изолируемся так же, как соседний тест реестра.
        monkeypatch.setattr("gex.adapters.providers.sec_edgar.get_redis", lambda: None)
        get_company_facts("320193")
        assert captured["url"] == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"

    def test_http_get_403_raises_edgart_error(self, monkeypatch):
        class FakeResp:
            status_code = 403

            def json(self):  # pragma: no cover
                return {}

        def fake_get(url: str, timeout: int):
            return FakeResp()

        monkeypatch.setattr("gex.adapters.providers.sec_edgar._http_get", fake_get)
        # Та же изоляция: с настоящим Redis ответ придёт из кэша и 403 не случится.
        monkeypatch.setattr("gex.adapters.providers.sec_edgar.get_redis", lambda: None)
        from gex.adapters.providers.sec_edgar import EdgartError

        with pytest.raises(EdgartError):
            get_company_facts("320193")


# ══════════════════════════════════════════════════════════════════════
#  2. XBRL tag normalizer (все метрики)
# ══════════════════════════════════════════════════════════════════════
class TestNormalize:
    def test_revenue_extracted(self):
        facts = _facts_multi({"RevenueFromContractWithCustomerExcludingAssessedTax": [_fact_row()]})
        out = normalize_metrics(facts)
        rows = out["revenue"]
        assert len(rows) == 1
        assert rows[0]["metric"] == "revenue"
        assert rows[0]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
        assert rows[0]["val"] == 383285000000.0
        assert rows[0]["form"] == "10-K"

    def test_ignores_non_report_forms(self):
        """8-K и прочие формы не попадают в показатели."""
        facts = _facts_multi({"Revenues": [_fact_row(form="8-K"), _fact_row(form="10-Q")]})
        rows = normalize_metrics(facts)["revenue"]
        assert len(rows) == 1
        assert rows[0]["form"] == "10-Q"

    def test_tags_priority_order(self):
        """Порядок REVENUE_TAGS — приоритет выбора тега (первый с данными)."""
        facts = _facts_multi({
            "SalesRevenueNet": [_fact_row(val=2.0)],
            "Revenues": [_fact_row(val=1.0)],
        })
        out = normalize_metrics(facts)
        assert out["revenue"][0]["val"] == 1.0
        assert out["revenue"][0]["tag"] == "Revenues"
        assert REVENUE_TAGS.index("Revenues") < REVENUE_TAGS.index("SalesRevenueNet")

    def test_financial_fallback_for_banks(self):
        facts = _facts_multi({"InterestIncome": [_fact_row()]})
        out = normalize_metrics(facts)
        assert out["revenue"][0]["tag"] == "InterestIncome"
        assert "InterestIncome" in FINANCIAL_REVENUE_TAGS

    def test_all_metrics_normalized(self):
        """Каждая метрика получает свой первый тег с данными."""
        facts = _apple_facts()
        out = normalize_metrics(facts)
        expected = {
            "revenue", "net_income", "operating_income", "eps_basic",
            "cfo", "capex", "dda", "lt_debt", "st_debt", "cash",
            "st_investments", "equity", "shares_outstanding",
        }
        assert expected <= set(out.keys())
        assert out["eps_basic"][0]["tag"] == "EarningsPerShareBasic"
        assert out["cash"][0]["tag"] == "CashAndCashEquivalentsAtCarryingValue"

    def test_empty_facts(self):
        assert normalize_metrics({"facts": {"us-gaap": {}}}) == {}
        assert normalize_metrics({}) == {}
        assert normalize_metrics({"facts": {}}) == {}

    def test_missing_start_ok(self):
        """10-K без start (редкие компании) не падает."""
        facts = _facts_multi({"Revenues": [_fact_row(start=None)]})
        rows = normalize_metrics(facts)["revenue"]
        assert len(rows) == 1
        assert rows[0]["start"] is None

    def test_non_usd_units(self):
        """EPS (USD/shares) и shares (shares) — юниты не только USD."""
        data = {"cik": 320193, "entityName": "APPLE INC", "facts": {"us-gaap": {
            "EarningsPerShareBasic": {"units": {"USD/shares": [_fact_row(val=6.08)]}},
            "CommonStockSharesOutstanding": {"units": {"shares": [_fact_row(val=15_419_532_000, start=None)]}},
        }}}
        out = normalize_metrics(data)
        assert out["eps_basic"][0]["val"] == pytest.approx(6.08)
        assert out["shares_outstanding"][0]["val"] == pytest.approx(15_419_532_000)

    def test_st_investments_modern_tag_fallback(self):
        """ShortTermInvestments устарел — Apple использует MarketableSecuritiesCurrent."""
        data = {"cik": 320193, "entityName": "APPLE INC", "facts": {"us-gaap": {
            "MarketableSecuritiesCurrent": {"units": {"USD": [_fact_row(val=31_601_000_000, start=None)]}},
        }}}
        out = normalize_metrics(data)
        assert out["st_investments"][0]["val"] == pytest.approx(31_601_000_000)

    def test_tag_selected_by_freshness(self):
        """Смена тега компанией (CapEx AMZN): свежий тег ведёт ряд, старый дополняет историю.

        PaymentsToAcquirePropertyPlantAndEquipment заканчивается 2016, а
        PaymentsToAcquireProductiveAssets — актуальный (2025): ряд склеивается
        в непрерывную историю, свежие годы — из нового тега.
        """
        stale = _fact_row(end="2016-12-31", start="2016-01-01", val=6.7e9, fy=2016)
        fresh = _fact_row(end="2025-12-31", start="2025-01-01", val=131.8e9, fy=2025)
        facts = _facts_multi({
            "PaymentsToAcquirePropertyPlantAndEquipment": [stale],
            "PaymentsToAcquireProductiveAssets": [fresh],
        })
        out = normalize_metrics(facts)
        capex = out["capex"]
        assert len(capex) == 2  # оба тега: история + актуальные годы
        by_year = {r["end"].year: r for r in capex}
        assert by_year[2016]["tag"] == "PaymentsToAcquirePropertyPlantAndEquipment"
        assert by_year[2016]["val"] == pytest.approx(6.7e9)
        assert by_year[2025]["tag"] == "PaymentsToAcquireProductiveAssets"
        assert by_year[2025]["val"] == pytest.approx(131.8e9)

    def test_tag_priority_when_equal_freshness(self):
        """Одинаковая свежесть — побеждает первый тег (порядок METRICS)."""
        facts = _facts_multi({
            "SalesRevenueNet": [_fact_row(val=2.0)],
            "Revenues": [_fact_row(val=1.0)],
        })
        out = normalize_metrics(facts)
        assert out["revenue"][0]["tag"] == "Revenues"
        assert out["revenue"][0]["val"] == pytest.approx(1.0)

    def test_tag_switch_merged_series(self):
        """Смена тега (ASC 606): ряды склеиваются в непрерывную историю без дублей."""
        old = [
            _fact_row(end="%d-12-31" % y, start="%d-01-01" % y, val=100.0 + y, fy=y, filed="%d-02-01" % (y + 1))
            for y in range(2010, 2018)
        ]
        new = [
            _fact_row(end="%d-12-31" % y, start="%d-01-01" % y, val=100.0 + y, fy=y, filed="2026-02-01")
            for y in range(2017, 2027)
        ]
        facts = _facts_multi({
            "SalesRevenueNet": old,
            "RevenueFromContractWithCustomerExcludingAssessedTax": new,
        })
        out = normalize_metrics(facts)
        rev = out["revenue"]
        years = [r["end"].year for r in rev]
        assert years[0] == 2010 and years[-1] == 2026
        assert len(years) == len(set(years))  # период не дублируется
        # Пересечение 2017 — покрыт один раз, тегом-«новичком» (свежее подача)
        rows_2017 = [r for r in rev if r["end"].year == 2017]
        assert len(rows_2017) == 1
        assert rows_2017[0]["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"

    def test_inconsistent_overlap_not_mixed(self):
        """Разные концепты с одинаковым периодом не смешиваются (без двойного счёта)."""
        old = _fact_row(end="2016-12-31", start="2016-01-01", val=10.0, fy=2016, filed="2017-02-01")
        new_2016 = _fact_row(end="2016-12-31", start="2016-01-01", val=999.0, fy=2016, filed="2026-02-01")
        new_2025 = _fact_row(end="2025-12-31", start="2025-01-01", val=999.0, fy=2025, filed="2026-02-01")
        facts = _facts_multi({
            "SalesRevenueNet": [old],
            "Revenues": [new_2016, new_2025],
        })
        out = normalize_metrics(facts)
        rev = out["revenue"]
        assert len(rev) == 2  # 2016 (только свежий тег) + 2025
        rows_2016 = [r for r in rev if r["end"].year == 2016]
        assert len(rows_2016) == 1
        assert rows_2016[0]["tag"] == "Revenues"
        assert rows_2016[0]["val"] == pytest.approx(999.0)

    def test_unknown_tag_discovered_by_keywords(self):
        """Тег вне курируемого списка находится автопоиском по ключевым словам."""
        facts = _facts_multi({
            "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets": [
                _fact_row(end="2025-12-31", start="2025-01-01", val=-50e9, fy=2025, filed="2026-02-01"),
            ],
        })
        out = normalize_metrics(facts)
        capex = out["capex"]
        assert capex and capex[0]["tag"] == "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"
        assert capex[0]["val"] == pytest.approx(-50e9)

    def test_bank_revenue_no_double_count(self):
        """Банк: InterestIncome и NoninterestIncome — разные концепты, ряд НЕ удваивается."""
        facts = _facts_multi({
            "InterestIncome": [
                _fact_row(end="2024-12-31", start="2024-01-01", val=10e9, fy=2024),
                _fact_row(end="2025-12-31", start="2025-01-01", val=11e9, fy=2025),
            ],
            "NoninterestIncome": [
                _fact_row(end="2024-12-31", start="2024-01-01", val=2e9, fy=2024),
                _fact_row(end="2025-12-31", start="2025-01-01", val=2.2e9, fy=2025),
            ],
        })
        out = normalize_metrics(facts)
        rev = out["revenue"]
        assert len(rev) == 2
        assert all(r["tag"] == "InterestIncome" for r in rev)
        assert rev[-1]["val"] == pytest.approx(11e9)

    def test_hard_negative_excludes_wrong_concept(self):
        """Автопоиск не подхватывает теги чужих концептов (NetIncome ≠ revenue)."""
        facts = _facts_multi({
            "NetIncomeLoss": [_fact_row(end="2025-12-31", val=5e9, fy=2025, filed="2026-02-01")],
        })
        out = normalize_metrics(facts)
        assert "revenue" not in out  # NetIncomeLoss не считается выручкой
        assert out["net_income"][0]["tag"] == "NetIncomeLoss"


# ══════════════════════════════════════════════════════════════════════
#  3. Deduplicator (latest filed per period)
# ══════════════════════════════════════════════════════════════════════
class TestDedupe:
    def test_latest_filed_wins(self):
        """Переподача: одна отчётность, два filed — берём поздний."""
        rows = [
            _dated_row(end="2023-09-30", filed="2023-11-03", val=383.0),
            _dated_row(end="2023-09-30", filed="2024-05-10", val=385.0),
        ]
        out = _dedupe_rows(rows)
        assert len(out) == 1
        assert out[0]["val"] == 385.0
        assert out[0]["filed"].isoformat() == "2024-05-10"

    def test_quarterly_3m_and_9m_both_kept(self):
        """10-Q: одна end, разные start (3M и 9M накопительно) — обе строки."""
        rows = [
            _dated_row(end="2024-06-29", start="2024-03-31", form="10-Q", fp="Q3", val=85.0),
            _dated_row(end="2024-06-29", start="2023-10-01", form="10-Q", fp="Q3", val=208.0),
        ]
        out = _dedupe_rows(rows)
        assert len(out) == 2

    def test_sorted_by_end(self):
        rows = [
            _dated_row(end="2022-09-30", val=1.0),
            _dated_row(end="2023-09-30", val=2.0),
            _dated_row(end="2021-09-30", val=3.0),
        ]
        out = _dedupe_rows(rows)
        assert [r["end"].isoformat() for r in out] == ["2021-09-30", "2022-09-30", "2023-09-30"]

    def test_no_filed_uses_oldest(self):
        """Строка без filed не перебивает строку с filed (date.min fallback)."""
        rows = [
            _dated_row(end="2023-09-30", filed=None, val=380.0),
            _dated_row(end="2023-09-30", filed="2024-05-10", val=385.0),
        ]
        out = _dedupe_rows(rows)
        assert len(out) == 1
        assert out[0]["val"] == 385.0

    def test_filter_period(self):
        fy = _dated_row(form="10-K")
        q = _dated_row(form="10-Q", end="2023-12-31", start="2023-10-01")
        assert len(filter_period([fy, q], "FY")) == 1
        assert len(filter_period([fy, q], "Q")) == 1
        assert len(filter_period([fy, q], "ALL")) == 2


# ══════════════════════════════════════════════════════════════════════
#  4. Расчёты производных показателей (чистые функции)
# ══════════════════════════════════════════════════════════════════════
class TestCalculations:
    def _by_metric(self) -> dict[str, list[dict]]:
        return normalize_metrics(_apple_facts())

    def test_annual_rows(self):
        annual = build_annual(self._by_metric())
        assert len(annual) == 2
        last = annual[-1]
        assert last["fy"] == 2024
        assert last["end"].isoformat() == "2024-09-28"

        # Потоковые
        assert last["revenue"] == pytest.approx(391_035_000_000)
        assert last["net_income"] == pytest.approx(93_736_000_000)
        assert last["operating_income"] == pytest.approx(123_216_000_000)
        assert last["eps_basic"] == pytest.approx(6.08)
        assert last["cfo"] == pytest.approx(118_254_000_000)
        assert last["capex"] == pytest.approx(-9_447_000_000)

        # Производные: FCF = CFO − |CapEx|; EBITDA = EBIT + D&A; margin; ROE
        assert last["free_cash_flow"] == pytest.approx(118_254_000_000 - 9_447_000_000)
        assert last["ebitda"] == pytest.approx(123_216_000_000 + 11_445_000_000)
        assert last["operating_margin"] == pytest.approx(123_216_000_000 / 391_035_000_000)
        assert last["roe"] == pytest.approx(93_736_000_000 / 56_950_000_000)

        # Баланс на конец года
        assert last["total_debt"] == pytest.approx(85_750_000_000 + 9_812_000_000)
        assert last["net_debt"] == pytest.approx(
            85_750_000_000 + 9_812_000_000 - 29_943_000_000 - 31_601_000_000
        )
        assert last["cash"] == pytest.approx(29_943_000_000)
        assert last["equity"] == pytest.approx(56_950_000_000)
        assert last["shares_outstanding"] == pytest.approx(15_419_532_000)

    def test_annual_sorted_asc(self):
        annual = build_annual(self._by_metric())
        assert [r["end"] for r in annual] == sorted(r["end"] for r in annual)

    def test_ebitda_without_dda(self):
        """Нет D&A → EBITDA = EBIT (fallback)."""
        facts = _apple_facts()
        facts["facts"]["us-gaap"].pop("DepreciationDepletionAndAmortization", None)
        by = normalize_metrics(facts)
        assert "dda" not in by
        last = build_annual(by)[-1]
        assert last["ebitda"] == pytest.approx(123_216_000_000)

    def test_fcf_without_capex_is_none(self):
        facts = _apple_facts()
        facts["facts"]["us-gaap"].pop("PaymentsToAcquirePropertyPlantAndEquipment", None)
        by = normalize_metrics(facts)
        last = build_annual(by)[-1]
        assert last["free_cash_flow"] is None

    def test_total_debt_combined_fallback(self):
        """Если есть DebtLongtermAndShorttermCombined — берём его, не сумму."""
        facts = _apple_facts()
        facts["facts"]["us-gaap"]["DebtLongtermAndShorttermCombined"] = {
            "units": {"USD": [_fact_row(end="2024-09-28", start="2023-10-01", val=90_000_000_000, filed="2024-11-01")]}
        }
        by = normalize_metrics(facts)
        last = build_annual(by)[-1]
        assert last["total_debt"] == pytest.approx(90_000_000_000)

    def test_latest_balance(self):
        bal = latest_balance(self._by_metric())
        assert bal["end"].isoformat() == "2024-09-28"
        assert bal["cash"] == pytest.approx(29_943_000_000)
        assert bal["st_investments"] == pytest.approx(31_601_000_000)
        assert bal["total_debt"] == pytest.approx(95_562_000_000)
        assert bal["net_debt"] == pytest.approx(95_562_000_000 - 61_544_000_000)
        assert bal["equity"] == pytest.approx(56_950_000_000)
        assert bal["shares_outstanding"] == pytest.approx(15_419_532_000)

    def test_latest_balance_empty(self):
        bal = latest_balance({})
        assert bal["end"] is None
        assert bal["total_debt"] is None
        assert bal["net_debt"] is None


# ══════════════════════════════════════════════════════════════════════
#  5. Сервис: конвейер + PG + рыночные мультипликаторы
# ══════════════════════════════════════════════════════════════════════
class TestService:
    def _patch_edgar(self, monkeypatch, facts: dict | None = None, tickers: dict | None = None):
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: tickers or {"AAPL": "0000320193", "JPM": "0000019617"},
        )
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_company_facts",
            lambda cik: facts if facts is not None else _facts_multi(),
        )

    def test_full_pipeline(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch)
        svc = SecFundamentalsService(redis_client=None)
        out = svc.get_revenue("aapl", "FY")  # lowercase — нормализуется
        assert out["ticker"] == "AAPL"
        assert out["cik"] == "0000320193"
        assert out["period"] == "FY"
        assert out["count"] >= 1
        assert out["source_tag"] == "Revenues"
        assert out["rows"][0]["val"] == 383285000000.0
        assert out["rows"][0]["end"].isoformat() == "2023-09-30"

        # Персистент в PostgreSQL/SQLite (company_metrics)
        with SessionLocal() as db:
            n = db.query(CompanyMetric).filter(
                CompanyMetric.ticker == "AAPL", CompanyMetric.metric == "revenue"
            ).count()
            assert n == out["count"]

    def test_freshness_skips_edgar(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch)
        svc = SecFundamentalsService(redis_client=None)
        assert svc.get_revenue("AAPL", "FY")["count"] >= 1

        # Повторный вызов в окне TTL: EDGAR не дёргается, данные из PG
        def _boom(cik: str):  # pragma: no cover
            raise AssertionError("EDGAR вызван повторно при свежих данных")

        monkeypatch.setattr("gex.application.sec.sec_fundamentals.get_company_facts", _boom)
        out = svc.get_revenue("AAPL", "FY")
        assert out["count"] >= 1

    def test_unknown_ticker_raises_valueerror(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch)
        svc = SecFundamentalsService(redis_client=None)
        with pytest.raises(ValueError):
            svc.get_revenue("ZZZZ", "FY")

    def test_quarterly_period_keeps_3m_9m(self, db_ready, monkeypatch):
        facts = _facts_multi({"Revenues": [
            _fact_row(form="10-K", fp="FY"),
            _fact_row(end="2024-06-29", start="2024-03-31", form="10-Q", fp="Q3", val=85.0),
            _fact_row(end="2024-06-29", start="2023-10-01", form="10-Q", fp="Q3", val=208.0),
        ]})
        self._patch_edgar(monkeypatch, facts=facts)
        svc = SecFundamentalsService(redis_client=None)

        q = svc.get_revenue("AAPL", "Q")
        assert q["count"] == 2
        assert all(r["form"] == "10-Q" for r in q["rows"])
        assert len({r["end"] for r in q["rows"]}) == 1  # одна end, две start

        fy = svc.get_revenue("AAPL", "FY")
        assert fy["count"] == 1

    def test_bank_financial_fallback(self, db_ready, monkeypatch):
        facts = _facts_multi({"InterestIncome": [_fact_row()]})
        self._patch_edgar(monkeypatch, facts=facts)
        svc = SecFundamentalsService(redis_client=None)
        out = svc.get_revenue("JPM", "FY")
        assert out["source_tag"] == "InterestIncome"
        assert out["count"] >= 1

    def test_tag_priority_when_multiple(self, db_ready, monkeypatch):
        """Несколько тегов с данными → берётся первый по приоритету REVENUE_TAGS."""
        facts = _facts_multi({
            "SalesRevenueNet": [_fact_row(val=2.0)],
            "Revenues": [_fact_row(val=999.0)],
        })
        self._patch_edgar(monkeypatch, facts=facts)
        svc = SecFundamentalsService(redis_client=None)
        out = svc.get_revenue("AAPL", "FY")
        assert out["source_tag"] == "Revenues"  # приоритетнее SalesRevenueNet
        assert out["rows"][0]["val"] == 999.0

    def test_edgar_failure_propagates(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch)

        def _boom(cik: str):  # pragma: no cover
            raise RuntimeError("SEC EDGAR недоступен")

        monkeypatch.setattr("gex.application.sec.sec_fundamentals.get_company_facts", _boom)
        svc = SecFundamentalsService(redis_client=None)
        with pytest.raises(RuntimeError):
            svc.get_revenue("AAPL", "FY")

    # ── Fundamentals: core + market ──────────────────────────────────
    def test_fundamentals_core(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch, facts=_apple_facts())
        svc = SecFundamentalsService(redis_client=None)
        core = svc.get_fundamentals_core("AAPL")
        assert core["ticker"] == "AAPL"
        assert core["cik"] == "0000320193"
        assert len(core["annual"]) == 2
        assert core["annual"][-1]["free_cash_flow"] == pytest.approx(108_807_000_000)
        assert core["balance"]["net_debt"] == pytest.approx(34_018_000_000)
        assert core["balance"]["shares_outstanding"] == pytest.approx(15_419_532_000)

    def test_market_ratios(self):
        core = {"ticker": "AAPL", "cik": "0000320193", "annual": [], "balance": {}}
        svc = SecFundamentalsService(redis_client=None)
        out = svc.with_market(core, price=232.0)
        assert out["market_cap"] is None          # нет shares → нет mcap
        assert out["ratios"]["pe"] is None
        assert out["price"]["value"] == 232.0

    def test_market_ratios_full(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch, facts=_apple_facts())
        svc = SecFundamentalsService(redis_client=None)
        core = svc.get_fundamentals_core("AAPL")
        out = svc.with_market(core, price=232.0)

        mcap = 232.0 * 15_419_532_000
        assert out["market_cap"] == pytest.approx(mcap)
        assert out["ratios"]["pe"] == pytest.approx(232.0 / 6.08)
        assert out["ratios"]["ps"] == pytest.approx(mcap / 391_035_000_000)
        ebitda = 123_216_000_000 + 11_445_000_000
        net_debt = 34_018_000_000
        assert out["ratios"]["net_debt_to_ebitda"] == pytest.approx(net_debt / ebitda)

    def test_fetch_price(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch, facts=_apple_facts())
        # fetch_price импортирует TATimeframesFetcher локально → патчим атрибут модуля
        monkeypatch.setattr("gex.adapters.fetchers.ta_fetcher.TATimeframesFetcher", _FakeSpotFetcher)
        svc = SecFundamentalsService(redis_client=None)
        assert svc.fetch_price("AAPL") == pytest.approx(232.5)

    def test_fetch_price_failure_returns_none(self, db_ready, monkeypatch):
        self._patch_edgar(monkeypatch, facts=_apple_facts())

        class _FailingFetcher:
            def __init__(self, redis_client=None):
                pass

            def fetch_spot(self, ticker: str) -> float:
                raise ValueError("нет данных")

        monkeypatch.setattr("gex.adapters.fetchers.ta_fetcher.TATimeframesFetcher", _FailingFetcher)
        svc = SecFundamentalsService(redis_client=None)
        assert svc.fetch_price("AAPL") is None


class _FakeSpotFetcher:
    def __init__(self, redis_client=None):
        pass

    def fetch_spot(self, ticker: str) -> float:
        return 232.5


# ══════════════════════════════════════════════════════════════════════
#  6. API: GET /companies/{ticker}/revenue|fundamentals (TestClient + BASIC)
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def client(monkeypatch):
    """TestClient с изоляцией от реального Redis (result_cache не пишет/не читает)."""
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


def _register_and_subscribe(client, email: str = "sec@test.local", password: str = "pass1234") -> str:
    """Зарегистрировать юзера, выдать BASIC-подписку, вернуть access_token."""
    r = client.post("/auth/register", json={
        "email": email,
        "password": password,
        "accept_terms": True,
        "telegram_username": "@secbot",
    })
    assert r.status_code == 201
    token = r.json()["access_token"]

    db = SessionLocal()
    # Email-гейт (security-аудит 2026-09-04): data-ручки требуют верификацию.
    db.query(User).filter(User.email == email).update({
        "subscription_status": "BASIC",
        "is_email_verified": True,
        "subscription_activated_at": datetime.now(UTC),
        "subscription_expires_at": datetime.now(UTC) + timedelta(days=30),
    })
    db.commit()
    db.close()
    return token


class TestRevenueApi:
    def test_requires_auth(self, client):
        r = client.get("/companies/AAPL/revenue")
        assert r.status_code == 401

    def test_requires_subscription(self, client):
        r = client.post("/auth/register", json={
            "email": "sec_inactive@test.local", "password": "pass1234",
            "accept_terms": True, "telegram_username": "@secbot2",
        })
        assert r.status_code == 201
        token = r.json()["access_token"]  # INACTIVE
        resp = client.get("/companies/AAPL/revenue", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_revenue_ok(self, client, monkeypatch):
        token = _register_and_subscribe(client)
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: {"AAPL": "0000320193"},
        )
        monkeypatch.setattr("gex.application.sec.sec_fundamentals.get_company_facts", lambda cik: _facts_multi())

        r = client.get("/companies/AAPL/revenue", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["cik"] == "0000320193"
        assert data["period"] == "FY"
        assert data["count"] >= 1
        assert data["source_tag"] == "Revenues"
        row = data["rows"][0]
        assert row["end"] == "2023-09-30"  # дата сериализована в ISO
        assert row["val"] == 383285000000.0
        assert row["tag"] == "Revenues"

    def test_unknown_ticker_404(self, client, monkeypatch):
        token = _register_and_subscribe(client, email="sec_404@test.local")
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: {"AAPL": "0000320193"},
        )
        r = client.get("/companies/ZZZZ/revenue", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 404

    def test_bad_period_422(self, client):
        token = _register_and_subscribe(client, email="sec_422@test.local")
        r = client.get("/companies/AAPL/revenue?period=XX", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 422


class TestFundamentalsApi:
    def _patch_all(self, monkeypatch):
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: {"AAPL": "0000320193"},
        )
        monkeypatch.setattr("gex.application.sec.sec_fundamentals.get_company_facts", lambda cik: _apple_facts())
        monkeypatch.setattr(
            SecFundamentalsService, "fetch_price", lambda self, ticker: 232.0
        )

    def test_fundamentals_ok(self, client, monkeypatch):
        token = _register_and_subscribe(client)
        self._patch_all(monkeypatch)

        r = client.get("/companies/AAPL/fundamentals", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["cik"] == "0000320193"
        assert data["price"]["value"] == 232.0
        assert data["price"]["source"] == "yfinance"

        last = data["annual"][-1]
        assert last["fy"] == 2024
        assert last["revenue"] == pytest.approx(391_035_000_000)
        assert last["free_cash_flow"] == pytest.approx(108_807_000_000)
        assert last["ebitda"] == pytest.approx(134_661_000_000)
        assert last["operating_margin"] == pytest.approx(0.3151, abs=1e-3)
        assert last["roe"] == pytest.approx(93_736_000_000 / 56_950_000_000)
        assert last["total_debt"] == pytest.approx(95_562_000_000)

        assert data["balance"]["net_debt"] == pytest.approx(34_018_000_000)
        assert data["market_cap"] == pytest.approx(232.0 * 15_419_532_000)
        assert data["ratios"]["pe"] == pytest.approx(232.0 / 6.08)
        assert data["ratios"]["ps"] == pytest.approx(232.0 * 15_419_532_000 / 391_035_000_000)
        assert data["ratios"]["net_debt_to_ebitda"] == pytest.approx(
            34_018_000_000 / 134_661_000_000
        )

    def test_fundamentals_price_unavailable(self, client, monkeypatch):
        """yfinance недоступен → price=None, мультипликаторы null, annual отдаётся."""
        token = _register_and_subscribe(client, email="sec_noprice@test.local")
        self._patch_all(monkeypatch)
        monkeypatch.setattr(SecFundamentalsService, "fetch_price", lambda self, ticker: None)

        r = client.get("/companies/AAPL/fundamentals", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["price"] is None
        assert data["market_cap"] is None
        assert data["ratios"]["pe"] is None
        assert data["annual"][-1]["revenue"] == pytest.approx(391_035_000_000)

    def test_fundamentals_unknown_ticker_404(self, client, monkeypatch):
        token = _register_and_subscribe(client, email="sec_f404@test.local")
        monkeypatch.setattr(
            "gex.application.sec.sec_fundamentals.get_ticker_to_cik",
            lambda redis=None: {"AAPL": "0000320193"},
        )
        r = client.get("/companies/ZZZZ/fundamentals", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 404
