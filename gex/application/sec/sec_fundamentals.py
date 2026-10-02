"""SEC EDGAR fundamentals: нормализация XBRL-показателей, дедуп, расчёты.

Конвейер (backend-часть модуля фундаментального анализа)::

    Ticker input
      ↓
    Ticker → CIK resolver (company_tickers.json + Redis 24ч)
      ↓
    SEC EDGAR API client (data.sec.gov, UA + rate limit)
      ↓
    XBRL tag normalizer (us-gaap теги → метрики: revenue, net_income, ...)
      ↓
    Deduplicator / latest filed selector (по (end, start))
      ↓
    Storage: PostgreSQL (company_metrics) + Redis SWR-кэш ответа
      ↓
    GET /companies/{ticker}/revenue | /companies/{ticker}/fundamentals

Показатели (производные считаются на лету):
  * free cash flow      = CFO − |CapEx|
  * EBITDA              = Operating Income + D&A (fallback: без D&A)
  * Operating Margin    = Operating Income / Revenue
  * Net Debt            = Total Debt − (Cash + Short-term investments)
  * ROE                 = Net Income / Stockholders Equity
  * P/E                 = Price / EPS (последний FY, fallback NI / shares)
  * P/S                 = Market Cap / Revenue (последний FY)
  * Net Debt / EBITDA   = Net Debt (последний баланс) / EBITDA (последний FY)

Текущая цена берётся из yfinance через ``TATimeframesFetcher.fetch_spot``
(Redis-кэш 300с) — переиспользуется существующий фетчер.

Грабли, учтённые в реализации:

1. **Разные XBRL-тэги** — у каждого показателя список тегов по приоритету
   (``METRICS``); для каждого показателя берётся ПЕРВЫЙ тег с данными.
2. **Дубликаты (переподача)** — для каждого периода (end, start) берётся
   последняя версия по ``filed``.
3. **Годовой vs квартальный** — 10-K (год) и 10-Q (квартал); у 10-Q на одну
   дату end бывает две строки (3M и 9M, разные start) — обе сохраняются.
4. **Rate limit + User-Agent** — bucket ``sec`` (5 req/s) и обязательный
   ``SEC_USER_AGENT`` в :mod:`gex.sec_edgar`.
5. **Даты** — SEC отдаёт ISO-строки, SQLAlchemy Date требует date-объекты
   (``_parse_date`` в нормализаторе).
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from gex.auth.config import settings
from gex.adapters.persistence.database import SessionLocal
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.providers.sec_edgar import get_company_facts, get_ticker_to_cik
from gex.adapters.persistence.sec_models import CompanyMetric

logger = logging.getLogger(__name__)

#: Основные us-gaap теги выручки в порядке приоритета
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]

#: Для банков/финансовых компаний revenue лежит в других полях
FINANCIAL_REVENUE_TAGS = [
    "TotalRevenue",
    "InterestIncome",
    "NoninterestIncome",
]

#: Конфиг XBRL-показателей: metric → (теги по приоритету, тип).
#: flow — потоковые (за период), balance — балансовые (на дату).
METRICS: dict[str, tuple[list[str], str]] = {
    "revenue": (REVENUE_TAGS + FINANCIAL_REVENUE_TAGS, "flow"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "flow"),
    "operating_income": (["OperatingIncomeLoss"], "flow"),
    "eps_basic": (["EarningsPerShareBasic"], "flow"),
    "eps_diluted": (["EarningsPerShareDiluted"], "flow"),
    "cfo": (
        [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "flow",
    ),
    "capex": (
        ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
        "flow",
    ),
    "dda": (
        [
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
        ],
        "flow",
    ),
    "debt_combined": (["DebtLongtermAndShorttermCombined"], "balance"),
    "lt_debt": (["LongTermDebtNoncurrent"], "balance"),
    "st_debt": (["LongTermDebtCurrent", "ShortTermBorrowings"], "balance"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], "balance"),
    "st_investments": (["ShortTermInvestments", "MarketableSecuritiesCurrent"], "balance"),
    "equity": (
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "balance",
    ),
    "shares_outstanding": (["CommonStockSharesOutstanding"], "balance"),
}

FLOW_METRICS: set[str] = {m for m, (_, kind) in METRICS.items() if kind == "flow"}
BALANCE_METRICS: set[str] = {m for m, (_, kind) in METRICS.items() if kind == "balance"}

#: Периоды, поддерживаемые эндпоинтом revenue
PERIODS = ("FY", "Q", "ALL")

#: Базовый семантический балл курируемых тегов (доменное знание надёжнее
#: автопоиска: "Revenues" не проиграет "SalesRevenueGoodsNet" при равной свежести)
CURATED_SEMANTIC = 2.5

# ══════════════════════════════════════════════════════════════════════ #
#  Семантические профили XBRL-тегов (глубокий анализ соответствия)
# ══════════════════════════════════════════════════════════════════════ #
# Компании часто МЕНЯЮТ us-gaap теги (ASC 606, смена презентации, ребрендинг
# таксономии): Amazon перешла с PaymentsToAcquirePropertyPlantAndEquipment на
# PaymentsToAcquireProductiveAssets, многие компании после 2018 перешли с
# SalesRevenueNet на RevenueFromContractWithCustomer...
#
# Профиль метрики:
#   pos          — ключевые слова КОНЦЕПТА (автопоиск + семантический скоринг);
#   neg_penalty  — менее желательные варианты того же концепта (штраф 0.5);
#   neg_hard     — ДРУГИЕ концепты: теги с ними исключаются из АВТОПОИСКА
#                  (курируемые теги в METRICS участвуют всегда);
#   positive     — балансовая метрика, ожидаемо > 0 (штраф за ≤ 0).
TAG_KEYWORDS: dict[str, dict[str, tuple[str, ...] | bool]] = {
    "revenue": {
        "pos": ("Revenue", "Revenues", "Sales", "Turnover"),
        "neg_penalty": ("Unearned", "Deferred", "Liability", "Return", "Allowance",
                         "Discount", "Refund", "PerShare", "Segment", "CostOfGoods"),
        "neg_hard": ("Comprehensive", "NetIncome", "Expense", "Financing", "Investing",
                      "InterestIncome", "NoninterestIncome", "OperatingCashFlow"),
    },
    "net_income": {
        "pos": ("NetIncome", "ProfitLoss", "IncomeLoss"),
        "neg_penalty": ("Comprehensive", "Noncontrolling", "AvailableToCommon",
                         "ContinuingOperations", "Discontinued", "PerShare",
                         "AttributableToParent", "Minority"),
        "neg_hard": ("Revenue", "OperatingIncome", "GrossProfit", "IncomeTax",
                      "BeforeIncomeTaxes", "Pretax", "ComprehensiveIncome"),
    },
    "operating_income": {
        "pos": ("OperatingIncome", "OperatingIncomeLoss"),
        "neg_penalty": ("BeforeIncomeTaxes", "Nonoperating", "Other", "GainLoss",
                         "Interest", "EquityMethod", "Including"),
        "neg_hard": ("NetIncome", "GrossProfit", "IncomeTax", "Revenue"),
    },
    "eps_basic": {
        "pos": ("EarningsPerShareBasic", "PerBasicShare", "BasicEarningsPerShare",
                 "IncomeLossPerBasicShare"),
        "neg_penalty": ("Diluted", "Comprehensive", "Continuing", "Discontinued",
                         "Noncontrolling"),
        "neg_hard": ("EarningsPerShareDiluted", "PerDilutedShare", "Revenue",
                      "NetIncomeLoss", "GrossProfit"),
    },
    "eps_diluted": {
        "pos": ("EarningsPerShareDiluted", "PerDilutedShare"),
        "neg_penalty": ("Comprehensive", "Continuing", "Discontinued", "Noncontrolling"),
        "neg_hard": ("EarningsPerShareBasic", "PerBasicShare", "Revenue", "NetIncomeLoss"),
    },
    "cfo": {
        "pos": ("NetCashProvidedByUsedInOperatingActivities",
                 "CashProvidedByUsedInOperatingActivities", "OperatingActivities"),
        "neg_penalty": ("Discontinued", "ContinuingOperations"),
        "neg_hard": ("Investing", "Financing", "CapitalExpenditure", "Dividend", "Interest"),
    },
    "capex": {
        "pos": ("PaymentsToAcquire", "CapitalExpenditure", "AdditionsTo", "PurchaseOfProperty",
                 "PaymentsForProperty", "PropertyPlantAndEquipment", "ProductiveAssets"),
        "neg_penalty": ("Intangible", "Business", "AndIntangibleAssets", "Combined", "Lease"),
        # Балансовые/чужие концепты: Accumulated/Gross/Net PP&E, покупка ценных бумаг
        "neg_hard": ("Accumulated", "Depreciation", "Gross", "Net", "Securities",
                      "MarketableSecurities", "Business", "BusinessCombination", "Investments",
                      "Debt", "Repayments", "Dividends", "Repurchase", "PaymentsToAcquireInvestments",
                      "Proceeds", "DeferredTax", "Liabilities"),
    },
    "dda": {
        "pos": ("Depreciation", "Amortization", "Depletion", "DepreciationAndAmortization"),
        "neg_penalty": ("Accretion", "Impairment", "Disposal", "Write", "Intangible",
                         "FinancingCost"),
        "neg_hard": ("Accumulated", "Gross", "IncomeTax", "OperatingIncome", "NetIncome",
                      "Interest"),
    },
    "debt_combined": {
        "pos": ("LongtermAndShortterm", "TotalDebt", "CombinedDebt", "DebtTotal", "Combined"),
        "neg_penalty": ("Lease", "Guarantee", "Commitment", "Contingency", "Unpaid", "Allowance"),
        # Компонентные теги (LongTermDebtNoncurrent, LongTermDebtCurrent, ...)
        # НЕ являются суммарным долгом — в автопоиск не попадают
        "neg_hard": ("LongTerm", "ShortTerm", "Noncurrent", "Current", "Lease", "Convertible",
                      "AccountsPayable", "Payables", "Accrued", "Revenue", "Income", "Equity",
                      "Cash", "Receivable", "Expense", "Deferred", "Deposit"),
    },
    "lt_debt": {
        "pos": ("LongTermDebt", "LongTermBorrowings", "LongTermObligations", "NoncurrentDebt"),
        "neg_penalty": ("Lease", "Convertible", "DueWithin"),
        "neg_hard": ("Current", "ShortTerm", "Proceeds", "Repayments", "Payments", "Issuance",
                      "Repurchase", "Allocation", "PurchasePrice", "Acquisition", "Liabilities",
                      "Interest", "Expense", "AccountsPayable", "Equity", "Cash", "Revenue"),
    },
    "st_debt": {
        "pos": ("ShortTermDebt", "ShortTermBorrowings", "CurrentDebt", "LongTermDebtCurrent",
                 "CurrentPortion"),
        "neg_penalty": ("Noncurrent",),
        "neg_hard": ("LongTermDebtNoncurrent", "Proceeds", "Repayments", "Payments", "Issuance",
                      "Repurchase", "Interest", "Expense", "AccountsPayable", "Equity", "Cash", "Revenue"),
    },
    "cash": {
        "pos": ("CashAndCashEquivalents", "CashCashEquivalents", "CashAtCarryingValue"),
        "neg_penalty": ("Restricted", "Discontinued", "HeldForSale"),
        "neg_hard": ("Investment", "Receivable", "Payable", "Revenue", "Income",
                      "MarketableSecurities", "PeriodIncreaseDecrease", "Increase", "Decrease",
                      "Effect", "ExchangeRate"),
        "positive": True,
    },
    "st_investments": {
        "pos": ("ShortTermInvestments", "MarketableSecuritiesCurrent", "InvestmentsCurrent",
                 "AvailableForSaleSecuritiesCurrent", "TradingSecuritiesCurrent",
                 "HeldToMaturitySecuritiesCurrent"),
        "neg_penalty": ("Other", "Total", "Noncurrent"),
        "neg_hard": ("LongTerm", "EquityMethod", "CostMethod", "Cash", "Receivable"),
        "positive": True,
    },
    "equity": {
        "pos": ("StockholdersEquity", "ShareholdersEquity", "EquityAttributable"),
        "neg_penalty": ("AttributableToParent", "CommonStock", "PreferredStock",
                         "RetainedEarnings", "AdditionalPaidInCapital", "AccumulatedOther",
                         "Treasury"),
        "neg_hard": ("CommonStock", "PreferredStock", "RetainedEarnings",
                      "AdditionalPaidInCapital", "AccumulatedOther", "Treasury", "Deferred",
                      "Note", "Ratio", "Conversion", "StockSplit", "PerShare",
                      "Revenue", "Income", "Cash", "Debt"),
        "positive": True,
    },
    "shares_outstanding": {
        "pos": ("SharesOutstanding", "SharesIssued", "CommonStockShares"),
        "neg_penalty": ("Authorized", "Treasury", "Warrants", "Restricted", "Voting",
                         "HeldBy", "Options"),
        "neg_hard": ("Authorized", "Treasury", "Warrants", "Restricted", "Voting",
                      "HeldBy", "Options", "Convertible", "WeightedAverage", "Diluted",
                      "Average", "Basic", "Noncash", "Acquisition", "Consideration", "Instrument",
                      "Revenue", "Income", "Debt", "Cash"),
        "positive": True,
    },
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(dt: datetime) -> datetime:
    """SQLite отдаёт naive datetime — нормализуем к UTC перед вычитанием."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parse_date(value: str | None) -> date | None:
    """ISO-строка из XBRL → date. SEC отдаёт 'YYYY-MM-DD', а SQLAlchemy
    Date (и SQLite, и PostgreSQL) требует date-объекты."""
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════ #
#  XBRL tag normalizer + deduplicator (чистые функции)
# ══════════════════════════════════════════════════════════════════════ #
def _semantic_score(tag: str, profile: dict) -> float | None:
    """Семантический скоринг тега по ключевым словам профиля метрики.

    Returns
    -------
    float | None
        ``None`` — тег относится к ДРУГОМУ концепту (есть hard-negative слово);
        иначе число: +1.0 за каждое положительное слово (кап 3.0), −0.5 за
        каждое нежелательное (кап −2.0).
    """
    for kw in profile.get("neg_hard", ()):
        if kw in tag:
            return None
    score = 0.0
    for kw in profile.get("pos", ()):
        if kw in tag:
            score += 1.0
    score = min(score, 3.0)
    for kw in profile.get("neg_penalty", ()):
        if kw in tag:
            score -= 0.5
    return max(score, -2.0)


def _analyze_tag_rows(rows: list[dict]) -> dict:
    """Анализ ряда тега: свежесть, покрытие истории, свежесть подачи."""
    annual = [r for r in rows if r.get("end") and _is_annual_row(r)]
    return {
        "annual_count": len(annual),
        "latest_annual": max((r["end"] for r in annual), default=None),
        "latest_end": max((r["end"] for r in rows if r.get("end")), default=None),
        "latest_filed": max((r["filed"] for r in rows if r.get("filed")), default=None),
        "latest_val": rows[-1]["val"] if rows else None,
    }


def _tag_score(analysis: dict, semantic: float, now: datetime, positive: bool = False) -> float:
    """Взвешенный рейтинг тега: свежесть + покрытие + семантика + подача.

    * свежесть годовых строк (актуальные данные — главный приоритет): до 10;
    * покрытие истории (число годовых строк): до 8;
    * семантическое соответствие концепту: до ~15;
    * свежесть подачи (недавно переизданные отчёты): до 2;
    * балансовая метрика с последним значением ≤ 0: −2 (сомнительный тег).
    """
    score = 0.0
    la = analysis["latest_annual"]
    if la is not None:
        age = max(0, now.year - la.year)
        score += max(0.0, 10.0 - 2.0 * age)
    else:
        le = analysis["latest_end"]
        if le is not None:
            age = max(0, now.year - le.year)
            score += max(0.0, 4.0 - 0.5 * age)
    score += min(analysis["annual_count"], 10) * 0.8
    score += semantic * 5.0
    lf = analysis["latest_filed"]
    if lf is not None:
        days = max(0, (now.date() - lf).days)
        score += max(0.0, 2.0 - days / 365.0)
    if positive and analysis["latest_val"] is not None and analysis["latest_val"] <= 0:
        score -= 2.0
    return score


def _merge_tag_rows(candidates: list[tuple[float, str, list[dict]]]) -> list[dict]:
    """Объединить ряды тегов в один непрерывный ряд (по рейтингу).

    Период берётся из ЛУЧШЕГО тега, где он есть; пропуски заполняются
    следующими по рейтингу тегами. ``end``, уже покрытый более высоким тегом,
    не дублируется — это исключает двойной счёт РАЗНЫХ концептов с одинаковыми
    периодами (InterestIncome vs NoninterestIncome у банков) и сохраняет
    историю при смене тега (SalesRevenueNet 2010-2017 + RevenueFromContract...
    2018-2026 → непрерывный ряд).
    """
    merged: dict[tuple, dict] = {}
    covered_ends: dict[date, str] = {}  # end -> тег, покрывший период
    for _score, tag, rows in candidates:  # отсортированы по убыванию рейтинга
        for row in rows:
            key = (row["end"], row["start"])
            if key in merged:
                continue
            end = row["end"]
            # end уже покрыт ДРУГИМ тегом: не дублируем (исключает двойной счёт
            # разных концептов). Строки ТОГО ЖЕ тега (3M и 9M на одну end) — ок.
            if end is not None and end in covered_ends and covered_ends[end] != tag:
                continue
            if end is not None:
                covered_ends[end] = tag
            merged[key] = row
    return sorted(merged.values(), key=lambda r: (r["end"] or date.min, r["start"] or date.min))


def _select_metric_rows(
    us_gaap: dict,
    metric: str,
    curated_tags: list[str],
    profile: dict,
    now: datetime,
) -> list[dict]:
    """Лучший ряд метрики: анализ всех кандидатов + умная склейка.

    Алгоритм:

    1. **Кандидаты** — курируемые теги (доменное знание, всегда участвуют)
       + автопоиск по ключевым словам профиля среди ВСЕХ тегов companyfacts
       (компании используют неожиданные варианты: смена тега, редкие таксономии);
    2. **Анализ** каждого кандидата: свежесть годовых строк, покрытие истории,
       свежесть подачи, знак значений (для балансовых метрик);
    3. **Рейтинг** — взвешенный балл (актуальность данных главнее всего,
       семантика — гарантия «того самого» концепта, при равенстве — порядок
       курируемого списка);
    4. **Склейка** — ряд строится из лучшего тега, пропуски заполняются из
       остальных (только непересекающиеся периоды) → непрерывная история даже
       при смене XBRL-тега.
    """
    candidates: list[tuple[float, str, list[dict]]] = []
    considered: set[str] = set()

    def _extract(tag: str) -> list[dict]:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            return []
        rows: list[dict] = []
        for unit_rows in tag_data.get("units", {}).values():
            for row in unit_rows:
                if row.get("form") not in ("10-K", "10-Q"):
                    continue
                rows.append({
                    "metric": metric,
                    "tag": tag,
                    "end": _parse_date(row.get("end")),
                    "start": _parse_date(row.get("start")),
                    "val": row.get("val"),
                    "fy": row.get("fy"),
                    "fp": row.get("fp"),
                    "form": row.get("form"),
                    "frame": row.get("frame"),
                    "filed": _parse_date(row.get("filed")),
                })
        return _dedupe_rows(rows)

    def _consider(tag: str, semantic: float) -> None:
        if tag in considered:
            return
        considered.add(tag)
        deduped = _extract(tag)
        if not deduped:
            return
        analysis = _analyze_tag_rows(deduped)
        score = _tag_score(analysis, semantic, now, bool(profile.get("positive")))
        candidates.append((score, tag, deduped))

    # 1) Курируемые теги — базовый семантический балл (доменное знание)
    for tag in curated_tags:
        _consider(tag, CURATED_SEMANTIC)
    # 2) Автопоиск: остальные теги companyfacts, семантически близкие концепту
    if profile.get("pos"):
        for tag in us_gaap:
            if tag in considered:
                continue
            semantic = _semantic_score(tag, profile)
            if semantic is None or semantic <= 0.3:
                continue
            _consider(tag, semantic)

    if not candidates:
        return []
    # Стабильная сортировка: при равенстве баллов — порядок добавления
    # (курируемый приоритет и порядок списка тегов)
    candidates.sort(key=lambda c: -c[0])
    return _merge_tag_rows(candidates)


def normalize_metrics(company_facts: dict) -> dict[str, list[dict]]:
    """Вытащить из XBRL-фактов все показатели (глубокий анализ тегов).

    Для каждого показателя:

    * кандидаты — курируемые us-gaap теги + автопоиск по ключевым словам
      (компании меняют теги: ASC 606, смена презентации);
    * каждый кандидат анализируется по свежести (годовые строки в первую
      очередь), покрытию истории, свежести подачи и знаку значений;
    * выбирается тег с ЛУЧШИМ рейтингом, ряды тегов склеиваются в
      непрерывную историю (пропуски заполняются из других тегов без
      дублей периодов и без смешивания разных концептов).

    Returns
    -------
    dict[metric → list[dict]]
        Для каждого показателя строки: ``{metric, tag, end, start, val, fy,
        fp, form, frame, filed}``. Берутся только отчётные формы 10-K / 10-Q.
    """
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    result: dict[str, list[dict]] = {}
    now = _utcnow()

    for metric, (tags, _kind) in METRICS.items():
        profile = TAG_KEYWORDS.get(metric, {})
        rows = _select_metric_rows(us_gaap, metric, tags, profile, now)
        if rows:
            result[metric] = rows

    return result


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    """Оставить последнюю поданную версию (по ``filed``) для каждого периода.

    Ключ периода — ``(end, start)``: у 10-Q на одну дату end бывает две
    строки — за 3 месяца и за 9 месяцев нарастающим итогом (разные start),
    обе нужны. У 10-K start обычно задан; если нет — ключ ``(end, None)``.
    """
    latest: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("end"), row.get("start"))
        prev = latest.get(key)
        filed = row.get("filed") or date.min
        if prev is None:
            latest[key] = row
            continue
        prev_filed = prev.get("filed") or date.min
        if filed > prev_filed:
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (r.get("end") or date.min, r.get("start") or date.min))


def _is_annual_row(row: dict) -> bool:
    """Годовая строка: duration ≈ 360 дней ИЛИ годовой XBRL frame.

    SEC помечает в companyfacts КВАРТАЛЬНЫЕ строки, перепечатанные в 10-K,
    как ``form=10-K`` (Apple публикует в 10-K ретроспективные кварталы) —
    фильтр по форме недостаточен. Надёжные признаки:

    * длительность периода (end−start ≈ 360 дней);
    * XBRL frame: годовой ``CY2017``/``FY2017`` vs квартальный ``CY2017Q4``
      (суффикс Qn) — используется, когда start отсутствует.
    """
    start, end = row.get("start"), row.get("end")
    if start is not None and end is not None:
        return 340 <= (end - start).days <= 400
    frame = row.get("frame") or ""
    if re.search(r"Q\d$", frame):
        return False
    return row.get("form") == "10-K"


def _is_quarter_row(row: dict) -> bool:
    """Квартальная строка: duration < года ИЛИ XBRL frame с Q-суффиксом.

    10-K публикует ретроспективные кварталы (frame ``CY2017Q4``, duration
    ≈ 90д) — включаются. Накопительные 10-Q (6M/9M, duration 180/270д) —
    тоже квартальная отчётность. Годовые (340-400д) — не квартальные.
    """
    start, end = row.get("start"), row.get("end")
    if start is not None and end is not None:
        days = (end - start).days
        return 70 <= days <= 330
    frame = row.get("frame") or ""
    if re.search(r"Q\d$", frame):
        return True
    return row.get("form") == "10-Q"


def filter_period(rows: list[dict], period: str) -> list[dict]:
    """Отфильтровать строки по периоду: ``FY`` (годовые) | ``Q`` (квартальные) | ``ALL``."""
    if period == "FY":
        return [r for r in rows if _is_annual_row(r)]
    if period == "Q":
        return [r for r in rows if _is_quarter_row(r)]
    return rows


# ══════════════════════════════════════════════════════════════════════ #
#  Расчёт производных показателей (чистые функции)
# ══════════════════════════════════════════════════════════════════════ #
def _row_at_end(rows: list[dict], end: date) -> dict | None:
    """Строка метрики с данным ``end`` (max start — защита от смены фин.года)."""
    candidates = [r for r in rows if r.get("end") == end]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("start") or date.min)


def _last_row(rows: list[dict]) -> dict | None:
    """Последняя строка по (end, start) — для балансовых показателей."""
    if not rows:
        return None
    return max(rows, key=lambda r: (r.get("end") or date.min, r.get("start") or date.min))


def _total_debt_by_end(by_metric: dict[str, list[dict]], end: date) -> float | None:
    """Total Debt на дату: combined (если есть) или lt_debt + st_debt."""
    combined = _row_at_end(by_metric.get("debt_combined", []), end)
    if combined is not None:
        return combined["val"]
    lt = _row_at_end(by_metric.get("lt_debt", []), end)
    st = _row_at_end(by_metric.get("st_debt", []), end)
    if lt is None and st is None:
        return None
    return (lt["val"] if lt else 0.0) + (st["val"] if st else 0.0)


def _net_debt_by_end(by_metric: dict[str, list[dict]], end: date) -> float | None:
    """Net Debt на дату = Total Debt − (Cash + Short-term investments)."""
    total = _total_debt_by_end(by_metric, end)
    if total is None:
        return None
    cash = _row_at_end(by_metric.get("cash", []), end)
    st_inv = _row_at_end(by_metric.get("st_investments", []), end)
    liquid = (cash["val"] if cash else 0.0) + (st_inv["val"] if st_inv else 0.0)
    return total - liquid


def _val_at(vals: dict[str, dict | None], metric: str) -> float | None:
    """Значение метрики из словаря строк (None если строки нет)."""
    row = vals.get(metric)
    return row["val"] if row else None


def _fy_of(vals: dict[str, dict | None]) -> int | None:
    for row in vals.values():
        if row and row.get("fy") is not None:
            return row["fy"]
    return None


def build_annual(by_metric: dict[str, list[dict]]) -> list[dict]:
    """Годовые строки (по 10-K end): потоковые за год + баланс на конец года.

    Производные: free_cash_flow, ebitda, operating_margin, net_debt, roe.
    """
    flow_rows: dict[str, list[dict]] = {
        m: [r for r in by_metric.get(m, []) if _is_annual_row(r)]
        for m in FLOW_METRICS
    }
    balance_rows: dict[str, list[dict]] = {m: by_metric.get(m, []) for m in BALANCE_METRICS}

    ends = sorted({r.get("end") for rows in flow_rows.values() for r in rows if r.get("end")})
    annual: list[dict] = []

    for end in ends:
        vals = {m: _row_at_end(rows, end) for m, rows in flow_rows.items()}

        revenue = _val_at(vals, "revenue")
        net_income = _val_at(vals, "net_income")
        operating_income = _val_at(vals, "operating_income")
        cfo = _val_at(vals, "cfo")
        capex = _val_at(vals, "capex")
        dda = _val_at(vals, "dda")

        # Производные
        free_cash_flow = None
        if cfo is not None and capex is not None:
            free_cash_flow = cfo - abs(capex)
        ebitda = None
        if operating_income is not None:
            ebitda = operating_income + (dda if dda is not None else 0.0)
        operating_margin = operating_income / revenue if operating_income is not None and revenue else None

        total_debt = _total_debt_by_end(by_metric, end)
        net_debt = _net_debt_by_end(by_metric, end)
        equity_row = _row_at_end(balance_rows.get("equity", []), end)
        equity = equity_row["val"] if equity_row else None
        roe = net_income / equity if net_income is not None and equity else None

        annual.append({
            "end": end,
            "fy": _fy_of(vals),
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "eps_basic": _val_at(vals, "eps_basic"),
            "eps_diluted": _val_at(vals, "eps_diluted"),
            "cfo": cfo,
            "capex": capex,
            "free_cash_flow": free_cash_flow,
            "ebitda": ebitda,
            "operating_margin": operating_margin,
            "total_debt": total_debt,
            "net_debt": net_debt,
            "cash": (_row_at_end(balance_rows.get("cash", []), end) or {}).get("val"),
            "equity": equity,
            "shares_outstanding": (_row_at_end(balance_rows.get("shares_outstanding", []), end) or {}).get("val"),
            "roe": roe,
        })

    return annual


def latest_balance(by_metric: dict[str, list[dict]]) -> dict:
    """Последний доступный баланс (по всем формам) + производные total/net debt."""
    end = date.min
    for m in BALANCE_METRICS:
        row = _last_row(by_metric.get(m, []))
        if row and (row["end"] or date.min) > end:
            end = row["end"]
    if end == date.min:
        end = None

    def _at(m: str) -> float | None:
        row = _row_at_end(by_metric.get(m, []), end) if end else None
        return row["val"] if row else None

    cash = _at("cash")
    st_inv = _at("st_investments")
    total_debt = _total_debt_by_end(by_metric, end) if end else None
    net_debt = None
    if total_debt is not None:
        net_debt = total_debt - ((cash or 0.0) + (st_inv or 0.0))

    return {
        "end": end,
        "cash": cash,
        "st_investments": st_inv,
        "total_debt": total_debt,
        "net_debt": net_debt,
        "equity": _at("equity"),
        "shares_outstanding": _at("shares_outstanding"),
    }


# ══════════════════════════════════════════════════════════════════════ #
#  Сервис: оркестрация + PG + рыночные мультипликаторы
# ══════════════════════════════════════════════════════════════════════ #
class SecFundamentalsService:
    """Полный конвейер: кэш → CIK → EDGAR → normalize → PG → расчёты.

    Свежесть данных в PG — ``SEC_FACTS_TTL_HOURS`` (12ч): в течение окна
    EDGAR не дёргается. Поверх core-данных — SWR-кэш ответа в Redis
    (роутер). Цена (yfinance) всегда свежая — Redis-кэш 300с у фетчера.
    """

    def __init__(self, redis_client: RedisClient | None = None):
        self.redis = redis_client

    # ── Public API ────────────────────────────────────────────────────
    def get_revenue(self, ticker: str, period: str = "FY") -> dict:
        """Выручка тикера: dict для :class:`RevenueResponse` (совместимость)."""
        ticker = ticker.strip().upper()
        period = period.upper() if period.upper() in PERIODS else "FY"

        self._ensure_fresh(ticker)
        with SessionLocal() as db:
            rows = self._read_metrics(db, ticker, "revenue")
            cik = self._get_cik(db, ticker)

        rows = filter_period(rows, period)
        # Схема RevenueRow не принимает служебный ключ "metric" (extra=forbid)
        rows = [{k: v for k, v in r.items() if k != "metric"} for r in rows]
        return {
            "ticker": ticker,
            "cik": cik,
            "period": period,
            "count": len(rows),
            "source_tag": rows[-1]["tag"] if rows else None,
            "rows": rows,
        }

    def get_fundamentals_core(self, ticker: str) -> dict:
        """Core-данные (без цены): annual-ряды + последний баланс. Кэшируется."""
        ticker = ticker.strip().upper()

        self._ensure_fresh(ticker)
        with SessionLocal() as db:
            all_rows = self._read_metrics(db, ticker)
            cik = self._get_cik(db, ticker)

        by_metric: dict[str, list[dict]] = {}
        for r in all_rows:
            by_metric.setdefault(r["metric"], []).append(r)

        return {
            "ticker": ticker,
            "cik": cik,
            "annual": build_annual(by_metric),
            "balance": latest_balance(by_metric),
        }

    def fetch_price(self, ticker: str) -> float | None:
        """Текущая цена из yfinance (Redis-кэш 300с). None при недоступности."""
        try:
            from gex.orchestrator.timeframes_fetcher import create_timeframes_fetcher

            fetcher = create_timeframes_fetcher(redis_client=self.redis)
            return float(fetcher.fetch_spot(ticker))
        except Exception as exc:
            logger.warning("Не удалось получить цену %s: %s", ticker, exc)
            return None

    def with_market(self, core: dict, price: float | None) -> dict:
        """Добавить рыночные мультипликаторы (P/E, P/S, NetDebt/EBITDA)."""
        annual = core.get("annual") or []
        balance = core.get("balance") or {}
        last = annual[-1] if annual else {}

        market_cap = None
        if price and balance.get("shares_outstanding"):
            market_cap = price * balance["shares_outstanding"]

        # P/E: цена / EPS последнего FY (fallback: NI / shares)
        pe = None
        if price:
            eps = last.get("eps_basic")
            if eps is None and last.get("net_income") is not None and balance.get("shares_outstanding"):
                eps = last["net_income"] / balance["shares_outstanding"]
            if eps:
                pe = price / eps

        # P/S: Market Cap / Revenue последнего FY
        ps = None
        if market_cap and last.get("revenue"):
            ps = market_cap / last["revenue"]

        # Net Debt / EBITDA: последний баланс / EBITDA последнего FY
        net_debt_to_ebitda = None
        if balance.get("net_debt") is not None and last.get("ebitda"):
            net_debt_to_ebitda = balance["net_debt"] / last["ebitda"]

        return {
            **core,
            "price": {"value": price, "source": "yfinance", "currency": "USD"} if price is not None else None,
            "market_cap": market_cap,
            "ratios": {
                "pe": pe,
                "ps": ps,
                "net_debt_to_ebitda": net_debt_to_ebitda,
            },
        }

    # ── Внутреннее ────────────────────────────────────────────────────
    def _ensure_fresh(self, ticker: str) -> None:
        """Обновить строки тикера в PG, если они старше TTL.

        Если тикер не найден в реестре SEC — ``ValueError`` (→ 404).
        Если EDGAR недоступен — ``RuntimeError`` (→ 502).

        B-04: сетевой вызов EDGAR идёт БЕЗ открытой сессии БД. Раньше сессия жила всё время
        запроса к EDGAR (секунды), держа соединение пула и транзакцию «idle in transaction»;
        при параллельных запросах пул исчерпывался. БД нужна только на чтение TTL и на upsert —
        каждая из этих операций берёт собственную короткую сессию.
        """
        latest = self._latest_updated_at(ticker)
        if latest is not None:
            age = (_utcnow() - _as_utc(latest)).total_seconds()
            if age < settings.SEC_FACTS_TTL_HOURS * 3600:
                return

        cik = self._resolve_cik(ticker)  # реестр CIK — Redis, БД не нужна
        facts = get_company_facts(cik)  # сеть: сессия БД не открыта
        normalized = normalize_metrics(facts)
        with SessionLocal() as db:
            self._upsert(ticker, cik, normalized, db)  # _upsert сам коммитит

    def _latest_updated_at(self, ticker: str) -> datetime | None:
        """Момент последнего обновления строк тикера (короткая сессия, затем закрывается)."""
        with SessionLocal() as db:
            return db.query(func.max(CompanyMetric.updated_at)).filter(
                CompanyMetric.ticker == ticker
            ).scalar()

    def _resolve_cik(self, ticker: str) -> str:
        ticker_map = get_ticker_to_cik(self.redis)
        cik = ticker_map.get(ticker)
        if not cik:
            raise ValueError(f"Тикер {ticker} не найден в реестре SEC EDGAR")
        return cik

    def _upsert(self, ticker: str, cik: str, normalized: dict[str, list[dict]], db: Session) -> None:
        """Replace-семантика: удалить строки тикера и вставить свежий снапшот."""
        db.query(CompanyMetric).filter(CompanyMetric.ticker == ticker).delete()
        now = _utcnow()
        total = 0
        for metric, rows in normalized.items():
            for r in rows:
                if not r.get("end"):
                    continue
                db.add(CompanyMetric(
                    ticker=ticker,
                    cik=cik,
                    metric=metric,
                    tag=r.get("tag"),
                    end=r.get("end"),
                    start=r.get("start"),
                    val=r.get("val"),
                    fy=r.get("fy"),
                    fp=r.get("fp"),
                    form=r.get("form"),
                    frame=r.get("frame"),
                    filed=r.get("filed"),
                    updated_at=now,
                ))
                total += 1
        db.commit()
        logger.info("SEC metrics upsert: %s (CIK %s) — %d строк", ticker, cik, total)

    def _read_metrics(self, db: Session, ticker: str, metric: str | None = None) -> list[dict]:
        """Строки тикера (все метрики или одна), по end↑. Делегат в модульную `read_metrics`."""
        return read_metrics(db, ticker, metric)

    @staticmethod
    def _get_cik(db: Session, ticker: str) -> str | None:
        row = db.query(CompanyMetric.cik).filter(CompanyMetric.ticker == ticker).first()
        return row[0] if row else None


def read_metrics(db: Session, ticker: str, metric: str | None = None) -> list[dict]:
    """Строки тикера (все метрики или одна), по end↑.

    Именно модульная функция, а не только метод: её вызывает `sec_forecast` для квартального
    роста выручки. Раньше там импортировался `_read_metrics` — метод, которого на уровне модуля
    не существует; ImportError глушился `except Exception`, и `_quarterly_revenue_growth`
    молча всегда возвращал None (BUG-STATIC-01).
    """
    q = db.query(CompanyMetric).filter(CompanyMetric.ticker == ticker)
    if metric:
        q = q.filter(CompanyMetric.metric == metric)
    rows = q.order_by(CompanyMetric.end, CompanyMetric.start).all()
    return [r.as_dict() for r in rows]
