"""Разбор XBRL из ответа SEC: чистые функции, без сети (ring: application).

Зачем отдельно от клиента
-------------------------
`sec_edgar.py` умеет **достать** данные (HTTP + кэш), а разбор фактов жил в
`sec_forecast.py` (1352 строки), вперемешку с регрессиями, оценками и вёрсткой ответа.
Из-за этого «правильно ли мы читаем XBRL» нельзя было проверить без сети: любая проверка
начиналась с запроса к data.sec.gov.

Здесь только разбор: вход — уже полученный JSON (`companyfacts`), выход — ряды значений.
Ни HTTP, ни базы, ни кэша, поэтому проверяется синтетическими payload'ами.

Две ловушки XBRL, из-за которых этот модуль существует
-----------------------------------------------------
1. **Один и тот же показатель лежит в нескольких тегах.** Выручка бывает в
   ``Revenues``, ``RevenueFromContractWithCustomerExcludingAssessedTax`` и
   ``SalesRevenueNet`` — и разные компании используют разные. Поэтому тег параметр,
   а ``extract_first`` перебирает варианты по порядку.
2. **Годовые и квартальные факты лежат в одном массиве.** Если их не разделить, «год»
   окажется кварталом, и все производные (рост, маржа, CAGR) поедут. Поэтому период
   различается по длительности окна (``fy``/``Q1..Q4`` плюс контроль по датам).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "AnnualFact",
    "extract_first",
    "extract_series",
    "latest_value",
    "parse_period",
]

#: Годовое окно: от 300 дней (не 365 — у части компаний «год» 52/53 недели).
ANNUAL_MIN_DAYS = 300

#: Квартальное окно: от 60 до 120 дней (13 недель ± сдвиги календаря).
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 120


@dataclass(frozen=True)
class AnnualFact:
    """Годовой факт: конец периода, значение и тег, откуда оно взято."""

    end: date
    value: float
    tag: str = ""
    form: str = ""
    fiscal_year: Optional[int] = None

    @property
    def key(self) -> str:
        return self.end.isoformat()


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def parse_period(fact: dict) -> Optional[tuple[date, date]]:
    """(начало, конец) периода факта или ``None``, если даты не читаются."""
    start = _as_date(fact.get("start"))
    end = _as_date(fact.get("end"))
    if end is None:
        return None
    return (start or end, end)


def _is_annual(fact: dict) -> bool:
    period = parse_period(fact)
    if period is None:
        return False
    start, end = period
    return (end - start).days >= ANNUAL_MIN_DAYS


def _is_quarter(fact: dict) -> bool:
    period = parse_period(fact)
    if period is None:
        return False
    start, end = period
    return QUARTER_MIN_DAYS <= (end - start).days <= QUARTER_MAX_DAYS


def extract_series(
    companyfacts: dict,
    tag: str,
    *,
    unit: str = "USD",
    annual: bool = True,
    forms: Sequence[str] = ("10-K",),
) -> list[AnnualFact]:
    """Ряд годовых (или квартальных) фактов по тегу, от старых к новым.

    Parameters
    ----------
    annual : bool
        ``True`` — только годовые окна (``>= 300`` дней), ``False`` — только квартальные
        (60–120 дней). Смешивать их нельзя: «год» из квартала ломает все производные.
    forms : sequence of str
        Какие формы принимаем. По умолчанию годовой отчёт; квартальный — ``("10-Q",)``.
    """
    payload = (companyfacts.get("facts") or {}).get("us-gaap") or {}
    blocks = payload.get(tag) or {}
    units = blocks.get("units") or {}
    facts = units.get(unit) or []

    predicate = _is_annual if annual else _is_quarter
    out: dict[date, tuple[str, AnnualFact]] = {}  # конец периода → (дата подачи, факт)
    for fact in facts:
        if fact.get("form") not in forms:
            continue
        if not predicate(fact):
            continue
        period = parse_period(fact)
        value = fact.get("val")
        if period is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        end = period[1]
        # Один период встречается несколько раз (пересдача формы): берём последнюю по дате
        # подачи — это и есть актуальное значение, а не первое попавшееся.
        filed = str(fact.get("filed") or "")
        previous = out.get(end)
        if previous is None or filed >= previous[0]:
            out[end] = (
                filed,
                AnnualFact(
                    end=end,
                    value=float(value),
                    tag=tag,
                    form=str(fact.get("form") or ""),
                    fiscal_year=fact.get("fy") if isinstance(fact.get("fy"), int) else None,
                ),
            )
    return [fact for _filed, fact in (out[key] for key in sorted(out))]


def extract_first(
    companyfacts: dict,
    tags: Iterable[str],
    **kwargs: Any,
) -> tuple[Optional[AnnualFact], list[AnnualFact], str]:
    """Первый тег из списка, по которому есть данные: ``(последний факт, ряд, тег)``.

    Компании называют одно и то же по-разному (``Revenues`` / ``RevenueFromContract...``
    / ``SalesRevenueNet``), поэтому тег перебирается: выбрать «правильный» заранее нельзя.
    """
    for tag in tags:
        series = extract_series(companyfacts, tag, **kwargs)
        if series:
            return series[-1], series, tag
    return None, [], ""


def latest_value(companyfacts: dict, tags: Iterable[str], **kwargs: Any) -> Optional[float]:
    """Значение последнего доступного факта по любому из тегов."""
    fact, _series, _tag = extract_first(companyfacts, tags, **kwargs)
    return fact.value if fact is not None else None
