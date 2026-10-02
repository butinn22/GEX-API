"""Payload'ы страниц, вынесенных из запроса: широта рынка, композит секторов, широта IMOEX.

Зачем отдельный модуль
----------------------
Три страницы объединяет одно: их расчёт стоит минуты, а не миллисекунды, и живые данные
им не нужны.

* ``/breadth`` — широта S&P 500: ~500 бумаг за 3 года одним батчем + McClellan;
* ``/sector`` — 12 секторальных ETF с MACD на ADL (по замерам роутера — 14–22 с);
* ``/breadth-imoex`` — состав IMOEX через ISS с суточной квотой обращений.

Пока расчёт жил в обработчике запроса, промах кэша означал «пользователь ждёт столько,
сколько понадобится» — при недоступном Redis это была **каждая** загрузка страницы. Здесь
собрана ровно та часть, которую можно выполнять вне запроса: построение payload'а и его
запись в хранилище. Чтение — ``gex.ports.cache.SnapshotPort`` (без вычисления), запуск —
фоновая задача (``gex/workers/tasks/market_pages.py``), ключ — ``gex.adapters.cache.keys.page_key``.

Модуль намеренно не знает ни про Redis, ни про очередь, ни про формат ключа: хранилище
приходит портом, ключ — строкой. Так одну и ту же функцию зовут и воркер, и ручной прогрев
из CLI, и тест.

Метаданные ответа
-----------------
К payload'у добавляется ``meta``: когда посчитано, на какую торговую дату данные, откуда
взято. Это не украшение, а необходимость: страница теперь может показывать вчерашние данные
(честно помеченные), а не ждать свежих — и пользователь обязан видеть, что именно он смотрит.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from gex.domain.freshness import policy_for_page

logger = logging.getLogger(__name__)

__all__ = [
    "BREADTH_MODES",
    "PAGE_BREADTH",
    "PAGE_IMOEX",
    "PAGE_SECTOR",
    "PAGES",
    "RefreshOutcome",
    "build_payload",
    "data_day",
    "refresh_snapshot",
    "retention_s",
    "with_meta",
]

#: Имена страниц совпадают с ``gex.domain.freshness.PAGE_CLASS``: окна свежести и форма
#: ключа обязаны называть страницу одинаково, иначе «страница» получает два разных TTL.
PAGE_BREADTH = "breadth"
PAGE_SECTOR = "sector"
PAGE_IMOEX = "breadth-imoex"

#: Режим «лидеров» на /breadth (``?giants=mags|top10``) — часть идентичности payload'а.
BREADTH_MODES = ("mags", "top10")

PAGES = (PAGE_BREADTH, PAGE_SECTOR, PAGE_IMOEX)


@dataclass(frozen=True)
class RefreshOutcome:
    """Результат фонового пересчёта — то, что воркер возвращает в хранилище результатов."""

    page: str
    mode: Optional[str]
    ok: bool
    data_day: Optional[str] = None
    retention_s: int = 0
    error: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"page": self.page, "ok": self.ok}
        if self.mode:
            out["mode"] = self.mode
        if self.data_day:
            out["data_day"] = self.data_day
        if self.retention_s:
            out["retention_s"] = self.retention_s
        if self.reason:
            out["reason"] = self.reason
        if self.error:
            out["error"] = self.error
        return out


def retention_s(page: str) -> int:
    """Сколько секунд payload обязан переживать в хранилище.

    Берём **внесессионное** окно ``stale_max`` из ``gex.domain.freshness``: это максимальный
    возраст, при котором значение ещё разрешено показывать. Ключ, живущий меньше, означал бы
    не «страница устарела», а «страницы нет» — то есть 503 там, где данные были вчера.
    """
    policy = policy_for_page(page, market_open=False)
    return int(policy.stale_max)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_day(payload: dict) -> Optional[str]:
    """Торговая дата, к которой относятся данные (для метаданных и для логов)."""
    current = payload.get("current") or {}
    day = current.get("day")
    if day:
        return str(day)
    meta = payload.get("meta") or {}
    if meta.get("last_completed_day"):
        return str(meta["last_completed_day"])
    composite = (payload.get("composite") or {}).get("dates") or []
    if composite:
        return str(composite[-1])
    dates = (payload.get("market") or {}).get("dates") or []
    return str(dates[-1]) if dates else None


def with_meta(payload: dict, meta: dict) -> dict:
    """Добавить/дополнить блок ``meta`` в payload (не мутируя исходный словарь).

    Существующий ``meta`` (его пишет расчёт IMOEX) сохраняется: служебные поля страницы
    дополняют его, а не затирают.
    """
    out = dict(payload)
    merged: dict[str, Any] = dict(payload.get("meta") or {})
    merged.update(meta)
    out["meta"] = merged
    return out


# ═════════════════════════════════════════════════════════════════════════
# Построение payload'ов (тяжёлая часть — только вне запроса)
# ═════════════════════════════════════════════════════════════════════════
def _build_breadth(mode: str) -> dict:
    """Широта рынка США. Каждый блок (market/stocks/corr/pulse) кэшируется внутри сервиса."""
    from gex.application.breadth_service import get_breadth_v2

    return get_breadth_v2(mode=mode)


def _build_sector(service: Any = None) -> dict:
    """Композит 12 секторальных ETF: композит, EMA20, MACD на ADL.

    Преобразование отчёта в ответ страницы живёт здесь, а не в роутере: роутер теперь
    ничего не считает, а воркер обязан отдать ровно тот же контракт, что отдавал роутер.
    """
    from gex.application.sector_service import SectorBreadthService

    svc = service or SectorBreadthService()
    report = svc.analyze()

    macd_df = report.macd_analysis
    macd_data = {
        "dates": [str(d.date()) for d in macd_df.index] if not macd_df.empty else [],
        "macd": [round(float(v), 4) for v in macd_df["macd"].values] if not macd_df.empty else [],
        "signal": [round(float(v), 4) for v in macd_df["signal"].values] if not macd_df.empty else [],
        "hist": [round(float(v), 4) for v in macd_df["hist"].values] if not macd_df.empty else [],
        "tl": [round(float(v), 4) for v in macd_df["tl"].values] if not macd_df.empty else [],
    }

    ema20: list[float] = []
    if hasattr(report, "composite_series") and len(report.composite_series) >= 20:
        # EMA20 по ПОЛНОМУ ряду, обрезаем до отображаемого окна (dates) — раньше брался
        # tail(200), из-за чего линия не совпадала с датами.
        ema20 = [
            round(float(v), 4)
            for v in report.composite_series.ewm(span=20, adjust=False).mean()
            .tail(len(report.dates))
            .values
        ]

    return {
        "composite": {
            "dates": report.dates,
            "values": report.values,
            "current": report.current,
            "n_days": report.n_days,
        },
        "ema20": ema20,
        "per_sector": report.per_sector,
        "trend": report.trend_analysis,
        "macd": macd_data,
    }


def build_payload(
    page: str,
    mode: Optional[str] = None,
    *,
    sector_service: Any = None,
    force: bool = True,
    bypass_rate_limit: bool = False,
) -> dict:
    """Собрать payload страницы. Тяжело: вызывать только из фонового пересчёта.

    ``sector_service`` — уже собранный сервис секторов (у воркера и у API он свой):
    передаётся снаружи, чтобы модуль не тянул ``gex.orchestrator`` (кольцо application
    не имеет права ходить в инфраструктуру).

    ``force``/``bypass_rate_limit`` осмысленны только для IMOEX (у остальных страниц
    источник — рыночные котировки, а не квотируемый ISS): см. :func:`_build_imoex`.
    """
    if page == PAGE_BREADTH:
        mode = mode or BREADTH_MODES[0]
        if mode not in BREADTH_MODES:
            raise ValueError(f"неизвестный режим лидеров: {mode!r} (ожидается {BREADTH_MODES})")
        return _build_breadth(mode)
    if page == PAGE_SECTOR:
        return _build_sector(sector_service)
    if page == PAGE_IMOEX:
        return _build_imoex(force=force, bypass_rate_limit=bypass_rate_limit)
    raise ValueError(f"страница {page!r} не описана как фоновая (известны {PAGES})")


def _build_imoex(*, force: bool = True, bypass_rate_limit: bool = False) -> dict:
    """Широта MOEX: расчёт оркестратора (ISS, суточная квота обращений — 3 в 24 ч).

    ``force=False`` — плановый режим: оркестратор сам решает, нужен ли ISS (у него есть
    кэш сырых данных на 6 часов). ``bypass_rate_limit=True`` — только ручное обновление
    из админки: это осознанное действие человека, а не расписание, и квота ему не указ.

    Неудачный расчёт **не** означает «страницы нет»: последний удачный остаётся на месте —
    ровно это и делает «если свежих данных нет, отдаём устаревшие». Пусто и там — ошибка.
    """
    from gex.application.breadth_imoex_service import ensure_warm, get_latest

    if not ensure_warm(force_refresh=force, bypass_rate_limit=bypass_rate_limit):
        stale = get_latest()
        if not stale:
            raise RuntimeError("IMOEX: расчёт не прошёл и предыдущих данных нет")
        logger.warning("IMOEX: обновление не удалось — оставляю предыдущий расчёт")
    latest = get_latest()
    if not latest:
        raise RuntimeError("IMOEX: расчёт прошёл, но результат не сохранён")
    return latest


# ═════════════════════════════════════════════════════════════════════════
# Запись в хранилище
# ═════════════════════════════════════════════════════════════════════════
def refresh_snapshot(
    store: Any,
    key: str,
    page: str,
    mode: Optional[str] = None,
    *,
    sector_service: Any = None,
    force: bool = True,
    bypass_rate_limit: bool = False,
) -> RefreshOutcome:
    """Посчитать payload и положить его в хранилище под ``key``.

    Ошибки не летят наружу молча: они логируются здесь и возвращаются в исходе, потому что
    у этого вызова два разных потребителя. Фоновая задача по исходу решает, повторять ли
    попытку; ручной прогрев — печатает результат. Исключение тут означало бы либо
    «проглотили сбой», либо «упал весь батч страниц из-за одной».
    """
    try:
        payload = build_payload(
            page, mode, sector_service=sector_service,
            force=force, bypass_rate_limit=bypass_rate_limit,
        )
    except Exception as exc:  # noqa: BLE001 — сбой страницы не должен ронять вызывающего
        logger.warning("Пересчёт страницы %s (mode=%s) не удался: %s", page, mode, exc, exc_info=True)
        return RefreshOutcome(page=page, mode=mode, ok=False, error=f"{type(exc).__name__}: {exc}")

    # Метаданные ставит тот, кто пишет: вычислитель может быть вызван и в тесте, и из CLI,
    # и «когда посчитано» — свойство записи, а не расчёта.
    payload = with_meta(
        payload,
        {
            "page": page,
            "mode": mode or "",
            "computed_at": _now_iso(),
            "data_day": data_day(payload) or "",
            "source": "background",
        },
    )

    ttl = retention_s(page)
    try:
        store.write(key, payload, fresh=ttl, source="background")
    except Exception as exc:  # noqa: BLE001 — запись кэша не критична
        logger.warning("Payload %s не сохранён: %s", key, exc, exc_info=True)
        return RefreshOutcome(page=page, mode=mode, ok=False, error=f"write: {exc}")

    day = data_day(payload)
    logger.info(
        "Страница %s обновлена: key=%s, data_day=%s, retention=%s c",
        page, key, day, ttl,
    )
    return RefreshOutcome(page=page, mode=mode, ok=True, data_day=day, retention_s=ttl)
