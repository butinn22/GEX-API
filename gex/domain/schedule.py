"""Слоты расписания: вхождения по московскому времени (ring: domain).

Почему в домене
---------------
Расчёт «сейчас внутри окна слота 23:00 МСК» — чистая функция от момента времени, без сети
и Redis. Она понадобилась двум потребителям: планировщику (публикация в окне) и воркеру
прогрева (аренда на вхождение). Держать её в планировщике значило бы либо дублировать
расчёт, либо импортировать планировщик из application — то есть завести цикл импортов.

Окно вхождения
--------------
Слот «23:00» срабатывает в интервале ``[23:00, 24:00)`` МСК. Снаружи окна вхождение не
считается: пропущенный слот **не догоняется** посреди сессии — полночный парсинг MOEX
в 15:00 был бы всплеском нагрузки без свежих данных на выходе.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

#: Москва: фиксированный UTC+3, без перехода на летнее время.
MSK = timezone(timedelta(hours=3))

#: Ширина окна вхождения слота: час после назначенного времени.
GRACE_MINUTES = 60

#: Фиксированные слоты периодических расчётов по МСК: страница -> [(час, минута), ...].
#:
#: Здесь, а не в планировщике, потому что потребителей двое и они в разных процессах:
#: планировщик приложения публикует прогрев в окне слота, а фоновая очередь запускает тот же
#: расчёт по своему расписанию. Две копии таблицы разошлись бы на первой же правке слотов,
#: и «23:00» означало бы разное время в разных планировщиках.
#:
#: Слот есть только у широты IMOEX: ISS отдаёт дневные данные и держит суточную квоту
#: обращений (3/24 ч), поэтому парсинг привязан к публикации данных, а не к таймеру «раз в
#: пять минут». Страницы США обновляются по интервалу — их окна заданы в ``freshness``.
PERIODIC_SLOTS: dict[str, tuple[tuple[int, int], ...]] = {
    "breadth-imoex": ((23, 0), (8, 0)),
}

__all__ = ["GRACE_MINUTES", "MSK", "PERIODIC_SLOTS", "slot_occurrence", "slots_due"]


def slot_occurrence(
    hours_minutes: Iterable[tuple[int, int]],
    now_msk: Optional[datetime] = None,
) -> Optional[str]:
    """Ключ вхождения ``YYYY-MM-DD HH:MM``, если сейчас окно одного из слотов.

    ``None`` — вне окна.
    """
    now = now_msk or datetime.now(MSK)
    if now.tzinfo is None:
        now = now.replace(tzinfo=MSK)
    for hour, minute in hours_minutes:
        occurrence = datetime(now.year, now.month, now.day, hour, minute, tzinfo=MSK)
        delta = now - occurrence
        if timedelta(0) <= delta < timedelta(minutes=GRACE_MINUTES):
            return occurrence.strftime("%Y-%m-%d %H:%M")
    return None


def slots_due(
    schedules: dict[str, Iterable[tuple[int, int]]],
    now_msk: Optional[datetime] = None,
) -> dict[str, str]:
    """Все слоты, чьё окно открыто сейчас: имя → ключ вхождения."""
    out: dict[str, str] = {}
    for name, hours_minutes in schedules.items():
        occurrence = slot_occurrence(hours_minutes, now_msk)
        if occurrence is not None:
            out[name] = occurrence
    return out
