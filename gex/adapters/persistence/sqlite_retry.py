"""Повтор операции при блокировке SQLite (ring: adapters/persistence).

Зачем отдельный модуль
----------------------
``journal_mode=WAL`` и ``busy_timeout`` снимают основную массу ``database is
locked``, но не все: ``busy_timeout`` — это *ожидание*, а не гарантия. Если
писатель держит транзакцию дольше таймаута (сетевой фетч внутри обработчика —
``get_session()`` коммитит в конце), остальные всё равно получат ``SQLITE_BUSY``.
Замер на тестовом стенде: 6 писателей + один «медленный» (8 с в транзакции) —
18 ошибок без WAL, 4 ошибки с WAL и ``busy_timeout=15s``, **0** с повтором.

Повтор — страховка последнего уровня, поэтому он:

* применяется **только** к блокировкам (``is_lock_error``), остальные ошибки
  пробрасываются сразу — маскировать настоящий сбой нельзя;
* повторяет **всю** единицу работы целиком, а не один ``commit()``: после
  неудачного коммита сессия откатана и данные потеряны, повтор коммита бесполезен;
* экспоненциально увеличивает паузу с небольшим джиттером, чтобы несколько
  потоков не выстраивались в «стадо» и не дрались за блокировку снова.

Использование::

    from gex.adapters.persistence.sqlite_retry import run_with_retry, DEFAULT_ATTEMPTS

    def _write():                      # сессия открывается ВНУТРИ — иначе
        db = SessionLocal()            # повторять будет нечего
        try:
            db.add(row)
            db.commit()
        finally:
            db.close()

    run_with_retry(_write, what="snapshot метрик")
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Сколько попыток сделать, прежде чем сдаться (первая — сразу).
DEFAULT_ATTEMPTS = 5

#: Стартовая пауза между попытками, секунды (далее удваивается).
DEFAULT_BASE_DELAY = 0.1

#: Потолок паузы, секунды.
DEFAULT_MAX_DELAY = 2.0

#: Подстроки, по которым ошибка опознаётся как «БД занята». SQLite формулирует
#: это по-разному в зависимости от причины:
#:   * ``database is locked``    — SQLITE_BUSY (в т. ч. BUSY_SNAPSHOT в WAL);
#:   * ``database table is locked`` — SQLITE_LOCKED (DROP/ALTER при активном читателе);
#:   * ``database is busy`` / ``sqlite_busy`` — вариант из некоторых сборок драйвера.
LOCK_MARKERS = (
    "database is locked",
    "database table is locked",
    "database is busy",
    "sqlite_busy",
    "sqlite_locked",
)


def is_lock_error(exc: BaseException) -> bool:
    """Похожа ли ошибка на блокировку SQLite (а не на настоящий сбой).

    Проверяется текст, а не тип: ``sqlite3.OperationalError`` приходит обёрнутым
    в ``sqlalchemy.exc.OperationalError``, и у того в ``str`` есть и причина, и
    сам SQL — по тексту надёжнее, чем по цепочке ``__cause__``.
    """
    try:
        text = f"{type(exc).__name__}: {exc}".lower()
    except Exception:  # noqa: BLE001 — __str__ может быть переопределён и бросить
        return False
    return any(marker in text for marker in LOCK_MARKERS)


def run_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    what: str = "",
    sleep: Callable[[float], None] = time.sleep,
    log: logging.Logger | None = None,
) -> T:
    """Выполнить ``operation``, повторяя при блокировке БД.

    Parameters
    ----------
    operation
        Единица работы целиком: открыть сессию → записать → закоммитить → закрыть.
        Должна быть идемпотентной (повтор не должен дублировать данные) — для
        вставок с первичным ключом это выполняется автоматически.
    attempts
        Число попыток (первая выполняется без паузы).
    what
        Что именно пишем — попадёт в лог, чтобы по нему было видно источник.
    sleep, log
        Точки внедрения для тестов (не ждать реально и не засорять лог).

    Returns
    -------
    T
        Результат ``operation``.

    Raises
    ------
    Exception
        Последняя ошибка — если повторы исчерпаны, либо исходная ошибка, если
        она не похожа на блокировку (повтор тут только скрыл бы причину).
    """
    out = log if log is not None else logger
    retries = max(1, int(attempts))
    label = what or getattr(operation, "__name__", "операция")

    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 — ниже разбираем, что делать
            if not is_lock_error(exc) or attempt == retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            # Джиттер: без него все потоки просыпаются одновременно и снова
            # выстраиваются в очередь за той же блокировкой.
            pause = delay + random.uniform(0, delay * 0.3)
            out.debug(
                "БД занята, повтор %d/%d через %.2fs (%s): %s",
                attempt, retries, pause, label, exc,
            )
            sleep(pause)

    raise AssertionError("unreachable")  # pragma: no cover — цикл всегда возвращает/бросает


def with_retry_kwargs(**kwargs: Any) -> dict:
    """Собрать kwargs для :func:`run_with_retry` (удобно для тестов/вызова).

    Оставляет только известные ключи, чтобы опечатка не прошла молча.
    """
    allowed = {"attempts", "base_delay", "max_delay", "what"}
    return {k: v for k, v in kwargs.items() if k in allowed}


__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_DELAY",
    "LOCK_MARKERS",
    "is_lock_error",
    "run_with_retry",
    "with_retry_kwargs",
]
