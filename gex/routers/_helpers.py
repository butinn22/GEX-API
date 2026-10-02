"""
Shared helpers for route handlers: error mapping + optional Telegram notify.

Usage::

    from ._helpers import handle

    @router.get("/example/{ticker}")
    def get_example(ticker: str, notify: bool = False, background_tasks: BackgroundTasks = None):
        return handle(
            lambda: service.analyze(ticker),
            notify_fn=notify_gex_analysis,
            notify=notify,
            background_tasks=background_tasks,
            error_src="yfinance",
        )
"""
from __future__ import annotations

import logging
import math
from typing import Any
from collections.abc import Callable

from fastapi import BackgroundTasks, HTTPException

logger = logging.getLogger(__name__)


def json_safe(obj: Any) -> Any:
    """Рекурсивно привести ответ к JSON-совместимому виду.
    Starlette сериализует с allow_nan=False: NaN/Infinity в ответе роняют
    эндпоинт с 500. Заменяем не-конечные float на None, numpy-скаляры — в python-типы.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    # numpy-скаляры (float32/int64/...)
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        return json_safe(obj.item())
    return obj


_ENTRY_SIDES: dict[str, str] = {"entry_long": "long", "entry_short": "short"}
_EXIT_SIDES: dict[str, str] = {"exit_long": "long", "exit_short": "short"}
_ADD_SIDES: dict[str, str] = {"add_long": "long", "add_short": "short"}


def drop_unanchored_signals(signals: list[dict]) -> list[dict]:
    """Убрать выходы/добавления без видимого входа — порядок ХРОНОЛОГИЧЕСКИЙ.

    Персональный flat-фильтр («отсекать без тренда») может скрыть блокированный
    ВХОД, оставив его выход — на странице снова получался «выход без входа».
    Ведём учёт видимых позиций: выход/добавление проходит только если в списке
    есть открывающий его вход того же направления; иначе строка отбрасывается.
    """
    if not signals:
        return signals
    side: str | None = None
    out: list[dict] = []
    for sig in signals:
        ot = str(sig.get("order_type") or "")
        if ot in _ENTRY_SIDES:
            side = _ENTRY_SIDES[ot]
            out.append(sig)
        elif ot in _EXIT_SIDES:
            if side == _EXIT_SIDES[ot]:
                side = None
                out.append(sig)
        elif ot in _ADD_SIDES:
            if side == _ADD_SIDES[ot]:
                out.append(sig)
        else:
            out.append(sig)
    return out


def handle(
    fn: Callable[[], Any],
    *,
    notify: bool = False,
    background_tasks: BackgroundTasks | None = None,
    notify_fn: Callable | None = None,
    chat_id: str | None = None,
    error_src: str = "данных",
) -> Any:
    """Execute fn() with unified error mapping → HTTPException.

    KeyError   → 404 (ticker not found)
    ValueError → 404 (invalid params / no data)
    RuntimeError → 502 (external API failure)

    Optionally schedules a Telegram notification via background_tasks.
    If chat_id is provided, notification goes to that specific chat.
    """
    try:
        result = fn()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'"))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка получения {error_src}: {e}")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in %s", error_src)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

    if notify and notify_fn is not None and background_tasks is not None:
        if chat_id:
            background_tasks.add_task(notify_fn, result, chat_id=chat_id)
        else:
            background_tasks.add_task(notify_fn, result)

    return json_safe(result)
