"""Модель фоновой задачи: вид работы и её отображение на порт очереди (ring: application).

Здесь живёт **что** мы делаем (``FetchTask``) и в какую очередь это кладётся; **как**
доставляется — дело адаптера (:mod:`gex.adapters.queue.redis_streams`) за портом
(:mod:`gex.ports.job_queue`). Раньше и то и другое лежало в одном ``gex/task_queue.py``,
и транспорт (``LPUSH``/``BRPOP``, цикл потребителя, переподключение Redis) был вшит
в модель задачи.

Зачем отдельный ``idempotency_key``
-----------------------------------
Доставка at-least-once: сообщение может прийти дважды (повторная выдача зависшего
сообщения, ретрай воркера). Ключ делается **детерминированным** от состава задачи —
тогда повторная доставка той же задачи за окно дедупликации не приводит к повторному
фетчу, а следующий цикл прогрева (другое окно) снова видит её как новую.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from gex.ports.job_queue import Job, Priority

__all__ = [
    "CONSUMER_PROFILES",
    "DEFAULT_QUEUE",
    "INTERACTIVE_PRIORITY",
    "QUEUE_FOR_TASK",
    "QUEUE_KINDS",
    "FetchTask",
    "idempotency_key_for",
    "job_to_task",
    "queue_for",
    "task_to_job",
]

#: Приоритет, начиная с которого задача считается интерактивной (пользователь ждёт).
INTERACTIVE_PRIORITY = 2

#: Куда кладётся задача каждого вида. ``vol`` — общая очередь «объёмных» расчётов
#: (sector/breadth/composite): они быстрые и делят один лимит провайдера.
QUEUE_FOR_TASK: dict[str, str] = {
    "ohlcv": "ohlcv",
    "chain": "chain",
    "vol": "vol",
    "gex_profile": "gex",
    "sector": "vol",
    "breadth": "vol",
    "breadth_imoex": "vol",
    "composite": "vol",
}

#: Очередь для вида задачи, которого нет в таблице (неизвестное не теряем, а кладём в общую).
DEFAULT_QUEUE = "default"

#: Виды очередей — этот список передаётся адаптеру, чтобы он знал, какие потоки читать.
QUEUE_KINDS: tuple[str, ...] = tuple(dict.fromkeys([*QUEUE_FOR_TASK.values(), DEFAULT_QUEUE]))

#: Профили потребителей очереди: имя → виды задач. Каждый потребитель читает только свои
#: очереди, поэтому «тяжёлые» расчёты (sector/breadth/composite — десятки секунд на задачу)
#: больше не стоят в очереди перед свежими свечами дашборда: market overview обновляется
#: своим потребителем, не дожидаясь разбора расчётных задач. Публикация общая — вид задачи
#: по-прежнему определяется в :data:`QUEUE_FOR_TASK`, профили влияют только на разбор.
#:
#: ``fast`` — прогрев market overview (свечи карточек); ``heavy`` — цепочки и расчёты,
#: где время ожидания результата не критично. Профили не пересекаются по стримам: у каждого
#: свой consumer-статус, а группа потребителей общая — дедупликация (``should_process``)
#: остаётся сквозной. Если chain понадобится ускорить отдельно — вид переезжает между
#: профилями одной строкой.
CONSUMER_PROFILES: dict[str, tuple[str, ...]] = {
    "fast": ("ohlcv",),
    "heavy": ("chain", "vol", "gex", "default"),
}


@dataclass
class FetchTask:
    """Задача для фонового фетчинга.

    Attributes
    ----------
    task_type : str
        ``ohlcv``, ``chain``, ``vol``, ``gex_profile``, ``sector``, ``breadth``, …
    provider : str
        ``yfinance``, ``bybit``, ``moex_iss``, ``webull``.
    ticker : str
        Тикер/актив.
    params : dict
        Дополнительные параметры (timeframe, max_expiries, days, ...).
    priority : int
        0=низкий (превентивный), 1=нормальный, 2=высокий (по запросу пользователя).
    """

    task_type: str
    provider: str
    ticker: str
    params: dict = field(default_factory=dict)
    priority: int = 1

    @property
    def queue(self) -> str:
        """Вид очереди для этой задачи."""
        return queue_for(self.task_type)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "FetchTask":
        return cls(**json.loads(data))

    @property
    def idempotency_key(self) -> str:
        """Детерминированный ключ дедупликации (см. :func:`idempotency_key_for`)."""
        return idempotency_key_for(self)


def queue_for(task_type: str) -> str:
    """Вид очереди для вида задачи (неизвестный вид — общая очередь, а не потеря)."""
    return QUEUE_FOR_TASK.get(task_type, DEFAULT_QUEUE)


def idempotency_key_for(task: FetchTask) -> str:
    """Ключ дедупликации: состав задачи без приоритета.

    Приоритет в ключ не входит: это свойство *доставки*, а не работы — иначе одна и та же
    задача, поставленная один раз превентивно и один раз по запросу пользователя, считалась
    бы двумя разными и выполнялась дважды.

    Тикер и провайдер нормализуются (``spy`` → ``SPY``, ``YF`` → ``yf``): без этого
    ``ohlcv:yfinance:spy`` и ``ohlcv:yfinance:SPY`` — два разных ключа, и дедупликация
    пропускала бы ровно те повторы, ради которых она нужна. Нормализация здесь своя
    (а не из ``adapters.cache.keys``), потому что слой application не имеет права
    импортировать adapters.
    """
    ticker = str(task.ticker or "").strip().upper()
    provider = str(task.provider or "").strip().lower()
    params = json.dumps(task.params or {}, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(params.encode("utf-8")).hexdigest()[:12]
    return f"{task.task_type}:{provider}:{ticker}:{digest}"


def task_to_job(task: FetchTask) -> Job:
    """Задача → сообщение очереди (тикер и параметры уезжают в ``payload``)."""
    return Job(
        task_type=task.task_type,
        idempotency_key=task.idempotency_key,
        payload={"ticker": task.ticker, "params": dict(task.params or {})},
        priority=Priority.INTERACTIVE if task.priority >= INTERACTIVE_PRIORITY else Priority.BACKGROUND,
        provider=task.provider or None,
    )


def job_to_task(job: Job) -> FetchTask:
    """Сообщение очереди → задача. Обратная операция к :func:`task_to_job`.

    Приоритет — **сужающее** преобразование: у порта два уровня, у задачи три, поэтому
    «низкий (0)» и «нормальный (1)» при возврате неразличимы и становятся ``1``.
    Значимое различие (``2`` — пользователь ждёт) сохраняется: оно определяет и потоки,
    и порядок разбора.
    """
    payload: dict[str, Any] = job.payload if isinstance(job.payload, dict) else {}
    params = payload.get("params")
    return FetchTask(
        task_type=job.task_type,
        provider=job.provider or "",
        ticker=str(payload.get("ticker") or ""),
        params=params if isinstance(params, dict) else {},
        priority=INTERACTIVE_PRIORITY if job.priority == Priority.INTERACTIVE else 1,
    )
