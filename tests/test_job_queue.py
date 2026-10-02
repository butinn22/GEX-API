"""Очередь задач: Streams, at-least-once, DLQ, дедупликация (итерация 28).

Что проверяется и почему именно это
-----------------------------------
Прежний транспорт (``LPUSH``/``BRPOP``) терял задачу при падении воркера: сообщение
исчезало из Redis **до** обработки. Порт обещает другое, и обещания надо проверять:

* **at-least-once** — неподтверждённое сообщение остаётся выданным и возвращается;
* **DLQ** — упавшая задача не исчезает, а попадает в отдельный поток с причиной;
* **дедупликация** — повторная доставка не выполняет работу дважды;
* **приоритет** — интерактивные разбираются раньше фоновых;
* **деградация** — без Redis очередь сообщает «недоступна», а не падает.

Наборы без внешних зависимостей не поднимают Redis, поэтому здесь используется фейк,
воспроизводящий семантику групп потребителей (включая pending и ``XAUTOCLAIM``):
без pending проверка at-least-once была бы декоративной.

    python tests/test_job_queue.py
    pytest tests/test_job_queue.py -q
"""
from __future__ import annotations

import ast
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.queue.redis_streams import (  # noqa: E402
    StreamJobQueue,
    base_kind,
    dlq_name,
    stream_name,
)
from gex.application.jobs import (  # noqa: E402
    CONSUMER_PROFILES,
    QUEUE_KINDS,
    FetchTask,
    job_to_task,
    queue_for,
    task_to_job,
)
from gex.application.queue import TaskPublisher  # noqa: E402
from gex.application.worker import TaskConsumer  # noqa: E402
from gex.ports.job_queue import Delivery, Job, JobQueuePort, Priority  # noqa: E402

KINDS = ("ohlcv", "chain")


class FakeStreamsRedis:
    """Клиент с семантикой Streams и групп потребителей.

    Моделирует **RedisClient** (тот, с кем работает адаптер), а не сырой redis-py:
    сигнатуры методов и поведение при сбое совпадают — при ``fail`` клиент возвращает
    нейтральное значение, потому что он сам ловит ошибки Redis. Фейк, который расходится
    с клиентом, проверяет не тот интерфейс: именно так был пропущен дефект ``script_load``
    в итер. 27.
    """

    def __init__(self):
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self.groups: dict[tuple[str, str], int] = {}          # (stream, group) → индекс последнего выданного
        self.pending: dict[tuple[str, str], dict[str, str]] = {}  # (stream, group) → {id: consumer}
        self.kv: dict[str, str] = {}
        self.fail = False
        self._seq = 0
        self._lock = threading.Lock()

    # -- Streams -------------------------------------------------------- #
    def xadd(self, name, fields, maxlen=None):
        if self.fail:
            return None
        with self._lock:
            self._seq += 1
            message_id = f"{self._seq}-0"
            self.streams.setdefault(name, []).append((message_id, dict(fields)))
            if maxlen and len(self.streams[name]) > maxlen:
                del self.streams[name][: len(self.streams[name]) - maxlen]
            return message_id

    def xgroup_create(self, name, group, id="0", mkstream=True):
        if self.fail:
            return False
        with self._lock:
            self.groups.setdefault((name, group), 0)
            return True

    def xreadgroup(self, group, consumer, streams, count=None, block_ms=None):
        # Сигнатура повторяет RedisClient.xreadgroup: фейк обязан совпадать с клиентом,
        # иначе тест проверяет не тот интерфейс, что в проде.
        if self.fail:
            return []
        out = []
        with self._lock:
            for stream, start in streams.items():
                if start != ">":
                    continue
                entries = self.streams.get(stream, [])
                last = self.groups.get((stream, group), 0)
                fresh = entries[last:][: count or 10]
                if not fresh:
                    continue
                self.groups[(stream, group)] = last + len(fresh)
                pend = self.pending.setdefault((stream, group), {})
                for message_id, _fields in fresh:
                    pend[message_id] = consumer
                out.append((stream, [(mid, dict(f)) for mid, f in fresh]))
        return out

    def xack(self, name, group, *message_ids):
        if self.fail:
            return 0
        with self._lock:
            pend = self.pending.setdefault((name, group), {})
            return sum(1 for mid in message_ids if pend.pop(mid, None) is not None)

    def xautoclaim(self, name, group, consumer, min_idle_ms=None, count=None):
        """Отдаём всё, что висит на другом потребителе (idle в фейке не моделируется).

        Возврат — как у ``RedisClient.xautoclaim`` (список пар), а не как у сырого
        redis-py (там ещё курсор и список удалённых): адаптер работает именно с клиентом.
        """
        if self.fail:
            return []
        claimed = []
        with self._lock:
            pend = self.pending.setdefault((name, group), {})
            entries = dict(self.streams.get(name, []))
            for message_id, owner in list(pend.items()):
                if owner == consumer:
                    continue
                claimed.append((message_id, dict(entries.get(message_id, {}))))
                pend[message_id] = consumer
                if count and len(claimed) >= count:
                    break
        return claimed

    def xlen(self, name):
        return len(self.streams.get(name, []))

    def xpending(self, name, group):
        return {"pending": len(self.pending.get((name, group), {}))}

    def xtrim(self, name, maxlen):
        with self._lock:
            entries = self.streams.get(name, [])
            removed = max(len(entries) - maxlen, 0)
            self.streams[name] = entries[removed:]
            return removed

    def xdel(self, name, *message_ids):
        with self._lock:
            entries = self.streams.get(name, [])
            keep = [e for e in entries if e[0] not in message_ids]
            self.streams[name] = keep
            return len(entries) - len(keep)

    # -- строки ключей (дедупликация) ----------------------------------- #
    def set(self, key, value, ex=None, nx=False):
        if self.fail:
            return None
        with self._lock:
            if nx and key in self.kv:
                return None
            self.kv[key] = value
            return True

    def get(self, key):
        return self.kv.get(key)

    def delete(self, key):
        """Redis DEL удаляет ключ любого типа: и строку, и поток."""
        with self._lock:
            removed = self.kv.pop(key, None) is not None
            if key in self.streams:
                self.streams.pop(key, None)
                self.pending = {k: v for k, v in self.pending.items() if k[0] != key}
                self.groups = {k: v for k, v in self.groups.items() if k[0] != key}
                removed = True
            return removed


def _queue(redis=None, **kwargs) -> StreamJobQueue:
    q = StreamJobQueue(redis if redis is not None else FakeStreamsRedis(), kinds=KINDS, **kwargs)
    q.ensure_groups()
    return q


def _job(task_type="ohlcv", key="k1", priority=Priority.INTERACTIVE, ticker="SPY") -> Job:
    return Job(
        task_type=task_type,
        idempotency_key=key,
        payload={"ticker": ticker, "params": {"timeframe": "1h"}},
        priority=priority,
        provider="yfinance",
    )


# ====================================================================== #
# 1. Контракт порта
# ====================================================================== #
def test_queue_implements_the_port():
    """Адаптер обязан удовлетворять протоколу: иначе порт — фикция."""
    assert isinstance(_queue(), JobQueuePort)


def test_empty_kinds_rejected():
    """Очередь без видов нечего читать — это ошибка сборки, а не пустая очередь."""
    try:
        StreamJobQueue(FakeStreamsRedis(), kinds=[])
    except ValueError:
        return
    raise AssertionError("пустой список видов принят")


def test_stream_naming_roundtrip():
    assert stream_name("ohlcv") == "gex:q:ohlcv"
    assert stream_name("ohlcv", Priority.BACKGROUND) == "gex:q:ohlcv:bg"
    assert base_kind("gex:q:ohlcv:bg") == "ohlcv"
    assert dlq_name("ohlcv") == "gex:q:dlq:ohlcv"


# ====================================================================== #
# 2. Публикация и разбор
# ====================================================================== #
def test_publish_then_read_keeps_job_intact():
    q = _queue()
    message_id = q.publish(_job(), queue="ohlcv")
    assert message_id is not None

    deliveries = q.read()
    assert len(deliveries) == 1
    delivery = deliveries[0]
    assert delivery.stream == "gex:q:ohlcv"
    assert delivery.message_id == message_id
    assert delivery.job.task_type == "ohlcv"
    assert delivery.job.provider == "yfinance"
    assert delivery.job.payload["ticker"] == "SPY"


def test_publish_to_unknown_queue_is_refused():
    """Неизвестная очередь — отказ с логом, а не молчаливая потеря задачи."""
    q = _queue()
    assert q.publish(_job(), queue="неведомая") is None
    assert q.read() == []


def test_read_returns_empty_list_when_no_messages():
    assert _queue().read() == []


def test_priority_interactive_is_read_before_background():
    """Контракт: приоритет влияет на порядок разбора, а не только на имя потока."""
    q = _queue()
    q.publish(_job(key="bg", priority=Priority.BACKGROUND), queue="ohlcv")
    q.publish(_job(key="hi", priority=Priority.INTERACTIVE), queue="ohlcv")

    deliveries = q.read(count=10)
    assert [d.job.idempotency_key for d in deliveries] == ["hi"]
    assert deliveries[0].stream.endswith(":bg") is False

    # Фоновое сообщение не потерялось — оно просто разбирается вторым.
    assert [d.job.idempotency_key for d in q.read()] == ["bg"]


def test_read_does_not_return_a_message_twice():
    q = _queue()
    q.publish(_job(), queue="ohlcv")
    assert len(q.read()) == 1
    assert q.read() == [], "сообщение выдано повторно без перехвата"


def test_corrupt_message_is_skipped_not_crashed():
    """Битое сообщение не должно ронять цикл разбора."""
    redis = FakeStreamsRedis()
    redis.xadd(stream_name("ohlcv"), {"payload": "не json", "task_type": "ohlcv"})
    q = _queue(redis)
    assert q.read() == []


# ====================================================================== #
# 3. At-least-once, DLQ, перехват
# ====================================================================== #
def test_unacked_message_stays_pending():
    """Ключевое отличие от BRPOP: без подтверждения сообщение не исчезает."""
    q = _queue()
    q.publish(_job(), queue="ohlcv")
    deliveries = q.read()
    assert q.pending("ohlcv")[stream_name("ohlcv")] == 1

    assert q.ack(deliveries[0]) is True
    assert q.pending("ohlcv")[stream_name("ohlcv")] == 0


def test_claim_stale_returns_messages_of_dead_consumer():
    """Задача, выданная упавшему воркеру, обязана вернуться в работу."""
    redis = FakeStreamsRedis()
    dead = StreamJobQueue(redis, kinds=KINDS, consumer="dead-worker")
    dead.ensure_groups()
    dead.publish(_job(), queue="ohlcv")
    dead.read()  # выдали «умершему» воркеру и не подтвердили

    alive = StreamJobQueue(redis, kinds=KINDS, consumer="alive-worker")
    claimed = alive.claim_stale(min_idle_ms=0)
    assert len(claimed) == 1
    assert claimed[0].redelivered is True
    assert claimed[0].job.task_type == "ohlcv"


def test_dead_letter_writes_and_acks():
    """Провал уходит в DLQ с причиной и подтверждается (иначе зависнет)."""
    redis = FakeStreamsRedis()
    q = _queue(redis)
    q.publish(_job(), queue="ohlcv")
    delivery = q.read()[0]

    assert q.dead_letter(delivery, "источник недоступен") is True
    assert q.dead_letters("ohlcv") == 1
    assert q.pending("ohlcv")[stream_name("ohlcv")] == 0

    fields = redis.streams[dlq_name("ohlcv")][0][1]
    assert fields["error"] == "источник недоступен"
    assert fields["source_id"] == delivery.message_id
    assert fields["task_type"] == "ohlcv"


def test_depth_reports_stream_lengths():
    q = _queue()
    q.publish(_job(key="a"), queue="ohlcv")
    q.publish(_job(key="b"), queue="chain")
    depth = q.depth()
    assert depth[stream_name("ohlcv")] == 1
    assert depth[stream_name("chain")] == 1
    assert q.describe()["available"] is True


def test_clear_empties_streams():
    q = _queue()
    q.publish(_job(), queue="ohlcv")
    q.clear()
    assert q.depth()[stream_name("ohlcv")] == 0


# ====================================================================== #
# 4. Дедупликация
# ====================================================================== #
def test_duplicate_is_processed_once():
    """at-least-once означает повторную доставку: повтор не должен выполнять работу."""
    q = _queue()
    job = _job(key="dup")
    assert q.should_process(job) is True
    assert q.should_process(job) is False
    assert q.duplicates_skipped == 1


def test_different_tasks_are_not_deduplicated():
    q = _queue()
    assert q.should_process(_job(key="a")) is True
    assert q.should_process(_job(key="b")) is True


def test_job_without_key_is_always_processed():
    """Без ключа дедуплицировать нечего: задача обрабатывается."""
    q = _queue()
    job = Job(task_type="ohlcv", idempotency_key="", payload={})
    assert q.should_process(job) is True


def test_forget_allows_reprocessing():
    q = _queue()
    job = _job(key="retry")
    q.should_process(job)
    assert q.forget(job) is True
    assert q.should_process(job) is True


def test_idempotency_key_is_normalized():
    """`spy` и `SPY` — одна задача: иначе дедупликация пропускает нужные повторы."""
    a = task_to_job(FetchTask("ohlcv", "yfinance", "spy", {"timeframe": "1h"}))
    b = task_to_job(FetchTask("ohlcv", "YFINANCE", "SPY", {"timeframe": "1h"}))
    assert a.idempotency_key == b.idempotency_key


# ====================================================================== #
# 5. Деградация без Redis
# ====================================================================== #
def test_queue_without_redis_degrades_quietly():
    """Приложение обязано подниматься без Redis: очередь сообщает «недоступна»."""
    q = StreamJobQueue(None, kinds=KINDS)
    assert q.publish(_job(), queue="ohlcv") is None
    assert q.read() == []
    assert q.depth() == {}
    assert q.dead_letters() == 0
    assert q.claim_stale() == []
    assert q.should_process(_job()) is True, "без Redis обработка не блокируется"
    assert q.describe()["available"] is False


def test_redis_failure_is_not_an_exception():
    """Сбой Redis у клиента — нейтральный ответ: очередь не бросает исключений наружу."""
    redis = FakeStreamsRedis()
    q = _queue(redis)
    redis.fail = True
    assert q.publish(_job(), queue="ohlcv") is None
    assert q.read() == []
    assert q.ack(Delivery(stream="s", message_id="1-0", job=_job())) is False


# ====================================================================== #
# 6. Модель задачи: очередь и приоритет
# ====================================================================== #
def test_task_routing_table():
    assert queue_for("ohlcv") == "ohlcv"
    assert queue_for("sector") == "vol"
    assert queue_for("нечто") == "default"
    assert "default" in QUEUE_KINDS


def test_task_to_job_priority_mapping():
    low = task_to_job(FetchTask("ohlcv", "yfinance", "SPY", {}, priority=0))
    high = task_to_job(FetchTask("ohlcv", "yfinance", "SPY", {}, priority=2))
    assert low.priority == Priority.BACKGROUND
    assert high.priority == Priority.INTERACTIVE


def test_job_to_task_restores_body():
    task = FetchTask("chain", "bybit", "BTC", {"max_expiries": 3}, priority=2)
    restored = job_to_task(task_to_job(task))
    assert (restored.task_type, restored.provider, restored.ticker) == ("chain", "bybit", "BTC")
    assert restored.params == {"max_expiries": 3}
    assert restored.priority == 2


# ====================================================================== #
# 7. Публикатор (фасад для планировщика)
# ====================================================================== #
def test_publisher_routes_to_task_queue():
    """Планировщик ставит превентивные задачи (priority=1) — это фоновый поток."""
    redis = FakeStreamsRedis()
    publisher = TaskPublisher(_queue(redis))
    assert publisher.publish(FetchTask("ohlcv", "yfinance", "SPY", {"timeframe": "1h"})) is True
    assert redis.xlen(stream_name("ohlcv", Priority.BACKGROUND)) == 1
    assert redis.xlen(stream_name("ohlcv")) == 0, "превентивная задача не должна попасть в интерактивный поток"


def test_publisher_sends_interactive_task_to_interactive_stream():
    """Задача по запросу пользователя (priority=2) идёт в интерактивный поток."""
    redis = FakeStreamsRedis()
    publisher = TaskPublisher(_queue(redis))
    task = FetchTask("ohlcv", "yfinance", "SPY", {"timeframe": "1h"}, priority=2)
    assert publisher.publish(task) is True
    assert redis.xlen(stream_name("ohlcv")) == 1


def test_publisher_reports_failure_without_raising():
    publisher = TaskPublisher(StreamJobQueue(None, kinds=KINDS))
    assert publisher.publish(FetchTask("ohlcv", "yfinance", "SPY")) is False
    assert publisher.queue_length() == 0
    assert publisher.dead_letters() == 0


def test_publisher_publish_many_counts_successes():
    redis = FakeStreamsRedis()
    publisher = TaskPublisher(_queue(redis))
    tasks = [FetchTask("ohlcv", "yfinance", "SPY"), FetchTask("chain", "bybit", "BTC")]
    assert publisher.publish_many(tasks) == 2


# ====================================================================== #
# 8. Потребитель
# ====================================================================== #
def test_consumer_handles_and_acks():
    redis = FakeStreamsRedis()
    q = _queue(redis)
    q.publish(_job(), queue="ohlcv")

    seen: list[FetchTask] = []
    consumer = TaskConsumer(q, handler=seen.append, block_ms=None)
    assert consumer.handle_once() == 1
    assert [t.ticker for t in seen] == ["SPY"]
    assert consumer.stats.processed == 1
    assert q.pending("ohlcv")[stream_name("ohlcv")] == 0, "сообщение не подтверждено"


def test_consumer_sends_failed_task_to_dlq():
    """Исключение обработчика — это DLQ, а не потерянная задача и не залипшая очередь."""
    redis = FakeStreamsRedis()
    q = _queue(redis)
    q.publish(_job(), queue="ohlcv")

    def boom(_task):
        raise RuntimeError("провайдер лёг")

    consumer = TaskConsumer(q, handler=boom, block_ms=None)
    assert consumer.handle_once() == 0
    assert consumer.stats.failed == 1
    assert consumer.stats.dead_lettered == 1
    assert q.dead_letters("ohlcv") == 1
    assert q.pending("ohlcv")[stream_name("ohlcv")] == 0


def test_consumer_skips_duplicate_without_calling_handler():
    redis = FakeStreamsRedis()
    q = _queue(redis)
    job = _job(key="dup")
    calls: list[FetchTask] = []

    consumer = TaskConsumer(q, handler=calls.append, block_ms=None)
    q.publish(job, queue="ohlcv")
    consumer.handle_once()
    # Повторная доставка того же ключа (как при at-least-once)
    q.publish(job, queue="ohlcv")
    consumer.handle_once()

    assert len(calls) == 1, "дубликат выполнен повторно"
    assert consumer.stats.skipped == 1


def test_consumer_recovers_stale_messages():
    """Потребитель обязан подобрать то, что осталось у умершего воркера."""
    redis = FakeStreamsRedis()
    dead = StreamJobQueue(redis, kinds=KINDS, consumer="dead")
    dead.ensure_groups()
    dead.publish(_job(), queue="ohlcv")
    dead.read()  # выдано «умершему» и не подтверждено

    alive_port = StreamJobQueue(redis, kinds=KINDS, consumer="alive")
    seen: list[FetchTask] = []
    consumer = TaskConsumer(alive_port, handler=seen.append, block_ms=None, min_idle_ms=0, clock=lambda: 1e9)
    consumer.handle_once()
    assert len(seen) == 1
    assert consumer.stats.claimed == 1


def test_consumer_stats_are_observable():
    redis = FakeStreamsRedis()
    q = _queue(redis)
    consumer = TaskConsumer(q, handler=lambda t: None, block_ms=None)
    stats = consumer.stats.as_dict()
    assert set(stats) >= {"processed", "skipped", "failed", "dead_lettered", "claimed"}


def test_consumer_run_stops_on_event():
    """Цикл в потоке должен останавливаться по событию, а не жить вечно."""
    q = _queue()
    consumer = TaskConsumer(q, handler=lambda t: None, block_ms=None, claim_interval_s=0.01)
    thread = consumer.start(name="test-consumer")
    time.sleep(0.05)
    consumer.stop(timeout=5)
    assert not thread.is_alive()



# ====================================================================== #
# 9. Контракт с RedisClient: адаптер не может звать то, чего у клиента нет
# ====================================================================== #
def test_adapter_uses_only_declared_client_methods():
    """Методы, которые адаптер зовёт у Redis, обязаны быть у ``RedisClient``.

    Этот класс дефекта встречался трижды (``connect_timeout`` в итер. 22, ``nx``/``px``
    в 26, ``script_load`` в 27): адаптер опирается на метод клиента, которого нет,
    и в проде это выглядит как «Redis недоступен». Проверка статическая: набор обязан
    работать без pandas.
    """
    def _methods(path: str, cls: str) -> set[str]:
        src = (ROOT / path).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                return {i.name for i in node.body if isinstance(i, ast.FunctionDef)}
        return set()

    # Streams-операции приходят миксином, поэтому смотрим и класс, и его базовый миксин:
    # проверка «только по телу RedisClient» пропустила бы их отсутствие.
    methods = _methods("gex/adapters/cache/redis_client.py", "RedisClient") | _methods(
        "gex/adapters/queue/streams_client.py", "StreamsClientMixin"
    )
    assert {"xadd", "xack", "xreadgroup", "xautoclaim", "xgroup_create", "xlen"} <= methods, (
        f"Streams-операций нет ни у RedisClient, ни у миксина: {sorted(methods)}"
    )

    adapter = (ROOT / "gex" / "adapters" / "queue" / "redis_streams.py").read_text(encoding="utf-8")
    called: set[str] = set()
    for node in ast.walk(ast.parse(adapter)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_redis"
        ):
            called.add(node.func.attr)
    assert called, "адаптер не обращается к Redis — проверка потеряла смысл"
    undeclared = called - methods
    assert not undeclared, f"адаптер зовёт методы, которых нет у RedisClient: {sorted(undeclared)}"


def test_fake_matches_client_signatures():
    """Фейк должен совпадать с клиентом по именам аргументов.

    Расхождение уже приводило к ложным падениям (``block`` против ``block_ms``), а
    обратная ситуация хуже: тест зелёный, а продовый вызов падает по ``TypeError``.
    """
    def _params(path: str, cls: str) -> dict[str, set[str]]:
        src = (ROOT / path).read_text(encoding="utf-8")
        out: dict[str, set[str]] = {}
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        out[item.name] = {a.arg for a in item.args.args + item.args.kwonlyargs}
        return out

    # Streams-методы живут в миксине: без него карта параметров пустая (KeyError на xadd)
    client_params = _params("gex/adapters/cache/redis_client.py", "RedisClient") | _params(
        "gex/adapters/queue/streams_client.py", "StreamsClientMixin"
    )

    fake_src = (ROOT / "tests" / "test_job_queue.py").read_text(encoding="utf-8")
    fake_params: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(fake_src)):
        if isinstance(node, ast.ClassDef) and node.name == "FakeStreamsRedis":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    fake_params[item.name] = {a.arg for a in item.args.args + item.args.kwonlyargs}

    for method in ("xadd", "xgroup_create", "xreadgroup", "xack", "xautoclaim", "xlen"):
        assert method in fake_params, f"в фейке нет {method}"
        missing = client_params[method] - fake_params[method]
        assert not missing, f"фейк {method} не принимает {sorted(missing)} (как клиент)"


# ====================================================================== #
#  Профили потребителей: market overview не ждёт расчётов
# ====================================================================== #
def test_consumer_profiles_cover_all_queues_without_overlap():
    """Каждая очередь разбирается ровно одним профилем — без пропусков и дублей."""
    covered = [kind for kinds in CONSUMER_PROFILES.values() for kind in kinds]
    assert len(covered) == len(set(covered)), "вид задачи попал в два профиля"
    assert set(covered) == set(QUEUE_KINDS), "не все очереди разобраны профилями"


def test_profile_streams_do_not_intersect():
    """Стримы профилей не пересекаются: каждый потребитель читает только свои очереди."""
    streams = {
        name: set(StreamJobQueue(None, kinds=kinds, consumer=f"w-{name}").all_streams())
        for name, kinds in CONSUMER_PROFILES.items()
    }
    names = list(streams)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert streams[a].isdisjoint(streams[b]), f"профили {a}/{b} читают общий стрим"


def test_profile_consumers_split_overview_from_calculations():
    """Расчёт не стоит в очереди перед свечами: fast видит только свои задачи.

    Сценарий, ради которого профили появились: один потребитель разбирал все очереди
    последовательно, и долгая breadth-задача (десятки секунд) задерживала разбор
    свежих свечей market overview. Теперь это два независимых чтения.
    """
    redis = FakeStreamsRedis()
    # Публикация идёт общим портом (все очереди) — как в проде bootstrap-публикатором.
    pub = TaskPublisher(StreamJobQueue(redis, kinds=QUEUE_KINDS, consumer="gex-worker"))
    pub.publish(FetchTask("ohlcv", "yfinance", "SPY"))
    pub.publish(FetchTask("chain", "webull", "SPY"))
    pub.publish(FetchTask("breadth", "yfinance", "ALL"))

    fast = StreamJobQueue(redis, kinds=CONSUMER_PROFILES["fast"], consumer="gex-worker-fast")
    heavy = StreamJobQueue(redis, kinds=CONSUMER_PROFILES["heavy"], consumer="gex-worker-heavy")
    fast.ensure_groups()
    heavy.ensure_groups()

    fast_types = [d.job.task_type for d in fast.read(count=10)]
    heavy_types = sorted(d.job.task_type for d in heavy.read(count=10))

    assert fast_types == ["ohlcv"], "быстрый потребитель должен видеть только свечи"
    assert heavy_types == ["breadth", "chain"], "тяжёлый потребитель — цепочки и расчёты"


def test_fast_consumer_cannot_be_beaten_by_slow_queue():
    """Обратная сторона контракта: heavy-потребитель не «съедает» задачи fast.

    Если бы оба профиля читали общий стрим одной группой, тяжёлый поток мог бы
    разбирать свечи — и маршрутизация «по профилям» ничего бы не гарантировала.
    """
    redis = FakeStreamsRedis()
    pub = TaskPublisher(StreamJobQueue(redis, kinds=QUEUE_KINDS, consumer="gex-worker"))
    pub.publish(FetchTask("ohlcv", "yfinance", "QQQ"))

    fast = StreamJobQueue(redis, kinds=CONSUMER_PROFILES["fast"], consumer="gex-worker-fast")
    heavy = StreamJobQueue(redis, kinds=CONSUMER_PROFILES["heavy"], consumer="gex-worker-heavy")
    fast.ensure_groups()
    heavy.ensure_groups()

    assert heavy.read(count=10) == [], "heavy не должен видеть очереди fast-профиля"
    assert [d.job.task_type for d in fast.read(count=10)] == ["ohlcv"]


# ====================================================================== #
# Доступность очереди: Redis удалён/лежит ≠ «задача не поставлена»
# ====================================================================== #
class _ConnFlagRedis(FakeStreamsRedis):
    """Клиент с признаком подключения (как ``RedisClient.connected``).

    Отключённый клиент — не «исключение», а нейтральный ответ: настоящий
    ``RedisClient`` при ``connected=False`` возвращает ``None``, иначе приложение
    не поднялось бы без Redis. Фейк обязан повторять это, иначе проверяется не
    тот интерфейс.
    """

    def __init__(self, connected: bool):
        super().__init__()
        self.connected = connected

    def xadd(self, name, fields, maxlen=None):
        if not self.connected:
            return None
        return super().xadd(name, fields, maxlen=maxlen)


def test_available_follows_connection_not_just_presence():
    """Клиент есть, но соединения нет — очередь недоступна.

    Без этого планировщик не отличал «Redis удалён» от «задача не принята»
    и писал предупреждение на каждую задачу.
    """
    assert StreamJobQueue(_ConnFlagRedis(True), kinds=KINDS).available() is True
    assert StreamJobQueue(_ConnFlagRedis(False), kinds=KINDS).available() is False
    assert StreamJobQueue(None, kinds=KINDS).available() is False


def test_available_true_for_clients_without_a_flag():
    """Клиент без признака подключения (тестовый) не считается отключённым."""
    assert StreamJobQueue(FakeStreamsRedis(), kinds=KINDS).available() is True


def test_publisher_does_not_flood_log_when_queue_is_gone():
    """Очереди нет — одна строка на интервал, а не по строке на задачу.

    Собственный перехватчик вместо ``caplog``: файл запускается и как
    ``python tests/test_job_queue.py``, где фикстур нет.
    """
    import logging

    import gex.application.queue as queue_mod

    queue_mod._last_unavailable_log = 0.0
    queue_mod._unavailable_suppressed = 0
    publisher = TaskPublisher(StreamJobQueue(_ConnFlagRedis(False), kinds=KINDS))

    collected: list[str] = []

    class _Catcher(logging.Handler):
        def emit(self, record):
            collected.append(record.getMessage())

    handler = _Catcher(level=logging.WARNING)
    queue_mod.logger.addHandler(handler)
    try:
        for _ in range(50):
            publisher.publish(FetchTask(task_type="ohlcv", provider="yfinance", ticker="SPY"))
    finally:
        queue_mod.logger.removeHandler(handler)

    about_queue = [m for m in collected if "недоступна" in m]
    assert len(about_queue) == 1, f"ожидалось одно сообщение, получено {len(about_queue)}"
    assert "SPY" not in about_queue[0]  # по задаче не пишем — таких были бы десятки


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- job queue: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
