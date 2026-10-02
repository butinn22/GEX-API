"""Unit-тесты: RateLimiter, TaskQueue, background_fetcher, scheduler."""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from gex.adapters.ratelimit.rate_limiter import TokenBucket, RateLimiter


# ====================================================================== #
#  Token Bucket
# ====================================================================== #
class TestTokenBucket:
    def test_init(self):
        tb = TokenBucket(rate=5.0, burst=10, name="test")
        assert tb.rate == 5.0
        assert tb.burst == 10
        assert tb.available <= 10.0

    def test_acquire_success(self):
        tb = TokenBucket(rate=1000.0, burst=1000, name="fast")
        assert tb.acquire(tokens=1, blocking=False)
        assert tb.available < 1000.0  # уменьшилось

    def test_acquire_fail_no_block(self):
        tb = TokenBucket(rate=0.001, burst=1, name="slow")
        tb.acquire(tokens=1, blocking=False)  # забираем единственный
        assert not tb.acquire(tokens=1, blocking=False)  # не хватает

    def test_acquire_blocking(self):
        """Блокирующий acquire — ждёт пополнения."""
        tb = TokenBucket(rate=100.0, burst=2, name="fast")
        tb.acquire(tokens=2, blocking=False)  # обнулили
        start = time.monotonic()
        assert tb.acquire(tokens=1, blocking=True)  # должен подождать ~0.01с
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # должно быть быстро

    def test_negative_rate(self):
        with pytest.raises(ValueError):
            TokenBucket(rate=-1, burst=10)

    def test_zero_burst(self):
        with pytest.raises(ValueError):
            TokenBucket(rate=1, burst=0)

    def test_reset(self):
        tb = TokenBucket(rate=10, burst=5, name="r")
        tb.acquire(tokens=5, blocking=False)
        assert tb.available < 5
        tb.reset()
        assert tb.available == 5

    def test_refill_over_time(self):
        """Ведро пополняется со временем."""
        tb = TokenBucket(rate=10.0, burst=5, name="r")
        tb.acquire(tokens=5, blocking=False)  # 0 токенов
        time.sleep(0.3)  # ~3 токена должно накопиться
        assert tb.available >= 2.0  # как минимум 2


# ====================================================================== #
#  RateLimiter
# ====================================================================== #
class TestRateLimiter:
    def test_default_providers(self):
        rl = RateLimiter()
        assert "yfinance" in rl.providers
        assert "bybit" in rl.providers
        assert "moex_iss" in rl.providers

    def test_unknown_provider_always_allows(self):
        rl = RateLimiter()
        assert rl.acquire("nonexistent", blocking=False)

    def test_wait_blocking(self):
        rl = RateLimiter()
        rl.acquire("yfinance", tokens=10, blocking=False)  # исчерпали
        start = time.monotonic()
        # Ждём 1 токен — должно быть быстро (rate=4/сек)
        rl.wait("yfinance")
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_reset_all(self):
        rl = RateLimiter()
        rl.acquire("yfinance", tokens=8, blocking=False)
        rl.reset_all()
        assert rl.get_bucket("yfinance").available == 8  # полный burst

    def test_get_bucket(self):
        rl = RateLimiter()
        b = rl.get_bucket("bybit")
        assert b is not None
        assert b.rate == 8.0


# ====================================================================== #
#  FetchTask (dataclass)
# ====================================================================== #
class TestFetchTask:
    def test_from_json(self):
        from gex.application.jobs import FetchTask
        task = FetchTask("ohlcv", "yfinance", "SPY", {"timeframe": "1d"}, priority=1)
        json_str = task.to_json()
        restored = FetchTask.from_json(json_str)
        assert restored.task_type == "ohlcv"
        assert restored.provider == "yfinance"
        assert restored.ticker == "SPY"
        assert restored.params == {"timeframe": "1d"}

    def test_queue_mapping(self):
        from gex.application.jobs import FetchTask
        assert FetchTask("ohlcv", "y", "SPY").queue == "ohlcv"
        assert FetchTask("chain", "y", "BTC").queue == "chain"
        assert FetchTask("vol", "y", "ALL").queue == "vol"
        assert FetchTask("unknown", "y", "X").queue == "default"


# ====================================================================== #
#  TaskPublisher (очередь за портом)
# ====================================================================== #
class FakePort:
    """Порт очереди в памяти: тесты публикации не требуют Redis."""

    def __init__(self):
        self.messages: list[tuple] = []
        self.depth_calls = 0
        self.clear_calls = 0
        self.fail = False

    def publish(self, job, *, queue=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.messages.append((queue, job))
        return f"{len(self.messages)}-0"

    def depth(self, queue=None):
        self.depth_calls += 1
        return {f"gex:q:{queue}": len(self.messages)} if queue else {"gex:q:ohlcv": len(self.messages)}

    def dead_letters(self, queue=None):
        return 0

    def clear(self, queue=None):
        self.clear_calls += 1
        self.messages.clear()


class TestTaskPublisher:
    def test_publishes_task_to_its_queue(self):
        """Задача уходит в очередь своего вида, тело сохраняется."""
        from gex.application.jobs import FetchTask
        from gex.application.queue import TaskPublisher

        port = FakePort()
        publisher = TaskPublisher(port)
        task = FetchTask("ohlcv", "yfinance", "SPY", {"timeframe": "1h"})

        assert publisher.publish(task) is True
        queue, job = port.messages[0]
        assert queue == "ohlcv"
        assert job.task_type == "ohlcv"
        assert job.provider == "yfinance"
        assert job.payload["ticker"] == "SPY"

    def test_publish_many_counts_successes(self):
        from gex.application.jobs import FetchTask
        from gex.application.queue import TaskPublisher

        port = FakePort()
        publisher = TaskPublisher(port)
        tasks = [FetchTask("ohlcv", "yfinance", "SPY"), FetchTask("chain", "bybit", "BTC")]

        assert publisher.publish_many(tasks) == 2
        assert [q for q, _ in port.messages] == ["ohlcv", "chain"]

    def test_failure_is_reported_not_raised(self):
        """Отказ очереди не должен ронять цикл планировщика."""
        from gex.application.jobs import FetchTask
        from gex.application.queue import TaskPublisher

        port = FakePort()
        port.fail = True
        publisher = TaskPublisher(port)
        assert publisher.publish(FetchTask("ohlcv", "yfinance", "SPY")) is False

    def test_queue_length_sums_depths(self):
        from gex.application.queue import TaskPublisher

        port = FakePort()
        publisher = TaskPublisher(port)
        assert publisher.queue_length() == 0

    def test_clear_queues_delegates(self):
        from gex.application.queue import TaskPublisher

        port = FakePort()
        publisher = TaskPublisher(port)
        publisher.clear_queues()
        assert port.clear_calls == 1

    def test_unknown_task_type_falls_back_to_default_queue(self):
        """Неизвестный вид задачи не теряется, а едет в общую очередь."""
        from gex.application.jobs import FetchTask
        from gex.application.queue import TaskPublisher

        port = FakePort()
        TaskPublisher(port).publish(FetchTask("неведомый_вид", "yfinance", "X"))
        assert port.messages[0][0] == "default"


# ====================================================================== #
#  Scheduler
# ====================================================================== #
class TestScheduler:
    def test_scheduler_start_stop(self):
        """Scheduler стартует и останавливается без ошибок."""
        from gex.deps import get_task_queue
        from gex.application.scheduler import Scheduler

        q = get_task_queue()
        sched = Scheduler(queue=q, interval_multiplier=100.0)  # почти никогда
        sched.start()
        assert sched.is_running
        sched.stop()
        assert not sched.is_running

    def test_trigger_prewarm(self):
        from gex.application.scheduler import trigger_prewarm
        from gex.adapters.cache.redis_client import get_redis

        redis = get_redis()
        if not redis or not redis.connected:
            pytest.skip("Redis not available")

        result = trigger_prewarm()
        assert "published" in result
        assert result["total"] > 10
