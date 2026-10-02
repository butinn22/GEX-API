"""Unit/integration tests for the central orchestrator core.

Uses fakeredis so no external network or Redis server is required.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fakeredis.aioredis import FakeRedis

from gex.orchestrator.cache import OrchestratorCache, build_cache_key
from gex.orchestrator.circuit_breaker import RedisCircuitBreaker
from gex.orchestrator.queue import OrchestratorQueue
from gex.orchestrator.rate_limiter import RedisRateLimiter
from gex.orchestrator.schemas import (
    OrchestratorRequest,
    OrchestratorTask,
    RateLimitRuleConfig,
)
from gex.orchestrator.service import Orchestrator
from gex.orchestrator.adapters.base import BaseProviderAdapter, ProviderAdapterRegistry
from gex.orchestrator.schemas import ProviderResponse


@pytest.fixture
async def fake_redis():
    redis = FakeRedis()
    yield redis
    await redis.aclose()


@pytest.fixture
def cache_key_request() -> OrchestratorRequest:
    return OrchestratorRequest(
        provider="bybit",
        data_type="candles",
        symbol="BTC",
        params={"interval": "1h", "limit": 100, "extra": {"a": 1}},
    )


# ----------------------------------------------------------------------
# Cache key builder
# ----------------------------------------------------------------------
class TestCacheKeyBuilder:
    def test_deterministic(self, cache_key_request):
        key1 = build_cache_key(cache_key_request)
        key2 = build_cache_key(cache_key_request.model_copy(deep=True))
        assert key1 == key2

    def test_includes_parts_and_version(self, cache_key_request):
        key = build_cache_key(cache_key_request)
        assert key.startswith("orchestrator:cache:bybit:candles:BTC:1h:-:-:")
        assert key.endswith(":v1")

    def test_symbol_uppercased(self, cache_key_request):
        req = cache_key_request.model_copy(deep=True)
        req.symbol = "btc"
        assert build_cache_key(req).startswith("orchestrator:cache:bybit:candles:BTC:")


# ----------------------------------------------------------------------
# Rate limiter
# ----------------------------------------------------------------------
class TestRedisRateLimiter:
    async def test_token_bucket_allows_first_denies_second(self, fake_redis):
        limiter = RedisRateLimiter(
            fake_redis,
            rules=[
                RateLimitRuleConfig(
                    name="rps",
                    endpoint_pattern="*",
                    rate_per_second=0.001,
                    burst=1,
                    cost=1,
                )
            ],
        )
        first = await limiter.allow("yfinance", "*")
        second = await limiter.allow("yfinance", "*")
        assert first.allowed is True
        assert second.allowed is False
        assert second.retry_after_ms > 0

    async def test_fixed_window_quota(self, fake_redis):
        limiter = RedisRateLimiter(
            fake_redis,
            rules=[
                RateLimitRuleConfig(
                    name="hourly",
                    endpoint_pattern="*",
                    window_seconds=3600,
                    max_requests=2,
                    cost=1,
                )
            ],
        )
        assert (await limiter.allow("iss", "*")).allowed is True
        assert (await limiter.allow("iss", "*")).allowed is True
        third = await limiter.allow("iss", "*")
        assert third.allowed is False

    async def test_cost_greater_than_one(self, fake_redis):
        limiter = RedisRateLimiter(
            fake_redis,
            rules=[
                RateLimitRuleConfig(
                    name="rps",
                    endpoint_pattern="*",
                    rate_per_second=0.001,
                    burst=2,
                    cost=2,
                )
            ],
        )
        assert (await limiter.allow("bybit", "*", cost=2)).allowed is True
        assert (await limiter.allow("bybit", "*", cost=1)).allowed is False


# ----------------------------------------------------------------------
# Circuit breaker
# ----------------------------------------------------------------------
class TestCircuitBreaker:
    async def test_open_after_failures(self, fake_redis):
        breaker = RedisCircuitBreaker(fake_redis, failure_threshold=2, cooldown_seconds=3600)
        assert await breaker.allow("bybit", "candles") is True
        await breaker.record_failure("bybit", "candles")
        assert await breaker.allow("bybit", "candles") is True
        await breaker.record_failure("bybit", "candles")
        assert await breaker.allow("bybit", "candles") is False
        assert await breaker.state("bybit", "candles") == "open"

    async def test_success_closes(self, fake_redis):
        breaker = RedisCircuitBreaker(fake_redis, failure_threshold=1, cooldown_seconds=3600)
        await breaker.record_failure("iss", "candles")
        assert await breaker.state("iss", "candles") == "open"
        await breaker.record_success("iss", "candles")
        assert await breaker.state("iss", "candles") == "closed"


# ----------------------------------------------------------------------
# Queue
# ----------------------------------------------------------------------
class TestOrchestratorQueue:
    async def test_enqueue_read_ack(self, fake_redis):
        queue = OrchestratorQueue(fake_redis)
        task = OrchestratorTask(
            request_id="00000000-0000-0000-0000-000000000001",
            provider="yfinance",
            data_type="candles",
            symbol="SPY",
            params={"interval": "1h"},
            priority="interactive",
            cache_key="orchestrator:cache:test",
        )
        mid = await queue.enqueue(task)
        assert mid is not None
        entries = await queue.read("yfinance", "consumer-1", "interactive", count=1, block_ms=100)
        assert len(entries) == 1
        entry_mid, read_task = entries[0]
        assert read_task.symbol == "SPY"
        assert await queue.ack("yfinance", "interactive", entry_mid) is True

    async def test_depth_counts(self, fake_redis):
        queue = OrchestratorQueue(fake_redis)
        task = OrchestratorTask(
            request_id="00000000-0000-0000-0000-000000000002",
            provider="webull",
            data_type="symbol_info",
            symbol="AAPL",
            priority="background",
            cache_key="orchestrator:cache:test2",
        )
        await queue.enqueue(task)
        depth = await queue.depth("webull")
        assert depth["orchestrator:queue:webull:background"] == 1


# ----------------------------------------------------------------------
# Singleflight / service
# ----------------------------------------------------------------------
class CountingAdapter(BaseProviderAdapter):
    provider_code = "bybit"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, task: OrchestratorTask) -> ProviderResponse:
        self.calls += 1
        await asyncio.sleep(0.01)
        return ProviderResponse(
            provider=self.provider_code,
            data_type=task.data_type,
            data={"symbol": task.symbol, "value": self.calls},
            status_code=200,
        )


class TestSingleflightAndService:
    async def test_100_identical_requests_trigger_one_external_call(self, fake_redis):
        registry = ProviderAdapterRegistry()
        adapter = CountingAdapter()
        registry.register(adapter)
        orch = Orchestrator(fake_redis, adapter_registry=registry, inline_execution=True)
        await orch.queue.clear()

        async def one_request() -> Any:
            return await orch.fetch_candles("BTC", "bybit", "1h")

        results = await asyncio.gather(*[one_request() for _ in range(100)])
        assert adapter.calls == 1
        assert all(r.data["symbol"] == "BTC" for r in results)

    async def test_cache_hit_does_not_call_adapter(self, fake_redis):
        registry = ProviderAdapterRegistry()
        adapter = CountingAdapter()
        registry.register(adapter)
        orch = Orchestrator(fake_redis, adapter_registry=registry, inline_execution=True)
        await orch.queue.clear()
        await orch.fetch_candles("ETH", "bybit", "1h")
        calls_after_first = adapter.calls
        await orch.fetch_candles("ETH", "bybit", "1h")
        assert calls_after_first == 1
        assert adapter.calls == 1

    async def test_stale_returns_stale_and_submits_revalidation(self, fake_redis):
        registry = ProviderAdapterRegistry()
        adapter = CountingAdapter()
        registry.register(adapter)
        orch = Orchestrator(fake_redis, adapter_registry=registry, inline_execution=True)
        await orch.queue.clear()

        key = "orchestrator:cache:bybit:candles:DOGE:1h:-:-:abc:v1"
        # Manually create a stale cache entry.
        await orch.cache.set(
            key,
            {"old": True},
            provider="bybit",
            data_type="candles",
            symbol="DOGE",
            fresh_ttl_seconds=1,
            stale_ttl_seconds=60,
        )
        # Pretend it is stale by inserting a very old fresh expiry but still within stale TTL.
        # cache.set computes now, so we directly rewrite the envelope fields via Redis.
        import json
        raw = await fake_redis.get(key)
        payload = json.loads(raw)
        from datetime import datetime, timedelta, timezone
        payload["fetched_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        payload["expires_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        payload["stale_until"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        await fake_redis.set(key, json.dumps(payload))

        req = OrchestratorRequest(
            provider="bybit",
            data_type="candles",
            symbol="DOGE",
            params={"interval": "1h"},
            priority="interactive",
        )
        req.cache_key = key
        result = await orch.execute(req)
        assert result.cache_status.value == "STALE"
        assert result.data == {"old": True}
