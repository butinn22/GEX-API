"""Tests for the token-bucket rate limiter."""
from __future__ import annotations

import pytest

from trading.adapters.ratelimit import RateLimiter, TokenBucket


def _clock_at(store: dict) -> "callable":
    return lambda: store["t"]


def test_token_bucket_drains_and_blocks():
    t = {"t": 0.0}
    b = TokenBucket(capacity=3.0, refill_rate=1.0, clock=_clock_at(t))
    assert b.try_acquire(1.0) == (True, 0.0)
    allowed, retry = b.try_acquire(3.0)  # only 2 left
    assert allowed is False
    assert retry == pytest.approx(1.0)  # need 1 more token at 1/sec


def test_token_bucket_refills_over_time():
    t = {"t": 0.0}
    b = TokenBucket(capacity=2.0, refill_rate=1.0, clock=_clock_at(t))
    assert b.try_acquire(2.0)[0] is True
    assert b.try_acquire(1.0)[0] is False
    t["t"] = 1.5
    assert b.try_acquire(1.0)[0] is True  # refilled 1.5 → capped 2 → 1 left


def test_token_bucket_caps_at_capacity():
    t = {"t": 0.0}
    b = TokenBucket(capacity=1.0, refill_rate=10.0, clock=_clock_at(t))
    t["t"] = 100.0  # long time passed, still capped at capacity
    assert b.try_acquire(1.0)[0] is True
    assert b.try_acquire(1.0)[0] is False


def test_token_bucket_rejects_bad_params():
    with pytest.raises(ValueError):
        TokenBucket(0.0, 1.0)
    with pytest.raises(ValueError):
        TokenBucket(1.0, 0.0)


def test_rate_limiter_scopes_are_independent():
    t = {"t": 0.0}
    rl = RateLimiter(clock=_clock_at(t))
    assert rl.acquire("a", 1.0, 1.0)[0] is True
    assert rl.acquire("a", 1.0, 1.0)[0] is False  # scope a exhausted
    assert rl.acquire("b", 1.0, 1.0)[0] is True  # scope b untouched
