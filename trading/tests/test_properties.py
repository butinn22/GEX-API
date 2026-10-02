"""Property-based tests (hypothesis) for value-object and metric invariants."""
from __future__ import annotations

import numpy as np
from hypothesis import given, strategies as st

from trading.adapters.ratelimit import TokenBucket
from trading.application.backtest.metrics import max_drawdown
from trading.domain.money import Price, Quantity


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
       st.floats(min_value=1e-6, max_value=1000, allow_nan=False))
def test_price_rounding_is_idempotent(value, tick):
    p = Price(value, tick)
    assert Price(p.value, tick).value == p.value


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
       st.floats(min_value=1e-6, max_value=1000, allow_nan=False))
def test_quantity_rounding_is_idempotent(value, lot):
    q = Quantity(value, lot)
    assert Quantity(q.value, lot).value == q.value


@given(st.floats(min_value=0.01, max_value=1000, allow_nan=False),
       st.integers(min_value=1, max_value=500))
def test_token_bucket_stays_within_capacity(capacity, n):
    b = TokenBucket(capacity=capacity, refill_rate=capacity / 10)
    for _ in range(n):
        b.try_acquire(capacity / 50)
    assert 0.0 <= b._tokens <= capacity


@given(st.lists(st.floats(min_value=-0.3, max_value=0.3, allow_nan=False),
                min_size=2, max_size=200))
def test_max_drawdown_bounds(returns):
    equity = 100.0 * np.cumprod(1.0 + np.asarray(returns))
    equity = np.concatenate([[100.0], equity])
    mdd = max_drawdown(equity)
    assert 0.0 <= mdd <= 1.0
