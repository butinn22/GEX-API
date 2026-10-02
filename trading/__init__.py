"""Trading bounded context — orders, execution, backtest, Monte-Carlo.

Separate from ``gex`` (the analytics engine). This package depends on ``gex``
only through its **ports** (rate limiter, market data, cache), never through its
adapters, so the analytics engine can evolve independently.

Layout mirrors ``gex`` (hexagonal): ``domain`` (pure), ``ports`` (interfaces),
``adapters`` (brokers/fetchers/ratelimit), ``application`` (engine/backtest),
``api`` (routers + websockets), ``tasks`` (celery).
"""
