"""HTTP transports (ring: adapters) — единственное место, где выполняются исходящие HTTP-вызовы.

Правило «один транспорт»: requests-сессия, политика timeout/retry/backoff и deadline-обёртка для yfinance
живут здесь; сервисы и домен инфраструктуру не видят.

Содержимое по итерациям 22–24 плана:
  * :mod:`gex.adapters.transport.http` — единый HTTP-транспорт (таймаут, ретраи, пул) — готово;
  * ``yf_transport`` — deadline-обёртка над yfinance (итерация 23);
  * ``loop`` — ``AsyncBridge`` вместо ``asyncio.run`` на каждый фетч (итерация 24).
"""
