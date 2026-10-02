"""Domain ring — чистые доменные сущности и правила.

ПРАВИЛА КОЛЬЦА (проверяются scripts/quality/ast_guard.py):
  * разрешено: stdlib, numpy, pandas, scipy;
  * запрещено: requests, httpx, redis, sqlalchemy, fastapi, yfinance, aiohttp, urllib.request;
  * запрещено импортировать gex.application, gex.adapters, gex.ports (домен ничего не знает о внешних слоях).

Наполняется по итерациям: indicators/, analysis/, options/, freshness.py, models.py, errors.py.
"""
