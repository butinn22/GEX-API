"""Application ring — use-cases (сценарии) приложения.

ПРАВИЛА КОЛЬЦА (проверяется scripts/quality/ast_guard.py):
  * разрешено: gex.domain, gex.ports, stdlib, typing, dataclasses;
  * запрещено: fastapi, sqlalchemy, redis, requests, httpx, yfinance;
  * запрещено импортировать gex.adapters (только через порты, полученные в конструкторе).

Транзакции (Unit of Work) открываются здесь и никогда не охватывают провайдерный I/O.
"""
