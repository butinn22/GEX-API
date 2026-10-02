"""Ports ring — Protocol-интерфейсы, которыми домен/application описывают внешний мир.

ПРАВИЛА:
  * только typing.Protocol + dataclass-DTO; ни одной реализации и ни одного импорта инфраструктуры;
  * реализация порта живёт в gex.adapters.** (владелец ресурса), сборка — в bootstrap.py.

Состав (по итерациям): market_data, option_chain, cache, rate_limit, job_queue, notifier, llm.
"""
