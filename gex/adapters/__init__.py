"""Adapters ring — реализации портов: HTTP-транспорт, провайдеры, кэш, лимиты, очередь, БД, HTTP-контроллеры.

ПРАВИЛА:
  * эти модули импортирует только bootstrap.py / gex.deps.py / main.py;
  * gex.domain и gex.application не должны импортировать gex.adapters (проверяет ast_guard.py);
  * один ресурс — один адаптер (реестр владельцев в docs/ARCHITECTURE.md).

Подпакеты: transport/, providers/, cache/, ratelimit/, queue/, persistence/, http/.
"""
