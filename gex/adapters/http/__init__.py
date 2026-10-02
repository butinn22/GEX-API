"""HTTP controllers (ring: adapters) — тонкие FastAPI-роутеры и pydantic-схемы.

Правило: только валидация, маппинг DTO и вызов use-case; никакого SQL, HTTP и бизнес-логики внутри.
"""
