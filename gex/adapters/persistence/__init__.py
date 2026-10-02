"""Persistence adapters (ring: adapters) — SQLAlchemy-реализации репозиториев и Unit of Work.

Правило: сессия БД не охватывает провайдерный I/O; роутеры не содержат SQL.
"""
