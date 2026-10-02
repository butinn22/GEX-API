"""Middleware adapters (ring: adapters) — ASGI middleware and process-level concerns.

Includes HTTP metrics, visit tracking, demo-scope enforcement, system metrics
collection, GeoIP detection, and structured logging configuration.

Note: visits.py and system_metrics.py also define SQLAlchemy models registered
in gex.adapters.persistence.database.init_db().
"""
