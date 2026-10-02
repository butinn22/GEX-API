"""Fetcher adapters (ring: adapters) — high-level data fetchers that combine
provider calls with caching and data loading.

Each fetcher wraps a provider adapter (gex/adapters/providers/) with Redis caching
(gex/adapters/cache/) and data normalization (gex/domain/data_loader.py).
Fetchers are injected into application-layer services via constructor DI.
"""
