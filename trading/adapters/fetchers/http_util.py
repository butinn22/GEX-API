"""Shared async HTTP helpers for fetchers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from trading.domain import DataFetchError
from trading.observability import DATA_FETCH_ERRORS

__all__ = ["get_json"]


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 15.0,
) -> Any:
    resp = await client.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        # The "source" label is the host, so cardinality stays bounded while an
        # operator can still tell which upstream is failing.
        DATA_FETCH_ERRORS.labels(source=httpx.URL(url).host or "unknown").inc()
        raise DataFetchError(f"{url} -> HTTP {resp.status_code}")
    return resp.json()
