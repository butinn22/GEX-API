"""Shared async HTTP helpers for fetchers."""
from __future__ import annotations

from typing import Any, Mapping

import httpx

from trading.domain import DataFetchError

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
        raise DataFetchError(f"{url} -> HTTP {resp.status_code}")
    return resp.json()
