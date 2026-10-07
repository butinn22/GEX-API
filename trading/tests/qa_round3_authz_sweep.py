"""QA Round-3 — INDEPENDENT dynamic anonymous-mutation sweep.

Unlike ``test_authz_regression.py`` (a hand-written param list), this enumerates
**every** HTTP route from the live ``app.openapi()`` spec and probes each one
anonymously, so a route added without the mount guard fails loudly even if the
hand-written list is not updated.

Run explicitly (NOT collected by ``pytest trading/tests`` because the filename
does not match ``test_*.py`` — this keeps the frozen suite count stable)::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_authz_sweep.py -q -p no:randomly
"""
from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from trading.adapters.persistence import database as db
from trading.main import app

pytestmark = pytest.mark.real_auth

# Route (method, path) pairs that are DELIBERATELY public (must NOT be 401).
PUBLIC_EXCEPTIONS = {
    ("POST", "/api/v1/auth/token"),            # login
    ("POST", "/api/v1/backtest/cancel/{token}"),  # sendBeacon capability token
    ("GET", "/health"),                        # liveness
    ("GET", "/metrics"),                       # token-gated separately (dev: open)
}

# Concrete values substituted for path params during the probe.
_PARAM_VALUES = {
    "key_id": "999999",
    "preset_id": "999999",
    "result_id": "999999",
    "order_id": "999999",
    "run_id": "999999",
    "symbol": "BTC",
    "name": "sma_crossover",
    "key": "deadbeefdeadbeef",
    "token": "some-run-token",
}


def _all_routes() -> list[tuple[str, str]]:
    spec = app.openapi()
    out: list[tuple[str, str]] = []
    for path, ops in spec["paths"].items():
        for method in ops:
            if method.lower() in ("head", "options", "parameters"):
                continue
            out.append((method.upper(), path))
    return sorted(set(out), key=lambda x: (x[1], x[0]))


def _fill(path: str) -> str:
    out = path
    for name, val in _PARAM_VALUES.items():
        out = out.replace("{" + name + "}", val)
    return out


BUSINESS_ROUTES = [
    (m, p) for (m, p) in _all_routes() if (m, p) not in PUBLIC_EXCEPTIONS
]


@pytest_asyncio.fixture
async def client():
    await db.init_db()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.parametrize(
    "method,path", BUSINESS_ROUTES, ids=[f"{m} {p}" for m, p in BUSINESS_ROUTES]
)
async def test_every_business_route_rejects_anonymous(client, method, path):
    url = _fill(path)
    kwargs = {"json": {}} if method in ("POST", "PATCH", "PUT") else {}
    r = await client.request(method, url, **kwargs)
    assert r.status_code in (401, 403), (
        f"{method} {path} -> {r.status_code}: reachable without credentials; "
        "must depend on require_auth"
    )
    if r.status_code == 401:
        # anonymous (no creds) must be exactly "Not authenticated"
        assert r.json().get("detail") == "Not authenticated", r.text


@pytest.mark.parametrize(
    "method,path", sorted(PUBLIC_EXCEPTIONS), ids=[f"{m} {p}" for m, p in sorted(PUBLIC_EXCEPTIONS)]
)
async def test_public_exceptions_stay_public(client, method, path):
    url = _fill(path)
    kwargs = {"json": {}} if method in ("POST", "PATCH", "PUT") else {}
    r = await client.request(method, url, **kwargs)
    assert r.status_code != 401, (
        f"{method} {path} unexpectedly returned 401 — a deliberate public route "
        "regressed"
    )


def test_route_inventory_printed(capsys):
    """Emit the full inventory so the report can cite it."""
    print(f"\nTOTAL HTTP ROUTES ENUMERATED: {len(_all_routes())}")
    print(f"BUSINESS (must-401) ROUTES: {len(BUSINESS_ROUTES)}")
    print(f"PUBLIC EXCEPTIONS: {len(PUBLIC_EXCEPTIONS)}")
    assert len(BUSINESS_ROUTES) >= 60, "route enumeration looks incomplete"
