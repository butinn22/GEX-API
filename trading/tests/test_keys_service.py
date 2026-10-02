"""Tests for API-key management (encrypt-at-rest, CRUD, credential resolution)."""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import ApiKeyRow
from trading.application.keys_service import KeysService, mask_secret

SECRET = "test-secret-key"


@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(ApiKeyRow))  # clean slate per test
        await s.commit()
        yield s


def test_mask_secret():
    assert mask_secret("PUBKEY123456") == "PUBK…3456"
    assert mask_secret("short") == "*****"


async def test_add_and_resolve(session):
    svc = KeysService(SECRET)
    row = await svc.add_key(session, exchange="bingx", label="main",
                            api_key="PUBKEY123456", api_secret="SECRET", extra=None)
    assert "PUBKEY123456" not in row.api_key_encrypted  # encrypted at rest
    creds = await svc.resolve_credentials(session, "bingx")
    assert creds["api_key"] == "PUBKEY123456"
    assert creds["api_secret"] == "SECRET"
    assert await svc.resolve_credentials(session, "tbank") is None


async def test_add_rejects_unknown_exchange(session):
    svc = KeysService(SECRET)
    with pytest.raises(ValueError):
        await svc.add_key(session, exchange="coinbase", label="x",
                          api_key="k", api_secret="s")


async def test_list_and_delete(session):
    svc = KeysService(SECRET)
    await svc.add_key(session, exchange="tbank", label="t", api_key="tk",
                      api_secret="acct", extra={"account_id": "123"})
    rows = await svc.list_keys(session)
    assert len(rows) == 1
    assert await svc.delete_key(session, rows[0].id) is True
    assert await svc.list_keys(session) == []
    assert await svc.delete_key(session, 9999) is False


async def test_resolve_returns_extra(session):
    svc = KeysService(SECRET)
    await svc.add_key(session, exchange="tbank", label="t", api_key="token",
                      api_secret="", extra={"account_id": "acc-1", "sandbox": True})
    creds = await svc.resolve_credentials(session, "tbank")
    assert creds["extra"] == {"account_id": "acc-1", "sandbox": True}
