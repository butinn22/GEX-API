"""Security-hardening regression tests (Round 2).

Covers the WS-2 controls: production fail-fast on default secrets, the split
between the JWT signing key and the Fernet at-rest key, and broker-credential
validation (BingX needs a secret, TBANK needs an account id).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import delete

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import ApiKeyRow
from trading.application.keys_service import KeysService
from trading.config import Settings

SECRET = "test-secret-key"


# ── production fail-fast ──────────────────────────────────────────────
def test_production_rejects_default_secret():
    with pytest.raises(RuntimeError):
        Settings(env="production", secret_key="dev-secret-change-me",
                 admin_username="admin", admin_password="admin")


def test_production_rejects_default_admin_creds():
    with pytest.raises(RuntimeError):
        Settings(env="production", secret_key="a-strong-key",
                 admin_username="admin", admin_password="admin")


def test_production_accepts_hardened_config():
    s = Settings(env="production", secret_key="a-strong-key",
                 admin_username="ops", admin_password="s3cr3t!")
    assert s.is_production is True


def test_dev_env_never_fails_fast():
    s = Settings(env="dev")
    assert s.is_production is False


# ── JWT vs Fernet key split ───────────────────────────────────────────
def test_encryption_secret_falls_back_to_jwt_key():
    assert Settings(secret_key="jwt", broker_key_secret="").encryption_secret == "jwt"


def test_encryption_secret_prefers_dedicated_key():
    s = Settings(secret_key="jwt", broker_key_secret="fernet-key")
    assert s.encryption_secret == "fernet-key"


# ── broker-credential validation ──────────────────────────────────────
@pytest_asyncio.fixture
async def session():
    await db.init_db()
    async with db._session_factory() as s:
        await s.execute(delete(ApiKeyRow))
        await s.commit()
        yield s


async def test_bingx_requires_secret(session):
    svc = KeysService(SECRET)
    with pytest.raises(ValueError):
        await svc.add_key(session, exchange="bingx", label="x",
                          api_key="PUB", api_secret="")


async def test_tbank_requires_account_id(session):
    svc = KeysService(SECRET)
    with pytest.raises(ValueError):
        await svc.add_key(session, exchange="tbank", label="x",
                          api_key="tok", api_secret="", extra={})


async def test_tbank_accepts_blank_secret_with_account_id(session):
    svc = KeysService(SECRET)
    row = await svc.add_key(session, exchange="tbank", label="x",
                            api_key="tok", api_secret="",
                            extra={"account_id": "acc-1"})
    assert row.id is not None
