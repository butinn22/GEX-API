"""Payment requisites: админка GET/PUT /auth/admin/payment-requisites,
публичные methods в /payment/plans, создание платежей BANK/CRYPTO/SBP."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from gex.auth.config import settings
from gex.auth.models import User
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.auth.router import seed_master_admin

from main import app

FIELDS = (
    "sbp_phone", "sbp_bank", "sbp_name",
    "crypto_usdt_trc20", "crypto_usdt_bep20",
    "bank_name", "bank_bic", "bank_account", "bank_recipient",
)


@pytest.fixture(autouse=True)
def clean_db():
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    yield


@pytest.fixture
def client():
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    return TestClient(app)


def _admin_token(client) -> str:
    db = SessionLocal()
    seed_master_admin(db)
    db.close()
    r = client.post("/auth/login", json={"email": "sadisting", "password": settings.MASTER_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _register_verified(client, email="pay@test.com") -> dict:
    r = client.post("/auth/register", json={
        "email": email, "password": "pass1234",
        "telegram_username": "@paytest", "accept_terms": True,
    })
    assert r.status_code == 201, r.text
    db = SessionLocal()
    u = db.query(User).filter(User.email == email).first()
    u.is_email_verified = True
    db.commit()
    db.close()
    return r.json()


def _auth(client, tok: str):
    return {"Authorization": f"Bearer {tok}"}


def _put(client, tok: str, values: dict) -> dict:
    body = {f: values.get(f, "") for f in FIELDS}
    r = client.put("/auth/admin/payment-requisites", json=body, headers=_auth(client, tok))
    assert r.status_code == 200, r.text
    return r.json()


def _method_ids(data: dict) -> list[str]:
    return [m["id"] for m in data.get("methods", [])]


def _crypto(data: dict) -> dict | None:
    for m in data.get("methods", []):
        if m["id"] == "CRYPTO":
            return m
    return None


class TestAdminRequisites:
    def test_get_requires_admin(self, client):
        reg = _register_verified(client, "user@test.com")
        r = client.get("/auth/admin/payment-requisites", headers=_auth(client, reg["access_token"]))
        assert r.status_code == 403

    def test_put_and_get_echo(self, client):
        tok = _admin_token(client)
        up = _put(client, tok, {
            "sbp_phone": "+79990001122", "sbp_bank": "Т-Банк", "sbp_name": "Иван",
            "crypto_usdt_trc20": "T" * 30, "crypto_usdt_bep20": "0x" + "b" * 40,
            "bank_name": "Альфа-Банк", "bank_bic": "044525593",
            "bank_account": "40817810099910004321", "bank_recipient": "Иванов Иван",
        })
        assert up["sbp_phone"] == "+79990001122"
        assert up["bank_account"] == "40817810099910004321"
        # эхо через GET
        r = client.get("/auth/admin/payment-requisites", headers=_auth(client, tok))
        assert r.status_code == 200
        assert r.json()["crypto_usdt_bep20"] == "0x" + "b" * 40

    def test_placeholder_values_hidden(self, client):
        """Значения с '*' (плейсхолдеры) не включают способ оплаты."""
        tok = _admin_token(client)
        up = _put(client, tok, {"sbp_phone": "+791****1132"})
        assert "SBP" not in _method_ids(up)
        up = _put(client, tok, {"sbp_phone": "+79990001122"})
        assert "SBP" in _method_ids(up)

    def test_methods_availability_rules(self, client):
        tok = _admin_token(client)
        # всё пусто
        up = _put(client, tok, {})
        assert _method_ids(up) == []
        # крипта только с адресом; банк только с банком+счётом
        up = _put(client, tok, {"crypto_usdt_trc20": "T" * 30})
        cr = _crypto(up)
        assert cr is not None and cr["networks"] == ["USDT_TRC20"]
        assert _method_ids(up) == ["CRYPTO"]
        up = _put(client, tok, {"bank_account": "40817810099910004321"})
        assert "BANK" not in _method_ids(up)
        up = _put(client, tok, {"bank_account": "40817810099910004321", "bank_name": "Сбербанк"})
        assert "BANK" in _method_ids(up)


class TestPlansPublic:
    def test_plans_show_configured_methods(self, client):
        r = client.get("/payment/plans")
        assert r.status_code == 200
        assert isinstance(r.json().get("methods"), list)

    def test_plans_reflect_admin_put(self, client):
        tok = _admin_token(client)
        _put(client, tok, {"sbp_phone": "+79990001122", "bank_account": "40817810099910004321", "bank_name": "Т-Банк"})
        d = client.get("/payment/plans").json()
        ids = [m["id"] for m in d["methods"]]
        assert ids == ["SBP", "BANK"]


class TestInitPayments:
    def _full_reqs(self, client, tok):
        return _put(client, tok, {
            "sbp_phone": "+79990001122", "sbp_bank": "Т-Банк", "sbp_name": "Иван",
            "crypto_usdt_trc20": "T" * 30,
            "bank_name": "Альфа-Банк", "bank_bic": "044525593",
            "bank_account": "40817810099910004321", "bank_recipient": "Иванов Иван",
        })

    def test_init_bank(self, client):
        tok = _admin_token(client)
        self._full_reqs(client, tok)
        reg = _register_verified(client)
        r = client.post("/payment/init", json={
            "plan": "BASIC", "method": "BANK", "accept_terms": True,
        }, headers=_auth(client, reg["access_token"]))
        assert r.status_code == 201, r.text
        p = r.json()
        assert p["method"] == "BANK"
        assert p["bank_account"] == "40817810099910004321"
        assert p["bank_bic"] == "044525593"
        assert p["crypto_address"] is None

    def test_init_crypto_network_rules(self, client):
        tok = _admin_token(client)
        self._full_reqs(client, tok)
        reg = _register_verified(client, "cr@test.com")
        # BEP20 не настроен → 400
        r = client.post("/payment/init", json={
            "plan": "EXTENDED", "method": "CRYPTO",
            "crypto_currency": "USDT_BEP20", "accept_terms": True,
        }, headers=_auth(client, reg["access_token"]))
        assert r.status_code == 400
        assert "USDT_TRC20" in r.json()["detail"]
        # дефолтная сеть = единственная настроенная
        r = client.post("/payment/init", json={
            "plan": "EXTENDED", "method": "CRYPTO", "accept_terms": True,
        }, headers=_auth(client, reg["access_token"]))
        assert r.status_code == 201
        assert r.json()["crypto_currency"] == "USDT_TRC20"
        assert r.json()["crypto_address"]

    def test_init_disabled_method_400(self, client):
        tok = _admin_token(client)
        _put(client, tok, {})  # всё очищено
        reg = _register_verified(client, "no@test.com")
        r = client.post("/payment/init", json={
            "plan": "BASIC", "method": "SBP", "accept_terms": True,
        }, headers=_auth(client, reg["access_token"]))
        assert r.status_code == 400
        assert "СБП не настроено" in r.json()["detail"]
