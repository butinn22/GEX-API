"""Comprehensive auth tests: unit + integration + edge cases."""
from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

# Auth module imports
from gex.auth.config import settings
from gex.auth.models import User, SubscriptionStatus
from gex.adapters.persistence.database import init_db, SessionLocal, Base, recreate_tables
from gex.adapters.persistence.database import engine as _engine
from gex.auth.service import (
    AuthService,
    hash_password,
    check_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    is_master_admin,
    check_subscription,
    can_bypass_barriers,
)
from gex.auth.dependencies import require_verified_email, require_subscription, require_verified_and
from gex.auth.router import router, seed_master_admin
from gex.auth.schemas import RegisterIn, LoginIn, TokenOut, UserOut, VerifyEmailIn, RefreshIn

# Main app for integration tests
from main import app


# ================================================================= #
#  Helpers
# ================================================================= #
def _activate_from_db(access_token: str) -> None:
    """Тестовый шорткат: подтвердить email по user_id из access-токена.

    Self-verify по bearer удалён из API (security-аудит 2026-09-04 — верификация
    только по одноразовому токену из письма/Telegram), поэтому в тестах, где
    верификация не является предметом проверки, подтверждаем напрямую в БД.
    Настоящий флоу покрыт тестами активации (email-ссылка / telegram webhook).
    """
    payload = decode_token(access_token)
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == payload.get("sub")).first()
        if u is not None:
            u.is_email_verified = True
            db.commit()
    finally:
        db.close()


_WEBHOOK_SECRET = os.environ.get("GEX_TEST_WEBHOOK_SECRET", "test-webhook-secret")


def _webhook_headers() -> dict:
    """Заголовок подлинности для /telegram/webhook (см. security-аудит 2026-09-04)."""
    return {"X-Telegram-Bot-Api-Secret-Token": _WEBHOOK_SECRET}


# ================================================================= #
#  Fixtures
# ================================================================= #
@pytest.fixture(autouse=True)
def clean_db():
    """Clean database before each test (drop + recreate for schema changes)."""
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    yield


@pytest.fixture
def db():
    recreate_tables()
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def svc(db):
    return AuthService(db)


@pytest.fixture
def client():
    """FastAPI TestClient (integration tests)."""
    # Ensure DB is clean before each integration test
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    return TestClient(app)


# ================================================================= #
#  1. UNIT: Password hashing
# ================================================================= #
class TestPassword:
    def test_hash_and_check(self):
        pwd = "mySecretPass123!@#"
        h = hash_password(pwd)
        assert h != pwd
        assert isinstance(h, str)
        assert h.startswith("$2b$") or h.startswith("$2a$")
        assert check_password(pwd, h)

    def test_wrong_password(self):
        h = hash_password("realpass1")
        assert not check_password("wrong", h)

    def test_empty_password(self):
        h = hash_password("x")
        assert check_password("x", h)
        assert not check_password("", h)

    def test_unicode_password(self):
        pwd = "пароль🔐"
        h = hash_password(pwd)
        assert check_password(pwd, h)

    def test_long_password(self):
        pwd = "a" * 72  # bcrypt max 72 bytes
        h = hash_password(pwd)
        assert check_password(pwd, h)
        assert not check_password("a" * 71, h)


# ================================================================= #
#  2. UNIT: JWT tokens
# ================================================================= #
class TestJWT:
    def test_access_token_creation(self):
        token = create_access_token("user-abc", "test@example.com")
        payload = decode_token(token)
        assert payload["sub"] == "user-abc"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_refresh_token_creation(self):
        token = create_refresh_token("user-abc")
        payload = decode_token(token)
        assert payload["sub"] == "user-abc"
        assert payload["type"] == "refresh"
        assert "exp" in payload

    def test_access_token_expires_in_15min(self):
        token = create_access_token("u1", "a@b.com")
        payload = decode_token(token)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        diff = exp - iat
        assert timedelta(minutes=14) <= diff <= timedelta(minutes=16)

    def test_refresh_token_expires_in_7days(self):
        token = create_refresh_token("u1")
        payload = decode_token(token)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        diff = exp - iat
        assert timedelta(days=6, hours=23) <= diff <= timedelta(days=7, hours=1)

    def test_decode_invalid_token(self):
        with pytest.raises(Exception):  # JWTError
            decode_token("not-a-valid-jwt")

    def test_decode_tampered_token(self):
        token = create_access_token("u1", "a@b.com")
        with pytest.raises(Exception):
            decode_token(token + "tampered")

    def test_different_users_get_different_tokens(self):
        t1 = create_access_token("u1", "a@b.com")
        t2 = create_access_token("u2", "c@d.com")
        assert t1 != t2
        p1 = decode_token(t1)
        p2 = decode_token(t2)
        assert p1["sub"] != p2["sub"]


# ================================================================= #
#  3. UNIT: AuthService
# ================================================================= #
class TestAuthService:
    def test_register_success(self, svc):
        user = svc.register("new@test.com", "ValidPass1!", "@test1")
        assert user.email == "new@test.com"
        assert user.is_email_verified is False
        assert user.subscription_status == "INACTIVE"
        assert user.password_hash is not None
        assert user.id is not None

    def test_register_duplicate(self, svc):
        svc.register("dup@test.com", "pass1234", "@test2")
        with pytest.raises(ValueError, match="уже существует"):
            svc.register("dup@test.com", "another_pass1", "@test3")

    def test_register_master_admin_blocked(self, svc):
        with pytest.raises(ValueError, match="зарезервирован"):
            svc.register("sadisting", "anypass1", "@test4")

    def test_register_different_case(self, svc):
        svc.register("Case@Test.Com", "pass1234", "@test5")
        # Same email different case
        with pytest.raises(ValueError, match="уже существует"):
            svc.register("case@test.com", "pass1234", "@test6")

    def test_login_success(self, svc):
        svc.register("user@test.com", "pass1234", "@test7")
        user = svc.login("user@test.com", "pass1234")
        assert user.email == "user@test.com"

    def test_login_wrong_password(self, svc):
        svc.register("user@test.com", "realpass1", "@test8")
        with pytest.raises(ValueError, match="Неверный"):
            svc.login("user@test.com", "wrongpass")

    def test_login_nonexistent(self, svc):
        with pytest.raises(ValueError, match="Неверный"):
            svc.login("nobody@test.com", "pass")

    def test_login_case_insensitive(self, svc):
        svc.register("User@Test.Com", "pass1234", "@test9")
        user = svc.login("user@test.com", "pass1234")
        assert user.email == "user@test.com"

    def test_verify_email(self, svc):
        user = svc.register("verify@test.com", "pass1234", "@test10")
        assert user.is_email_verified is False
        svc.verify_email(user)
        assert user.is_email_verified is True

    def test_verify_already_verified(self, svc):
        user = svc.register("test@test.com", "pass1234", "@test11")
        svc.verify_email(user)
        svc.verify_email(user)  # повтор — не должен упасть
        assert user.is_email_verified is True

    def test_refresh_tokens_valid(self, db, svc):
        user = svc.register("test@test.com", "pass1234", "@test12")
        # Email-гейт (security-аудит 2026-09-04): refresh доступен только
        # подтверждённым пользователям.
        user.is_email_verified = True
        db.commit()
        refresh = create_refresh_token(user.id)
        new_access, new_refresh = svc.refresh_tokens(refresh)
        assert len(new_access) > 0
        assert len(new_refresh) > 0
        assert new_access != refresh
        assert new_refresh != refresh

    def test_refresh_unverified_blocked(self, db, svc):
        """Неверифицированный пользователь не может обновлять токены."""
        user = svc.register("unverified@test.com", "pass1234", "@test12b")
        refresh = create_refresh_token(user.id)
        with pytest.raises(ValueError, match="не подтверждён"):
            svc.refresh_tokens(refresh)

    def test_refresh_invalid_token(self, svc):
        with pytest.raises(ValueError, match="Недействительный"):
            svc.refresh_tokens("totally-invalid")

    def test_refresh_wrong_type(self, svc):
        access = create_access_token("user-1", "test@test.com")
        with pytest.raises(ValueError, match="Неверный тип"):
            svc.refresh_tokens(access)

    def test_get_user_profile(self, svc):
        user = svc.register("profile@test.com", "pass1234", "@test13")
        profile = svc.get_user_profile(user)
        assert profile["email"] == "profile@test.com"
        assert profile["is_email_verified"] is False
        assert profile["subscription_status"] == "INACTIVE"
        assert profile["is_master_admin"] is False


# ================================================================= #
#  4. UNIT: Master Admin
# ================================================================= #
class TestMasterAdmin:
    def test_seed_master_admin(self, db):
        seed_master_admin(db)
        user = db.query(User).filter(User.email == "sadisting").first()
        assert user is not None
        assert user.is_email_verified is True
        assert user.subscription_status == "ADMIN"
        assert user.password_hash is not None

    def test_seed_idempotent(self, db):
        seed_master_admin(db)
        seed_master_admin(db)  # повтор — не должен создать дубликат
        count = db.query(User).filter(User.email == "sadisting").count()
        assert count == 1

    def test_master_admin_login(self, db):
        from gex.auth.config import settings

        seed_master_admin(db)
        svc = AuthService(db)
        user = svc.master_admin_login(settings.MASTER_PASSWORD)
        assert user.email == "sadisting"
        assert user.subscription_status == "ADMIN"

    def test_master_admin_wrong_password(self, db):
        seed_master_admin(db)
        svc = AuthService(db)
        with pytest.raises(ValueError, match="Неверный"):
            svc.master_admin_login("wrongpass")

    def test_is_master_admin_true(self, db):
        seed_master_admin(db)
        user = db.query(User).filter(User.email == "sadisting").first()
        assert is_master_admin(user) is True

    def test_is_master_admin_false(self, svc):
        user = svc.register("normal@test.com", "pass1234", "@test14")
        assert is_master_admin(user) is False

    def test_can_bypass_true(self, db):
        seed_master_admin(db)
        user = db.query(User).filter(User.email == "sadisting").first()
        assert can_bypass_barriers(user) is True

    def test_can_bypass_false(self, svc):
        user = svc.register("normal@test.com", "pass1234", "@test15")
        assert can_bypass_barriers(user) is False


# ================================================================= #
#  5. UNIT: OAuth
# ================================================================= #
class TestOAuth:
    def test_oauth_new_user(self, svc):
        user = svc.oauth_login_or_register("google", "oauth_new@test.com", "g123")
        assert user.email == "oauth_new@test.com"
        assert user.is_email_verified is True
        assert user.oauth_provider == "google"
        assert user.oauth_id == "g123"
        assert user.password_hash is None  # OAuth user has no password

    def test_oauth_existing_user(self, svc):
        svc.register("existing@test.com", "pass1234", "@test16")
        # Existing user logins via OAuth
        user = svc.oauth_login_or_register("github", "existing@test.com", "gh456")
        assert user.oauth_provider == "github"
        assert user.oauth_id == "gh456"
        assert user.is_email_verified is True  # Was set to True

    def test_oauth_link_twice(self, svc):
        user = svc.oauth_login_or_register("google", "test@test.com", "g1")
        # Same email, different provider
        user = svc.oauth_login_or_register("github", "test@test.com", "gh1")
        assert user.oauth_provider == "github"  # Last provider wins

    def test_oauth_user_no_password(self, svc):
        user = svc.oauth_login_or_register("google", "onlyoauth@test.com", "g_abc")
        assert user.password_hash is None


# ================================================================= #
#  6. UNIT: Subscription Barriers
# ================================================================= #
class TestBarriers:
    def test_subscription_basic(self):
        class MockUser:
            subscription_status = "BASIC"
        u = MockUser()
        assert check_subscription(u, "INACTIVE") is True
        assert check_subscription(u, "BASIC") is True
        assert check_subscription(u, "EXTENDED") is False
        assert check_subscription(u, "ADMIN") is False

    def test_subscription_admin(self):
        class MockUser:
            subscription_status = "ADMIN"
        u = MockUser()
        assert check_subscription(u, "ADMIN") is True
        assert check_subscription(u, "INACTIVE") is True

    def test_subscription_inactive(self):
        class MockUser:
            subscription_status = "INACTIVE"
        u = MockUser()
        assert check_subscription(u, "INACTIVE") is True
        assert check_subscription(u, "BASIC") is False
        assert check_subscription(u, "EXTENDED") is False

    def test_unknown_subscription(self):
        class MockUser:
            subscription_status = "UNKNOWN"
        u = MockUser()
        assert check_subscription(u, "INACTIVE") is False  # Unknown treated as < INACTIVE

    def test_require_verified_dependency(self):
        assert callable(require_verified_email)

    def test_require_subscription_dependency(self):
        dep = require_subscription("BASIC")
        assert callable(dep)

    def test_require_verified_and_dependency(self):
        dep = require_verified_and("EXTENDED")
        assert callable(dep)


# ================================================================= #
#  7. INTEGRATION: API Endpoints via TestClient
# ================================================================= #
class TestAuthAPI:
    """Integration tests using FastAPI TestClient."""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_register_endpoint(self, client):
        resp = client.post("/auth/register", json={
            "email": "api_test@example.com",
            "password": "StrongPass1!", 
            "telegram_username": "@test18",
            "accept_terms": True,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_endpoint(self, client):
        client.post("/auth/register", json={
            "email": "dup@example.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test19", 
        })
        resp = client.post("/auth/register", json={
            "email": "dup@example.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test20", 
        })
        assert resp.status_code == 409

    def test_login_endpoint(self, client):
        r = client.post("/auth/register", json={
            "email": "login_test@example.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test21", 
        })
        # Подтверждаем email (логин без верификации запрещён)
        _activate_from_db(r.json()["access_token"])
        resp = client.post("/auth/login", json={
            "email": "login_test@example.com", "password": "pass1234",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert len(data["access_token"]) > 20

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={
            "email": "user@example.com", "password": "realpass1", "accept_terms": True,"telegram_username": "@test22", 
        })
        resp = client.post("/auth/login", json={
            "email": "user@example.com", "password": "wrong",
        })
        assert resp.status_code == 401

    def test_login_master_admin(self, client):
        db = SessionLocal()
        seed_master_admin(db)
        db.close()

        resp = client.post("/auth/login", json={
            "email": "sadisting", "password": settings.MASTER_PASSWORD,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    def test_get_me_authenticated(self, client):
        r = client.post("/auth/register", json={
            "email": "me@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test23", 
        })
        token = r.json()["access_token"]
        # Подтверждаем email (логин без верификации запрещён)
        _activate_from_db(token)
        login_resp = client.post("/auth/login", json={
            "email": "me@test.com", "password": "pass1234",
        })
        token = login_resp.json()["access_token"]

        resp = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "me@test.com"
        assert data["is_email_verified"] is True
        assert data["subscription_status"] == "INACTIVE"

    def test_get_me_unauthorized(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_verify_email_endpoint(self, client):
        """Верификация — только по одноразовому токену (не по bearer)."""
        r = client.post("/auth/register", json={
            "email": "verify_me@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test24", 
        })
        token = r.json()["access_token"]

        # Self-verify по bearer запрещён (security-аудит 2026-09-04)
        resp = client.get("/auth/verify-email", headers={
            "Authorization": f"Bearer {token}",
        })
        assert resp.status_code == 403

        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
        assert me["is_email_verified"] is False

        # Реальный флоу: одноразовый токен из письма
        vtoken = r.json()["verification_url"].split("token=")[1]
        resp2 = client.get(f"/auth/verify-email?token={vtoken}", follow_redirects=False)
        assert resp2.status_code in (200, 307)

        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
        assert me["is_email_verified"] is True

    def test_verify_email_without_token_rejected(self, client):
        """GET /auth/verify-email без токена отклоняется (403), даже с авторизацией."""
        r = client.post("/auth/register", json={
            "email": "no_verify@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test24b", 
        })
        resp = client.get("/auth/verify-email", headers={
            "Authorization": f"Bearer {r.json()['access_token']}",
        })
        assert resp.status_code == 403

    def test_verify_email_invalid_token(self, client):
        resp = client.get("/auth/verify-email?token=deadbeef")
        assert resp.status_code == 404

    def test_refresh_endpoint(self, client):
        r = client.post("/auth/register", json={
            "email": "refresh@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test25", 
        })
        _activate_from_db(r.json()["access_token"])
        login_resp = client.post("/auth/login", json={
            "email": "refresh@test.com", "password": "pass1234",
        })
        refresh_token = login_resp.json()["refresh_token"]

        resp = client.post("/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_with_access_token(self, client):
        r = client.post("/auth/register", json={
            "email": "refresh2@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test26", 
        })
        _activate_from_db(r.json()["access_token"])
        login_resp = client.post("/auth/login", json={
            "email": "refresh2@test.com", "password": "pass1234",
        })
        access_token = login_resp.json()["access_token"]

        resp = client.post("/auth/refresh", json={
            "refresh_token": access_token,
        })
        assert resp.status_code == 401

    def test_admin_endpoint_not_accessible(self, client):
        resp = client.get("/auth/me/admin")
        assert resp.status_code == 401

    def test_admin_endpoint_master_admin(self, client):
        db = SessionLocal()
        seed_master_admin(db)
        db.close()

        login_resp = client.post("/auth/login", json={
            "email": "sadisting", "password": settings.MASTER_PASSWORD,
        })
        token = login_resp.json()["access_token"]

        resp = client.get("/auth/me/admin", headers={
            "Authorization": f"Bearer {token}",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_master_admin"] is True
        assert "admin_stats" in data

    def test_logout_endpoint(self, client):
        resp = client.get("/auth/logout")
        assert resp.status_code == 200
        assert "Выход" in resp.json()["message"]


# ================================================================= #
#  8. INTEGRATION: Full auth flow
# ================================================================= #
class TestAuthFlow:
    """End-to-end auth flow simulation."""

    def test_full_flow(self, client):
        # 1. Register
        r1 = client.post("/auth/register", json={
            "email": "fullflow@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test27", 
        })
        assert r1.status_code == 201
        tokens = r1.json()

        # 2. Access protected route
        me = client.get("/auth/me", headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        })
        assert me.status_code == 200
        assert me.json()["is_email_verified"] is False

        # 3. Verify email — только по одноразовому токену из письма
        vtoken = tokens["verification_url"].split("token=")[1]
        v = client.get(f"/auth/verify-email?token={vtoken}", follow_redirects=False)
        assert v.status_code in (200, 307)

        # 4. Check verified status via fresh DB query
        db = SessionLocal()
        user = db.query(User).filter(User.email == "fullflow@test.com").first()
        db.close()
        assert user is not None
        assert user.is_email_verified is True

        # 5. Refresh tokens
        r_refresh = client.post("/auth/refresh", json={
            "refresh_token": tokens["refresh_token"],
        })
        assert r_refresh.status_code == 200
        new_tokens = r_refresh.json()
        assert new_tokens["access_token"] != tokens["access_token"]

    def test_barrier_email_not_verified(self, client):
        r = client.post("/auth/register", json={
            "email": "unverified@test.com", "password": "pass1234", "accept_terms": True,"telegram_username": "@test28", 
        })
        token = r.json()["access_token"]

        me = client.get("/auth/me", headers={
            "Authorization": f"Bearer {token}",
        })
        assert me.status_code == 200
        assert me.json()["is_email_verified"] is False

    def test_db_persistence(self, db):
        svc = AuthService(db)
        svc.register("persist@test.com", "pass1234", "@test17")
        db.commit()

        db2 = SessionLocal()
        user = db2.query(User).filter(User.email == "persist@test.com").first()
        assert user is not None
        assert user.email == "persist@test.com"
        assert user.telegram_username == "@test17"
        db2.close()


# ================================================================= #
#  9. INTEGRATION: Telegram-регистрация и активация
# ================================================================= #
class TestTelegramRegistration:
    """Обязательная привязка Telegram: валидация ника, активация через /start."""

    def test_register_requires_telegram(self, client):
        r = client.post("/auth/register", json={
            "email": "notelegram@test.com", "password": "pass1234", "accept_terms": True,
        })
        assert r.status_code == 422  # telegram_username обязателен

    def test_register_invalid_telegram_format(self, client):
        r = client.post("/auth/register", json={
            "email": "badnick@test.com", "password": "pass1234",
            "telegram_username": "sadisting", "accept_terms": True,  # без @
        })
        assert r.status_code == 422

        r2 = client.post("/auth/register", json={
            "email": "badnick2@test.com", "password": "pass1234",
            "telegram_username": "@ab", "accept_terms": True,  # слишком короткий
        })
        assert r2.status_code == 422

    def test_register_weak_password(self, client):
        r = client.post("/auth/register", json={
            "email": "weak@test.com", "password": "short1",
            "telegram_username": "@weakuser", "accept_terms": True,
        })
        assert r.status_code == 422  # < 8 символов

        r2 = client.post("/auth/register", json={
            "email": "weak2@test.com", "password": "aaaaaaaa",
            "telegram_username": "@weakuser2", "accept_terms": True,  # без цифр
        })
        assert r2.status_code == 422

    def test_register_duplicate_telegram(self, client):
        client.post("/auth/register", json={
            "email": "tg1@test.com", "password": "pass1234",
            "telegram_username": "@same_nick", "accept_terms": True,
        })
        r = client.post("/auth/register", json={
            "email": "tg2@test.com", "password": "pass1234",
            "telegram_username": "@SAME_NICK", "accept_terms": True,  # регистр не важен
        })
        assert r.status_code == 409

    def test_register_stores_telegram_and_returns_link(self, client):
        r = client.post("/auth/register", json={
            "email": "storetg@test.com", "password": "pass1234",
            "telegram_username": "@sadisting", "accept_terms": True,
        })
        assert r.status_code == 201
        data = r.json()
        assert "telegram_activation_url" in data
        assert data["telegram_activation_url"].startswith("https://t.me/")
        assert "start=activate_" in data["telegram_activation_url"]

        db = SessionLocal()
        user = db.query(User).filter(User.email == "storetg@test.com").first()
        db.close()
        assert user is not None
        assert user.telegram_username == "@sadisting"
        assert user.is_email_verified is False

    def test_activation_via_telegram_webhook(self, client):
        """Полный сценарий: регистрация → /start activate_<token> → активация."""
        r = client.post("/auth/register", json={
            "email": "act@test.com", "password": "pass1234",
            "telegram_username": "@sadisting", "accept_terms": True,
        })
        assert r.status_code == 201

        db = SessionLocal()
        user = db.query(User).filter(User.email == "act@test.com").first()
        token = user.verification_token
        db.close()
        assert token

        # Бот получает /start activate_<token> от чата @sadisting
        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 1,
            "message": {
                "text": f"/start activate_{token}",
                "chat": {"id": "123456789", "username": "sadisting", "type": "private"},
            },
        })
        assert resp.status_code == 200

        db = SessionLocal()
        user = db.query(User).filter(User.email == "act@test.com").first()
        db.close()
        assert user.is_email_verified is True
        assert user.telegram_chat_id == "123456789"
        assert user.telegram_username == "@sadisting"
        assert user.verification_token is None

        # Логин теперь работает
        login = client.post("/auth/login", json={
            "email": "act@test.com", "password": "pass1234",
        })
        assert login.status_code == 200

    def test_activation_invalid_token(self, client):
        r = client.post("/auth/register", json={
            "email": "badact@test.com", "password": "pass1234",
            "telegram_username": "@badact", "accept_terms": True,
        })
        assert r.status_code == 201

        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 2,
            "message": {
                "text": "/start activate_deadbeef",
                "chat": {"id": "999", "username": "badact", "type": "private"},
            },
        })
        assert resp.status_code == 200

        db = SessionLocal()
        user = db.query(User).filter(User.email == "badact@test.com").first()
        db.close()
        assert user.is_email_verified is False

    def test_activation_keeps_email_link(self, client):
        """Старый email-путь верификации продолжает работать."""
        r = client.post("/auth/register", json={
            "email": "emaily@test.com", "password": "pass1234",
            "telegram_username": "@emaily", "accept_terms": True,
        })
        assert r.status_code == 201
        data = r.json()
        assert "verification_url" in data
        assert "/auth/verify-email?token=" in data["verification_url"]

        # Верификация по email-ссылке тоже активирует аккаунт
        token = data["verification_url"].split("token=")[1]
        resp = client.get(f"/auth/verify-email?token={token}")
        assert resp.status_code in (200, 307)  # redirect на фронтенд

        db = SessionLocal()
        user = db.query(User).filter(User.email == "emaily@test.com").first()
        db.close()
        assert user.is_email_verified is True

    def test_send_telegram_activation_resend(self, client):
        r = client.post("/auth/register", json={
            "email": "resend@test.com", "password": "pass1234",
            "telegram_username": "@resenduser", "accept_terms": True,
        })
        assert r.status_code == 201
        token = r.json()["access_token"]

        resp = client.post("/auth/send-telegram-activation", headers={
            "Authorization": f"Bearer {token}",
        })
        assert resp.status_code == 200
        assert "Telegram" in resp.json()["message"]

    def test_send_telegram_activation_already_activated(self, client):
        r = client.post("/auth/register", json={
            "email": "resa@test.com", "password": "pass1234",
            "telegram_username": "@resauser", "accept_terms": True,
        })
        assert r.status_code == 201

        db = SessionLocal()
        user = db.query(User).filter(User.email == "resa@test.com").first()
        user.is_email_verified = True
        db.commit()
        db.close()

        resp = client.post("/auth/send-telegram-activation", headers={
            "Authorization": f"Bearer {r.json()['access_token']}",
        })
        assert resp.status_code == 400


# ================================================================= #
#  Смена email в личном кабинете
# ================================================================= #
class TestEmailChange:
    """PUT /auth/profile/email: смена почты с проверкой пароля + ре-верификация."""

    @staticmethod
    def _register(client, email="change@test.com", password="pass1234", tg="@changeme"):
        r = client.post("/auth/register", json={
            "email": email, "password": password,
            "telegram_username": tg, "accept_terms": True,
        })
        assert r.status_code == 201
        return r.json()["access_token"]

    def test_email_change_success(self, client, monkeypatch):
        # Не дёргаем реальный Telegram-бот при отправке ссылки активации
        monkeypatch.setattr(
            "gex.auth.router._send_telegram_activation",
            lambda user, link, db=None: False,
        )
        token = self._register(client)

        resp = client.put("/auth/profile/email", json={
            "email": "newaddr@test.com", "password": "pass1234",
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user"]["email"] == "newaddr@test.com"
        assert data["user"]["is_email_verified"] is False
        assert data["access_token"] and data["refresh_token"]

        # Новые токены работают и возвращают новый email
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {data['access_token']}"})
        assert me.status_code == 200
        assert me.json()["email"] == "newaddr@test.com"

        db = SessionLocal()
        user = db.query(User).filter(User.email == "newaddr@test.com").first()
        db.close()
        assert user is not None
        assert user.is_email_verified is False
        assert user.verification_token is not None  # ре-верификация

    def test_email_change_wrong_password(self, client, monkeypatch):
        monkeypatch.setattr(
            "gex.auth.router._send_telegram_activation",
            lambda user, link, db=None: False,
        )
        token = self._register(client)
        resp = client.put("/auth/profile/email", json={
            "email": "newaddr@test.com", "password": "wrongpass1",
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

        # Email не изменился
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.json()["email"] == "change@test.com"

    def test_email_change_duplicate(self, client, monkeypatch):
        monkeypatch.setattr(
            "gex.auth.router._send_telegram_activation",
            lambda user, link, db=None: False,
        )
        self._register(client, email="first@test.com", tg="@firstuser")
        token2 = self._register(client, email="second@test.com", tg="@seconduser")

        resp = client.put("/auth/profile/email", json={
            "email": "first@test.com", "password": "pass1234",
        }, headers={"Authorization": f"Bearer {token2}"})
        assert resp.status_code == 409

    def test_email_change_invalid_email(self, client):
        token = self._register(client, email="badchange@test.com", tg="@badchange")
        resp = client.put("/auth/profile/email", json={
            "email": "not-an-email", "password": "pass1234",
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 422

    def test_email_change_master_email_blocked(self, client):
        # MASTER_EMAIL ("sadisting") не является email — валидация формата отклоняет
        token = self._register(client, email="mastertry@test.com", tg="@mastertry")
        resp = client.put("/auth/profile/email", json={
            "email": settings.MASTER_EMAIL, "password": "pass1234",
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 422

    def test_email_change_requires_auth(self, client):
        resp = client.put("/auth/profile/email", json={
            "email": "nobody@test.com", "password": "pass1234",
        })
        assert resp.status_code == 401


# ================================================================= #
#  Telegram в личном кабинете: ник, уведомления, проверка связи
# ================================================================= #
class TestTelegramProfile:
    """Ник @username, deep-link /start, статус, тест и персональные уведомления."""

    @staticmethod
    def _register(client, email="tgprof@test.com", password="pass1234", tg="@tgprof"):
        r = client.post("/auth/register", json={
            "email": email, "password": password,
            "telegram_username": tg, "accept_terms": True,
        })
        assert r.status_code == 201
        return r.json()["access_token"]

    @staticmethod
    def _auth(token):
        return {"Authorization": f"Bearer {token}"}

    def test_nickname_save_and_status(self, client):
        token = self._register(client)
        resp = client.put("/auth/telegram/nickname", json={
            "telegram_username": "@new_nick_123",
        }, headers=self._auth(token))
        assert resp.status_code == 200
        assert resp.json()["username"] == "@new_nick_123"

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_username == "@new_nick_123"

        st = client.get("/auth/telegram/status", headers=self._auth(token))
        assert st.status_code == 200
        assert st.json()["connected"] is False
        assert st.json()["username"] == "@new_nick_123"

    def test_nickname_invalid_format(self, client):
        token = self._register(client)
        for bad in ("nickwithoutat", "@ab", "@with space!", ""):
            resp = client.put("/auth/telegram/nickname", json={
                "telegram_username": bad,
            }, headers=self._auth(token))
            assert resp.status_code == 422, f"expected 422 for {bad!r}"

    def test_nickname_duplicate(self, client):
        token1 = self._register(client, email="u1@test.com", tg="@taken_nick")
        token2 = self._register(client, email="u2@test.com", tg="@other_nick")

        # u2 пытается занять ник u1
        resp = client.put("/auth/telegram/nickname", json={
            "telegram_username": "@TAKEN_NICK",  # регистр не важен
        }, headers=self._auth(token2))
        assert resp.status_code == 409

        # u1 может сохранить свой же ник повторно
        resp = client.put("/auth/telegram/nickname", json={
            "telegram_username": "@taken_nick",
        }, headers=self._auth(token1))
        assert resp.status_code == 200

    def test_notify_toggle(self, client):
        token = self._register(client)
        on = client.put("/auth/telegram/notify", json={"notify": True}, headers=self._auth(token))
        assert on.status_code == 200
        assert on.json()["notify"] is True

        off = client.put("/auth/telegram/notify", json={"notify": False}, headers=self._auth(token))
        assert off.json()["notify"] is False

    def test_connect_link_generation(self, client):
        token = self._register(client)
        resp = client.post("/auth/telegram/connect", headers=self._auth(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False
        assert data["connect_url"].startswith("https://t.me/")
        assert "?start=" in data["connect_url"]
        assert data["expires_in_minutes"] == 15

        # Токен сохранён в БД
        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_connect_token
        assert user.telegram_connect_token in data["connect_url"]

    def test_connect_binds_matching_nickname(self, client, monkeypatch):
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)
        token = self._register(client, tg="@mynick")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@MyNick"}, headers=self._auth(token))

        client.post("/auth/telegram/connect", headers=self._auth(token))
        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()

        # Бот получает /start <token> от чата с тем же ником (регистр не важен)
        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 1,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "777", "username": "mynick", "type": "private"}},
        })
        assert resp.status_code == 200

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_chat_id == "777"
        assert user.telegram_username == "@mynick"
        assert user.telegram_connect_token is None

        st = client.get("/auth/telegram/status", headers=self._auth(token))
        assert st.json()["connected"] is True

    def test_connect_rejects_nickname_mismatch(self, client, monkeypatch):
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)
        token = self._register(client, tg="@myrealnick")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@myrealnick"}, headers=self._auth(token))
        client.post("/auth/telegram/connect", headers=self._auth(token))

        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()

        # Пришёл чат с ДРУГИМ ником — привязка запрещена
        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 2,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "888", "username": "somebody_else", "type": "private"}},
        })
        assert resp.status_code == 200

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_chat_id is None
        assert user.telegram_connect_token == connect_token  # токен не сожжён

    def test_connect_binds_chat_without_username(self, client, monkeypatch):
        """Чат без публичного ника: привязка по токену (токен — доказательство)."""
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)
        token = self._register(client, tg="@privacyuser")
        client.post("/auth/telegram/connect", headers=self._auth(token))

        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()

        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 3,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "999", "first_name": "NoName", "type": "private"}},
        })
        assert resp.status_code == 200

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_chat_id == "999"

    def test_connect_already_connected(self, client, monkeypatch):
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)
        token = self._register(client, tg="@already")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@already"}, headers=self._auth(token))
        client.post("/auth/telegram/connect", headers=self._auth(token))

        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()
        client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 4,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "111", "username": "already", "type": "private"}},
        })

        # Повторный connect: уже подключён — ссылка не нужна
        resp = client.post("/auth/telegram/connect", headers=self._auth(token))
        assert resp.json()["connected"] is True

    def test_test_message_requires_connection(self, client):
        token = self._register(client)
        resp = client.post("/auth/telegram/test", headers=self._auth(token))
        assert resp.status_code == 400

    def test_test_message_success(self, client, monkeypatch):
        sent = {}

        def fake_send(text, *, parse_mode=None, chat_id=None):
            sent["text"] = text
            sent["chat_id"] = chat_id
            return {"success": True, "batches_sent": 1, "errors": None, "raw_responses": [{"ok": True}]}

        monkeypatch.setattr("gex.adapters.notifications.telegram_sender.send_telegram_message", fake_send)
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)

        token = self._register(client, tg="@testuser")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@testuser"}, headers=self._auth(token))
        client.post("/auth/telegram/connect", headers=self._auth(token))
        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()
        client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 5,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "222", "username": "testuser", "type": "private"}},
        })

        resp = client.post("/auth/telegram/test", headers=self._auth(token))
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert sent.get("chat_id") == "222"
        assert "Связь работает" in sent.get("text", "")

        # Время последней проверки зафиксировано
        st = client.get("/auth/telegram/status", headers=self._auth(token))
        assert st.json()["test_at"] is not None

    def test_test_message_failure(self, client, monkeypatch):
        def fake_send(text, *, parse_mode=None, chat_id=None):
            return {"success": False, "batches_sent": 0, "errors": ["TELEGRAM_BOT_TOKEN not configured"], "raw_responses": []}

        monkeypatch.setattr("gex.adapters.notifications.telegram_sender.send_telegram_message", fake_send)
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)

        token = self._register(client, tg="@failuser")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@failuser"}, headers=self._auth(token))
        client.post("/auth/telegram/connect", headers=self._auth(token))
        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()
        client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 6,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "333", "username": "failuser", "type": "private"}},
        })

        resp = client.post("/auth/telegram/test", headers=self._auth(token))
        assert resp.status_code == 200
        assert resp.json()["ok"] is False
        st = client.get("/auth/telegram/status", headers=self._auth(token))
        assert st.json()["test_at"] is None

    def test_webhook_test_command_updates_status(self, client, monkeypatch):
        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        monkeypatch.setattr("gex.adapters.notifications.telegram_sender.send_telegram_message",
                            lambda text, *, parse_mode=None, chat_id=None: {"success": True, "batches_sent": 1, "errors": None, "raw_responses": []})

        token = self._register(client, tg="@webtest")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@webtest"}, headers=self._auth(token))
        client.post("/auth/telegram/connect", headers=self._auth(token))
        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()
        client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 7,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "444", "username": "webtest", "type": "private"}},
        })

        # Пользователь пишет боту /test
        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 8,
            "message": {"text": "/test",
                        "chat": {"id": "444", "username": "webtest", "type": "private"}},
        })
        assert resp.status_code == 200
        assert len(replies) == 2  # /start connect + /test
        assert "Связь работает" in replies[1][1]

        st = client.get("/auth/telegram/status", headers=self._auth(token))
        assert st.json()["test_at"] is not None

    def test_disconnect_clears_state(self, client, monkeypatch):
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)
        monkeypatch.setattr("gex.adapters.notifications.telegram_sender.send_telegram_message",
                            lambda text, *, parse_mode=None, chat_id=None: {"success": True, "batches_sent": 1, "errors": None, "raw_responses": []})

        token = self._register(client, tg="@discuser")
        client.put("/auth/telegram/nickname", json={"telegram_username": "@discuser"}, headers=self._auth(token))
        client.post("/auth/telegram/connect", headers=self._auth(token))
        client.put("/auth/telegram/notify", json={"notify": True}, headers=self._auth(token))
        db = SessionLocal()
        connect_token = db.query(User).filter(User.email == "tgprof@test.com").first().telegram_connect_token
        db.close()
        client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 9,
            "message": {"text": f"/start {connect_token}",
                        "chat": {"id": "555", "username": "discuser", "type": "private"}},
        })
        client.post("/auth/telegram/test", headers=self._auth(token))

        resp = client.post("/auth/telegram/disconnect", headers=self._auth(token))
        assert resp.status_code == 200
        st = client.get("/auth/telegram/status", headers=self._auth(token))
        assert st.json()["connected"] is False
        assert st.json()["notify"] is False
        assert st.json()["test_at"] is None

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_chat_id is None
        assert user.telegram_username is None

    # ── /start боту: сервис узнаёт о пользователе и подтверждает регистрацию ──

    def test_webhook_start_no_token_binds_and_sends_activation(self, client, monkeypatch):
        """Пользователь нажал /start (без токена) → сервис привязал чат и
        отправил ссылку активации, подтвердив регистрацию."""
        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        token = self._register(client, tg="@startuser")

        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 1,
            "message": {"text": "/start",
                        "chat": {"id": "111222", "username": "startuser", "type": "private"}},
        })
        assert resp.status_code == 200

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_chat_id == "111222"  # сервис подтянул /start

        assert len(replies) == 1
        reply = replies[0][1]
        assert "Мы получили ваш /start" in reply
        assert "activate_" in reply  # ссылка активации в ответе бота
        assert user.verification_token in reply

        # Логин всё ещё запрещён (email не подтверждён)
        login = client.post("/auth/login", json={
            "email": "tgprof@test.com", "password": "pass1234",
        })
        assert login.status_code == 403

    def test_webhook_start_no_token_active_user(self, client, monkeypatch):
        """Активированный пользователь нажал /start → чат привязан + ответ бота."""
        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        token = self._register(client, tg="@activeuser")
        # Активируем аккаунт
        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        user.is_email_verified = True
        db.commit()
        db.close()

        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 2,
            "message": {"text": "/start",
                        "chat": {"id": "333444", "username": "activeuser", "type": "private"}},
        })
        assert resp.status_code == 200
        assert "активирован" in replies[0][1]

        db = SessionLocal()
        user = db.query(User).filter(User.email == "tgprof@test.com").first()
        db.close()
        assert user.telegram_chat_id == "333444"

    def test_webhook_start_records_start_for_unknown_username(self, client, monkeypatch):
        """/start от незарегистрированного ника → welcome, запись в telegram_starts."""
        from gex.auth.models import TelegramStart
        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 3,
            "message": {"text": "/start",
                        "chat": {"id": "555666", "username": "stranger_nick", "type": "private"}},
        })
        assert resp.status_code == 200
        assert "Добро пожаловать" in replies[0][1]

        db = SessionLocal()
        row = db.query(TelegramStart).filter(TelegramStart.username == "stranger_nick").first()
        db.close()
        assert row is not None
        assert row.chat_id == "555666"

    def test_register_uses_known_chat_id_from_start(self, client, monkeypatch):
        """Пользователь нажал /start ДО регистрации → сервис знает chat_id по нику
        и шлёт активацию прямо в чат + привязывает его к аккаунту."""
        from gex.auth.models import TelegramStart
        sent = {}

        def fake_send(text, *, parse_mode=None, chat_id=None):
            sent["chat_id"] = chat_id
            sent["text"] = text
            return {"success": True, "batches_sent": 1, "errors": None, "raw_responses": [{"ok": True}]}

        # router.py использует alias _send_tg (импортирован на старте модуля)
        monkeypatch.setattr("gex.auth.router._send_tg", fake_send)
        monkeypatch.setattr("gex.auth.telegram_router._send_telegram_reply", lambda chat_id, text: None)

        # Сначала stranger нажимает /start
        client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 4,
            "message": {"text": "/start",
                        "chat": {"id": "777888", "username": "earlybird", "type": "private"}},
        })

        # Теперь регистрируется с этим ником
        r = client.post("/auth/register", json={
            "email": "early@test.com", "password": "pass1234",
            "telegram_username": "@earlybird", "accept_terms": True,
        })
        assert r.status_code == 201
        assert sent.get("chat_id") == "777888"  # активация ушла в чат /start

        db = SessionLocal()
        user = db.query(User).filter(User.email == "early@test.com").first()
        db.close()
        assert user.telegram_chat_id == "777888"

    def test_webhook_start_activation_confirms_registration(self, client, monkeypatch):
        """Полный сценарий: регистрация → /start activate_<token> → ответ бота
        подтверждает регистрацию."""
        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        r = client.post("/auth/register", json={
            "email": "confirm@test.com", "password": "pass1234",
            "telegram_username": "@confirmuser", "accept_terms": True,
        })
        assert r.status_code == 201

        db = SessionLocal()
        user = db.query(User).filter(User.email == "confirm@test.com").first()
        db.close()
        resp = client.post("/telegram/webhook", headers=_webhook_headers(), json={
            "update_id": 5,
            "message": {"text": f"/start activate_{user.verification_token}",
                        "chat": {"id": "999000", "username": "confirmuser", "type": "private"}},
        })
        assert resp.status_code == 200
        assert len(replies) == 1
        assert "Регистрация подтверждена" in replies[0][1]
        assert "confirm@test.com" in replies[0][1]
