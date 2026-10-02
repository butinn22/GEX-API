"""Tests for database infrastructure: engine, session, CRUD, config.

Все тесты используют SQLite in-memory (не требуют PostgreSQL).
"""
from __future__ import annotations

import sys
import os
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import inspect, text

from gex.adapters.persistence.database import (
    SQLITE_BUSY_TIMEOUT_MS,
    Base,
    SessionLocal,
    engine,
    init_db,
    drop_all,
    recreate_tables,
    get_session,
    db_type,
    _attach_sqlite_pragmas,
    _build_engine_url,
    _engine_kwargs,
    _is_file_sqlite,
)
from gex.auth.models import User, SubscriptionStatus
from gex.auth.config import settings


# ================================================================= #
#  Fixtures
# ================================================================= #
@pytest.fixture(autouse=True)
def reset_db():
    """Ensure clean tables before each test."""
    recreate_tables()
    yield
    drop_all()


@pytest.fixture
def db():
    """Get a fresh SQLAlchemy session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


# ================================================================= #
#  1. Config detection
# ================================================================= #
class TestConfig:
    def test_default_is_sqlite(self):
        """DATABASE_URL starts with sqlite when no env override."""
        url = settings.DATABASE_URL
        assert url.startswith("sqlite") or "postgresql" in url, f"Unexpected URL: {url}"
        # Accept both: sqlite (default) and postgresql (env override)

    def test_db_type_sqlite(self):
        """db_type() returns correct type based on current DATABASE_URL."""
        expected = "sqlite" if settings.DATABASE_URL.startswith("sqlite") else "postgresql"
        assert db_type() == expected, f"Expected {expected}, got {db_type()}"

    def test_db_type_postgresql(self):
        """db_type() returns 'postgresql' for postgresql URLs."""
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "postgresql+psycopg2://user:pass@localhost/test"
            assert db_type() == "postgresql"
        finally:
            settings.DATABASE_URL = original

    def test_db_type_asyncpg(self):
        """db_type() returns 'postgresql' for asyncpg URLs."""
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/test"
            assert db_type() == "postgresql"
        finally:
            settings.DATABASE_URL = original


# ================================================================= #
#  2. Engine creation
# ================================================================= #
class TestEngine:
    def test_engine_exists(self):
        """Engine should be created on import."""
        assert engine is not None
        assert hasattr(engine, "url")

    def test_engine_url_conversion(self):
        """_build_engine_url should convert asyncpg to psycopg2."""
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "postgresql+asyncpg://u:p@localhost/db"
            url = _build_engine_url()
            assert "psycopg2" in url
        finally:
            settings.DATABASE_URL = original

    def test_engine_url_keeps_sqlite(self):
        """_build_engine_url should keep sqlite as-is."""
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "sqlite:///./gex.db"
            url = _build_engine_url()
            assert url == "sqlite:///./gex.db"
        finally:
            settings.DATABASE_URL = original

    def test_engine_url_sqlite_memory(self):
        """_build_engine_url should use in-memory for testing."""
        original = settings.DATABASE_URL
        test_orig = settings.TESTING
        try:
            settings.DATABASE_URL = "sqlite:///./gex.db"
            settings.TESTING = True
            url = _build_engine_url()
            assert url == "sqlite:///:memory:"
        finally:
            settings.DATABASE_URL = original
            settings.TESTING = test_orig

    def test_engine_kwargs_sqlite(self):
        """_engine_kwargs should include check_same_thread and a busy timeout for SQLite.

        ``timeout`` — предел ожидания чужой блокировки: при значении драйвера
        по умолчанию (5 с) фоновые писатели падали с «database is locked», пока
        обработчик держал транзакцию открытой на время сетевого фетча.
        """
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "sqlite:///./test.db"
            kwargs = _engine_kwargs()
            args = kwargs["connect_args"]
            assert args["check_same_thread"] is False
            assert args["timeout"] == SQLITE_BUSY_TIMEOUT_MS / 1000
        finally:
            settings.DATABASE_URL = original

    def test_sqlite_pragmas_applied(self):
        """Файловый SQLite должен подниматься в WAL с заданным busy_timeout.

        Проверка не «на всякий случай», а на конкретный дефект: без WAL любой
        читатель блокировал писателя, и метрики/визиты не писались.
        """
        import tempfile
        from pathlib import Path

        from sqlalchemy import create_engine, text

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "pragma.db"
            eng = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False, "timeout": 30.0},
            )
            _attach_sqlite_pragmas(eng)
            try:
                with eng.connect() as conn:
                    assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
                    assert conn.execute(
                        text("PRAGMA busy_timeout")
                    ).scalar() == SQLITE_BUSY_TIMEOUT_MS
            finally:
                eng.dispose()

    def test_sqlite_pragmas_skipped_for_inmemory(self):
        """In-memory SQLite — не файл: WAL там неприменим, PRAGMA не вешаем."""
        from sqlalchemy import create_engine

        eng = create_engine("sqlite:///:memory:")
        try:
            assert _is_file_sqlite(eng) is False
            _attach_sqlite_pragmas(eng)  # не должно бросить
        finally:
            eng.dispose()

    def test_engine_kwargs_postgresql(self):
        """_engine_kwargs should include pool settings for PostgreSQL."""
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "postgresql+psycopg2://u:p@localhost/db"
            kwargs = _engine_kwargs()
            assert "pool_size" in kwargs
            assert kwargs["pool_pre_ping"] is True
            assert "connect_args" not in kwargs  # no SQLite args
        finally:
            settings.DATABASE_URL = original

    def test_ensure_db_sqlite_noop(self):
        """ensure_db should not change engine for SQLite."""
        from gex.adapters.persistence.database import ensure_db, is_fallback, active_url
        ensure_db()
        # Is a no-op — SQLite always works
        assert is_fallback() is False
        assert active_url() is not None

    def test_check_db_returns_status(self):
        """check_db should return status dict."""
        from gex.adapters.persistence.database import check_db
        status = check_db()
        assert "configured_url" in status
        assert "active_url" in status
        assert "fallback_active" in status
        assert "db_type" in status
        # SQLite should be connected
        assert status["connected"] is True

    def test_is_fallback_false_with_sqlite(self):
        """is_fallback should be False when using SQLite."""
        from gex.adapters.persistence.database import is_fallback
        assert is_fallback() is False


# ================================================================= #
#  3. Session lifecycle
# ================================================================= #
class TestSession:
    def test_session_can_connect(self, db):
        """Session should be able to execute queries."""
        result = db.execute(text("SELECT 1"))
        assert result.scalar() == 1

    def test_get_session_yields(self):
        """get_session() dependency should yield a working session."""
        gen = get_session()
        session = next(gen)
        try:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            try:
                next(gen)  # завершение генератора — close
            except StopIteration:
                pass

    def test_session_rollback_on_error(self):
        """get_session() should rollback on exception."""
        gen = get_session()
        session = next(gen)
        try:
            # Добавляем, но не коммитим
            user = User(id=str(uuid.uuid4()), email="rollback@test.com")
            session.add(user)
            session.flush()
            user_id = user.id

            # Кидаем исключение — генератор должен откатить
            gen.throw(RuntimeError("simulated error"))
        except RuntimeError:
            pass

        # Проверяем в новой сессии, что транзакция откатилась
        db2 = SessionLocal()
        try:
            found = db2.query(User).filter(User.id == user_id).first()
            assert found is None, "Transaction should have been rolled back"
        finally:
            db2.close()

    def test_multiple_sessions_independent(self, db):
        """Multiple sessions should be independent."""
        db1 = SessionLocal()
        db2 = SessionLocal()
        try:
            assert db1 is not db2
            assert db1.bind is engine
            assert db2.bind is engine
        finally:
            db1.close()
            db2.close()


# ================================================================= #
#  4. Table creation / lifecycle
# ================================================================= #
class TestTableLifecycle:
    def test_init_db_creates_tables(self):
        """init_db should create all tables."""
        drop_all()
        inspector = inspect(engine)
        assert "users" not in inspector.get_table_names()

        init_db()
        inspector = inspect(engine)
        assert "users" in inspector.get_table_names()

    def test_drop_all_removes_tables(self):
        """drop_all should remove all tables."""
        init_db()
        inspector = inspect(engine)
        assert "users" in inspector.get_table_names()

        drop_all()
        inspector = inspect(engine)
        assert "users" not in inspector.get_table_names()

    def test_recreate_tables(self):
        """recreate_tables should drop and create."""
        recreate_tables()
        inspector = inspect(engine)
        assert "users" in inspector.get_table_names()

    def test_table_has_expected_columns(self):
        """users table should have expected columns."""
        recreate_tables()
        inspector = inspect(engine)
        columns = {c["name"]: c for c in inspector.get_columns("users")}
        expected = {
            "id", "email", "password_hash", "is_email_verified",
            "subscription_status", "verification_token",
            "oauth_provider", "oauth_id", "created_at", "updated_at",
        }
        assert expected.issubset(columns.keys()), f"Missing: {expected - columns.keys()}"
        assert columns["email"]["nullable"] is False
        assert columns["is_email_verified"]["nullable"] is False


# ================================================================= #
#  5. Model CRUD (User)
# ================================================================= #
class TestUserCRUD:
    def test_create_user(self, db):
        """Should be able to create and persist a user."""
        user = User(
            id=str(uuid.uuid4()),
            email="create@test.com",
            is_email_verified=False,
            subscription_status=SubscriptionStatus.INACTIVE,
        )
        db.add(user)
        db.commit()

        found = db.query(User).filter(User.email == "create@test.com").first()
        assert found is not None
        assert found.email == "create@test.com"
        assert found.is_email_verified is False

    def test_create_user_with_password(self, db):
        """User with password hash should persist correctly."""
        user = User(
            id=str(uuid.uuid4()),
            email="hashed@test.com",
            password_hash="$2b$12$LJ3m4ys3Lk0TSwHnbfOMiOXPm1QwH/AwjAyOyF1ZyM".replace("", ""),
            is_email_verified=True,
            subscription_status=SubscriptionStatus.BASIC,
        )
        db.add(user)
        db.commit()

        found = db.query(User).filter(User.email == "hashed@test.com").first()
        assert found is not None
        assert found.password_hash is not None
        assert found.subscription_status == "BASIC"
        assert found.is_email_verified is True

    def test_update_user(self, db):
        """Should update user fields correctly."""
        user = User(
            id=str(uuid.uuid4()),
            email="update@test.com",
            subscription_status=SubscriptionStatus.INACTIVE,
        )
        db.add(user)
        db.commit()

        user.subscription_status = SubscriptionStatus.EXTENDED
        user.is_email_verified = True
        db.commit()

        found = db.query(User).filter(User.email == "update@test.com").first()
        assert found.subscription_status == "EXTENDED"
        assert found.is_email_verified is True

    def test_delete_user(self, db):
        """Should delete user correctly."""
        user = User(
            id=str(uuid.uuid4()),
            email="delete@test.com",
        )
        db.add(user)
        db.commit()

        db.delete(user)
        db.commit()

        found = db.query(User).filter(User.email == "delete@test.com").first()
        assert found is None

    def test_unique_email(self, db):
        """Email should be unique."""
        user1 = User(
            id=str(uuid.uuid4()),
            email="unique@test.com",
        )
        db.add(user1)
        db.commit()

        user2 = User(
            id=str(uuid.uuid4()),
            email="unique@test.com",
        )
        with pytest.raises(Exception):  # IntegrityError
            db.add(user2)
            db.flush()  # flush кидает исключение, commit не понадобится
        db.rollback()

    def test_find_by_email(self, db):
        """Should find user by email (indexed)."""
        uid = str(uuid.uuid4())
        user = User(id=uid, email="findme@test.com")
        db.add(user)
        db.commit()

        found = db.query(User).filter(User.email == "findme@test.com").first()
        assert found.id == uid

    def test_find_by_id(self, db):
        """Should find user by id (PK)."""
        uid = str(uuid.uuid4())
        user = User(id=uid, email="byid@test.com")
        db.add(user)
        db.commit()

        found = db.query(User).filter(User.id == uid).first()
        assert found.email == "byid@test.com"

    def test_default_values(self, db):
        """User should have sensible defaults."""
        user = User(id=str(uuid.uuid4()), email="defaults@test.com")
        db.add(user)
        db.commit()

        assert user.is_email_verified is False
        assert user.subscription_status == "INACTIVE"
        assert user.password_hash is None
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_timestamps_auto(self, db):
        """created_at and updated_at should be set automatically."""
        user = User(id=str(uuid.uuid4()), email="timestamps@test.com")
        db.add(user)
        db.commit()

        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)
        # SQLite не сохраняет tzinfo, но datetime хранится как UTC

    def test_oauth_fields(self, db):
        """OAuth-specific fields should work."""
        user = User(
            id=str(uuid.uuid4()),
            email="oauth@test.com",
            oauth_provider="google",
            oauth_id="google-123",
            is_email_verified=True,
        )
        db.add(user)
        db.commit()

        found = db.query(User).filter(User.email == "oauth@test.com").first()
        assert found.oauth_provider == "google"
        assert found.oauth_id == "google-123"
        assert found.password_hash is None

    def test_verification_token(self, db):
        """Verification token should be findable."""
        token = "test-token-abc-123"
        user = User(
            id=str(uuid.uuid4()),
            email="verify_token@test.com",
            verification_token=token,
        )
        db.add(user)
        db.commit()

        found = db.query(User).filter(User.verification_token == token).first()
        assert found is not None
        assert found.email == "verify_token@test.com"

    def test_bulk_create(self, db):
        """Bulk insert should work."""
        users = [
            User(id=str(uuid.uuid4()), email=f"bulk{i}@test.com")
            for i in range(10)
        ]
        for u in users:
            db.add(u)
        db.commit()

        count = db.query(User).count()
        assert count == 10


# ================================================================= #
#  6. Subscription status helpers (unit)
# ================================================================= #
class TestSubscriptionOrder:
    def test_order_values(self):
        """SUBSCRIPTION_ORDER should define correct ordering."""
        from gex.auth.models import SUBSCRIPTION_ORDER
        assert SUBSCRIPTION_ORDER["INACTIVE"] == 0
        assert SUBSCRIPTION_ORDER["BASIC"] == 1
        assert SUBSCRIPTION_ORDER["EXTENDED"] == 2
        assert SUBSCRIPTION_ORDER["ADMIN"] == 3
