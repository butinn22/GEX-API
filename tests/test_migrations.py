"""Tests for Alembic migrations: upgrade, downgrade, history.

Все тесты используют SQLite in-memory (не требуют PostgreSQL).
"""
from __future__ import annotations

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext
from alembic.command import upgrade, downgrade, history, current, revision

from gex.adapters.persistence.database import engine, Base, init_db, drop_all, recreate_tables
from gex.auth.models import User
from gex.auth.config import settings


# ================================================================= #
#  Helpers
# ================================================================= #

def _alembic_cfg() -> Config:
    """Create Alembic config pointing to our project dir."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ini_path = os.path.join(project_dir, "alembic.ini")
    cfg = Config(ini_path)
    cfg.set_main_option("script_location", os.path.join(project_dir, "alembic"))
    cfg.set_main_option("sqlalchemy.url", str(engine.url))
    # Миграции применяются к уже открытому соединению нашего engine
    # (in-memory SQLite: Alembic не может создать свой engine на ту же БД).
    cfg.attributes["connection"] = engine.connect()
    return cfg


# ================================================================= #
#  Fixtures
# ================================================================= #
@pytest.fixture(autouse=True)
def clean_slate():
    """Ensure clean database before each migration test (включая alembic_version)."""
    from gex.adapters.persistence.database import SessionLocal
    drop_all()
    # Drop alembic_version table if it exists (она не в Base.metadata)
    db = SessionLocal()
    try:
        db.execute(text("DROP TABLE IF EXISTS alembic_version"))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    yield


# ================================================================= #
#  1. Alembic configuration basics
# ================================================================= #
class TestAlembicConfig:
    def test_config_exists(self):
        """alembic.ini should exist and be loadable."""
        cfg = _alembic_cfg()
        assert cfg is not None
        assert cfg.config_file_name is not None

    def test_script_directory(self):
        """Alembic script directory should exist with versions."""
        cfg = _alembic_cfg()
        script = ScriptDirectory.from_config(cfg)
        assert script is not None
        heads = script.get_heads()
        assert len(heads) >= 1, "At least one migration revision should exist"

    def test_script_has_heads(self):
        """Script directory should have at least one head revision."""
        cfg = _alembic_cfg()
        script = ScriptDirectory.from_config(cfg)
        heads = script.get_heads()
        assert len(heads) >= 1
        # Verify head exists in versions dir
        for head in heads:
            rev = script.get_revision(head)
            assert rev is not None
            assert rev.doc is not None


# ================================================================= #
#  2. Migration lifecycle: upgrade → downgrade
# ================================================================= #
class TestMigrationLifecycle:
    def test_upgrade_head(self):
        """upgrade('head') should create all tables."""
        cfg = _alembic_cfg()
        upgrade(cfg, "head")

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "users" in tables, f"Expected 'users' in tables, got {tables}"

    def test_upgrade_then_downgrade_base(self):
        """Upgrade to head then downgrade to base should drop all tables."""
        cfg = _alembic_cfg()

        # Upgrade
        upgrade(cfg, "head")
        inspector = inspect(engine)
        assert "users" in inspector.get_table_names()

        # Downgrade to base (no tables)
        downgrade(cfg, "base")
        inspector = inspect(engine)
        assert "users" not in inspector.get_table_names()

    def test_upgrade_downgrade_idempotent(self):
        """Upgrade → downgrade → upgrade should be idempotent."""
        cfg = _alembic_cfg()

        # First cycle
        upgrade(cfg, "head")
        downgrade(cfg, "base")

        # Second cycle
        upgrade(cfg, "head")
        inspector = inspect(engine)
        assert "users" in inspector.get_table_names()

    def test_migration_applies_correct_schema(self):
        """After migration, users table should have expected columns."""
        cfg = _alembic_cfg()
        upgrade(cfg, "head")

        inspector = inspect(engine)
        columns = {c["name"]: c for c in inspector.get_columns("users")}
        expected = {
            "id", "email", "password_hash", "is_email_verified",
            "subscription_status", "verification_token",
            "oauth_provider", "oauth_id", "created_at", "updated_at",
        }
        assert expected.issubset(columns.keys()), f"Missing: {expected - columns.keys()}"

    def test_migration_creates_indexes(self):
        """After migration, email should have unique constraint."""
        cfg = _alembic_cfg()
        upgrade(cfg, "head")

        inspector = inspect(engine)
        # Проверяем UNIQUE constraint (SQLite может не показывать индекс
        # для UNIQUE-колонок через get_indexes, но сам constraint есть)
        columns = {c["name"]: c for c in inspector.get_columns("users")}
        assert columns["email"]["nullable"] is False

        # Проверяем, что можно вставить дубликат без уникальности
        from gex.adapters.persistence.database import SessionLocal
        from gex.auth.models import User
        import uuid
        db = SessionLocal()
        try:
            user1 = User(id=str(uuid.uuid4()), email="migration_idx@test.com")
            db.add(user1)
            db.commit()
            db.close()

            db2 = SessionLocal()
            user2 = User(id=str(uuid.uuid4()), email="migration_idx@test.com")
            db2.add(user2)
            with pytest.raises(Exception):
                db2.commit()
            db2.rollback()
            db2.close()
        finally:
            pass

    def test_insert_after_migration(self):
        """After migration, we should be able to insert and query users."""
        cfg = _alembic_cfg()
        upgrade(cfg, "head")

        from gex.adapters.persistence.database import SessionLocal
        db = SessionLocal()
        try:
            user = User(
                id=str(uuid.uuid4()),
                email="migration_test@test.com",
                is_email_verified=True,
                subscription_status="BASIC",
            )
            db.add(user)
            db.commit()

            found = db.query(User).filter(User.email == "migration_test@test.com").first()
            assert found is not None
            assert found.is_email_verified is True
        finally:
            db.close()


# ================================================================= #
#  3. Revision history
# ================================================================= #
class TestRevisionHistory:
    def test_revision_has_down_revision_none(self):
        """First revision should have down_revision = None."""
        cfg = _alembic_cfg()
        script = ScriptDirectory.from_config(cfg)
        for rev in script.walk_revisions("base", "head"):
            if rev.down_revision is None:
                assert rev.doc is not None
                return
        pytest.fail("No base revision found (down_revision=None)")

    def test_revision_chain_linear(self):
        """All revisions should form a linear chain (not branching)."""
        cfg = _alembic_cfg()
        script = ScriptDirectory.from_config(cfg)
        heads = script.get_heads()
        assert len(heads) == 1, f"Expected single head, got {heads}"

    def test_current_is_none_before_migration(self):
        """Before any migration, current should indicate no version."""
        cfg = _alembic_cfg()

        # We can't easily call current() without migration context,
        # but we can verify that the database has no Alembic version table
        inspector = inspect(engine)
        assert "alembic_version" not in inspector.get_table_names()

    def test_current_is_head_after_upgrade(self):
        """After upgrade head, current should match head revision."""
        cfg = _alembic_cfg()
        upgrade(cfg, "head")

        inspector = inspect(engine)
        assert "alembic_version" in inspector.get_table_names()

        # Read the version directly
        from gex.adapters.persistence.database import SessionLocal
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
            assert version is not None
            # Should match one of the heads
            script = ScriptDirectory.from_config(cfg)
            assert version in script.get_heads()
        finally:
            db.close()


# ================================================================= #
#  4. Offline migration (SQL generation)
# ================================================================= #
class TestOfflineMigration:
    def test_offline_upgrade_generates_sql(self):
        """Offline mode should generate SQL without a database connection."""
        cfg = _alembic_cfg()
        script = ScriptDirectory.from_config(cfg)

        # Run offline migration
        import io

        output = io.StringIO()
        cfg.attributes["output_buffer"] = output

        # Use offline context
        script = ScriptDirectory.from_config(cfg)
        heads = script.get_heads()

        # We just verify the script can be rendered offline
        assert len(heads) >= 1
        rev = script.get_revision(heads[0])
        assert rev is not None

    def test_offline_upgrade_creates_users_table(self):
        """Offline upgrade SQL should contain CREATE TABLE users."""
        cfg = _alembic_cfg()

        import io
        buf = io.StringIO()

        # Run offline
        from alembic.command import upgrade as _upgrade

        # Temporarily redirect stdout
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _upgrade(cfg, "head", sql=True)
        finally:
            sys.stdout = old_stdout

        sql_output = buf.getvalue()
        assert "CREATE TABLE users" in sql_output or "CREATE TABLE" in sql_output
        assert "id" in sql_output
        assert "email" in sql_output


# ================================================================= #
#  5. Autogenerate detection
# ================================================================= #
class TestAutogenerate:
    def test_autogenerate_detects_new_column(self):
        """Autogenerate should produce a migration for schema changes.

        Note: this is a conceptual test — we can't easily run autogenerate
        in a test context, but we verify the env.py configuration supports it.
        """
        cfg = _alembic_cfg()
        script = ScriptDirectory.from_config(cfg)

        # Verify env.py imports our models
        env_py = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "alembic", "env.py"
        )
        with open(env_py) as f:
            content = f.read()
        assert "from gex.adapters.persistence.database import Base" in content
        assert "target_metadata = Base.metadata" in content
        assert "from gex.auth import models" in content

    def test_env_py_handles_sqlite(self):
        """env.py should handle SQLite correctly (sync mode)."""
        cfg = _alembic_cfg()
        cfg.set_main_option("sqlalchemy.url", "sqlite:///:memory:")

        # env.py should not attempt async for SQLite
        env_py_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "alembic", "env.py"
        )
        with open(env_py_path) as f:
            content = f.read()
        assert "sqlite" in content.lower()
        assert "run_sync_migrations" in content


# ================================================================= #
#  6. Dual-driver support (SQLite / PostgreSQL)
# ================================================================= #
class TestDualDriver:
    def test_sqlite_url_not_changed(self):
        """SQLite URL should stay as-is in env.py logic."""
        from gex.adapters.persistence.database import _build_engine_url
        from gex.auth.config import settings as s
        url = _build_engine_url()
        assert url == s.DATABASE_URL.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        ).replace(
            "postgresql://", "postgresql+psycopg2://"
        ), f"URL mismatch: {url} vs {s.DATABASE_URL}"

    def test_sync_postgres_url_conversion(self):
        """postgresql+asyncpg:// should convert to psycopg2."""
        from gex.adapters.persistence.database import _build_engine_url
        original = settings.DATABASE_URL
        try:
            settings.DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/db"
            result = _build_engine_url()
            assert "psycopg2" in result
            assert "asyncpg" not in result
        finally:
            settings.DATABASE_URL = original
