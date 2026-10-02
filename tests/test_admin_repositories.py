"""Репозитории админки на настоящей БД: SQLite в памяти, реальная модель (итерация 34).

Зачем отдельный набор
---------------------
``tests/test_admin_and_sec.py`` проверяет use-case с **подставными** репозиториями: он
отвечает на вопрос «правильно ли собирается статистика». Но сам SQL при этом не выполняется
ни разу — а именно туда, в репозитории, переехали все запросы админки. Фильтры (кто считается
активным, что попадает в окно, что считается выручкой) — это и есть та часть, которую нельзя
проверить заглушкой: заглушка вернёт то, что в неё положили, и подтвердит сама себя.

Поэтому здесь SQL выполняется по-настоящему: in-memory SQLite, настоящие ``User``/``Payment``
из ``gex.auth.models`` / ``gex.auth.payment_models``. Никакой сети и никаких файлов.

Почему SQLite, а не Postgres: проверяются условия и порядок, а не диалект. Отдельно
проверяется ветка ``db_type()`` для группировки по дням — она на SQLite идёт по ``date()``,
а не по ``timezone('UTC', ...)``.

    python tests/test_admin_repositories.py
    pytest tests/test_admin_repositories.py -q
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Skipped(Exception):
    """Проверка требует sqlalchemy (её нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_SA = True
_IMPORT_ERROR: Exception | None = None
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from gex.adapters.persistence.auth_repositories import PaymentRepository, UserRepository
    from gex.auth.models import SUBSCRIPTION_VALUES, User
    from gex.auth.payment_models import Payment
except ImportError as exc:  # sqlalchemy (или pydantic в цепочке настроек) отсутствует
    _HAS_SA = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_SA:
        raise Skipped(f"нужен sqlalchemy ({_IMPORT_ERROR})")


# ====================================================================== #
#  База в памяти
# ====================================================================== #
NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


class Db:
    """In-memory SQLite с настоящими таблицами; StaticPool — чтобы база не исчезала."""

    def __init__(self):
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        # Создаём только нужные таблицы: чужие модели могут использовать типы, которых
        # в SQLite нет, и падение на них не имело бы отношения к предмету проверки.
        User.__table__.create(self.engine)
        Payment.__table__.create(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False)

    def session(self):
        return self.Session()

    def close(self):
        self.engine.dispose()


def add_user(session, index: int = 0, **kw) -> User:
    """Пользователь со предсказуемым email и временем регистрации."""
    defaults = dict(
        id=str(uuid.uuid4()),
        email=kw.pop("email", f"user{index}@example.com"),
        subscription_status=kw.pop("subscription_status", "INACTIVE"),
        created_at=kw.pop("created_at", NOW - timedelta(days=index)),
        updated_at=NOW,
    )
    user = User(**{**defaults, **kw})
    session.add(user)
    session.flush()
    return user


def add_payment(session, *, status: str, amount_usd: float, confirmed_at=None, created_at=None):
    payment = Payment(
        id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        user_email="payer@example.com",
        plan="BASIC",
        amount_rub=amount_usd * 90.0,
        amount_usd=amount_usd,
        method="SBP",
        status=status,
        created_at=created_at or NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=30),
        confirmed_at=confirmed_at if confirmed_at is not None else (
            NOW - timedelta(days=1) if status == "CONFIRMED" else None
        ),
    )
    session.add(payment)
    session.flush()
    return payment


# ====================================================================== #
#  UserRepository: счётчики
# ====================================================================== #
def test_total_and_active_count():
    """«Активных» — все, кроме неактивных; граница задаётся константой модели."""
    _require()
    db = Db()
    s = db.session()
    add_user(s, 0, subscription_status="INACTIVE")
    add_user(s, 1, subscription_status="BASIC")
    add_user(s, 2, subscription_status="EXTENDED")
    s.commit()

    repo = UserRepository(s)
    assert repo.total() == 3, "неверный общий счётчик"
    assert repo.active_count() == 2, "активными считаются не все не-INACTIVE"
    assert repo.inactive_status() == "INACTIVE", "константа неактивности разошлась с моделью"
    db.close()


def test_by_status_seeds_all_statuses_with_zero():
    """Статусы без строк присутствуют с нулём: иначе в админке пропадёт колонка."""
    _require()
    db = Db()
    s = db.session()
    add_user(s, 0, subscription_status="BASIC")
    add_user(s, 1, subscription_status="BASIC")
    add_user(s, 2, subscription_status="ADMIN")
    s.commit()

    counts = UserRepository(s).by_status()
    assert set(counts) == set(SUBSCRIPTION_VALUES), "список статусов не совпадает с моделью"
    assert counts["BASIC"] == 2 and counts["ADMIN"] == 1, counts
    assert counts["EXTENDED"] == 0 and counts["INACTIVE"] == 0, "нулевые статусы потерялись"
    db.close()


def test_by_status_covers_every_value_of_the_model():
    """Новый статус в модели обязан появиться в ответе без правки репозитория.

    Это и есть смысл того, что список берётся из модели: вторая копия молча разошлась бы.
    """
    _require()
    db = Db()
    s = db.session()
    add_user(s, 0, subscription_status="INACTIVE")
    s.commit()

    counts = UserRepository(s).by_status()
    assert len(counts) == len(SUBSCRIPTION_VALUES) == 4, (counts, SUBSCRIPTION_VALUES)
    db.close()


def test_by_provider_names_email_users():
    """OAuth-провайдер как есть; пользователи без OAuth называются ``email``, а не ``None``."""
    _require()
    db = Db()
    s = db.session()
    add_user(s, 0, oauth_provider="google")
    add_user(s, 1, oauth_provider="google")
    add_user(s, 2, oauth_provider="yandex")
    add_user(s, 3)  # без OAuth → email
    add_user(s, 4)  # без OAuth → email
    s.commit()

    by_provider = UserRepository(s).by_provider()
    # Через .get, а не []: при ключе None тест обязан сообщить «email не найден»,
    # а не падать с KeyError, спрятав причину.
    assert by_provider.get("email") == 2, f"пользователи без OAuth не подписаны как email: {by_provider}"
    assert by_provider.get("google") == 2, by_provider
    assert by_provider.get("yandex") == 1, by_provider
    assert None not in by_provider, "None попал в ключи — подпись графика будет пустой"
    assert len(by_provider) == 3, f"лишние ключи в разбивке: {by_provider}"
    db.close()


def test_new_since_boundary_is_inclusive():
    """Окно «новые за сутки» включает ровно граничный момент и не берёт более ранних."""
    _require()
    db = Db()
    s = db.session()
    add_user(s, 0, created_at=NOW - timedelta(days=2))  # старый
    add_user(s, 1, created_at=NOW - timedelta(days=1))  # ровно на границе
    add_user(s, 2, created_at=NOW - timedelta(hours=1))  # свежий
    s.commit()

    repo = UserRepository(s)
    assert repo.new_since(NOW - timedelta(days=1)) == 2, "граница обработана неверно"
    assert repo.new_since(NOW - timedelta(days=7)) == 3, "окно в неделю потеряло строки"
    assert repo.new_since(NOW - timedelta(minutes=30)) == 0, "окно в 30 минут непустое"
    db.close()


def test_expiring_within_needs_all_three_conditions():
    """Истекающие: срок есть, он в будущем, и подписка не неактивна. Нужны все три условия.

    Убрать любое — и админка покажет то просроченные подписки, то пустые, то уже выключенные.
    """
    _require()
    db = Db()
    s = db.session()
    # В окне, активная — считается.
    add_user(s, 0, subscription_status="BASIC",
             subscription_expires_at=NOW + timedelta(days=3))
    # В окне, но уже неактивна — не считается.
    add_user(s, 1, subscription_status="INACTIVE",
             subscription_expires_at=NOW + timedelta(days=3))
    # Просрочена (в прошлом) — не считается, хотя подписка «активна».
    add_user(s, 2, subscription_status="BASIC",
             subscription_expires_at=NOW - timedelta(days=1))
    # Дальше горизонта — не считается.
    add_user(s, 3, subscription_status="BASIC",
             subscription_expires_at=NOW + timedelta(days=60))
    # Срока нет вовсе (бессрочная) — не считается.
    add_user(s, 4, subscription_status="ADMIN", subscription_expires_at=None)
    s.commit()

    repo = UserRepository(s)
    assert repo.expiring_within(timedelta(days=7), now=NOW) == 1, "неверный подсчёт истекающих"
    assert repo.expiring_within(timedelta(days=90), now=NOW) == 2, "горизонт не расширил окно"
    db.close()


def test_registrations_by_day_groups_and_skips_empty():
    """Группировка по дате отдаёт только непустые дни; заполнение — дело use-case."""
    _require()
    db = Db()
    s = db.session()
    add_user(s, 0, created_at=datetime(2026, 9, 16, 1, 0, tzinfo=timezone.utc))
    add_user(s, 1, created_at=datetime(2026, 9, 16, 23, 30, tzinfo=timezone.utc))  # тот же день
    add_user(s, 2, created_at=datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc))
    add_user(s, 3, created_at=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc))  # вне окна
    s.commit()

    counts = UserRepository(s).registrations_by_day(7, now=NOW)
    assert counts.get("2026-09-16") == 2, f"день схлопнулся или разъехался: {counts}"
    assert counts.get("2026-09-14") == 1, counts
    assert "2026-09-15" not in counts, "пустой день попал в результат репозитория"
    assert "2026-09-01" not in counts, "строка вне окна попала в результат"
    db.close()


# ====================================================================== #
#  UserRepository: постраничный список
# ====================================================================== #
def _seed_page(session):
    add_user(session, 0, email="alice@example.com", subscription_status="BASIC",
             created_at=NOW - timedelta(days=1))
    add_user(session, 1, email="bob@example.com", subscription_status="EXTENDED",
             created_at=NOW - timedelta(days=2))
    add_user(session, 2, email="carol@gmail.com", subscription_status="INACTIVE",
             created_at=NOW - timedelta(days=3))


def test_page_search_is_case_insensitive_and_uses_substring():
    """Поиск по email: подстрока и регистр не важны — иначе админ ищет «вручную»."""
    _require()
    db = Db()
    s = db.session()
    _seed_page(s)
    s.commit()

    repo = UserRepository(s)
    rows, total = repo.page(search="ALICE", offset=0, limit=50)
    assert total == 1 and len(rows) == 1, (total, [u.email for u in rows])
    assert rows[0].email == "alice@example.com"

    rows, total = repo.page(search="example.com", offset=0, limit=50)
    assert total == 2, f"подстрока не найдена: {total}"

    # Пробелы вокруг запроса не должны обнулять результат (в форму часто копируют с пробелом).
    rows, total = repo.page(search="  bob  ", offset=0, limit=50)
    assert total == 1 and rows[0].email == "bob@example.com", total
    db.close()


def test_page_status_filter_matches_exactly():
    """Фильтр по статусу — точное совпадение, а не LIKE (иначе BASIC поймал бы BASICPLUS)."""
    _require()
    db = Db()
    s = db.session()
    _seed_page(s)
    s.commit()

    rows, total = UserRepository(s).page(status="BASIC", offset=0, limit=50)
    assert total == 1 and rows[0].subscription_status == "BASIC", total
    db.close()


def test_page_filters_apply_before_pagination():
    """``total`` — число ПОДХОДЯЩИХ, а не число всех строк.

    Классическая ошибка: посчитать total до фильтра, и в пагинации появится лишняя страница,
    которая открывается пустой.
    """
    _require()
    db = Db()
    s = db.session()
    _seed_page(s)
    s.commit()

    repo = UserRepository(s)
    _, filtered_total = repo.page(search="example.com", offset=0, limit=1)
    _, all_total = repo.page(offset=0, limit=1)
    assert all_total == 3, all_total
    assert filtered_total == 2, f"total посчитан до фильтра: {filtered_total} вместо 2"
    db.close()


def test_page_orders_newest_first_and_paginates():
    """Свежие — первыми; offset/limit режут уже упорядоченный список."""
    _require()
    db = Db()
    s = db.session()
    _seed_page(s)
    s.commit()

    repo = UserRepository(s)
    rows, total = repo.page(offset=0, limit=2)
    assert total == 3
    assert [u.email for u in rows] == ["alice@example.com", "bob@example.com"], [u.email for u in rows]

    rows, _ = repo.page(offset=2, limit=2)
    assert [u.email for u in rows] == ["carol@gmail.com"], [u.email for u in rows]

    rows, _ = repo.page(offset=10, limit=2)
    assert rows == [], "за пределами списка должны быть пустые страницы, а не ошибка"
    db.close()


def test_page_search_and_status_combine():
    """Поиск и фильтр складываются (AND), а не заменяют друг друга."""
    _require()
    db = Db()
    s = db.session()
    _seed_page(s)
    s.commit()

    repo = UserRepository(s)
    rows, total = repo.page(search="example.com", status="BASIC", offset=0, limit=50)
    assert total == 1 and rows[0].email == "alice@example.com", total

    rows, total = repo.page(search="gmail.com", status="BASIC", offset=0, limit=50)
    assert total == 0 and rows == [], "фильтры не сложились: нашлось то, чего нет"
    db.close()


def test_page_empty_search_does_not_filter():
    """Пустой поиск и ``None`` — одно и то же: список, а не «найти пустую строку»."""
    _require()
    db = Db()
    s = db.session()
    _seed_page(s)
    s.commit()

    repo = UserRepository(s)
    assert repo.page(search="", offset=0, limit=50)[1] == 3
    assert repo.page(search="   ", offset=0, limit=50)[1] == 3
    assert repo.page(search=None, offset=0, limit=50)[1] == 3
    db.close()


# ====================================================================== #
#  PaymentRepository
# ====================================================================== #
def test_payment_counts_and_statuses():
    """Всего, «в процессе» (два статуса) и разбивка по статусам."""
    _require()
    db = Db()
    s = db.session()
    add_payment(s, status="CONFIRMED", amount_usd=10.0)
    add_payment(s, status="PENDING", amount_usd=20.0)
    add_payment(s, status="PAID_CLIENT", amount_usd=30.0)
    add_payment(s, status="REJECTED", amount_usd=40.0)
    s.commit()

    repo = PaymentRepository(s)
    assert repo.total() == 4, "неверный общий счётчик платежей"
    assert repo.pending_count() == 2, "в «процессе» должны быть PENDING и PAID_CLIENT"
    by_status = repo.by_status()
    assert by_status["CONFIRMED"] == 1 and by_status["REJECTED"] == 1, by_status
    db.close()


def test_payment_revenue_counts_confirmed_only():
    """Выручка — только подтверждённые платежи.

    Считать всё, что лежит в таблице, значит показывать в админке деньги, которых не поступало:
    PENDING это «ещё не перевёл», REJECTED — «перевод не найден».
    """
    _require()
    db = Db()
    s = db.session()
    add_payment(s, status="CONFIRMED", amount_usd=10.0)
    add_payment(s, status="CONFIRMED", amount_usd=5.5)
    add_payment(s, status="PENDING", amount_usd=100.0)
    add_payment(s, status="PAID_CLIENT", amount_usd=100.0)
    add_payment(s, status="REJECTED", amount_usd=100.0)
    s.commit()

    assert PaymentRepository(s).revenue() == 15.5, "в выручку попали неподтверждённые платежи"
    db.close()


def test_payment_revenue_window_uses_confirmation_moment():
    """Окно выручки — по моменту подтверждения, а не по моменту создания заявки.

    Разница в реальная: человек мог создать заявку в пятницу, а оплатить в понедельник —
    и «выручка за 7 дней» не должна ни терять его, ни приписывать его прошлой неделе.
    """
    _require()
    db = Db()
    s = db.session()
    # Создан давно, подтверждён только что → попадает в окно.
    add_payment(s, status="CONFIRMED", amount_usd=7.0,
                created_at=NOW - timedelta(days=40), confirmed_at=NOW - timedelta(hours=1))
    # Создан вчера, подтверждён давно (переоформление старой заявки) → вне окна.
    # Дата подтверждения взята заметно за границей (45 дней, а не ровно 30): на самой границе
    # платёж попадает в окно по праву, и тест проверял бы уже не то, что заявлено в названии.
    add_payment(s, status="CONFIRMED", amount_usd=3.0,
                created_at=NOW - timedelta(days=1), confirmed_at=NOW - timedelta(days=45))
    s.commit()

    repo = PaymentRepository(s)
    assert repo.revenue() == 10.0, "выручка за всё время неверна"
    assert repo.revenue(since=NOW - timedelta(days=30)) == 7.0, "окно считает по created_at"
    db.close()


def test_payments_are_empty_safe():
    """Пустая таблица — нули, а не падение и не ``None`` в ответе админки."""
    _require()
    db = Db()
    s = db.session()
    repo = PaymentRepository(s)
    assert repo.total() == 0 and repo.pending_count() == 0
    assert repo.by_status() == {} and repo.revenue() == 0.0
    assert repo.revenue(since=NOW - timedelta(days=7)) == 0.0
    assert UserRepository(s).total() == 0 and UserRepository(s).active_count() == 0
    assert UserRepository(s).page(offset=0, limit=10) == ([], 0)
    db.close()


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Skipped as exc:
            skipped += 1
            print(f"SKIP {fn.__name__}: {exc}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- admin repositories: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
