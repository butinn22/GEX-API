"""Прогрев: план расписания и «ровно один раз» на кластер (итерация 32).

Что проверяется
---------------
1. **План — объявление, а не неявный цикл.** Каждый слот имеет имя, интервал, цели и
   описание; проверка полноты («план ↔ обработчик») не находит проблем. Слот без описания
   или с дублирующимся именем — это объявление, которое ничего не делает, и такое должно
   ловиться на сборке.
2. **Слот выполняется ровно один раз на кластер.** Это замена прежнему маркеру
   ``GET`` → публикация → ``SET``: он не атомарен, поэтому после старта все реплики
   публиковали прогрев одновременно (худший случай для провайдера). Теперь аренда
   ``SET NX PX`` берётся **до** публикации.
3. **Освобождается только своя аренда.** ``DEL`` по ключу снял бы аренду, которую уже
   перехватил другой процесс, — и слот выполнился бы дважды.

Аренда тестируется настоящим :class:`RedisLease` поверх фейкового Redis: проверять
заглушку вместо механизма смысла нет.

    python tests/test_prewarm.py
    pytest tests/test_prewarm.py -q
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.cache.lease import MIN_TTL_S, RedisLease  # noqa: E402
from gex.application.jobs import FetchTask  # noqa: E402
from gex.application.prewarm import (  # noqa: E402
    PrewarmPlan,
    PrewarmSlot,
    PrewarmWorker,
)
from gex.domain.schedule import MSK, slot_occurrence  # noqa: E402


class Skipped(Exception):
    """Проверка требует pandas (нет в stdlib-прогоне) — пропуск, а не «зелёный»."""


class FakeRedis:
    """Redis с семантикой ``SET NX PX``: строки, истечение по монотонному времени."""

    def __init__(self, clock):
        self.kv: dict[str, tuple[bytes, float | None]] = {}
        self._clock = clock
        self.fail = False

    def _purge(self, key: str) -> None:
        entry = self.kv.get(key)
        if entry is not None and entry[1] is not None and self._clock() >= entry[1]:
            self.kv.pop(key, None)

    def set(self, key, value, ex=None, *, px=None, nx=False):
        if self.fail:
            raise ConnectionError("redis down")
        self._purge(key)
        if nx and key in self.kv:
            return None
        ttl = None
        if px is not None:
            ttl = self._clock() + px / 1000.0
        elif ex is not None:
            ttl = self._clock() + ex
        self.kv[key] = (value if isinstance(value, bytes) else str(value).encode(), ttl)
        return True

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        self._purge(key)
        entry = self.kv.get(key)
        return entry[0] if entry else None

    def delete(self, key):
        self._purge(key)
        return self.kv.pop(key, None) is not None


class Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingPublisher:
    """Публикатор, который запоминает, что и сколько раз отправил."""

    def __init__(self, *, published: int | None = None):
        self.calls: list[list[FetchTask]] = []
        self.published = published  # None — «сколько дали, столько и ушло»

    def publish_many(self, tasks):
        tasks = list(tasks)
        self.calls.append(tasks)
        return len(tasks) if self.published is None else self.published


def _slot(name="ohlcv_us", interval=600, *, tasks=None, fixed_msk=()) -> PrewarmSlot:
    return PrewarmSlot(
        name=name,
        interval_s=interval,
        tasks=tasks or (lambda: [FetchTask("ohlcv", "yfinance", "SPY", priority=0)]),
        description=f"слот {name} (тест)",
        fixed_msk=fixed_msk,
    )


def _worker(slots, publisher, clock, redis=None):
    lease = RedisLease(redis) if redis is not None else RedisLease(FakeRedis(clock))
    return PrewarmWorker(PrewarmPlan(slots=tuple(slots)), publisher, lease, clock=clock)


# ====================================================================== #
# 1. План: объявление и проверка полноты
# ====================================================================== #
def test_plan_validate_accepts_complete_plan():
    plan = PrewarmPlan(slots=(_slot("a", 60), _slot("b", 120)))
    assert plan.validate() == []
    assert plan.names() == ("a", "b")
    assert plan.intervals() == {"a": 60, "b": 120}


def test_plan_flags_duplicate_names():
    problems = PrewarmPlan(slots=(_slot("dup", 60), _slot("dup", 120))).validate()
    assert any("дважды" in p for p in problems), problems


def test_plan_flags_missing_description():
    """Слот без описания — объявление, которое ничего не объясняет: это ошибка сборки."""
    bare = PrewarmSlot(name="x", interval_s=60, tasks=lambda: [], description="")
    assert any("нет описания" in p for p in PrewarmPlan(slots=(bare,)).validate())


def test_slot_rejects_non_positive_interval_without_fixed_window():
    try:
        PrewarmSlot(name="x", interval_s=0, tasks=lambda: [], description="d")
    except ValueError as exc:
        assert "интервал" in str(exc)
        return
    raise AssertionError("нулевой интервал принят")


def test_plan_get_unknown_slot():
    try:
        PrewarmPlan(slots=(_slot("a"),)).get("нет")
    except KeyError:
        return
    raise AssertionError("неизвестный слот принят")


# ====================================================================== #
# 2. Ровно один раз на кластер (главное свойство итерации)
# ====================================================================== #
def test_slot_fires_once_across_workers():
    """Две реплики, один Redis: слот публикуется **один** раз, а не дважды."""
    clock = Clock()
    redis = FakeRedis(clock)
    publisher_a, publisher_b = RecordingPublisher(), RecordingPublisher()
    worker_a = _worker([_slot()], publisher_a, clock, redis)
    worker_b = _worker([_slot()], publisher_b, clock, redis)

    assert worker_a.run_once() == ["ohlcv_us"]
    assert worker_b.run_once() == [], "вторая реплика выполнила занятый слот"

    assert len(publisher_a.calls) == 1 and publisher_b.calls == []
    assert worker_b.skipped_lease_held == 1


def test_slot_becomes_available_after_the_interval():
    """Аренда живёт интервал: раньше его слот не повторяется, после — снова доступен."""
    clock = Clock()
    redis = FakeRedis(clock)
    worker = _worker([_slot(interval=600)], RecordingPublisher(), clock, redis)

    assert worker.run_once() == ["ohlcv_us"]
    clock.advance(300)
    assert worker.run_once() == [], "слот выполнен раньше интервала"
    clock.advance(301)  # аренда истекла (600 с), интервал прошёл
    assert worker.run_once() == ["ohlcv_us"]


def test_release_only_own_lease():
    """Освободить чужую аренду нельзя: иначе слот выполнится дважды."""
    clock = Clock()
    redis = FakeRedis(clock)
    lease = RedisLease(redis)

    token_a = lease.acquire("prewarm:ohlcv_us", ttl_s=60)
    assert token_a is not None
    assert lease.acquire("prewarm:ohlcv_us", ttl_s=60) is None, "аренда выдана дважды"

    assert lease.release("prewarm:ohlcv_us", "чужой-токен") is False, "сняли не свою аренду"
    assert lease.is_held("prewarm:ohlcv_us") is True

    assert lease.release("prewarm:ohlcv_us", token_a) is True
    assert lease.is_held("prewarm:ohlcv_us") is False


def test_lease_denies_on_redis_failure():
    """Redis недоступен — аренда не берётся: лучше не прогревать, чем прогревать N раз."""
    clock = Clock()
    redis = FakeRedis(clock)
    redis.fail = True
    assert RedisLease(redis).acquire("slot", ttl_s=60) is None


def test_minimum_lease_ttl():
    clock = Clock()
    lease = RedisLease(FakeRedis(clock))
    token = lease.acquire("slot", ttl_s=0)  # нулевой TTL превратился бы в отсутствие аренды
    assert token is not None
    clock.advance(MIN_TTL_S + 1)
    assert lease.is_held("slot") is False


# ====================================================================== #
# 3. Ограниченность и изоляция сбоев
# ====================================================================== #
def test_bounded_slots_per_tick():
    """На старте все слоты просрочены: без границы процесс выпустил бы весь прогрев разом."""
    clock = Clock()
    publisher = RecordingPublisher()
    slots = [_slot(f"s{i}", 600) for i in range(6)]
    worker = _worker(slots, publisher, clock)
    worker._max_slots_per_tick = 2

    assert worker.run_once() == ["s0", "s1"]
    assert worker.run_once() == ["s2", "s3"]


def test_failing_slot_does_not_stop_the_others_and_retries():
    """Сбой одной цели не должен мешать остальным, а её аренда — освобождаться."""
    clock = Clock()

    def boom():
        raise RuntimeError("вселенная недоступна")

    slots = [_slot("bad", 600, tasks=boom), _slot("good", 600)]
    publisher = RecordingPublisher()
    worker = _worker(slots, publisher, clock)

    assert worker.run_once() == ["good"]
    assert worker.failed.get("bad") == 1
    # Аренда сбоящего слота свободна — следующая попытка не будет ждать интервал.
    assert worker._lease.is_held("prewarm:bad") is False
    worker.run_once()
    assert worker.failed.get("bad") == 2


def test_nothing_published_releases_the_lease():
    """Очередь недоступна — аренду освобождаем, иначе слот «сгорит» до конца интервала."""
    clock = Clock()
    worker = _worker([_slot()], RecordingPublisher(published=0), clock)
    assert worker.run_once() == []
    assert worker._lease.is_held("prewarm:ohlcv_us") is False


def test_empty_targets_release_the_lease():
    clock = Clock()
    worker = _worker([_slot(tasks=lambda: [])], RecordingPublisher(), clock)
    assert worker.run_once() == []
    assert worker._lease.is_held("prewarm:ohlcv_us") is False


def test_worker_without_lease_still_works():
    """Аренда не настроена (нет Redis) — работаем как единственный процесс."""
    clock = Clock()
    publisher = RecordingPublisher()
    worker = PrewarmWorker(PrewarmPlan(slots=(_slot(),)), publisher, None, clock=clock)
    assert worker.run_once() == ["ohlcv_us"]
    assert len(publisher.calls) == 1


def test_describe_exposes_state():
    clock = Clock()
    worker = _worker([_slot()], RecordingPublisher(), clock)
    worker.run_once()
    state = worker.describe()
    assert state["fired"] == {"ohlcv_us": 1}
    assert state["slots"] == ["ohlcv_us"]


# ====================================================================== #
# 4. Фиксированные МСК-слоты
# ====================================================================== #
def test_fixed_slot_fires_only_inside_its_window():
    """23:00 МСК: слот срабатывает в окне, и вне окна не «догоняется»."""
    slot = _slot("breadth_imoex", 86400, fixed_msk=((23, 0),))
    plan = PrewarmPlan(slots=(slot,))

    inside = datetime(2026, 9, 16, 23, 30, tzinfo=MSK)
    outside = datetime(2026, 9, 16, 15, 0, tzinfo=MSK)
    assert slot_occurrence(slot.fixed_msk, inside) == "2026-09-16 23:00"
    assert slot_occurrence(slot.fixed_msk, outside) is None

    clock = Clock()
    publisher = RecordingPublisher()
    worker = PrewarmWorker(plan, publisher, RedisLease(FakeRedis(clock)), clock=clock,
                           wall_clock=inside.timestamp)
    assert worker.run_once() == ["breadth_imoex"]

    # Второй воркер в том же вхождении — уже не публикует (аренда на вхождение).
    other = PrewarmWorker(plan, RecordingPublisher(), RedisLease(FakeRedis(clock)), clock=clock,
                          wall_clock=inside.timestamp)
    other._lease = worker._lease  # тот же Redis
    assert other.run_once() == []
    assert len(publisher.calls) == 1


def test_fixed_slot_does_not_fire_outside_window():
    slot = _slot("breadth_imoex", 86400, fixed_msk=((23, 0),))
    clock = Clock()
    publisher = RecordingPublisher()
    outside = datetime(2026, 9, 16, 12, 0, tzinfo=MSK)
    worker = PrewarmWorker(PrewarmPlan(slots=(slot,)), publisher, RedisLease(FakeRedis(clock)),
                           clock=clock, wall_clock=outside.timestamp)
    assert worker.run_once() == []
    assert publisher.calls == []


def test_schedule_window_uses_msk_not_local_time():
    """Окно считается по МСК: тот же момент в UTC не должен попадать в слот."""
    moment_msk = datetime(2026, 9, 16, 23, 30, tzinfo=MSK)
    moment_utc = moment_msk.astimezone(timezone.utc)  # 20:30 UTC
    assert slot_occurrence(((23, 0),), moment_msk) is not None
    assert slot_occurrence(((23, 0),), moment_utc) is not None, "aware-время сравнивается верно"
    assert slot_occurrence(((23, 0),), moment_utc + timedelta(hours=1)) is None


# ====================================================================== #
# 5. Реальное расписание проекта (нужен pandas: вселенные живут в background_fetcher)
# ====================================================================== #
def test_project_plan_is_complete():
    """План проекта: 12 слотов, каждый с целями и описанием, без проблем полноты.

    Слотов было 13 — ушёл ``breadth_imoex``: парсинг ISS переехал в фиксированные слоты
    Celery Beat (``gex.domain.schedule.PERIODIC_SLOTS``). Причина в квоте провайдера: два
    планировщика (потоковый и Beat) означали бы четыре обращения к ISS в сутки при лимите
    три, то есть один слот гарантированно оставался бы без свежих данных.
    """
    try:
        from gex.application.scheduler import build_prewarm_plan
    except ImportError as exc:  # pandas
        raise Skipped(f"нужен pandas ({exc})") from None

    plan = build_prewarm_plan()
    assert plan.validate() == [], plan.validate()
    assert len(plan.slots) == 12, plan.names()
    assert "breadth_imoex" not in plan.names(), "парсинг ISS планирует Beat, а не потоковый план"
    # Каждый слот обязан уметь построить непустой список задач на реальных вселенных.
    for slot in plan.slots:
        if slot.fixed_msk:
            continue
        assert slot.tasks(), f"слот {slot.name} не строит задач"


def test_project_plan_matches_the_old_schedule():
    """Интервалы плана совпадают с прежним `_SCHEDULE`: поведение не изменилось."""
    try:
        from gex.application.scheduler import _SCHEDULE, build_prewarm_plan
    except ImportError as exc:
        raise Skipped(f"нужен pandas ({exc})") from None

    plan = build_prewarm_plan()
    for name, interval in _SCHEDULE.items():
        assert plan.intervals().get(name) == interval, f"{name}: {plan.intervals().get(name)} != {interval}"


def test_multiplier_scales_intervals():
    try:
        from gex.application.scheduler import _SCHEDULE, build_prewarm_plan
    except ImportError as exc:
        raise Skipped(f"нужен pandas ({exc})") from None

    base = _SCHEDULE["ohlcv_us"]
    plan = build_prewarm_plan(2.0)
    assert plan.intervals()["ohlcv_us"] == 2 * base


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
    print(f"--- prewarm: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
