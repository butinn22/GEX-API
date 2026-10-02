"""Тесты политики свежести (чистые, без pandas/numpy).

Проверяют инварианты, на которых держится требование «пользователь не ждёт и не видит устаревшее
молча»: у каждой страницы есть явный класс и окна, вне сессии окна не строже, чем в сессии,
расписание отделено от непрерывных данных.

    python tests/test_freshness_policy.py
    pytest tests/test_freshness_policy.py -q
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gex.domain import freshness as fr  # noqa: E402


def test_every_page_has_class_and_policy():
    for page in fr.all_pages():
        cls = fr.page_class(page)
        assert isinstance(cls, fr.Freshness)
        policy = fr.policy_for_page(page)
        assert policy.fresh > 0, f"{page}: fresh должен быть положительным"
        assert policy.stale_max >= policy.fresh, f"{page}: stale_max < fresh"
        assert policy.prewarm >= 0


def test_every_page_has_explicit_windows():
    """Ни одна страница не должна «проваливаться» в дефолт класса: окна задаются явно."""
    missing = [p for p in fr.all_pages() if p not in fr.PAGE_WINDOWS]
    assert missing == [], f"страницы без явных окон: {missing}"


def test_off_session_windows_are_not_stricter():
    """Вне сессии окна обязаны быть не уже, чем в сессии (иначе провайдера дёргаем зря)."""
    for page in fr.all_pages():
        if fr.is_periodic_page(page):
            continue  # у расписания сессий нет
        in_session = fr.policy_for_page(page, market_open=True)
        off_session = fr.policy_for_page(page, market_open=False)
        assert off_session.fresh >= in_session.fresh, f"{page}: вне сессии fresh меньше"
        assert off_session.stale_max >= in_session.stale_max, f"{page}: вне сессии stale_max меньше"


def test_live_pages_are_fresher_than_computed_slow():
    """Котировки обновляются быстрее, чем тяжёлые вычисления (иначе политика бессмысленна)."""
    live = fr.policy_for_page("quote").fresh
    slow = fr.policy_for_page("cone").fresh
    assert live < slow, f"quote.fresh={live} должен быть меньше cone.fresh={slow}"


def test_imoex_breadth_is_periodic():
    assert fr.is_periodic_page("breadth-imoex")
    policy = fr.policy_for_page("breadth-imoex", market_open=True)
    assert policy.fresh >= 3600, "расписание: окно свежести — часы, а не минуты"
    assert not fr.is_periodic_page("ta"), "непрерывные страницы не должны быть расписанием"


def test_unknown_page_raises_with_hint():
    try:
        fr.policy_for_page("no-such-page")
    except KeyError as exc:
        assert "не описана в PAGE_CLASS" in str(exc)
    else:
        raise AssertionError("для неизвестной страницы должно падать с подсказкой")


def test_timeframe_policies_ordering():
    p1m = fr.policy_for_timeframe("1m")
    p1d = fr.policy_for_timeframe("1d")
    assert p1m.fresh < p1d.fresh, "минутные свечи должны обновляться чаще дневных"
    assert p1m.stale_max <= p1d.stale_max
    unknown = fr.policy_for_timeframe("7m")  # не в таблице → фолбэк LIVE
    assert unknown.fresh > 0


def test_session_windows_msk():
    """US: 16:30–23:00 MSK; MOEX: 10:00–18:40 MSK; выходные закрыты."""
    monday_0900 = datetime(2026, 9, 14, 9, 0, tzinfo=fr.MSK)
    monday_1030 = datetime(2026, 9, 14, 10, 30, tzinfo=fr.MSK)
    monday_1700 = datetime(2026, 9, 14, 17, 0, tzinfo=fr.MSK)
    monday_1900 = datetime(2026, 9, 14, 19, 0, tzinfo=fr.MSK)
    monday_2330 = datetime(2026, 9, 14, 23, 30, tzinfo=fr.MSK)
    saturday_1700 = datetime(2026, 9, 19, 17, 0, tzinfo=fr.MSK)

    # US
    assert not fr.session_open("us", monday_0900), "09:00 MSK — US ещё закрыт"
    assert fr.session_open("us", monday_1700)
    assert not fr.session_open("us", monday_2330), "после 23:00 MSK US закрыт"
    # MOEX (открывается в 10:00 MSK, закрывается в 18:40)
    assert not fr.session_open("moex", monday_0900), "09:00 MSK — MOEX ещё не открылся"
    assert fr.session_open("moex", monday_1030)
    assert not fr.session_open("moex", monday_1900), "после 18:40 MSK MOEX закрыт"
    # выходные
    assert not fr.session_open("us", saturday_1700), "в субботу рынок закрыт"
    assert fr.market_open_now(("us", "moex"), monday_1700)
    assert not fr.market_open_now(("us", "moex"), monday_2330)


def test_session_open_accepts_utc_datetime():
    """UTC-момент корректно переводится в MSK (16:30 MSK = 13:30 UTC)."""
    utc_open = datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc)
    assert fr.session_open("us", utc_open)
    utc_before = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)  # 15:00 MSK
    assert not fr.session_open("us", utc_before)


def test_policy_rejects_invalid_windows():
    try:
        fr.Policy(fresh=100, stale_max=50, prewarm=10)
    except ValueError as exc:
        assert "некорректные окна" in str(exc)
    else:
        raise AssertionError("fresh > stale_max должен быть отклонён")


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- freshness policy: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
