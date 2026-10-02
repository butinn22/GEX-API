"""Use-case конуса: источник → параметры → калькулятор, и golden по числам (итерация 36).

Что проверяется
---------------
1. **Golden.** Use-case обязан воспроизвести ровно те числа, которые давал прежний
   обработчик: те же ``r``/``q``/множитель и тот же результат калькулятора. Проверка
   сравнивает результат, полученный через use-case, с результатом **прямого** вызова
   ``build_gex_cone`` с параметрами, выписанными по прежней логике ветвления. Так «вынесли
   в use-case» не превращается в «немного поменяли модель».
2. **Множитель контракта.** ``per_contract`` = 1 для крипты и 100 для акций. Ошибка здесь
   масштабирует профиль в 100 раз, не ломая ничего видимого (числа остаются правдоподобными),
   поэтому множитель проверяется отдельно — по каждому источнику.
3. **Модель Блэка.** Для MOEX и индексных фьючерсов ``q = r``: у опциона на фьючерс нет
   дивидендной доходности, её роль играет ставка. Проверяется явно, потому что «q = r»
   легко принять за опечатку и «исправить».
4. **Окно цепочки MOEX.** Обрезка по сроку применяется только к MOEX и не трогает остальные
   источники; окно = ``max(2×horizon, 45)`` дней.
5. **Резолвер ничего не решает молча.** Неизвестный тикер — это акция/ETF с дефолтными
   ставкой и доходностью (историческое поведение), а не ошибка и не «источник по умолчанию»
   с чужими параметрами.

Запуск::

    python tests/test_cone_usecase.py
    pytest tests/test_cone_usecase.py -q
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Skipped(Exception):
    """Проверка требует pandas/scipy (нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_DEPS = True
_IMPORT_ERROR: Exception | None = None
pd = None
build_gex_cone = None
ConeRequest = None
ComputeConeUseCase = None
resolve_source = None
OptionSnapshot = None
build_engine_params = None
stk_all_from_profile = None
GexEngineResult = None
GEXPipelineRunner = None
try:
    import pandas as pd

    from gex.application.cone import (
        ComputeConeUseCase,
        ConeRequest,
        resolve_source,
    )
    from gex.application.gex_engine import (
        GexEngineResult,
        build_engine_params,
        stk_all_from_profile,
    )
    from gex.application.pipeline_runner import GEXPipelineRunner
    from gex.domain.data_loader import OptionSnapshot
    from gex.domain.gexcone import build_gex_cone
except ImportError as exc:  # pandas/scipy отсутствуют
    _HAS_DEPS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_DEPS:
        raise Skipped(f"нужны pandas/scipy ({_IMPORT_ERROR})")


# ====================================================================== #
#  Синтетическая цепочка
# ====================================================================== #
def chain():
    """Цепочка с двумя выраженными страйками по OI и тремя сроками."""
    rows = []
    for T in (0.04, 0.10, 0.30):
        for strike in range(90, 111, 5):
            for kind in ("C", "P"):
                rows.append({
                    "strike": float(strike),
                    "type": kind,
                    "oi": float(8000 if strike in (100, 105) else 500),
                    "iv": 0.22,
                    "T": T,
                })
    return pd.DataFrame(rows)


def snapshot(name="TEST", spot=100.0):
    return OptionSnapshot(
        symbol=name, spot=spot, as_of=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
        chain=chain(),
    )


def engine_for(symbol, snap, source_name, days=14):
    """Канонический прогон GEX-движка — как у роутера (profile_provider).

    Тот же путь, что на проде: параметры источника → run_gex_profile_domain
    (filter по days → auto-params → pipeline.run) → GexEngineResult.
    """
    params = build_engine_params(symbol, snap.spot, source_name)
    runner = GEXPipelineRunner(direction_provider=None, summary_provider=None)
    engine = runner.run_gex_profile_domain(snap, days, params.pipeline)
    return GexEngineResult(
        profile=engine.profile,
        snapshot=engine.snapshot,
        r=params.r,
        q=params.q,
        horizon_years=engine.horizon_years,
        atm_vol=engine.atm_vol,
        stk_all=stk_all_from_profile(engine.profile),
    )


def use_case(*, hv=0.25, atr=3.5, seen=None):
    """Use-case с подставными цепочкой и статистикой; ``seen`` собирает вызовы фетчера."""
    calls = seen if seen is not None else []

    def fetch(symbol, source, expiries):
        calls.append({"symbol": symbol, "source": source.name, "expiries": expiries})
        return snapshot(symbol)

    return ComputeConeUseCase(
        fetch_snapshot=fetch,
        vol_stats=lambda symbol: (hv, atr),
        profile_provider=lambda symbol, snap, source_name: engine_for(
            symbol, snap, source_name, days=REQUEST["horizon_days"],
        ),
        build=build_gex_cone,
    )


REQUEST = dict(ticker="TEST", expiries=3, horizon_days=14, wall_decay=2.0, top_oi=2, oi_quantile=0.9)


def cone_numbers(cone) -> dict:
    """Числовая подпись конуса: всё, что видит пользователь, кроме меток времени.

    Сравниваются и **вероятности по страйкам** (``probs``): конус — это прежде всего
    лестница вероятностей, и golden по одним метаданным пропустил бы изменение модели,
    из-за которого числа в таблице поехали, а ``spot``/``iv_atm`` остались теми же.
    """
    return {
        "spot": round(float(cone.spot), 6),
        "r": round(float(cone.r), 6),
        "q": round(float(cone.q), 6),
        "iv_atm": round(float(cone.iv_atm), 6),
        "net_gex": round(float(cone.net_gex), 6),
        "total_ag": round(float(cone.total_ag), 6),
        "gamma_score": round(float(cone.gamma_score), 6),
        "regime": cone.regime,
        "call_wall": cone.call_wall,
        "put_wall": cone.put_wall,
        "gamma_flip": cone.gamma_flip,
        "axis": (cone.axis_min, cone.axis_max),
        "path": [(round(float(p.get("spot", 0)), 6), round(float(p.get("prob", 0)), 8))
                 for p in cone.cone_path],
        "global_levels": [(l.strike, round(float(l.oi), 3)) for l in cone.levels],
        "expiries": [
            (e.date, round(float(e.iv_atm), 6), round(float(e.gex_net), 6),
             round(float(e.ag), 6), round(float(e.vol_gex), 6))
            for e in cone.expirations
        ],
        # per-strike: главная часть «золота» — страйки, их GEX/AG и вероятности
        "per_strike": [
            (
                e.date, l.strike, l.kind, l.side,
                round(float(l.gex_net), 6), round(float(l.ag), 6),
                round(float(l.strength), 6),
                tuple(
                    (
                        round(float(pr.get("p_above") or 0), 8),
                        round(float(pr.get("p_below") or 0), 8),
                        round(float(pr.get("p_touch") or 0), 8),
                    )
                    for pr in l.probs
                ),
            )
            for e in cone.expirations for l in e.levels
        ],
    }


# ====================================================================== #
#  Резолвер источника: чистые проверки
# ====================================================================== #
def test_resolver_identifies_each_source():
    _require()
    assert resolve_source("BTC").name == "crypto"
    assert resolve_source("btc ").name == "crypto", "тикер не нормализован"
    assert resolve_source("SBER").name == "moex"
    assert resolve_source("ES").name == "futures"
    assert resolve_source("AAPL").name == "equity"
    assert resolve_source("SPY").name == "equity", "ETF — не фьючерс, хотя лежит в DEFAULT_ASSETS"


def test_contract_multiplier_is_per_source():
    """per_contract = 1 для крипты, 100 для акций: ошибка дала бы ×100 в GEX."""
    _require()
    assert resolve_source("BTC").per_contract == 1, "1 контракт Bybit = 1 монета"
    assert resolve_source("SBER").per_contract == 100
    assert resolve_source("ES").per_contract == 100
    assert resolve_source("AAPL").per_contract == 100


def test_futures_use_black_model_q_equals_r():
    """Опцион на фьючерс: q = r. Это модель, а не опечатка — и не дефолт «по невнимательности»."""
    _require()
    moex = resolve_source("SBER")
    assert moex.q == moex.r, f"MOEX: q={moex.q} при r={moex.r} — форвард получит дивидендный снос"
    es = resolve_source("ES")
    assert es.q == es.r, f"фьючерс: q={es.q} при r={es.r}"
    # Контроль: у акции q — это доходность, и она НЕ равна ставке.
    aapl = resolve_source("AAPL")
    assert aapl.q == 0.0 and aapl.r != 0.0, "у акции q подменилось ставкой"


def test_equity_without_config_gets_defaults_not_an_error():
    """Незнакомый тикер — акция с дефолтными ставкой и доходностью (историческое поведение)."""
    _require()
    src = resolve_source("ZZZZ")
    assert src.name == "equity" and src.r == 0.045 and src.q == 0.0


def test_moex_window_is_a_policy_of_the_source():
    """Окно обрезки задано только у MOEX и растёт с горизонтом (минимум 45 дней)."""
    _require()
    assert resolve_source("AAPL").chain_window_days is None, "акции обрезать нечем"
    assert resolve_source("SBER", horizon_days=14).chain_window_days == 45.0, "минимум 45 дней"
    assert resolve_source("SBER", horizon_days=30).chain_window_days == 60.0, "2×горизонт"


# ====================================================================== #
#  Golden: числа не должны измениться
# ====================================================================== #
def test_golden_equity_matches_direct_call():
    """Golden: результат через use-case равен прямому вызову с каноническим движком."""
    _require()
    case = use_case(hv=0.25, atr=3.5)
    got = case.execute(ConeRequest(**REQUEST))

    # ровно то, что делает use-case: канонический профиль движка (equity) → build.
    eng = engine_for("TEST", snapshot("TEST"), "equity", days=14)
    expected = build_gex_cone(
        eng.snapshot, r=eng.r, q=eng.q, atm_vol=eng.atm_vol,
        wall_decay=2.0, max_expiries=3, horizon_days=14,
        top_oi_per_expiry=2, oi_quantile=0.9, hv=0.25, atr=3.5,
        profile=eng.profile, stk_all=eng.stk_all,
    )
    assert cone_numbers(got) == cone_numbers(expected), "числа конуса разошлись с прямым вызовом"


def test_golden_crypto_matches_direct_call():
    """Golden для крипты: множитель 1 и ставки из таблицы активов (движок, как у /crypto/gex)."""
    _require()
    case = use_case()
    got = case.execute(ConeRequest(**{**REQUEST, "ticker": "BTC"}))

    eng = engine_for("BTC", snapshot("BTC"), "crypto", days=14)
    expected = build_gex_cone(
        eng.snapshot, r=eng.r, q=eng.q, atm_vol=eng.atm_vol,
        wall_decay=2.0, max_expiries=3, horizon_days=14,
        top_oi_per_expiry=2, oi_quantile=0.9, hv=0.25, atr=3.5,
        profile=eng.profile, stk_all=eng.stk_all,
    )
    assert cone_numbers(got) == cone_numbers(expected)


def test_golden_moex_matches_direct_call_and_trims_chain():
    """Golden для MOEX: q=r, множитель 100 и цепочка, обрезанная окном источника."""
    _require()
    case = use_case()
    got = case.execute(ConeRequest(**{**REQUEST, "ticker": "SBER"}))

    snap = snapshot("SBER")
    trimmed = snap.chain[snap.chain["T"] <= 45.0 / 365.0].reset_index(drop=True)
    from dataclasses import replace

    eng = engine_for("SBER", replace(snap, chain=trimmed), "moex", days=14)
    expected = build_gex_cone(
        eng.snapshot, r=eng.r, q=eng.q, atm_vol=eng.atm_vol,
        wall_decay=2.0, max_expiries=3, horizon_days=14,
        top_oi_per_expiry=2, oi_quantile=0.9, hv=0.25, atr=3.5,
        profile=eng.profile, stk_all=eng.stk_all,
    )
    assert cone_numbers(got) == cone_numbers(expected)


def test_crypto_and_equity_cones_differ_by_multiplier():
    """Один и тот же вход, разные источники ⇒ разница ровно в множителе контракта (×100).

    Проверка защищает от «причесали параметры и случайно выровняли крипту с акциями»:
    конус крипты, посчитанный с множителем 100, отличался бы от правильного в 100 раз.
    """
    _require()
    crypto = use_case().execute(ConeRequest(**{**REQUEST, "ticker": "BTC"}))

    # «Неправильный» движок: equity-пайплайн (per_contract=100) на крипто-цепочке.
    wrong_params = build_engine_params("BTC", snapshot("BTC").spot, "equity")
    wrong_runner = GEXPipelineRunner(direction_provider=None, summary_provider=None)
    wrong_engine = wrong_runner.run_gex_profile_domain(snapshot("BTC"), 14, wrong_params.pipeline)
    wrong = build_gex_cone(
        wrong_engine.snapshot, r=wrong_params.r, q=wrong_params.q, atm_vol=wrong_engine.atm_vol,
        wall_decay=2.0, max_expiries=3, horizon_days=14,
        top_oi_per_expiry=2, oi_quantile=0.9, hv=0.25, atr=3.5,
        profile=wrong_engine.profile, stk_all=stk_all_from_profile(wrong_engine.profile),
    )
    assert abs(float(crypto.total_ag)) > 0, "нулевой AG — сравнение ничего не значит"
    ratio = abs(float(wrong.total_ag)) / abs(float(crypto.total_ag))
    assert abs(ratio - 100.0) < 0.5, f"множитель контракта не 1: отношение {ratio:.1f}"


# ====================================================================== #
#  Источник решает use-case, а не фетчер
# ====================================================================== #
def test_fetcher_receives_the_resolved_source():
    """Фетчеру передаётся уже готовое описание источника — он не решает это второй раз."""
    _require()
    seen = []
    case = use_case(seen=seen)
    case.execute(ConeRequest(**{**REQUEST, "ticker": "BTC"}))
    case.execute(ConeRequest(**{**REQUEST, "ticker": "SBER"}))
    case.execute(ConeRequest(**{**REQUEST, "ticker": "AAPL"}))

    assert [c["source"] for c in seen] == ["crypto", "moex", "equity"], seen
    assert [c["symbol"] for c in seen] == ["BTC", "SBER", "AAPL"], "тикер ушёл ненормализованным"
    assert all(c["expiries"] == 3 for c in seen), "число экспираций должно дойти до фетчера"


def test_build_receives_exactly_the_source_parameters():
    """В калькулятор уходят канонический профиль движка и параметры источника.

    Через подставные ``profile_provider`` и ``build`` видно ровно то, что
    use-case передаёт дальше, поэтому проверка не зависит от математики конуса.
    """
    _require()
    captured = {}

    def fake_provider(symbol, snap, source_name):
        params = build_engine_params(symbol, snap.spot, source_name)
        from gex.application.gex_engine import GexEngineResult

        return GexEngineResult(
            profile=object(), snapshot=snap,
            r=params.r, q=params.q,
            horizon_years=0.1, atm_vol=0.22,
            stk_all=pd.DataFrame({"strike": [100.0], "gex_net": [1.0], "ag": [1.0]}),
        )

    def fake_build(snap, **kwargs):
        captured.clear()
        captured.update(kwargs)
        captured["symbol"] = snap.symbol
        return "ok"

    case = ComputeConeUseCase(
        fetch_snapshot=lambda symbol, source, expiries: snapshot(symbol),
        vol_stats=lambda symbol: (0.3, 4.0),
        profile_provider=fake_provider,
        build=fake_build,
    )
    case.execute(ConeRequest(**{**REQUEST, "ticker": "BTC"}))
    assert captured["profile"] is not None and "stk_all" in captured, captured
    assert (captured["r"], captured["q"]) == (0.045, 0.0), captured
    assert captured["atm_vol"] == 0.22, "ATM-вола движка не дошла до калькулятора"
    assert captured["hv"] == 0.3 and captured["atr"] == 4.0, "статистика волатильности не дошла"

    case.execute(ConeRequest(**{**REQUEST, "ticker": "SBER"}))
    assert (captured["r"], captured["q"]) == (0.16, 0.16), captured


def test_trim_happens_before_the_calculator():
    """Обрезка — до калькулятора: он не должен видеть сроки вне окна источника."""
    _require()
    seen = {}

    def fake_provider(symbol, snap, source_name):
        return type("Eng", (), {
            "profile": object(), "snapshot": snap,
            "r": 0.045, "q": 0.0, "atm_vol": 0.22,
            "stk_all": pd.DataFrame({"strike": [], "gex_net": [], "ag": []}),
        })()

    def fake_build(snap, **kwargs):
        seen["max_T"] = float(snap.chain["T"].max())
        seen["rows"] = len(snap.chain)
        return "ok"

    case = ComputeConeUseCase(
        fetch_snapshot=lambda symbol, source, expiries: snapshot(symbol),
        vol_stats=lambda symbol: (None, None),
        profile_provider=fake_provider,
        build=fake_build,
    )
    case.execute(ConeRequest(**{**REQUEST, "ticker": "SBER"}))
    assert seen["max_T"] <= 45.0 / 365.0 + 1e-12, f"цепочка не обрезана: max T = {seen['max_T']}"
    assert seen["rows"] < len(chain()), "строки вне окна остались"

    case.execute(ConeRequest(**{**REQUEST, "ticker": "AAPL"}))
    assert seen["rows"] == len(chain()), "акции обрезаны, хотя окна у источника нет"


def test_cache_parts_include_every_result_affecting_parameter():
    """Части ключа кэша — все параметры результата (дефект EC-8: `expiries` не входил)."""
    _require()
    parts = ConeRequest(**REQUEST).cache_parts()
    assert len(parts) == 6, parts
    assert parts[0] == "TEST", "тикер в ключе должен быть нормализован"
    for field, index in (("expiries", 5), ("horizon_days", 1), ("wall_decay", 2),
                         ("top_oi", 3), ("oi_quantile", 4)):
        other = ConeRequest(**{**REQUEST, field: REQUEST[field] + 1}).cache_parts()
        assert other[index] != parts[index], f"{field} не влияет на ключ кэша"


def test_missing_chain_does_not_crash_trimming():
    """Снапшот без колонки T (или без цепочки) не должен ронять обрезку."""
    _require()

    class Snap:
        symbol = "SBER"
        spot = 100.0
        chain = pd.DataFrame({"strike": [100.0]})

    case = ComputeConeUseCase(
        fetch_snapshot=lambda symbol, source, expiries: Snap(),
        vol_stats=lambda symbol: (None, None),
        profile_provider=lambda symbol, snap, source_name: type("Eng", (), {
            "profile": object(), "snapshot": snap,
            "r": 0.0, "q": 0.0, "atm_vol": None,
            "stk_all": pd.DataFrame({"strike": [], "gex_net": [], "ag": []}),
        })(),
        build=lambda snap, **kwargs: "ok",
    )
    assert case.execute(ConeRequest(**{**REQUEST, "ticker": "SBER"})) == "ok"


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
    print(f"--- cone use-case: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
