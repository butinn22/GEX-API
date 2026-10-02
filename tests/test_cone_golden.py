"""Golden GEX-конуса: числа после перевода GEX-ядра на единый движок.

Зачем отдельный эталон
----------------------
``tests/test_cone_usecase.py`` сравнивает результат use-case с **прямым вызовом**
``build_gex_cone``. Пока разбирается сам ``build_gex_cone``, обе стороны сравнения меняются
вместе, и такая проверка становится тавтологией. Здесь числа заморожены в фикстуре.

Сейчас эталон снят с версии, где GEX-ядро конуса (стены, Gamma Flip, режим,
Net GEX, AG) считается ЕДИНСТВЕННЫМ движком ``GEXPipelineRunner.run_gex_profile_domain`` —
тем же, что и главная страница GEX. Если движок изменится, конус обязан измениться
вместе с ним, и эталон перезаписывается осознанно: ``--write-golden``.

Что попадает в эталон
---------------------
Три конфигурации (разные спот, ``wall_decay``, ``top_oi``, ``oi_quantile``), потому что
разбор трогает и расчёт по экспирациям, и путь конуса, и вероятности — а разные параметры
включают разные ветки (например, ``top_oi`` управляет числом уровней на экспирацию).
Для каждой: метаданные, каждая экспирация, **каждый уровень со страйком** (включая лестницу
вероятностей), глобальные уровни и точки пути.

Отдельно проверяется невырожденность: в эталоне должны быть и уровни, и непустая лестница
вероятностей, и точки пути — иначе «всё совпало» означало бы, что считать было нечего.

    python tests/test_cone_golden.py
    python tests/test_cone_golden.py --write-golden
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "cone_golden.json"

NDIGITS = 6


class Skipped(Exception):
    """Нужны pandas/scipy (нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_DEPS = True
_IMPORT_ERROR: Exception | None = None
pd = None
build_gex_cone = None
OptionSnapshot = None
GEXPipelineRunner = None
build_engine_params = None
stk_all_from_profile = None
try:
    import pandas as pd

    from gex.application.gex_engine import build_engine_params, stk_all_from_profile
    from gex.application.pipeline_runner import GEXPipelineRunner
    from gex.domain.data_loader import OptionSnapshot
    from gex.domain.gexcone import build_gex_cone
except ImportError as exc:
    _HAS_DEPS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_DEPS:
        raise Skipped(f"нужны pandas/scipy ({_IMPORT_ERROR})")


def _num(value):
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, NDIGITS)


def chain(spot: float, seed: int = 20260916) -> "pd.DataFrame":
    """Цепочка с улыбкой по IV и асимметричным OI (асимметрия важна: см. итер. 39).

    OI сознательно менее концентрирован, чем в ранних версиях: после перевода
    GEX-ядра на единый движок цепочка сначала режется по days (как на главной
    странице), и на одной экспирации должно оставаться ≥3 страйков с каждой
    стороны спота, иначе 90%-квантиль OI оставляет единственный уровень.
    """
    rnd = random.Random(seed)
    rows = []
    base = int(round(spot))
    for T in (0.02, 0.05, 0.12, 0.30):
        for strike in range(base - 20, base + 21, 5):
            moneyness = abs(strike - spot) / spot
            iv = 0.20 + 1.1 * moneyness ** 2 + rnd.gauss(0, 0.002)
            for kind in ("C", "P"):
                oi = 1500.0 if strike in (base, base + 5) else 500.0 + 30.0 * (20 - abs(strike - spot))
                if kind == "P":
                    oi *= 0.75
                rows.append({"strike": float(strike), "type": kind, "oi": float(oi),
                             "iv": round(max(iv, 0.05), 5), "T": T})
    return pd.DataFrame(rows)


CONFIGS = [
    {"symbol": "TEST1", "spot": 100.0, "opts": dict(expiries=3, horizon_days=14, wall_decay=2.0, top_oi=3, oi_quantile=0.9)},
    {"symbol": "TEST2", "spot": 100.0, "opts": dict(expiries=4, horizon_days=7, wall_decay=0.0, top_oi=5, oi_quantile=0.5)},
    {"symbol": "TEST3", "spot": 52.5, "opts": dict(expiries=2, horizon_days=30, wall_decay=8.0, top_oi=2, oi_quantile=1.0)},
]


def _level(level) -> dict:
    return {
        "strike": _num(level.strike),
        "oi": _num(level.oi),
        "side": level.side,
        "strength": _num(level.strength),
        "gex_net": _num(level.gex_net),
        "ag": _num(level.ag),
        "kind": level.kind,
        "probs": [
            {k: _num(v) for k, v in sorted(pr.items())}
            for pr in (level.probs or [])
        ],
    }


def _expiry(expiry) -> dict:
    out = {}
    for name, value in vars(expiry).items():
        if name == "levels":
            out["levels"] = [_level(x) for x in (value or [])]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = _num(value)
        elif isinstance(value, bool):
            out[name] = value
        elif isinstance(value, (str, type(None))):
            out[name] = value
    return out


def collect() -> dict:
    out = {}
    for cfg in CONFIGS:
        if OptionSnapshot is None:  # pragma: no cover — защита от порядка импортов
            raise Skipped("нет OptionSnapshot")
        snap = OptionSnapshot(
            symbol=cfg["symbol"], spot=cfg["spot"],
            as_of=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
            chain=chain(cfg["spot"]),
        )
        opts = dict(cfg["opts"])
        # Канонический GEX-движок — тот же, что строит профиль главной
        # страницы GEX (equity): конус получает профиль + per-strike кадр.
        params = build_engine_params(snap.symbol, snap.spot, "equity")
        runner = GEXPipelineRunner(direction_provider=None, summary_provider=None)
        engine = runner.run_gex_profile_domain(snap, opts["horizon_days"], params.pipeline)
        cone = build_gex_cone(
            engine.snapshot,
            r=params.r, q=params.q, atm_vol=engine.atm_vol,
            wall_decay=opts["wall_decay"],
            max_expiries=opts["expiries"],
            horizon_days=opts["horizon_days"],
            top_oi_per_expiry=opts["top_oi"],
            oi_quantile=opts["oi_quantile"],
            hv=0.25, atr=3.5,
            profile=engine.profile,
            stk_all=stk_all_from_profile(engine.profile),
        )
        out[cfg["symbol"]] = {
            "opts": opts,
            "meta": {
                "ticker": cone.ticker, "spot": _num(cone.spot), "r": _num(cone.r),
                "q": _num(cone.q), "iv_atm": _num(cone.iv_atm), "regime": cone.regime,
                "net_gex": _num(cone.net_gex), "total_ag": _num(cone.total_ag),
                "gamma_score": _num(cone.gamma_score), "vol_mult": _num(cone.vol_mult),
                "call_wall": _num(cone.call_wall), "put_wall": _num(cone.put_wall),
                "gamma_flip": _num(cone.gamma_flip), "oi_quantile": _num(cone.oi_quantile),
                "hv": _num(cone.hv), "atr": _num(cone.atr),
                "axis": [_num(cone.axis_min), _num(cone.axis_max)],
            },
            "expiries": [_expiry(e) for e in cone.expirations],
            "global_levels": [_level(l) for l in cone.levels],
            "cone_path": [
                {k: _num(v) for k, v in sorted(p.items())} for p in cone.cone_path
            ],
        }
    return out


def _first_difference(a, b, path="") -> str:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                return _first_difference(a.get(key), b.get(key), f"{path}.{key}")
        return f"{path}: словари равны"
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: длина {len(a)} → {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_difference(x, y, f"{path}[{i}]")
        return f"{path}: списки равны"
    return f"{path}: {a!r} → {b!r}"


def test_golden_matches_fixture():
    """Числа конуса совпадают с эталоном, снятым до разбора файла."""
    _require()
    assert FIXTURE.exists(), f"нет эталона: {FIXTURE}"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = collect()
    for cfg in CONFIGS:
        symbol = cfg["symbol"]
        if expected.get(symbol) != got.get(symbol):
            raise AssertionError(f"расхождение в «{symbol}»: "
                                 f"{_first_difference(expected.get(symbol), got.get(symbol))}")


def test_golden_is_not_vacuous():
    """Эталон должен содержать посчитанные уровни, вероятности и путь."""
    _require()
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for symbol, block in expected.items():
        assert block["expiries"], f"{symbol}: экспирации пусты"
        assert block["global_levels"], f"{symbol}: глобальные уровни пусты"
        assert block["cone_path"], f"{symbol}: путь конуса пуст"
        assert block["meta"]["iv_atm"], f"{symbol}: iv_atm не посчитан"

        levels = [lv for e in block["expiries"] for lv in e["levels"]]
        assert levels, f"{symbol}: уровней на экспирациях нет"
        with_probs = [lv for lv in levels if lv["probs"]]
        assert with_probs, f"{symbol}: ни у одного уровня нет лестницы вероятностей"
        p_above = [pr.get("p_above") for lv in with_probs for pr in lv["probs"]]
        assert any(v is not None for v in p_above), f"{symbol}: вероятности не посчитаны"
        strengths = {lv["strength"] for lv in levels}
        assert len(strengths) > 1, f"{symbol}: силы уровней одинаковы ({strengths})"


def test_regenerate_when_asked():
    """`--write-golden` перезаписывает эталон (осознанное действие)."""
    _require()
    if "--write-golden" not in sys.argv:
        _skip_unless_regenerating("эталон перезаписывается только по флагу --write-golden")
    data = collect()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(f"эталон записан: {FIXTURE}")


def _skip_unless_regenerating(message: str) -> None:
    """Пропустить тест так, чтобы это понял и pytest, и собственный раннер.

    Различие обязательное: набор запускается и скриптом, и через pytest. Под pytest нужен
    его собственный ``skip`` (иначе он видит FAILED), вне pytest — наш ``Skipped`` (иначе
    исключение pytest не поймает раннер и запуск скриптом упадёт вместо SKIP).
    Признак «идёт pytest» — сам модуль ``pytest`` в ``sys.modules``: он появляется там
    только когда тесты запускает pytest, а не когда модуль просто установлен.
    """
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(message)
    raise Skipped(message)


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
            print(f"FAIL {fn.__name__}: {str(exc)[:400]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:400]}")
            failed += 1
    print(f"--- cone golden: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
