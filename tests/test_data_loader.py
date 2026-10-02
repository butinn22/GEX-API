"""Unit-тесты GEXDataLoader: нормализация, дедупликация, preserve_expiry.

Проверяем ключевое изменение Phase-3 (аудит 2026-09-17 §6 P0): при
``preserve_expiry=True`` экспирационная размерность сохраняется — два срока
одного (strike, type) остаются отдельными строками, ``T`` между ними не
усредняется. Поведение по умолчанию (``False``) не меняется для прочих
потребителей загрузчика.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from gex.domain.data_loader import GEXDataLoader


def _frame(expiry_days: tuple[float, ...]) -> pd.DataFrame:
    """Две экспирации × один (strike, type): проверочный случай коллапса."""
    rows = []
    for days in expiry_days:
        T = days / 365.0
        rows.append({"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.20, "T": T})
    return pd.DataFrame(rows)


class TestPreserveExpiry:
    """``preserve_expiry=True`` не схлопывает экспирации и не усредняет T."""

    def test_default_collapses_expiries(self):
        """По умолчанию две экспирации одного (strike,type) схлопываются, T = среднее."""
        loader = GEXDataLoader(spot=100.0, symbol="TEST")
        snap = loader.load_dataframe(_frame((30.0, 60.0)))
        assert len(snap.chain) == 1
        # T = mean(30/365, 60/365)
        expected_T = (30.0 / 365.0 + 60.0 / 365.0) / 2.0
        assert math.isclose(float(snap.chain.iloc[0]["T"]), expected_T, rel_tol=1e-9)

    def test_preserve_expiry_keeps_two_rows(self):
        """preserve_expiry=True: две экспирации остаются двумя строками."""
        loader = GEXDataLoader(spot=100.0, symbol="TEST")
        snap = loader.load_dataframe(_frame((30.0, 60.0)), preserve_expiry=True)
        assert len(snap.chain) == 2

    def test_preserve_expiry_does_not_average_t(self):
        """preserve_expiry=True: T каждой строки остаётся своим (не средним)."""
        loader = GEXDataLoader(spot=100.0, symbol="TEST")
        snap = loader.load_dataframe(_frame((30.0, 60.0)), preserve_expiry=True)
        ts = sorted(float(t) for t in snap.chain["T"])
        assert math.isclose(ts[0], 30.0 / 365.0, rel_tol=1e-6)
        assert math.isclose(ts[1], 60.0 / 365.0, rel_tol=1e-6)

    def test_preserve_expiry_sums_oi_only_within_bucket(self):
        """Дубликаты в одном бакете схлопываются; разные бакеты — нет."""
        # 30d и 30.2d попадают в один бакет (round(T*365)==30), 60d — в другой.
        df = pd.DataFrame([
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.20, "T": 30.0 / 365.0},
            {"strike": 100.0, "type": "C", "oi": 5.0, "iv": 0.22, "T": 30.2 / 365.0},
            {"strike": 100.0, "type": "C", "oi": 7.0, "iv": 0.24, "T": 60.0 / 365.0},
        ])
        loader = GEXDataLoader(spot=100.0, symbol="TEST")
        snap = loader.load_dataframe(df, preserve_expiry=True)
        assert len(snap.chain) == 2
        ois = sorted(float(o) for o in snap.chain["oi"])
        assert ois == [7.0, 15.0]  # 30-дневный бакет: 10+5; 60-дневный: 7

    def test_load_csv_accepts_preserve_expiry(self):
        """CSV-вход пробрасывает preserve_expiry в load_dataframe.

        Используем ``tempfile.TemporaryDirectory`` (не pytest tmp_path): на Windows
        cleanup pytest-овского tmp_path падает с PermissionError из-за локов.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as base:
            path = Path(base) / "chain.csv"
            pd.DataFrame([
                {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.2, "T": 30.0 / 365.0},
                {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.2, "T": 60.0 / 365.0},
            ]).to_csv(path, index=False)
            loader = GEXDataLoader(spot=100.0, symbol="TEST")
            snap = loader.load_csv(path, preserve_expiry=True)
            assert len(snap.chain) == 2


class TestInterpolateIv:
    """IV-интерполяция: пропуски заполняются ПО СТРАЙКУ внутри (type, T).

    Покрывает поведение, из-за которого fetcher-ы раньше выбрасывали ~94% цепочки:
    контракт с открытым интересом, но без IV, должен остаться в профиле.
    """

    @staticmethod
    def _loader() -> GEXDataLoader:
        return GEXDataLoader(spot=100.0, symbol="TEST")

    def test_row_without_iv_is_kept_and_filled(self):
        """Контракт с OI > 0 и iv = NaN остаётся и получает IV по соседям."""
        df = pd.DataFrame([
            {"strike": 90.0, "type": "C", "oi": 10.0, "iv": 0.20, "T": 0.02},
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": float("nan"), "T": 0.02},
            {"strike": 110.0, "type": "C", "oi": 10.0, "iv": 0.30, "T": 0.02},
        ])
        snap = self._loader().load_dataframe(df)
        assert len(snap.chain) == 1 + 1 + 1
        got = float(snap.chain[snap.chain["strike"] == 100.0]["iv"].iloc[0])
        assert math.isclose(got, 0.25, abs_tol=1e-9)  # середина между 0.20 и 0.30

    def test_interpolation_is_by_strike_not_by_row_position(self):
        """Сетка неравномерная: 91 ближе к 90, поэтому IV ближе к 0.215, а не к
        середине 0.2075 (что дала бы интерполяция по номеру строки)."""
        df = pd.DataFrame([
            {"strike": 90.0, "type": "C", "oi": 10.0, "iv": 0.215, "T": 0.02},
            {"strike": 91.0, "type": "C", "oi": 10.0, "iv": float("nan"), "T": 0.02},
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.200, "T": 0.02},
        ])
        snap = self._loader().load_dataframe(df)
        got = float(snap.chain[snap.chain["strike"] == 91.0]["iv"].iloc[0])
        assert math.isclose(got, 0.2135, abs_tol=1e-9)
        assert not math.isclose(got, 0.2075, abs_tol=1e-9)

    def test_edge_gap_uses_nearest_known(self):
        """Крайние пропуски (крыло за пределами известных страйков) — nearest."""
        df = pd.DataFrame([
            {"strike": 50.0, "type": "P", "oi": 10.0, "iv": float("nan"), "T": 0.02},
            {"strike": 90.0, "type": "P", "oi": 10.0, "iv": 0.40, "T": 0.02},
            {"strike": 100.0, "type": "P", "oi": 10.0, "iv": 0.30, "T": 0.02},
        ])
        snap = self._loader().load_dataframe(df)
        got = float(snap.chain[snap.chain["strike"] == 50.0]["iv"].iloc[0])
        assert math.isclose(got, 0.40, abs_tol=1e-9)

    def test_groups_are_independent(self):
        """Интерполяция идёт внутри (type, T):put-крыло не портит call-смайл."""
        df = pd.DataFrame([
            {"strike": 90.0, "type": "C", "oi": 10.0, "iv": 0.20, "T": 0.02},
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": float("nan"), "T": 0.02},
            {"strike": 90.0, "type": "P", "oi": 10.0, "iv": 0.80, "T": 0.02},
            {"strike": 100.0, "type": "P", "oi": 10.0, "iv": 0.90, "T": 0.02},
            # другая экспирация: свой T-бакет, своя интерполяция
            {"strike": 90.0, "type": "C", "oi": 10.0, "iv": 0.50, "T": 0.20},
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": float("nan"), "T": 0.20},
        ])
        snap = self._loader().load_dataframe(df, preserve_expiry=True)
        by_key = {
            (float(r["strike"]), r["type"], round(float(r["T"]), 3)): float(r["iv"])
            for _, r in snap.chain.iterrows()
        }
        assert math.isclose(by_key[(100.0, "C", 0.02)], 0.20, abs_tol=1e-9)
        assert math.isclose(by_key[(100.0, "C", 0.2)], 0.50, abs_tol=1e-9)
        assert math.isclose(by_key[(100.0, "P", 0.02)], 0.90, abs_tol=1e-9)

    def test_whole_group_without_iv_is_dropped(self):
        """Если в группе (type, T) нет ни одного известного IV — взять неоткуда,
        строки снимает фильтр iv > 0 (это существующий предохранитель)."""
        df = pd.DataFrame([
            {"strike": 90.0, "type": "C", "oi": 10.0, "iv": float("nan"), "T": 0.02},
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": float("nan"), "T": 0.02},
            {"strike": 90.0, "type": "P", "oi": 10.0, "iv": 0.30, "T": 0.02},
        ])
        snap = self._loader().load_dataframe(df)
        assert len(snap.chain) == 1
        assert snap.chain.iloc[0]["type"] == "P"

    def test_zero_oi_still_dropped(self):
        """Контракт без открытого интереса по-прежнему отбрасывается."""
        df = pd.DataFrame([
            {"strike": 100.0, "type": "C", "oi": 0.0, "iv": 0.30, "T": 0.02},
            {"strike": 100.0, "type": "P", "oi": 10.0, "iv": 0.30, "T": 0.02},
        ])
        snap = self._loader().load_dataframe(df)
        assert len(snap.chain) == 1
        assert snap.chain.iloc[0]["type"] == "P"

    def test_no_nans_left_when_nothing_is_missing(self):
        """Группы без пропусков не «шевелятся»: IV остаются исходными."""
        df = pd.DataFrame([
            {"strike": 90.0, "type": "C", "oi": 10.0, "iv": 0.21, "T": 0.02},
            {"strike": 100.0, "type": "C", "oi": 10.0, "iv": 0.19, "T": 0.02},
        ])
        snap = self._loader().load_dataframe(df)
        assert sorted(float(v) for v in snap.chain["iv"]) == [0.19, 0.21]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
