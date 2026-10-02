"""Парсинг, загрузка и нормализация опционных цепочек.

Поддерживаются три источника данных:
  1. CSV-файл со столбцами: ``strike,type,oi,iv,T`` (или ``expiry`` — дата экспирации);
  2. ``pandas.DataFrame`` с теми же столбцами;
  3. Синтетический генератор цепочки (для бэктестов/демо).

Ключевые операции предобработки
-------------------------------
* Приведение типов, отбрасывание строк с ``NaN`` в критичных полях
  (``strike``, ``oi``, ``T``). ``iv`` интерполируется по страйкам, если есть
  пропуски (smile-интерполяция).
* Фильтр арбитражных/неверных значений: ``iv>0``, ``oi>=0``, ``T>0``,
  ``strike>0``.
* Очистка дубликатов по ``(strike,type)``: OI суммируется, IV усредняется.
* Пересчёт ``expiry`` (даты) → ``T`` (лет) от ``as_of`` даты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLS = ("strike", "type", "oi", "iv", "T")
OPTION_TYPES = ("C", "P")


@dataclass
class OptionSnapshot:
    """Каноническое представление опционной цепи после очистки.

    Attributes
    ----------
    symbol : str
        Тикер базового актива (SPX, SPY, ...).
    spot : float
        Цена базового актива на момент снапшота.
    as_of : datetime
        Временная метка данных.
    chain : pd.DataFrame
        Очищенная цепочка: columns = ``strike,type,oi,iv,T``.
    meta : dict
        Произвольные метаданные (процент от float, и т.п.).
    """

    symbol: str
    spot: float
    as_of: datetime
    chain: pd.DataFrame
    meta: dict = field(default_factory=dict)


class GEXDataLoader:
    """Загрузчик и очиститель опционных цепочек.

    Parameters
    ----------
    spot : float
        Текущая цена базового актива.
    symbol : str
        Тикер.
    contract_multiplier : int
        Множитель контракта (100 для SPY/QQQ/DIA/IWM; 100 для SPX cash-settled).
    """

    def __init__(self, spot: float, symbol: str = "SPX", contract_multiplier: int = 100):
        if spot <= 0:
            raise ValueError(f"spot должен быть > 0, получено {spot}")
        self.spot = float(spot)
        self.symbol = symbol
        self.contract_multiplier = int(contract_multiplier)

    # ------------------------------------------------------------------ #
    #  Публичные точки входа
    # ------------------------------------------------------------------ #
    def load_csv(
        self, path: str | Path, as_of: Optional[datetime] = None,
        preserve_expiry: bool = False,
    ) -> OptionSnapshot:
        """Загрузить цепочку из CSV. См. заголовок модуля по формату столбцов.

        ``preserve_expiry=True`` включает экспирационный бакет в ключ дедупликации
        (см. :meth:`_aggregate_duplicates`) — разные экспирации одного (strike, type)
        остаются отдельными строками, ``T`` не усредняется между ними.
        """
        df = pd.read_csv(path)
        return self.load_dataframe(df, as_of=as_of, preserve_expiry=preserve_expiry)

    def load_dataframe(
        self, df: pd.DataFrame, as_of: Optional[datetime] = None,
        preserve_expiry: bool = False,
    ) -> OptionSnapshot:
        """Очистить и нормализовать готовый DataFrame.

        ``preserve_expiry=True`` сохраняет экспирационную размерность: дедупликация
        идёт по ``(strike, type, round(T*365))``, поэтому мульти-экспирационные
        цепочки не схлопываются до одного усреднённого ``T`` (важно для
        временного веса ``e^(-T/30)`` в расширенном GEX).
        """
        cleaned = self._clean(df, preserve_expiry=preserve_expiry)
        return OptionSnapshot(
            symbol=self.symbol,
            spot=self.spot,
            as_of=as_of or datetime.now(timezone.utc),
            chain=cleaned,
        )

    # ------------------------------------------------------------------ #
    #  Конвейер очистки
    # ------------------------------------------------------------------ #
    def _clean(self, df: pd.DataFrame, preserve_expiry: bool = False) -> pd.DataFrame:
        df = df.copy()

        # --- приведение типа опциона к каноническому C/P ---
        df["type"] = (
            df["type"].astype(str).str.strip().str.upper().str[0]
        )
        df = df[df["type"].isin(OPTION_TYPES)].copy()

        # --- если задана дата экспирации, переведём её в T (лет) ---
        if "expiry" in df.columns and "T" not in df.columns:
            df["T"] = self._expiry_to_years(df["expiry"])

        # --- проверка наличия обязательных столбцов ---
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"В цепочке отсутствуют столбцы: {missing}")

        # --- числовое приведение (type — категориальный, его не трогаем) ---
        for c in REQUIRED_COLS:
            if c == "type":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # type приводим к каноничному строковому типу, отбрасывая NaN
        df["type"] = df["type"].astype(str).str.strip().str.upper().str[0]
        df = df[df["type"].isin(OPTION_TYPES)].copy()

        # --- интерполяция IV по smile внутри (type, T) если есть пропуски ---
        df = self._interpolate_iv(df)

        # --- отбрасывание строк с NaN в критичных полях ---
        df = df.dropna(subset=REQUIRED_COLS).copy()

        # --- фильтр арбитражных/невозможных значений ---
        df = df[(df["strike"] > 0) & (df["oi"] > 0) & (df["T"] > 0) & (df["iv"] > 0)]

        # --- агрегация дубликатов: OI суммируем, IV взвешенно усредняем ---
        df = self._aggregate_duplicates(df, preserve_expiry=preserve_expiry)

        if df.empty:
            raise ValueError("После очистки цепочка пуста — проверьте входные данные.")

        df = df.reset_index(drop=True)
        return df[["strike", "type", "oi", "iv", "T"]]

    # ------------------------------------------------------------------ #
    #  Вспомогательные методы очистки
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expiry_to_years(expiry: pd.Series) -> pd.Series:
        """Перевод дат экспирации во время до экспирации в годах."""
        exp = pd.to_datetime(expiry, errors="coerce")
        now = pd.Timestamp.now('UTC').normalize()
        days = (exp - now).dt.total_seconds() / 86400.0
        # торговых дней ~ 252, но для греков берём календарное время
        return (days / 365.0).clip(lower=1e-6)

    @staticmethod
    def _interpolate_iv(df: pd.DataFrame) -> pd.DataFrame:
        """Заполнить пропуски IV линейной интерполяцией по СТРАЙКУ в каждой
        группе (type, T). Краевые провалы заполняются ближайшим известным
        значением (эквивалент forward/back-fill).

        Интерполяция идёт именно по значению страйка, а не по номеру строки:
        сетка страйков неравномерна (плотно у денег, редко в крыльях), поэтому
        интерполяция по позиции строки systematic underestimates крыло там, где
        smile крутой. Замер на страйке 91 между известными 90 и 100: по позиции
        строки 0.2075, по страйку 0.2135; для пута с крутым крылом 0.350 против
        0.390 — расхождение ~10% IV, то есть заметная ошибка в греках.

        Группы без пропусков не трогаем: валидные IV не должны «шевелиться».
        """
        if "iv" not in df.columns:
            return df

        df = df.copy()
        out = pd.to_numeric(df["iv"], errors="coerce").astype(float)
        if not out.isna().any():
            return df

        # Группируем по типу и (округлённому) T, чтобы smile был однородным.
        df["_Tkey"] = np.round(df["T"].fillna(-1), 5)
        for _, idx in df.groupby(["type", "_Tkey"], sort=False).groups.items():
            values = out.loc[idx]
            if not values.isna().any():
                continue
            # Сортировка по страйку внутри группы; stable сохраняет исходный
            # порядок для совпадающих страйков.
            order = df.loc[idx, "strike"].to_numpy().argsort(kind="stable")
            x = df.loc[idx, "strike"].to_numpy(dtype=float)[order]
            y = values.to_numpy(dtype=float)[order]
            out.loc[idx[order]] = GEXDataLoader._interp_by_x(x, y)
        df["iv"] = out
        return df.drop(columns="_Tkey")


    @staticmethod
    def _interp_by_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Линейная интерполяция ``y`` по узлам ``x`` с удержанием краёв.

        ``np.interp`` делает ровно то, что нужно smile: внутри диапазона —
        линейно по ``x``, за пределами — крайнее известное значение (nearest).
        """
        known = np.flatnonzero(~np.isnan(y))
        if known.size == 0:
            return y
        if known.size == 1:
            return np.full(y.shape, y[known[0]], dtype=float)
        return np.interp(x, x[known], y[known])

    @staticmethod
    def _aggregate_duplicates(
        df: pd.DataFrame, preserve_expiry: bool = False,
    ) -> pd.DataFrame:
        """Схлопнуть дублирующие ряды: OI → сумма, IV → OI-взвешенное.

        Ключ дедупликации по умолчанию — ``(strike, type)`` (историческое
        поведение, ``T`` усредняется между всеми экспирациями). При
        ``preserve_expiry=True`` в ключ добавляется экспирационный бакет
        ``round(T*365)`` (та же гранулярность, что у ``merge_chains`` /
        ``count_expiries``), поэтому разные экспирации одного (strike, type)
        остаются отдельными строками и ``T`` между ними **не** усредняется
        (внутри бакета значения ``T`` совпадают с точностью до дня).

        Полностью векторизовано и устойчиво к версиям pandas: считаем
        числитель (sum iv*oi) и знаменатель (sum oi) отдельно, затем делим.
        Группы с нулевым OI получают простое среднее IV (fallback).
        """
        df = df.copy()
        if preserve_expiry:
            # Экспирационный бакет в днях — как merge_chains/count_expiries.
            df["_tday"] = (df["T"].astype(float) * 365.0).round().astype("int64")
            keys = ["strike", "type", "_tday"]
        else:
            keys = ["strike", "type"]

        if not df.duplicated(subset=keys).any():
            if preserve_expiry:
                df = df.drop(columns="_tday")
            return df[["strike", "type", "oi", "iv", "T"]]

        grouper = [df[c] for c in keys]
        g = df.groupby(keys, as_index=False, sort=True)
        agg = g.agg(oi=("oi", "sum"), T=("T", "mean"))

        # OI-взвешенное среднее: числитель и знаменатель по тем же группам.
        wiv = (df["iv"] * df["oi"]).groupby(grouper).transform("sum")
        soi = df["oi"].groupby(grouper).transform("sum")
        # Усредняем по строкам внутри группы, затем берём первое значение группы
        # (все строки одной группы дадут одинаковый результат transform).
        df = df.assign(_wiv=wiv, _soi=soi)
        # Если OI в группе = 0, используем невзвешенное среднее IV по группе.
        df["_iv_agg"] = np.where(df["_soi"] > 0, df["_wiv"] / df["_soi"],
                                 df["iv"].groupby(grouper).transform("mean"))
        iv_per_group = df.groupby(keys, as_index=False)["_iv_agg"].first()
        agg = agg.merge(iv_per_group, on=keys)
        agg = agg.rename(columns={"_iv_agg": "iv"})
        return agg[["strike", "type", "oi", "iv", "T"]]

    # ------------------------------------------------------------------ #
    #  Синтетический генератор (для тестов/демо)
    # ------------------------------------------------------------------ #
    def synthetic_chain(
        self,
        strikes: Iterable[float],
        expiry_years: float,
        atm_iv: float = 0.18,
        skew: float = 1.0,
        oi_seed: int = 0,
        n_strikes: Optional[int] = None,
        as_of: Optional[datetime] = None,
    ) -> OptionSnapshot:
        """Сгенерировать синтетическую цепочку с правдоподобным smile и OI.

        Удобно для построения демонстрационных профилей GEX без живых данных.
        """
        rng = np.random.default_rng(oi_seed)
        K = np.atleast_1d(np.asarray(list(strikes), dtype=float))
        if n_strikes is not None:
            K = np.linspace(K.min(), K.max(), n_strikes)

        moneyness = np.log(K / self.spot)
        # Smile: крылья поднимаются, ATM минимум (полиномиальная аппроксимация)
        iv = atm_iv * (1.0 + skew * (moneyness ** 2) + 0.3 * np.abs(moneyness))
        iv = np.clip(iv, 0.02, 5.0)

        rows = []
        for k, v in zip(K, iv):
            # OI концентрируется вокруг ATM и падает в крыльях
            base = np.exp(-0.5 * ((np.log(k / self.spot)) / 0.08) ** 2)
            oi_c = int(base * rng.uniform(5_000, 20_000))
            oi_p = int(base * rng.uniform(5_000, 20_000))
            rows.append((k, "C", oi_c, v, expiry_years))
            rows.append((k, "P", oi_p, v, expiry_years))

        df = pd.DataFrame(rows, columns=list(REQUIRED_COLS))
        return self.load_dataframe(df, as_of=as_of)
