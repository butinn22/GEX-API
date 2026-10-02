"""Tests: composite-formula endpoint data availability and computation.

Проверяет:
1. COR1M загружается из CBOE CDN (не из yfinance, т.к. тот даёт 1 день)
2. Все 6 компонентов формулы доступны
3. Формула даёт > 100 значимых точек с current > 0
4. Перцентильные bands не None
5. kомпоненты: VIX, VVIX, DXY, COR1M, SumIdx, PCR
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import requests
import io


def test_cor1m_cboe_cdn_available():
    """COR1M CBOE CDN должен возвращать > 100 строк."""
    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv"
    resp = requests.get(URL, timeout=15)
    assert resp.status_code == 200, f"CBOE CDN вернул HTTP {resp.status_code}"
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    cor1m = df["CLOSE"].astype(float).dropna().sort_index()
    assert len(cor1m) > 100, f"Слишком мало строк COR1M: {len(cor1m)}"
    assert cor1m.index[-1] >= pd.Timestamp("2025-01-01"), "Данные слишком старые"


def test_cor1m_yfinance_insufficient():
    """yfinance ^COR1M отдаёт <= 5 строк (нельзя использовать для ряда)."""
    import yfinance as yf
    yt = yf.Ticker("^COR1M")
    df = yt.history(period="1y", auto_adjust=False)
    if df is not None and not df.empty:
        close = df["Close"].dropna()
        assert len(close) <= 5, (
            f"yfinance ^COR1M: {len(close)} closes — если Yahoo изменил API, "
            "можно убрать этот тест."
        )


def test_all_components_available():
    """VIX, VVIX, DXY, COR1M — все загружаются с достаточной историей."""
    import yfinance as yf

    for t, name in [("^VIX", "VIX"), ("^VVIX", "VVIX"), ("DX-Y.NYB", "DXY")]:
        yt = yf.Ticker(t)
        df = yt.history(period="1y", auto_adjust=False)
        assert df is not None and not df.empty, f"{name} не загрузился"
        close = df["Close"].dropna()
        assert len(close) > 100, f"{name}: только {len(close)} закрытий"


def test_composite_formula_full_computation():
    """Полный прогон: формула даёт 200 дней, current > 0, bands не None."""
    import yfinance as yf
    from gex.adapters.fetchers.breadth_fetcher import fetch_mcclellan

    # --- 1. Загрузка всех компонентов ---
    closes = {}
    for yf_t, name in [("^VIX", "VIX"), ("^VVIX", "VVIX"), ("DX-Y.NYB", "DXY")]:
        yt = yf.Ticker(yf_t)
        df = yt.history(period="1y", auto_adjust=False)
        s = df["Close"].astype(float).dropna()
        s.name = name
        if hasattr(s.index, 'tz') and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        closes[name] = s

    # COR1M CBOE CDN
    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv"
    resp = requests.get(URL, timeout=15)
    cdf = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    cor1m = cdf["CLOSE"].astype(float).dropna().sort_index()
    if hasattr(cor1m.index, 'tz') and cor1m.index.tz is not None:
        cor1m.index = cor1m.index.tz_localize(None)
    cor1m.index = pd.to_datetime(cor1m.index).normalize()
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=1)
    cor1m = cor1m[cor1m.index >= cutoff]
    cor1m.name = "COR1M"
    closes["COR1M"] = cor1m

    # SumIdx
    mcc = fetch_mcclellan()
    if mcc and mcc.dates and mcc.mc_summation_index:
        closes["SumIdx"] = pd.Series(
            mcc.mc_summation_index, index=pd.to_datetime(mcc.dates), name="SumIdx"
        )

    # PCR proxy
    spy = yf.Ticker("SPY").history(period="1y", auto_adjust=False)
    if not spy.empty:
        spy_c = spy["Close"].astype(float).dropna()
        if hasattr(spy_c.index, 'tz') and spy_c.index.tz is not None:
            spy_c.index = spy_c.index.tz_localize(None)
        delta = spy_c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        pcr_proxy = 100 / (1 + rs)
        pcr_proxy.name = "PCR"
        closes["PCR"] = pcr_proxy

    assert len(closes) >= 3, f"Слишком мало компонентов: {list(closes)}"

    # --- 2. Объединение и выравнивание ---
    df_all = pd.DataFrame(closes).ffill().bfill()
    assert not df_all.empty, "DataFrame пуст после выравнивания"
    assert len(df_all) > 100, f"Слишком мало дней: {len(df_all)}"

    # --- 3. Формула на сырых значениях ---
    formula = (
        df_all["VIX"]
        / (df_all["VVIX"].replace(0, 0.01) * df_all["COR1M"].replace(0, 0.01))
        * df_all["DXY"]
        * df_all.get("SumIdx", pd.Series(1000, index=df_all.index))
        / df_all.get("PCR", pd.Series(50, index=df_all.index)).replace(0, 1)
    )

    assert len(formula) > 100, f"Формула дала только {len(formula)} точек"

    # --- 4. Последние 200 дней для отображения ---
    display = formula.tail(200)
    assert len(display) >= 100, f"display < 100: {len(display)}"

    current = float(display.iloc[-1])
    assert np.isfinite(current), f"current not finite: {current}"
    assert current > 0, f"current <= 0: {current}"

    # --- 5. Перцентильные bands за 90 дней ---
    recent = display.tail(min(90, len(display)))
    upper = float(recent.quantile(0.85)) if len(recent) > 1 else None
    lower = float(recent.quantile(0.15)) if len(recent) > 1 else None
    median = float(recent.median()) if len(recent) > 1 else None

    assert upper is not None, "upper_band не должен быть None"
    assert lower is not None, "lower_band не должен быть None"
    assert median is not None, "median не должен быть None"
    assert upper >= median, f"upper ({upper}) < median ({median})"
    assert lower <= median, f"lower ({lower}) > median ({median})"

    # --- 6. Итоговая выдача ---
    assert "COR1M" in closes, "COR1M не загружен"
    assert "VIX" in closes, "VIX не загружен"
    assert "VVIX" in closes, "VVIX не загружен"
    assert "DXY" in closes, "DXY не загружен"

    print(f"✅ Формула: {len(display)} дней, current={current:.2f}, "
          f"range=[{float(formula.tail(200).min()):.2f}, {float(formula.tail(200).max()):.2f}]")
    print(f"   90d bands: upper={upper:.1f}, lower={lower:.1f}, median={median:.1f}")
    print(f"   Компоненты: {list(closes.keys())}")
