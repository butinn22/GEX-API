"""Unit-тесты товарных активов: WTI-псевдоним и переанкеринг UKOIL → WTI.

Проверяем изменение Phase-4 (аудит 2026-09-17): ключ ``UKOIL`` остаётся
стабильным, но его прокси/метка переносятся на WTI (``CL=F``), а псевдоним
``WTI`` резолвится в ``UKOIL`` на точках входа — без добавления в
``COMMODITY_TICKERS`` (иначе нефть посчиталась бы дважды в композите).
"""
from __future__ import annotations

from gex.commodity_assets import (
    COMMODITY_ASSETS,
    COMMODITY_TICKERS,
    COMMODITY_WITH_OPTIONS,
    resolve_commodity,
)


def test_wti_alias_resolves_to_ukoil():
    assert resolve_commodity("WTI") == "UKOIL"
    assert resolve_commodity("wti") == "UKOIL"
    assert resolve_commodity(" WTI ") == "UKOIL"


def test_ukoil_key_stable():
    assert resolve_commodity("UKOIL") == "UKOIL"
    assert resolve_commodity("ukoil") == "UKOIL"


def test_unknown_asset_passthrough():
    assert resolve_commodity("GOLD") == "GOLD"
    # «BRENT» не псевдоним — проходит как есть (не тихо-переписывается).
    assert resolve_commodity("BRENT") == "BRENT"


def test_ukoil_reanchored_to_wti():
    cfg = COMMODITY_ASSETS["UKOIL"]
    assert cfg["yf_symbol"] == "CL=F"
    assert cfg["label"] == "WTI Crude Oil"
    assert cfg["etf_proxy"] == "USO"
    assert cfg["unit"] == "$/bbl"


def test_wti_not_added_to_tickers():
    # Псевдоним не должен попадать в тикеры/список с опционами.
    assert "WTI" not in COMMODITY_TICKERS
    assert "WTI" not in COMMODITY_WITH_OPTIONS
    assert "UKOIL" in COMMODITY_TICKERS
    assert "UKOIL" in COMMODITY_WITH_OPTIONS
