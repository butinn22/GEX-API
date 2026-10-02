"""Конфигурация товарных активов для Commodity Service.

Два режима анализа:
  * **ETF-прокси** — товар имеет ETF с ликвидными опционами в yfinance.
    GEX-анализ строится по опционной цепочке ETF (GLD, SLV, USO, UNG, ...).
  * **Price-action** — опционы недоступны, анализ по OHLCV (тренд + уровни).

────────────────────────────────────────────────────────────────────────────
АУДИТ 2026-09-17: прокси и коэффициенты нормализации
────────────────────────────────────────────────────────────────────────────
Замер (``scripts/probe_commodity_proxies.py`` + проверка отношений цен)
показал три дефекта прежней конфигурации:

1. **Прокси-«акции» вместо прокси-товара.** COPX (медные добытчики) и URA
   (урановые добытчики) — это **корзины акций**, а не трекеры цены товара.
   Пересчёт их страйков в «цену меди/урана» через коэффициент даёт профиль,
   который не имеет смысла: опционы на COPX выражают гамму по акциям
   добытчиков. Заменены на трекеры самого товара:
   ``COPPER: COPX → CPER`` (United States Copper Index Fund, фьючерсы меди) и
   ``URANIUM: URA → U`` (Sprott Physical Uranium Trust, физический уран).

2. **Палладий и платина были отключены напрасно.** У ``PALL`` (Sprott Physical
   Palladium) и ``PPLT`` (abrdn Physical Platinum) есть опционы в yfinance —
   4 и 6 экспираций, OI на ближней 27k и 72k. Оба включены.

3. **Неверная полоса sanity-провокации коэффициента.** Полоса ``[0.2, 5.0]``
   исходила из того, что «1 акция ≈ 1 единица товара». Для трастов на
   драгметаллы это неверно: у GLD акция ≈ 1/11 унции, у PPLT ≈ 1/111.
   Измеренные коэффициенты:
       GC=F/GLD 11.0 · SI=F/SLV 1.12 · CL=F/USO 0.65 · NG=F/UNG 0.28 ·
       HG=F/CPER 0.167 · PA=F/PALL 55.6 · PL=F/PPLT 110.7 · U/U 1.02
   Поэтому полоса задана **на актив** (``ratio_band``); выход за неё означает
   рассинхронизацию, и цепочка НЕ масштабируется (см. ``commodity_fetcher``).

Недоступны (нет ETF с опционами в yfinance):
  * NICKEL — Nickel
"""
from __future__ import annotations

# Товарные активы: { ticker: { yf_symbol, etf_proxy, label, unit, category, has_options } }
COMMODITY_ASSETS: dict[str, dict] = {
    "UKOIL": {
        "yf_symbol": "CL=F",
        "etf_proxy": "USO",
        "label": "WTI Crude Oil",
        "unit": "$/bbl",
        "category": "energy",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [0.30, 1.50],   # измерено CL=F/USO = 0.65
    },
    "GOLD": {
        "yf_symbol": "GC=F",
        "etf_proxy": "GLD",
        "label": "Gold",
        "unit": "$/oz",
        "category": "precious",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [5.0, 20.0],    # измерено GC=F/GLD = 11.0 (акция ≈ 1/11 унции)
    },
    "SILVER": {
        "yf_symbol": "SI=F",
        "etf_proxy": "SLV",
        "label": "Silver",
        "unit": "$/oz",
        "category": "precious",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [0.50, 2.00],   # измерено SI=F/SLV = 1.12
    },
    "NATGAS": {
        "yf_symbol": "NG=F",
        "etf_proxy": "UNG",
        "label": "Natural Gas (Henry Hub)",
        "unit": "$/MMBtu",
        "category": "energy",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [0.10, 0.60],   # измерено NG=F/UNG = 0.28
    },
    "COPPER": {
        # CPER — трекер фьючерсов меди (был COPX — корзина акций добытчиков).
        "yf_symbol": "HG=F",
        "etf_proxy": "CPER",
        "label": "Copper",
        "unit": "$/lb",
        "category": "industrial",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [0.08, 0.35],   # измерено HG=F/CPER = 0.167
    },
    "PALLAD": {
        "yf_symbol": "PA=F",
        "etf_proxy": "PALL",          # Sprott Physical Palladium — опционы есть
        "label": "Palladium",
        "unit": "$/oz",
        "category": "precious",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [30.0, 90.0],   # измерено PA=F/PALL = 55.6
    },
    "PLAT": {
        "yf_symbol": "PL=F",
        "etf_proxy": "PPLT",          # abrdn Physical Platinum — опционы есть
        "label": "Platinum",
        "unit": "$/oz",
        "category": "precious",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [60.0, 180.0],  # измерено PL=F/PPLT = 110.7
    },
    "URANIUM": {
        # U — Sprott Physical Uranium Trust (физический уран);
        # раньше был URA — корзина акций урановых добытчиков.
        "yf_symbol": "U",
        "etf_proxy": "U",
        "label": "Uranium (physical trust)",
        "unit": "$",
        "category": "energy",
        "has_options": True,
        "r": 0.045,
        "q": 0.0,
        "ratio_band": [0.70, 1.40],   # база = тот же траст, коэффициент ≈ 1.0
    },
    "NICKEL": {
        "yf_symbol": "NIKL",
        "etf_proxy": None,
        "label": "Nickel",
        "unit": "$",
        "category": "industrial",
        "has_options": False,
        "r": 0.045,
        "q": 0.0,
    },
}

COMMODITY_TICKERS: list[str] = list(COMMODITY_ASSETS.keys())

# Только тикеры с доступными опционами (для GEX-анализа)
COMMODITY_WITH_OPTIONS: list[str] = [
    t for t, cfg in COMMODITY_ASSETS.items() if cfg.get("has_options")
]

#: Исторический псевдоним: ключ "UKOIL" исторически обозначал Brent, но его
#: прокси USO отслеживает **WTI** (аудит 2026-09-17). Ключ остаётся стабильным
#: (он фигурирует в URL/пресетах и фоновом фетчере), а "WTI" резолвится в него.
COMMODITY_ALIASES: dict[str, str] = {"WTI": "UKOIL"}


def resolve_commodity(asset: str) -> str:
    """Канонизировать имя товара, применяя псевдонимы (``WTI`` → ``UKOIL``).

    Используется на точках входа сервисов/фетчера, чтобы ``/commodity/*/WTI``
    работал без добавления псевдонима в ``COMMODITY_TICKERS`` (иначе композит
    товарной динамики посчитал бы нефть дважды).
    """
    key = (asset or "").strip().upper()
    return COMMODITY_ALIASES.get(key, key)
