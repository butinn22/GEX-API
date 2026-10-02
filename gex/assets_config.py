"""Общая конфигурация активов для субсервисов GEX.

Вынесена из ``service.py`` для переиспользования между всеми субсервисами
(``LiveService``, ``MOEXService``, ``CryptoService``, ``VolIndexService``).

Не содержит бизнес-логики — только словари конфигураций.
"""
from __future__ import annotations

# Параметры по умолчанию для основных тикеров (ATM spot + dividends proxy)
DEFAULT_ASSETS: dict[str, dict] = {
    "SPX": {"spot": 7477.0, "q": 0.012},
    "SPY": {"spot": 748.0,  "q": 0.012},
    "QQQ": {"spot": 710.0,  "q": 0.005},
    "IWM": {"spot": 299.0,  "q": 0.010},
    "DIA": {"spot": 529.0,  "q": 0.018},
    "MAGS": {"spot": 70.0,   "q": 0.002},
    "DXY":  {"spot": 104.0,  "q": 0.0, "yf_ticker": "DX-Y.NYB",
              "etf_proxy": "UUP", "note": "DXY Index via UUP ETF options proxy"},
    "ES":   {"spot": 7523.0, "r": 0.045, "q": 0.045, "yf_ticker": "^SPX",
              "per_contract": 100, "note": "E-mini S&P 500 Futures via SPX options proxy (Black model: q=r)"},
    "NQ":   {"spot": 28758.0, "r": 0.045, "q": 0.045, "yf_ticker": "^NDX",
              "per_contract": 100, "note": "E-mini Nasdaq 100 Futures via NDX options proxy (Black model: q=r)"},
}

# MOEX: опционы на фьючерсы RTS/MIX/CNY/Si (модель Блэка, q=r)
MOEX_ASSETS: dict[str, dict] = {
    "RTS": {"r": 0.16, "per_contract": 100},
    "MIX": {"r": 0.16, "per_contract": 100},
    "CNY": {"r": 0.16, "per_contract": 1000},
    "SI":  {"r": 0.16, "per_contract": 1},
}

# Опционы FORTS на фьючерсы российских акций (тикер акции → GEX-анализ).
# Тот же набор, что в moex_fetcher.py: _STOCK_FUTURE_OPTIONS (F-опционы на
# фьючерсы акций) + _STOCK_SPOT_OPTIONS (еженедельные опционы на акции).
# Перед правкой сверять с moex_fetcher.py и живым ISS (ASSETCODE/lot меняются
# при корп. событиях). Включены только активы с реальной ликвидностью
# (проверка OI на ISS 2026-09-07).
for _ticker in (
    "GAZP", "GMKN", "LKOH", "MOEX", "ROSN",
    "SBER", "SNGS", "SNGSP", "TATN", "VKCO", "VTBR",
):
    MOEX_ASSETS.setdefault(_ticker, {"r": 0.16, "per_contract": 100})
# Еженедельные опционы MOEX на сами акции (ASSETCODE 'S').
for _ticker in (
    "AFKS", "AFLT", "ALRS", "CHMF", "IRAO", "MAGN", "MSNG", "NLMK",
    "OZON", "POSI", "RTKM", "RUAL", "SVCB", "YDEX",
):
    MOEX_ASSETS.setdefault(_ticker, {"r": 0.16, "per_contract": 100})
del _ticker

# Индексы волатильности (VIX/VVIX) — cash-settled европейские опционы CBOE.
# Конвенция знаков инвертирована: call_sign=-1, put_sign=+1
VOL_INDEX_ASSETS: dict[str, dict] = {
    "VIX":  {
        "yf_ticker": "^VIX",  "r": 0.045, "q": 0.0, "per_contract": 100,
        "call_sign": -1.0, "put_sign": +1.0,
        "has_chain": True,
        "note": "CBOE VIX, cash-settled European, данные через yfinance",
    },
    "VVIX": {
        "yf_ticker": "^VVIX", "r": 0.045, "q": 0.0, "per_contract": 100,
        "call_sign": -1.0, "put_sign": +1.0,
        "has_chain": False,
        "note": "CBOE VVIX — цепочка опционов недоступна через yfinance",
    },
}


# ====================================================================== #
#  Криптовалюты (Bybit V5)
# ====================================================================== #
#: Параметры опционов на крипту. Жили в ``bybit_fetcher.py`` рядом с HTTP-клиентом,
#: но это **не свойство биржи**, а параметры актива: тот же ``per_contract=1`` нужен
#: и конусу (``gexcone``), и расширенному профилю (``extended``), и сервису крипты.
#: Держать их в модуле с сетевым клиентом означало, что любой потребитель параметров
#: (включая слой приложения) обязан импортировать модуль, который ходит в сеть.
#: ``bybit_fetcher._CRYPTO_ASSETS`` оставлен как псевдоним — это тот же объект.
CRYPTO_ASSETS: dict[str, dict] = {
    "BTC":  {"r": 0.045, "q": 0.0, "per_contract": 1, "call_sign": +1.0, "put_sign": -1.0},
    "ETH":  {"r": 0.045, "q": 0.0, "per_contract": 1, "call_sign": +1.0, "put_sign": -1.0},
    "SOL":  {"r": 0.045, "q": 0.0, "per_contract": 1, "call_sign": +1.0, "put_sign": -1.0},
    "XRP":  {"r": 0.045, "q": 0.0, "per_contract": 1, "call_sign": +1.0, "put_sign": -1.0},
    "DOGE": {"r": 0.045, "q": 0.0, "per_contract": 1, "call_sign": +1.0, "put_sign": -1.0},
}

#: Тикеры, опционы на которые считаются по **модели Блэка** (опцион на фьючерс,
#: ``q = r``): индексные фьючерсы торгуются как прокси через опционы индекса.
#: Признак вынесен отдельно, потому что ``DEFAULT_ASSETS`` содержит и ETF/индексы
#: (SPX/SPY/QQQ), для которых ``q`` — это дивидендная доходность, а не ставка.
FUTURES_TICKERS: frozenset[str] = frozenset({"ES", "NQ"})
