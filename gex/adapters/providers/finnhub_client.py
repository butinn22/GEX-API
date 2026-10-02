"""Finnhub API client: company profile2 + сверка количества акций с SEC.

Что даёт
--------
``GET /stock/profile2?symbol=`` — базовый профиль компании, включая
``shareOutstanding`` (количество акций в обращении) и
``marketCapitalization``. У Finnhub масштаб значений не фиксирован
(могут быть штуки / тысячи / миллионы / миллиарды) — для сверки с SEC
подбирается масштаб, при котором значение ближе всего к эталону.

Алгоритм сверки (``reconcile_shares``)
---------------------------------------
1. Если SEC не дал акции → берём Finnhub (нормированный);
2. Если SEC дал → сравниваем с Finnhub по относительной разнице,
   перебирая масштабы ×1 / ×10³ / ×10⁶ / ×10⁹ и выбирая ближайший;
3. Если расхождение ≤ tolerance (5%) — источники согласованы (match),
   используем SEC (официальная отчётность точнее);
4. Если расхождение больше — warning с указанием разницы и масштаба,
   значение остаётся от SEC.
"""
from __future__ import annotations

import logging

import requests

from gex.auth.config import settings
from gex.adapters.ratelimit.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"
PROFILE2_PATH = "/stock/profile2"

#: Возможные масштабы значений Finnhub: штуки / тысячи / миллионы / миллиарды
SHARES_SCALES = (1.0, 1e3, 1e6, 1e9)


class FinnhubError(RuntimeError):
    """Ошибка взаимодействия с Finnhub (→ 502 в роутере)."""


# ══════════════════════════════════════════════════════════════════════ #
#  HTTP-клиент
# ══════════════════════════════════════════════════════════════════════ #
def _http_get_json(path: str, params: dict | None = None, timeout: int = 15):
    """GET с токеном, rate-limit'ом и понятными ошибками."""
    if not settings.FINNHUB_API_KEY:
        raise FinnhubError("FINNHUB_API_KEY не задан (см. .env)")
    limiter = get_rate_limiter()
    limiter.wait("finnhub")
    url = FINNHUB_BASE + path
    query = {**(params or {}), "token": settings.FINNHUB_API_KEY}
    try:
        resp = requests.get(url, params=query, timeout=timeout)
    except requests.RequestException as exc:
        raise FinnhubError(f"Finnhub недоступен: {exc}") from exc

    if resp.status_code in (401, 403):
        raise FinnhubError("Finnhub 401/403: проверьте FINNHUB_API_KEY")
    if resp.status_code == 429:
        raise FinnhubError("Finnhub 429: превышен rate limit")
    if resp.status_code == 404:
        raise FinnhubError(f"Finnhub 404: тикер не найден ({path})")
    resp.raise_for_status()
    return resp.json()


def get_company_profile2(symbol: str) -> dict:
    """Профиль компании: name, exchange, marketCapitalization, shareOutstanding…"""
    data = _http_get_json(PROFILE2_PATH, {"symbol": symbol.strip().upper()})
    if not isinstance(data, dict) or not data.get("ticker"):
        raise FinnhubError(f"Finnhub не вернул профиль для {symbol}")
    return data


# ══════════════════════════════════════════════════════════════════════ #
#  Нормировка и сверка количества акций (чистые функции)
# ══════════════════════════════════════════════════════════════════════ #
def best_scale(raw_value: float, reference: float) -> dict:
    """Подобрать масштаб (×1/×10³/×10⁶/×10⁹), при котором raw ближе всего к reference.

    Returns
    -------
    dict
        ``{scale, value, diff}`` — масштаб, нормированное значение,
        относительная разница с reference (0 = идеально).
    """
    best: dict | None = None
    for scale in SHARES_SCALES:
        value = raw_value * scale
        if value <= 0 or not reference:
            continue
        diff = abs(value - reference) / reference
        if best is None or diff < best["diff"]:
            best = {"scale": scale, "value": value, "diff": diff}
    if best is None:
        return {"scale": None, "value": None, "diff": None}
    return best


def reconcile_shares(
    sec_shares: float | None,
    finnhub_shares: float | None,
    tolerance: float = 0.05,
) -> dict:
    """Сверить количество акций SEC vs Finnhub и выбрать источник.

    Parameters
    ----------
    sec_shares : float | None
        Акции из SEC EDGAR (последний баланс, абсолютное число).
    finnhub_shares : float | None
        Сырое значение shareOutstanding из Finnhub (масштаб неизвестен).
    tolerance : float
        Допустимая относительная разница (0.05 = 5%).

    Returns
    -------
    dict
        ``{source, shares, match, finnhub_raw, finnhub_shares, finnhub_scale,
        diff_pct, warning}``.
    """
    if not sec_shares or sec_shares <= 0:
        if not finnhub_shares or finnhub_shares <= 0:
            return {
                "source": None, "shares": None, "match": None,
                "finnhub_raw": finnhub_shares, "finnhub_shares": None,
                "finnhub_scale": None, "diff_pct": None,
                "warning": "Количество акций отсутствует в обоих источниках",
            }
        return {
            "source": "finnhub", "shares": float(finnhub_shares), "match": None,
            "finnhub_raw": finnhub_shares, "finnhub_shares": float(finnhub_shares),
            "finnhub_scale": 1.0, "diff_pct": None,
            "warning": "SEC не предоставил акции — использованы данные Finnhub (масштаб как в ответе)",
        }

    if not finnhub_shares or finnhub_shares <= 0:
        return {
            "source": "sec", "shares": float(sec_shares), "match": None,
            "finnhub_raw": None, "finnhub_shares": None,
            "finnhub_scale": None, "diff_pct": None,
            "warning": "Finnhub недоступен — использованы данные SEC",
        }

    best = best_scale(float(finnhub_shares), float(sec_shares))
    if best["value"] is None:
        return {
            "source": "sec", "shares": float(sec_shares), "match": None,
            "finnhub_raw": finnhub_shares, "finnhub_shares": None,
            "finnhub_scale": None, "diff_pct": None,
            "warning": "Не удалось сопоставить масштаб акций Finnhub с SEC",
        }

    match = best["diff"] <= tolerance
    warning = None
    if not match:
        warning = (
            f"Расхождение акций: SEC {float(sec_shares):,.0f} vs "
            f"Finnhub {best['value']:,.0f} (×{best['scale']:g}) — "
            f"разница {best['diff']:.1%}"
        )
    return {
        "source": "sec",  # SEC — официальная отчётность, точнее
        "shares": float(sec_shares),
        "match": match,
        "finnhub_raw": finnhub_shares,
        "finnhub_shares": best["value"],
        "finnhub_scale": best["scale"],
        "diff_pct": best["diff"],
        "warning": warning,
    }
