"""Векторизованные греки Блэка-Шоулза-Мертона (BSM).

Все функции принимают либо скаляры, либо numpy-массивы (broadcasting).
Дивидендная доходность ``q`` поддерживается для индексов (SPX) и ETF (SPY).

Обозначения
-----------
S : спот базового актива
K : страйк
T : время до экспирации, лет   (T>0)
r : безрисковая ставка, годовых
q : непрерывная дивидендная доходность, годовых (по умолчанию 0)
sigma : подразумеваемая волатильность, годовых
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def d1d2(S, K, T, r, sigma, q=0.0):
    """Возвращает кортеж (d1, d2) — стандартизованные расстояния Блэка-Шоулза.

    .. math::
        d_1 = \\frac{\\ln(S/K) + (r - q + \\tfrac12\\sigma^2)T}{\\sigma\\sqrt T},
        \\qquad d_2 = d_1 - \\sigma\\sqrt T
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sqrt_T = np.sqrt(T)
    # Защита: T<=0 → греки вырождаются, принудительно зануляем знаменатель-чувствительность.
    safe_T = np.where(T > 0, T, 1.0)
    sigma_sqrt = np.asarray(sigma) * np.sqrt(safe_T)
    sigma_sqrt = np.where(sigma_sqrt > 1e-10, sigma_sqrt, 1e-10)
    d1 = (np.log(S / K) + (r - q + 0.5 * np.asarray(sigma) ** 2) * T) / sigma_sqrt
    d2 = d1 - sigma_sqrt
    return d1, d2


def bs_gamma(S, K, T, r, sigma, q=0.0):
    r"""Гамма BSM: вторая производная по споту.

    .. math::
        \\Gamma = \\frac{e^{-qT}\\, n(d_1)}{S\\,\\sigma\\sqrt T},
        \\quad n(x)=\\frac{1}{\\sqrt{2\\pi}}e^{-x^2/2}
    """
    S = np.asarray(S, dtype=float)
    d1, _ = d1d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(np.where(T > 0, T, 1.0))
    denom = S * np.asarray(sigma) * sqrt_T
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    return np.exp(-q * T) * norm.pdf(d1) / denom


def bs_delta(S, K, T, r, sigma, q=0.0, is_call=True):
    r"""Дельта BSM (sign-aware по типу опциона).

    Для call: :math:`\Delta = e^{-qT} N(d_1)`
    Для put : :math:`\Delta = e^{-qT}(N(d_1)-1)`

    ``is_call`` может быть скаляром или массивом булевых значений (broadcasting
    поддерживается для векторного расчёта по всей цепочке).
    """
    d1, _ = d1d2(S, K, T, r, sigma, q)
    cdf = norm.cdf(d1)
    is_call_arr = np.asarray(is_call)
    if is_call_arr.ndim == 0:
        # Скалярный случай
        return np.exp(-q * T) * cdf if bool(is_call_arr) else np.exp(-q * T) * (cdf - 1.0)
    # Векторный случай: применяем корректировку −1 только к путам
    delta = np.exp(-q * T) * cdf
    delta = np.where(is_call_arr, delta, delta - np.exp(-q * T))
    return delta


def bs_vega(S, K, T, r, sigma, q=0.0):
    r"""Вега BSM (на 1.0 волатильности, не на 1%).

    .. math::
        \\mathcal{V} = S e^{-qT}\\, n(d_1)\\,\\sqrt T
    """
    S = np.asarray(S, dtype=float)
    d1, _ = d1d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(np.where(T > 0, T, 0.0))


# Удобный алиас для Z-плотности нормального распределения (используется в метриках).
def normal_pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / _SQRT_2PI
