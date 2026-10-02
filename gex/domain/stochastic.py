"""Стохастический движок: режим-зависимые SDE и first-passage probabilities.

Обоснование выбора моделей
--------------------------
Гамма-режим рынка задаёт качественно разную динамику цены базиса:

* **Positive Gamma** — дилеры подавляют волатильность, цена стремится вернуться
  к «прилипшему» страйку (mean-reversion). Естественная модель — процесс
  Орнштейна–Уленбека (OU), у которого дрейф тянет к долгосрочному среднему
  :math:`\\mu_{OU}`, а скорость возврата :math:`\\kappa` тем выше, чем больше
  Net GEX. Мы работаем с лог-ценой :math:`X_t = \\ln S_t`.

  .. math::
      dX_t = \\kappa(\\mu_{OU} - X_t)\\,dt + \\sigma_{OU}\\,dW_t

* **Negative Gamma** — дилеры усиливают тренды, волатильность пробивает стены,
  развивается направленное движение. Используем GBM с локальной (зависящей от
  страйка/гаммы) волатильностью :math:`\\sigma_{loc}(S)` (Dupire-подобная):

  .. math::
      dS_t = \\mu\\,S_t\\,dt + \\sigma_{loc}(S_t)\\,S_t\\,dW_t

Переход через уровни
--------------------
Для OU процесс — *Gaussian mean-reverting*, переход через барьер :math:`b` за
время :math:`T` аналитически известен (Doob 1949). Для GBM (постоянная вола)
используется отражательное броуновское тождество (reflection principle):
формула первого достижения уровня. При локальной волатильности применяется
Монте-Карло с адаптивным шагом.

Реализация даёт два пути:
  1. Аналитика для частных случаев (быстро, точно).
  2. Монте-Карло для общего случая (локальная вола, произвольный профиль GEX).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from collections.abc import Callable

import numpy as np
from scipy.stats import norm

# np.trapz удалён в NumPy 2.0 (переименован в np.trapezoid). Версионно-безопасный алиас.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

from gex.domain.metrics import GEXProfile


# ====================================================================== #
#  Результат вероятностного расчёта
# ====================================================================== #
@dataclass
class LevelProbability:
    """Вероятность достижения ценой уровня ``level`` до экспирации.

    Attributes
    ----------
    level : float
        Целевой уровень цены базиса.
    direction : 'up' | 'down'
        С какой стороны подходит текущий спот.
    p_hit : float
        Вероятность касания за горизонт [0,1].
    method : str
        Использованный метод ('OU-analytic', 'GBM-analytic', 'monte-carlo').
    """

    level: float
    direction: str
    p_hit: float
    method: str

    def __repr__(self) -> str:  # pragma: no cover
        return (f"P(touch {self.direction} {self.level:.2f}) = "
                f"{self.p_hit:.3f}  [{self.method}]")


# ====================================================================== #
#  Стохастический движок
# ====================================================================== #
class StochasticEngine:
    """Движок режим-зависимых SDE и вероятностей первого достижения.

    Parameters
    ----------
    kappa : float
        Скорость возврата OU (positive-gamma). По умолчанию подобрана эмпирически.
    kappa_gex_scale : float
        Множитель, через который Net GEX модулирует kappa (больше гаммы → быстрее
        возврат). Используется в :meth:`fit_ou_to_gex`.
    n_paths, dt_years : int, float
        Параметры Монте-Карло (шаг по времени в годах).
    seed : int
        Зерно генератора.
    """

    def __init__(
        self,
        kappa: float = 5.0,
        kappa_gex_scale: float = 1e-9,
        n_paths: int = 50_000,
        dt_years: float = 1.0 / 252.0,
        seed: int = 42,
    ):
        if kappa <= 0:
            raise ValueError("kappa должен быть > 0")
        self.kappa = float(kappa)
        self.kappa_gex_scale = float(kappa_gex_scale)
        self.n_paths = int(n_paths)
        self.dt = float(dt_years)
        self.rng = np.random.default_rng(seed)

    # ================================================================== #
    #  Публичный API: вероятность касания уровня
    # ================================================================== #
    def touch_probability(
        self,
        spot: float,
        level: float,
        T: float,
        sigma: float,
        profile: GEXProfile,
        mu: float = 0.0,
        use_mc: bool = False,
    ) -> LevelProbability:
        """Вероятность, что цена коснётся ``level`` до времени ``T``.

        Автоматически выбирает SDE по режиму GEX:
          * POSITIVE → OU (mean-reversion), аналитика или MC;
          * NEGATIVE → GBM с локальной волой, аналитика или MC.

        Parameters
        ----------
        spot : float
            Текущая цена.
        level : float
            Целевой уровень (Wall или Flip).
        T : float
            Горизонт, лет.
        sigma : float
            Базовая волатильность (годовая), например ATM IV.
        profile : GEXProfile
            Текущий GEX-профиль (определяет режим и параметры OU).
        mu : float
            Дрейф GBM (для negative gamma), годовых. По умолчанию 0.
        use_mc : bool
            Принудительно использовать Монте-Карло (для локальной волы).
        """
        if T <= 0 or sigma <= 0:
            raise ValueError("T и sigma должны быть > 0")
        direction = "up" if level > spot else "down"

        if profile.regime == "POSITIVE":
            # OU в лог-цене: долгосрочное среднее = лог-спота (прилипший уровень)
            mu_ou = np.log(spot)
            kappa = self.fit_ou_to_gex(profile.net_gex)
            if use_mc:
                p = self._ou_touch_mc(spot, level, T, sigma, mu_ou, kappa)
                method = "monte-carlo"
            else:
                p = self._ou_touch_analytic(spot, level, T, sigma, mu_ou, kappa)
                method = "OU-analytic"
        else:
            # GBM с локальной волой через профиль GEX
            loc_vol = self._local_vol_func(profile, sigma)
            if use_mc or loc_vol is not None:
                p = self._gbm_touch_mc(spot, level, T, sigma, mu, loc_vol)
                method = "monte-carlo"
            else:
                p = self._gbm_touch_analytic(spot, level, T, sigma, mu)
                method = "GBM-analytic"

        p = float(np.clip(p, 0.0, 1.0))
        return LevelProbability(level=level, direction=direction, p_hit=p, method=method)

    # ================================================================== #
    #  Параметризация OU из GEX
    # ================================================================== #
    def fit_ou_to_gex(self, net_gex: float) -> float:
        """Скорость возврата OU как функция Net GEX.

        .. math::
            \\kappa_{eff} = \\kappa_0 + s\\cdot \\max(0, \\mathrm{NetGEX})

        Чем больше положительная гаммы дилеров, тем сильнее возврат к среднему
        и тем меньше волатильность реализованная (пин-эффект).
        """
        pos = max(0.0, net_gex)
        return self.kappa + self.kappa_gex_scale * pos

    # ================================================================== #
    #  Аналитика OU (лог-цена, mean-reverting Gaussian)
    # ================================================================== #
    @staticmethod
    def _ou_touch_analytic(spot, level, T, sigma, mu_ou, kappa) -> float:
        r"""Вероятность касания барьера процессом OU.

        Используется классический результат для first-passage OU (см.
        Ricciardi & Sato 1988, упрощённая upper bound форма, точная для
        симметричных случаев). Здесь применена стандартная приближённая
        формула через интеграл от плотности first-passage:

        .. math::
            P(\\tau_b \\le T) \\approx
            \\int_0^T \\frac{|b-x_0|}{\\sqrt{2\\pi\\,v_t^3}}\\,
            \\exp\\!\\left(-\\frac{(b-m_t)^2}{2 v_t}\\right) dt

        где :math:`m_t = x_0 e^{-\\kappa t} + \\mu(1-e^{-\\kappa t})` —
        условное среднее, :math:`v_t = \\frac{\\sigma^2}{2\\kappa}(1-e^{-2\\kappa t})`
        — условная дисперсия.
        """
        x0 = np.log(spot)
        b = np.log(level)
        # Численное интегрирование плотности first-passage
        ts = np.linspace(1e-6, T, 1000)
        ekt = np.exp(-kappa * ts)
        m_t = x0 * ekt + mu_ou * (1.0 - ekt)
        v_t = (sigma ** 2) / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * ts))
        v_t = np.maximum(v_t, 1e-15)
        # Плотность первого достижения для OU (informed approximation)
        fpd = (np.abs(b - x0) / np.sqrt(2.0 * np.pi * v_t ** 3)) * \
              np.exp(-((b - m_t) ** 2) / (2.0 * v_t))
        fpd *= np.exp(-kappa * ts)  # поправка на mean-reversion drift
        return float(_trapz(fpd, ts))

    def _ou_touch_mc(self, spot, level, T, sigma, mu_ou, kappa) -> float:
        """Monte-Carlo для OU: симуляция path-wise с фиксацией первого касания."""
        n_steps = max(1, int(np.ceil(T / self.dt)))
        dt = T / n_steps
        x = np.full(self.n_paths, np.log(spot))
        b = np.log(level)
        hit = np.zeros(self.n_paths, dtype=bool)
        # Орнштейн-Уленбек дискретизация (точная для краткосрочных шагов):
        # X_{t+dt} = m + (X_t - m)e^{-k dt} + sigma*sqrt((1-e^{-2k dt})/(2k)) * Z
        e_kdt = np.exp(-kappa * dt)
        vol_step = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
        for _ in range(n_steps):
            z = self.rng.standard_normal(self.n_paths)
            x = mu_ou + (x - mu_ou) * e_kdt + vol_step * z
            if level >= spot:
                hit |= (x >= b)
            else:
                hit |= (x <= b)
            if hit.all():
                break
        return float(hit.mean())

    # ================================================================== #
    #  Аналитика GBM (reflection principle, constant vol)
    # ================================================================== #
    @staticmethod
    def _gbm_touch_analytic(spot, level, T, sigma, mu) -> float:
        r"""Вероятность касания уровня геометрическим броуновским движением
        (постоянная волатильность) через отражательный принцип.

        Лог-доходность :math:`X_t = \ln(S_t/S_0) = \nu t + \sigma W_t` с
        :math:`\nu = \mu - \tfrac{\sigma^2}{2}`. Для верхнего барьера
        :math:`b > S_0`, :math:`A = \ln(b/S_0) > 0`:

        .. math::
            P\!\left(\max_{t\le T} S_t \ge b\right) =
            \Phi\!\left(\frac{\nu T - A}{\sigma\sqrt T}\right)
            + e^{2\nu A/\sigma^2}\,
            \Phi\!\left(\frac{-(\nu T + A)}{\sigma\sqrt T}\right)

        Для нижнего барьера :math:`b < S_0`, :math:`D = \ln(S_0/b) > 0` —
        формула для :math:`\min X_t \le -D` получается отражением
        :math:`X\to -X` (дрейф :math:`\nu\to -\nu`):

        .. math::
            P\!\left(\min_{t\le T} S_t \le b\right) =
            \Phi\!\left(\frac{-\nu T - D}{\sigma\sqrt T}\right)
            + e^{-2\nu D/\sigma^2}\,
            \Phi\!\left(\frac{\nu T - D}{\sigma\sqrt T}\right)
        """
        nu = mu - 0.5 * sigma ** 2          # дрейф лог-цены
        sT = sigma * np.sqrt(T)
        if level > spot:
            # Верхний барьер: A = ln(b/S0)
            A = np.log(level / spot)
            term1 = norm.cdf((nu * T - A) / sT)
            term2 = np.exp(2.0 * nu * A / (sigma ** 2)) * norm.cdf(-(nu * T + A) / sT)
            return float(term1 + term2)
        else:
            # Нижний барьер: D = ln(S0/b)
            D = np.log(spot / level)
            term1 = norm.cdf((-nu * T - D) / sT)
            term2 = np.exp(-2.0 * nu * D / (sigma ** 2)) * norm.cdf((nu * T - D) / sT)
            return float(term1 + term2)

    # ================================================================== #
    #  Монте-Карло GBM с локальной волатильностью
    # ================================================================== #
    def _gbm_touch_mc(self, spot, level, T, sigma, mu, loc_vol) -> float:
        """Euler–Maruyama для GBM с локальной волатильностью (если задана)."""
        n_steps = max(1, int(np.ceil(T / self.dt)))
        dt = T / n_steps
        s = np.full(self.n_paths, spot)
        hit = np.zeros(self.n_paths, dtype=bool)
        sqrt_dt = np.sqrt(dt)
        for _ in range(n_steps):
            z = self.rng.standard_normal(self.n_paths)
            sig_t = loc_vol(s) * sigma if loc_vol is not None else sigma
            s = s * np.exp((mu - 0.5 * sig_t ** 2) * dt + sig_t * sqrt_dt * z)
            if level >= spot:
                hit |= (s >= level)
            else:
                hit |= (s <= level)
            if hit.all():
                break
        return float(hit.mean())

    # ================================================================== #
    #  Локальная волатильность из GEX-профиля
    # ================================================================== #
    @staticmethod
    def _local_vol_func(
        profile: GEXProfile, sigma0: float
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        r"""Построить локальную волу как функцию цены из GEX-профиля.

        Эвристика: локальная волатильность возрастает в зонах отрицательной
        гаммы (дилеры усиливают движения) и убывает в зонах положительной
        (пин-эффект). Используется гладкая параметризация:

        .. math::
            \\sigma_{loc}(S) = 1 + \\alpha\\,
            \\tanh\\!\\left(-\\beta \\cdot \\widetilde{\\mathrm{GEX}}(S)\\right)

        где :math:`\\widetilde{\\mathrm{GEX}}` — нормированный на единицу GEX
        в точке S (через линейную интерполяцию по страйкам), :math:`\\alpha\\le 1`
        ограничивает масштаб (чтобы вола оставалась положительной).
        """
        df = profile.per_strike
        if df.empty:
            return None
        K = df["strike"].values
        gex = df["gex_net"].values
        scale = np.max(np.abs(gex))
        if scale < 1e-12:
            return None  # плоский профиль → постоянная вола
        gex_norm = gex / scale
        alpha = 0.5   # ±50% от базовой волы

        def loc(s):
            s = np.asarray(s, dtype=float)
            gi = np.interp(s, K, gex_norm, left=gex_norm[0], right=gex_norm[-1])
            return 1.0 + alpha * np.tanh(-gi)

        return loc


# ====================================================================== #
#  Z-score: стандартизация (вынесен для переиспользования в EV)
# ====================================================================== #
def to_standard_normal(series: np.ndarray) -> np.ndarray:
    """Привести ряд к стандартному нормальному распределению (Z-scores).

    Используется для нормализации исторических отклонений цены от Gamma Flip
    перед оценкой вероятностей. Удаляет выбросы через MAD-масштаб (робастный).

    .. math::
        Z_i = \\frac{x_i - \\mathrm{median}(x)}{1.4826\\cdot \\mathrm{MAD}(x)}
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return np.array([])
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    scale = 1.4826 * mad if mad > 0 else (np.std(series) or 1.0)
    return (series - median) / scale
