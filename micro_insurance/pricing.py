"""Week 3/4 - Black-Scholes call-option replication of a micro-insurance policy.

Mapping (roadmap section 2.2)

    S       accumulated annual treatment cost (expected claim cost per insured)
    K       policy deductible
    T - t   policy term in years (typically 1)
    r       short-term sovereign risk-free rate (continuously compounded)
    sigma   annualised volatility of the treatment-cost process
    C(S,t)  pure micro-insurance premium

    C  = S N(d1) - K exp(-r tau) N(d2)
    d1 = [ln(S/K) + (r + sigma^2/2) tau] / (sigma sqrt(tau)),  d2 = d1 - sigma sqrt(tau)

Greeks
    Delta = N(d1)                     dC/dS
    Gamma = n(d1) / (S sigma sqrt(tau)) d2C/dS2
    Vega  = S n(d1) sqrt(tau)         dC/dsigma (per 1.00 of vol)
    Rho   = K tau exp(-r tau) N(d2)   dC/dr     (per 1.00 of rate)
    Theta = -S n(d1) sigma / (2 sqrt(tau)) - r K exp(-r tau) N(d2)   dC/dt (per year)

All functions are vectorised over numpy broadcasting. The degenerate limits
tau -> 0 or sigma -> 0 return the discounted intrinsic value
max(S - K exp(-r tau), 0) rather than NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .volatility import simulate_gbm


def _validate(s, k, tau, r, sigma):
    s, k, tau, r, sigma = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (s, k, tau, r, sigma)))
    if (s <= 0).any():
        raise ValueError("underlying S must be strictly positive")
    if (k < 0).any():
        raise ValueError("deductible K must be non-negative")
    if (tau < 0).any():
        raise ValueError("time to expiry must be non-negative")
    if (sigma < 0).any():
        raise ValueError("volatility must be non-negative")
    return s, k, tau, r, sigma


def _d1_d2(s, k, tau, r, sigma):
    vol_sqrt_t = sigma * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t, vol_sqrt_t


def _scalar(x):
    return float(x) if np.ndim(x) == 0 else x


def call_price(s, k, tau, r, sigma):
    """Black-Scholes European call value = pure micro-insurance premium."""
    s, k, tau, r, sigma = _validate(s, k, tau, r, sigma)
    disc_k = k * np.exp(-r * tau)
    intrinsic = np.maximum(s - disc_k, 0.0)
    regular = (k > 0) & (sigma * np.sqrt(tau) > 0)
    d1, d2, _ = _d1_d2(s, np.where(regular, k, 1.0), tau, r, np.where(regular, sigma, 1.0))
    bs = s * stats.norm.cdf(d1) - disc_k * stats.norm.cdf(d2)
    return _scalar(np.where(regular, bs, intrinsic))


@dataclass(frozen=True)
class Greeks:
    price: np.ndarray | float
    delta: np.ndarray | float
    gamma: np.ndarray | float
    vega: np.ndarray | float
    rho: np.ndarray | float
    theta: np.ndarray | float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("price", "delta", "gamma", "vega", "rho", "theta")}


def call_greeks(s, k, tau, r, sigma) -> Greeks:
    """Price and Greeks. Degenerate cases (tau=0, sigma=0, K=0) use the
    limiting values of the deterministic payoff (gamma = vega = 0)."""
    s, k, tau, r, sigma = _validate(s, k, tau, r, sigma)
    disc = np.exp(-r * tau)
    regular = (k > 0) & (sigma * np.sqrt(tau) > 0)
    k_safe = np.where(regular, k, 1.0)
    sig_safe = np.where(regular, sigma, 1.0)
    tau_safe = np.where(regular, tau, 1.0)
    d1, d2, vst = _d1_d2(s, k_safe, tau_safe, r, sig_safe)
    pdf1, cdf1, cdf2 = stats.norm.pdf(d1), stats.norm.cdf(d1), stats.norm.cdf(d2)

    itm = (s > k * disc).astype(float)  # deterministic payoff indicator
    price = call_price(s, k, tau, r, sigma)
    delta = np.where(regular, cdf1, itm)
    gamma = np.where(regular, pdf1 / (s * vst), 0.0)
    vega = np.where(regular, s * pdf1 * np.sqrt(tau_safe), 0.0)
    rho = np.where(regular, k * tau * disc * cdf2, k * tau * disc * itm)
    theta = np.where(
        regular,
        -s * pdf1 * sig_safe / (2 * np.sqrt(tau_safe)) - r * k * disc * cdf2,
        -r * k * disc * itm,
    )
    return Greeks(*(_scalar(v) for v in (price, delta, gamma, vega, rho, theta)))


def monte_carlo_call(s, k, tau, r, sigma, n_paths: int = 200_000, seed: int | None = 0) -> tuple[float, float]:
    """Risk-neutral Monte Carlo check of :func:`call_price`. Returns (price, std error)."""
    if tau <= 0:
        return float(max(s - k, 0.0)), 0.0
    terminal = simulate_gbm(s, r, sigma, tau, 1, n_paths, seed)[:, -1]
    payoff = np.exp(-r * tau) * np.maximum(terminal - k, 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths))


# --------------------------------------------------------------------------- #
# Contract-level wrapper
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MicroInsuranceContract:
    """One annual policy priced as a European call."""

    expected_cost: float  # S
    deductible: float  # K
    volatility: float  # sigma*
    risk_free_rate: float  # r
    term: float = 1.0  # T - t

    def greeks(self) -> Greeks:
        return call_greeks(self.expected_cost, self.deductible, self.term, self.risk_free_rate, self.volatility)

    @property
    def premium(self) -> float:
        return float(call_price(self.expected_cost, self.deductible, self.term, self.risk_free_rate, self.volatility))

    def loaded_premium(self, expense_loading: float = 0.0, risk_margin: float = 0.0) -> float:
        """Gross premium = pure premium x (1 + risk margin) / (1 - expense ratio)."""
        if not 0 <= expense_loading < 1 or risk_margin < 0:
            raise ValueError("expense_loading must be in [0, 1) and risk_margin >= 0")
        return self.premium * (1 + risk_margin) / (1 - expense_loading)
