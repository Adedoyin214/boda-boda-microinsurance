"""Week 1 - risk distribution fitting and bootstrap validation.

* Frequency: Poisson with exposure (claims per policyholder-year), with a
  Pearson chi-square dispersion check.
* Severity: Gamma and Log-Normal by maximum likelihood, compared by
  log-likelihood / AIC and the Kolmogorov-Smirnov statistic.
* Stop-loss (excess-of-deductible) expectations E[(X - K)+] for both severity
  families, used later as the actuarial benchmark for the option premium.
* Non-parametric bootstrap for small-sample validation of any statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Frequency
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PoissonFit:
    rate: float  # lambda per unit of exposure (claims per policyholder-year)
    total_claims: float
    total_exposure: float
    dispersion_statistic: float  # Pearson chi-square
    dispersion_df: int
    dispersion_p_value: float  # small p-value => over-dispersion vs Poisson

    @property
    def dispersion_ratio(self) -> float:
        return self.dispersion_statistic / self.dispersion_df if self.dispersion_df > 0 else float("nan")


def fit_poisson_frequency(counts, exposure=None) -> PoissonFit:
    """MLE of a Poisson rate with exposure: lambda_hat = sum(N) / sum(P)."""
    counts = np.asarray(counts, dtype=float).ravel()
    exposure = np.ones_like(counts) if exposure is None else np.asarray(exposure, dtype=float).ravel()
    if counts.shape != exposure.shape:
        raise ValueError("counts and exposure must have the same length")
    if counts.size == 0:
        raise ValueError("no observations")
    if (counts < 0).any() or (exposure <= 0).any():
        raise ValueError("counts must be >= 0 and exposure > 0")

    rate = counts.sum() / exposure.sum()
    expected = rate * exposure
    if rate == 0:
        stat, df, p = 0.0, counts.size - 1, 1.0
    else:
        stat = float(np.sum((counts - expected) ** 2 / expected))
        df = counts.size - 1
        p = float(stats.chi2.sf(stat, df)) if df > 0 else float("nan")
    return PoissonFit(float(rate), float(counts.sum()), float(exposure.sum()), stat, df, p)


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeverityFit:
    family: str  # "gamma" or "lognormal"
    params: dict  # gamma: shape, scale ; lognormal: mu, sigma
    log_likelihood: float
    aic: float
    ks_statistic: float
    ks_p_value: float
    n: int
    dist: object = field(repr=False, compare=False)  # frozen scipy distribution

    @property
    def mean(self) -> float:
        return float(self.dist.mean())

    @property
    def variance(self) -> float:
        return float(self.dist.var())

    def stop_loss(self, deductible: float) -> float:
        """E[(X - K)+] - expected claim payment above deductible K."""
        return stop_loss_expectation(self.family, self.params, deductible)


def _clean_positive(amounts) -> np.ndarray:
    x = np.asarray(amounts, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least 2 claim amounts to fit a severity distribution")
    if not np.all(np.isfinite(x)) or (x <= 0).any():
        raise ValueError("claim amounts must be finite and strictly positive")
    return x


def fit_lognormal(amounts) -> SeverityFit:
    """Closed-form MLE: mu = mean(log x), sigma = std(log x, ddof=0)."""
    x = _clean_positive(amounts)
    logs = np.log(x)
    mu, sigma = float(logs.mean()), float(logs.std(ddof=0))
    if sigma <= 0:
        raise ValueError("claim amounts are all identical; cannot fit a log-normal")
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    return _finish("lognormal", {"mu": mu, "sigma": sigma}, dist, x)


def fit_gamma(amounts) -> SeverityFit:
    """MLE with location fixed at zero."""
    x = _clean_positive(amounts)
    if np.allclose(x, x[0]):
        raise ValueError("claim amounts are all identical; cannot fit a gamma")
    shape, _, scale = stats.gamma.fit(x, floc=0)
    dist = stats.gamma(a=shape, scale=scale)
    return _finish("gamma", {"shape": float(shape), "scale": float(scale)}, dist, x)


def _finish(family: str, params: dict, dist, x: np.ndarray) -> SeverityFit:
    ll = float(np.sum(dist.logpdf(x)))
    ks = stats.kstest(x, dist.cdf)
    return SeverityFit(family, params, ll, 2 * 2 - 2 * ll, float(ks.statistic), float(ks.pvalue), x.size, dist)


def fit_severity(amounts) -> dict[str, SeverityFit]:
    """Fit both candidate families; returns {"gamma": ..., "lognormal": ...}."""
    return {"gamma": fit_gamma(amounts), "lognormal": fit_lognormal(amounts)}


def best_fit(fits: dict[str, SeverityFit]) -> SeverityFit:
    """Lowest AIC wins (both families have two parameters, so this is max likelihood)."""
    return min(fits.values(), key=lambda f: f.aic)


def stop_loss_expectation(family: str, params: dict, deductible: float) -> float:
    """Closed-form E[(X - K)+] for gamma / log-normal severity."""
    k = float(deductible)
    if k < 0:
        raise ValueError("deductible must be non-negative")
    if family == "lognormal":
        mu, sigma = params["mu"], params["sigma"]
        mean = np.exp(mu + 0.5 * sigma**2)
        if k == 0:
            return float(mean)
        d1 = (mu + sigma**2 - np.log(k)) / sigma
        d2 = d1 - sigma
        return float(mean * stats.norm.cdf(d1) - k * stats.norm.cdf(d2))
    if family == "gamma":
        a, theta = params["shape"], params["scale"]
        return float(
            a * theta * stats.gamma.sf(k, a + 1, scale=theta) - k * stats.gamma.sf(k, a, scale=theta)
        )
    raise ValueError(f"unknown family {family!r}")


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence: float
    replicates: np.ndarray = field(repr=False)


def bootstrap(
    data,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapResult:
    """Non-parametric bootstrap with percentile confidence interval."""
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("bootstrap needs at least 2 observations")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot < 10:
        raise ValueError("n_boot should be at least 10")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    reps = np.array([statistic(x[row]) for row in idx], dtype=float)
    alpha = (1 - confidence) / 2
    return BootstrapResult(
        estimate=float(statistic(x)),
        std_error=float(reps.std(ddof=1)),
        ci_low=float(np.quantile(reps, alpha)),
        ci_high=float(np.quantile(reps, 1 - alpha)),
        confidence=confidence,
        replicates=reps,
    )
