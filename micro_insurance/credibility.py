"""Week 2 - Empirical Bayes Credibility Theory (EBCT) Models I and II.

Notation follows the standard actuarial treatment (e.g. Gray & Pitts,
*Risk Modelling in General Insurance*; Institute and Faculty of Actuaries CS2):
N risks (fleets) i = 1..N observed for n years j = 1..n.

EBCT Model I (unweighted, Buhlmann)
    X_ij                           claims of risk i in year j
    E[m(theta)]   = X_bar          grand mean
    E[s^2(theta)] = 1/N sum_i 1/(n-1) sum_j (X_ij - X_bar_i)^2
    Var[m(theta)] = 1/(N-1) sum_i (X_bar_i - X_bar)^2 - E[s^2(theta)] / n
    Z             = n / (n + E[s^2(theta)] / Var[m(theta)])
    premium_i     = Z * X_bar_i + (1 - Z) * X_bar

EBCT Model II (volume-weighted, Buhlmann-Straub)
    X_ij = Y_ij / P_ij,  P_i = sum_j P_ij,  P = sum_i P_i
    P*            = 1/(Nn-1) sum_i P_i (1 - P_i / P)
    X_bar_i       = sum_j P_ij X_ij / P_i
    X_bar         = sum_ij P_ij X_ij / P
    E[s^2(theta)] = 1/N sum_i 1/(n-1) sum_j P_ij (X_ij - X_bar_i)^2
    Var[m(theta)] = 1/P* ( 1/(Nn-1) sum_ij P_ij (X_ij - X_bar)^2 - E[s^2(theta)] )
    Z_i           = P_i / (P_i + E[s^2(theta)] / Var[m(theta)])
    premium per unit_i = Z_i * X_bar_i + (1 - Z_i) * X_bar
    premium_i (next year) = P_i,n+1 * premium per unit_i

If the estimate of Var[m(theta)] is <= 0 the data show no evidence of
between-risk heterogeneity; it is floored at 0, giving Z = 0 (collective
premium for everybody). This is the conventional treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import to_matrices


@dataclass(frozen=True)
class CredibilityResult:
    model: str
    fleets: list[str]
    grand_mean: float  # E[m(theta)]
    within_variance: float  # E[s^2(theta)]
    between_variance: float  # Var[m(theta)] (floored at 0)
    between_variance_raw: float  # unfloored estimate, for diagnostics
    k: float  # E[s^2] / Var[m]  (inf when Var[m] = 0)
    risk_means: np.ndarray  # X_bar_i
    z: np.ndarray  # credibility factor per fleet (constant for Model I)
    premium_per_unit: np.ndarray  # credibility premium per unit of volume
    volume: np.ndarray | None = None  # P_i (Model II only)

    def table(self) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "fleet": self.fleets,
                "risk_mean": self.risk_means,
                "Z": self.z,
                "credibility_premium": self.premium_per_unit,
            }
        )
        if self.volume is not None:
            out.insert(1, "total_volume", self.volume)
        return out

    def forecast(self, next_volume=None) -> pd.DataFrame:
        """Premium forecast for the next policy year.

        Model I: premium_i is already the forecast of X_i,n+1.
        Model II: provide ``next_volume`` (scalar, array of length N, or a
        mapping fleet -> volume) to obtain total premiums P_i,n+1 * premium_i.
        """
        out = self.table()[["fleet", "credibility_premium"]].copy()
        if next_volume is None:
            return out
        if isinstance(next_volume, dict):
            vol = np.array([float(next_volume[f]) for f in self.fleets])
        else:
            vol = np.broadcast_to(np.asarray(next_volume, dtype=float), (len(self.fleets),)).copy()
        if (vol <= 0).any():
            raise ValueError("next-year volume must be positive")
        out["next_volume"] = vol
        out["total_premium"] = vol * out["credibility_premium"].to_numpy()
        return out


def _check_matrix(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (risks x years)")
    n_risks, n_years = x.shape
    if n_risks < 2:
        raise ValueError("EBCT needs at least 2 risks")
    if n_years < 2:
        raise ValueError("EBCT needs at least 2 years of data")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


def ebct_model_1(x, fleets=None) -> CredibilityResult:
    """EBCT Model I. ``x`` has shape (N risks, n years)."""
    x = _check_matrix(x, "x")
    n_risks, n_years = x.shape
    fleets = list(fleets) if fleets is not None else [f"risk_{i + 1}" for i in range(n_risks)]

    risk_means = x.mean(axis=1)
    grand_mean = float(risk_means.mean())
    within = float(np.mean(x.var(axis=1, ddof=1)))
    between_raw = float(risk_means.var(ddof=1) - within / n_years)
    between = max(between_raw, 0.0)

    if between > 0:
        k = within / between
        z = n_years / (n_years + k)
    else:
        k, z = float("inf"), 0.0
    premium = z * risk_means + (1 - z) * grand_mean
    return CredibilityResult(
        "EBCT Model I", fleets, grand_mean, within, between, between_raw, k,
        risk_means, np.full(n_risks, z), premium,
    )


def ebct_model_2(y, p, fleets=None) -> CredibilityResult:
    """EBCT Model II. ``y`` = aggregate claims Y_ij, ``p`` = volumes P_ij, shape (N, n)."""
    y = _check_matrix(y, "y")
    p = _check_matrix(p, "p")
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    if (p <= 0).any():
        raise ValueError("volumes P_ij must be strictly positive")
    n_risks, n_years = y.shape
    fleets = list(fleets) if fleets is not None else [f"risk_{i + 1}" for i in range(n_risks)]

    x = y / p
    p_i = p.sum(axis=1)
    p_total = p_i.sum()
    p_star = float(np.sum(p_i * (1 - p_i / p_total)) / (n_risks * n_years - 1))

    risk_means = (p * x).sum(axis=1) / p_i
    grand_mean = float((p * x).sum() / p_total)
    within = float(np.mean((p * (x - risk_means[:, None]) ** 2).sum(axis=1) / (n_years - 1)))
    total_dispersion = float((p * (x - grand_mean) ** 2).sum() / (n_risks * n_years - 1))
    between_raw = (total_dispersion - within) / p_star
    between = max(between_raw, 0.0)

    if between > 0:
        k = within / between
        z = p_i / (p_i + k)
    else:
        k, z = float("inf"), np.zeros(n_risks)
    premium = z * risk_means + (1 - z) * grand_mean
    return CredibilityResult(
        "EBCT Model II", fleets, grand_mean, within, between, float(between_raw), k,
        risk_means, z, premium, volume=p_i,
    )


def ebct_from_claims(df: pd.DataFrame, model: int = 2) -> CredibilityResult:
    """Convenience wrapper on the long fleet-year table.

    Model I is applied to the per-unit claims X_ij = Y_ij / P_ij (treating each
    year as equal weight); Model II uses Y and P directly.
    """
    fleets, _, y, p = to_matrices(df)
    if model == 1:
        return ebct_model_1(y / p, fleets)
    if model == 2:
        return ebct_model_2(y, p, fleets)
    raise ValueError("model must be 1 or 2")
