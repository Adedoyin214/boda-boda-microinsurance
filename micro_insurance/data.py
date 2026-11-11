"""Week 1 - data ingestion and standardisation.

The library works on a *long* claims table with one row per (fleet, year):

    fleet          identifier of the risk class (e.g. a boda-boda stage association)
    year           policy year
    claims_amount  Y_ij  aggregate annual claim amount for the fleet
    policyholders  P_ij  risk volume (number of insured operators)
    claim_count    N_ij  number of claims (optional, needed for frequency fitting)

plus, optionally, an *individual claims* table (one row per claim, column
``amount``) for severity fitting and a *treatment cost* time series (columns
``date``, ``cost_index``) for the volatility estimate used by Black-Scholes.

No real data ships with the project roadmap, so :func:`generate_synthetic_portfolio`
creates a reproducible, clearly-labelled synthetic portfolio with realistic
structure (heterogeneous fleets, growing volume, Poisson frequency, log-normal
severity, GBM cost index). Real data can be loaded with :func:`load_claims_csv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("fleet", "year", "claims_amount", "policyholders")


@dataclass(frozen=True)
class SyntheticPortfolio:
    """Container for a generated (or loaded) dataset."""

    claims: pd.DataFrame  # long fleet-year table
    individual_claims: pd.DataFrame  # one row per claim
    cost_index: pd.DataFrame  # daily treatment-cost index (date, cost_index)


# --------------------------------------------------------------------------- #
# Validation & standardisation
# --------------------------------------------------------------------------- #
def validate_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Check the long claims table and return a clean, sorted copy.

    Raises ``ValueError`` with a clear message on any structural problem so that
    errors surface at ingestion rather than deep inside the models.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"claims table is missing required columns: {missing}")

    out = df.copy()
    out["fleet"] = out["fleet"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["claims_amount"] = pd.to_numeric(out["claims_amount"], errors="raise").astype(float)
    out["policyholders"] = pd.to_numeric(out["policyholders"], errors="raise").astype(float)

    if out[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("claims table contains missing values in required columns")
    if (out["claims_amount"] < 0).any():
        raise ValueError("claims_amount (Y_ij) must be non-negative")
    if (out["policyholders"] <= 0).any():
        raise ValueError("policyholders (P_ij) must be strictly positive")
    if out.duplicated(["fleet", "year"]).any():
        raise ValueError("duplicate (fleet, year) rows found")
    if "claim_count" in out.columns:
        out["claim_count"] = pd.to_numeric(out["claim_count"], errors="raise")
        if (out["claim_count"] < 0).any():
            raise ValueError("claim_count must be non-negative")

    return out.sort_values(["fleet", "year"]).reset_index(drop=True)


def standardise_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-unit claim column ``X_ij = Y_ij / P_ij`` (``claims_per_unit``)."""
    out = validate_claims(df)
    out["claims_per_unit"] = out["claims_amount"] / out["policyholders"]
    return out


def to_matrices(df: pd.DataFrame) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """Pivot the long table into ``(fleets, years, Y, P)`` with Y, P of shape (N, n).

    EBCT requires a balanced panel (every fleet observed in every year); a
    ``ValueError`` is raised otherwise.
    """
    clean = validate_claims(df)
    y = clean.pivot(index="fleet", columns="year", values="claims_amount")
    p = clean.pivot(index="fleet", columns="year", values="policyholders")
    if y.isna().any().any() or p.isna().any().any():
        raise ValueError("EBCT needs a balanced panel: every fleet must have every year")
    return list(y.index), [int(c) for c in y.columns], y.to_numpy(float), p.to_numpy(float)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_claims_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a fleet-year claims CSV."""
    return validate_claims(pd.read_csv(path))


def save_portfolio(portfolio: SyntheticPortfolio, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    portfolio.claims.to_csv(folder / "claims.csv", index=False)
    portfolio.individual_claims.to_csv(folder / "individual_claims.csv", index=False)
    portfolio.cost_index.to_csv(folder / "treatment_cost_index.csv", index=False)


def load_portfolio(folder: str | Path) -> SyntheticPortfolio:
    folder = Path(folder)
    claims = load_claims_csv(folder / "claims.csv")
    individual = pd.read_csv(folder / "individual_claims.csv")
    cost = pd.read_csv(folder / "treatment_cost_index.csv", parse_dates=["date"])
    return SyntheticPortfolio(claims, individual, cost)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def generate_synthetic_portfolio(
    n_fleets: int = 8,
    n_years: int = 5,
    start_year: int = 2020,
    seed: int = 2026,
    base_frequency: float = 0.12,
    severity_median: float = 18_000.0,
    severity_sigma: float = 0.9,
    cost_index_days: int = 3 * 365,
    cost_drift: float = 0.08,
    cost_volatility: float = 0.25,
) -> SyntheticPortfolio:
    """Generate a reproducible synthetic boda-boda accident/health portfolio.

    * Each fleet has its own latent risk level theta_i (log-normal multiplier on
      the claim frequency), so true between-fleet variation exists.
    * Volume P_ij grows over time at a fleet-specific rate (EBCT II motivation).
    * Claim counts N_ij ~ Poisson(lambda_i * P_ij); severities are log-normal
      (amounts in KES); Y_ij is the sum of the severities.
    * A daily treatment-cost index follows a geometric Brownian motion.
    """
    if n_fleets < 2 or n_years < 2:
        raise ValueError("need at least 2 fleets and 2 years")
    rng = np.random.default_rng(seed)

    fleets = [f"Fleet-{chr(ord('A') + i)}" for i in range(n_fleets)]
    risk_multiplier = rng.lognormal(mean=0.0, sigma=0.35, size=n_fleets)
    initial_volume = rng.integers(40, 250, size=n_fleets)
    growth = rng.uniform(0.05, 0.40, size=n_fleets)

    rows, claim_rows = [], []
    mu_log = np.log(severity_median)
    for i, fleet in enumerate(fleets):
        for j in range(n_years):
            volume = int(round(initial_volume[i] * (1 + growth[i]) ** j))
            lam = base_frequency * risk_multiplier[i] * volume
            count = int(rng.poisson(lam))
            amounts = rng.lognormal(mu_log, severity_sigma, size=count)
            year = start_year + j
            rows.append(
                {
                    "fleet": fleet,
                    "year": year,
                    "claims_amount": float(np.round(amounts.sum(), 2)),
                    "policyholders": volume,
                    "claim_count": count,
                }
            )
            claim_rows.extend(
                {"fleet": fleet, "year": year, "amount": float(np.round(a, 2))} for a in amounts
            )

    # Daily GBM treatment-cost index starting at 100.
    dt = 1 / 365
    shocks = rng.standard_normal(cost_index_days)
    log_path = np.cumsum((cost_drift - 0.5 * cost_volatility**2) * dt + cost_volatility * np.sqrt(dt) * shocks)
    index = 100 * np.exp(np.concatenate([[0.0], log_path]))
    dates = pd.date_range(f"{start_year + n_years - 3}-01-01", periods=cost_index_days + 1, freq="D")

    return SyntheticPortfolio(
        claims=pd.DataFrame(rows),
        individual_claims=pd.DataFrame(claim_rows, columns=["fleet", "year", "amount"]),
        cost_index=pd.DataFrame({"date": dates, "cost_index": np.round(index, 6)}),
    )
