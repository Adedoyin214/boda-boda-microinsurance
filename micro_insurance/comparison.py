"""Week 3/4 - EBCT vs Black-Scholes comparison and sensitivity stress tests.

For each fleet the EBCT credibility premium per unit (expected annual claim
cost per operator) is used as the option underlying S. The option premium is
then the value of cover *above* the deductible K, so with K = 0 and r = 0 the
two approaches coincide (C = S). A third, purely actuarial benchmark is the
stop-loss premium per operator

    lambda_i * E[(X - K)+]

from the fitted claim-frequency and severity distributions, where the
deductible is applied per claim. Comparing the three shows how much of the
option premium is driven by the GBM/volatility assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .credibility import CredibilityResult
from .distributions import SeverityFit
from .pricing import call_greeks, call_price


def compare_premiums(
    cred: CredibilityResult,
    deductible: float,
    volatility: float,
    risk_free_rate: float,
    term: float = 1.0,
    frequency_by_fleet: dict[str, float] | None = None,
    severity: SeverityFit | None = None,
) -> pd.DataFrame:
    """Per-fleet table of EBCT premium, Black-Scholes premium and Greeks.

    If ``frequency_by_fleet`` (claims per operator-year) and a fitted
    ``severity`` are supplied, a ``stop_loss_premium`` column is added.
    """
    s = np.asarray(cred.premium_per_unit, dtype=float)
    if (s <= 0).any():
        raise ValueError("EBCT premiums must be positive to serve as option underlying")
    g = call_greeks(s, deductible, term, risk_free_rate, volatility)
    out = pd.DataFrame(
        {
            "fleet": cred.fleets,
            "Z": cred.z,
            "ebct_premium": s,
            "deductible": float(deductible),
            "bs_premium": g.price,
            "bs_to_ebct": g.price / s,
            "delta": g.delta,
            "gamma": g.gamma,
            "vega": g.vega,
            "rho": g.rho,
            "theta": g.theta,
        }
    )
    if frequency_by_fleet is not None and severity is not None:
        excess = severity.stop_loss(deductible)
        out["stop_loss_premium"] = [frequency_by_fleet[f] * excess for f in cred.fleets]
    return out


def frequency_by_fleet(claims: pd.DataFrame) -> dict[str, float]:
    """Claims per operator-year for each fleet (needs ``claim_count``)."""
    if "claim_count" not in claims:
        raise ValueError("claims table needs a claim_count column")
    g = claims.groupby("fleet")[["claim_count", "policyholders"]].sum()
    return (g["claim_count"] / g["policyholders"]).to_dict()


def sensitivity_grid(
    s: float,
    deductibles,
    volatilities,
    risk_free_rate: float,
    term: float = 1.0,
) -> pd.DataFrame:
    """Long table of premium and Greeks over a deductible x volatility grid."""
    k, v = np.meshgrid(np.asarray(deductibles, float), np.asarray(volatilities, float), indexing="ij")
    g = call_greeks(s, k, term, risk_free_rate, v)
    return pd.DataFrame(
        {
            "deductible": k.ravel(),
            "volatility": v.ravel(),
            **{name: np.ravel(val) for name, val in g.as_dict().items()},
        }
    ).rename(columns={"price": "premium"})


def stress_test(
    s: float,
    deductible: float,
    volatility: float,
    risk_free_rate: float,
    term: float = 1.0,
    cost_shocks=(-0.2, -0.1, 0.0, 0.1, 0.2, 0.5),
    vol_shocks=(-0.1, 0.0, 0.1, 0.25),
    rate_shocks=(-0.02, 0.0, 0.02, 0.05),
) -> pd.DataFrame:
    """One-factor-at-a-time stress scenarios relative to a base contract.

    Cost shocks are relative (+0.2 = costs 20% higher); volatility and rate
    shocks are absolute (+0.1 = 10 vol points, +0.02 = 200 bp).
    """
    base = float(call_price(s, deductible, term, risk_free_rate, volatility))
    rows = []
    for factor, shocks in (("cost", cost_shocks), ("volatility", vol_shocks), ("rate", rate_shocks)):
        for shock in shocks:
            s_, v_, r_ = s, volatility, risk_free_rate
            if factor == "cost":
                s_ = s * (1 + shock)
            elif factor == "volatility":
                v_ = max(volatility + shock, 0.0)
            else:
                r_ = risk_free_rate + shock
            if s_ <= 0:
                continue
            p = float(call_price(s_, deductible, term, r_, v_))
            rows.append(
                {"factor": factor, "shock": shock, "premium": p, "change": p - base,
                 "pct_change": (p - base) / base if base > 0 else np.nan}
            )
    return pd.DataFrame(rows)
