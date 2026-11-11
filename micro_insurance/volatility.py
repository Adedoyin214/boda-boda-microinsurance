"""Week 3 - mapping a treatment-cost time series to GBM parameters.

Under geometric Brownian motion dS = mu S dt + sigma S dW, log returns
r_t = ln(S_t / S_{t-1}) are i.i.d. Normal((mu - sigma^2/2) dt, sigma^2 dt), so

    sigma* = std(r_t) * sqrt(periods_per_year)
    mu     = mean(r_t) * periods_per_year + sigma*^2 / 2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GBMParams:
    mu: float  # annualised drift
    sigma: float  # annualised volatility sigma*
    s0: float  # latest observed level
    n_returns: int
    periods_per_year: float


def log_returns(values) -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 3:
        raise ValueError("need at least 3 observations to estimate volatility")
    if not np.all(np.isfinite(v)) or (v <= 0).any():
        raise ValueError("cost series must be finite and strictly positive")
    return np.diff(np.log(v))


def infer_periods_per_year(dates) -> float:
    """Calendar-based frequency from the median spacing of the dates."""
    d = pd.to_datetime(pd.Series(dates)).sort_values()
    spacing_days = d.diff().dt.total_seconds().dropna().median() / 86_400
    if not spacing_days or spacing_days <= 0:
        raise ValueError("cannot infer frequency: dates must be distinct")
    return 365.25 / spacing_days


def estimate_gbm(values, periods_per_year: float) -> GBMParams:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    r = log_returns(values)
    sigma = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    mu = float(r.mean() * periods_per_year + 0.5 * sigma**2)
    return GBMParams(mu, sigma, float(np.asarray(values, dtype=float).ravel()[-1]), r.size, float(periods_per_year))


def estimate_gbm_from_frame(df: pd.DataFrame, freq: str = "daily", value_col: str = "cost_index",
                            date_col: str = "date") -> GBMParams:
    """Estimate GBM parameters from a dated series at ``daily`` or ``monthly`` frequency.

    ``monthly`` uses month-end levels, which is more robust to noisy daily data.
    """
    if value_col not in df or date_col not in df:
        raise ValueError(f"frame needs columns {date_col!r} and {value_col!r}")
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.sort_values(date_col).set_index(date_col)[value_col]
    if freq == "daily":
        return estimate_gbm(s.to_numpy(), infer_periods_per_year(s.index))
    if freq == "monthly":
        monthly = s.resample("ME").last().dropna()
        return estimate_gbm(monthly.to_numpy(), 12.0)
    raise ValueError("freq must be 'daily' or 'monthly'")


def simulate_gbm(s0: float, mu: float, sigma: float, t: float, n_steps: int, n_paths: int,
                 seed: int | None = 0) -> np.ndarray:
    """Simulate GBM paths, returns array (n_paths, n_steps + 1) including S0."""
    if s0 <= 0 or sigma < 0 or t <= 0 or n_steps < 1 or n_paths < 1:
        raise ValueError("invalid GBM simulation parameters")
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    return s0 * np.exp(log_paths)
