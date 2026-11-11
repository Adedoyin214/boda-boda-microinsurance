"""Micro-insurance pricing library.

Week 1  :mod:`.data`, :mod:`.distributions`   ingestion, X_ij = Y_ij / P_ij, frequency/severity fits, bootstrap
Week 2  :mod:`.credibility`                   EBCT Models I and II
Week 3  :mod:`.volatility`, :mod:`.pricing`   GBM sigma*, Black-Scholes premium and Greeks
Week 4  :mod:`.comparison`, :mod:`.dashboard` EBCT vs option premiums, sensitivities, stress tests
        :mod:`.pipeline`                      end-to-end run writing all results
"""

from .comparison import compare_premiums, frequency_by_fleet, sensitivity_grid, stress_test
from .credibility import CredibilityResult, ebct_from_claims, ebct_model_1, ebct_model_2
from .data import (
    generate_synthetic_portfolio,
    load_claims_csv,
    load_portfolio,
    save_portfolio,
    standardise_claims,
    to_matrices,
    validate_claims,
)
from .distributions import (
    best_fit,
    bootstrap,
    fit_gamma,
    fit_lognormal,
    fit_poisson_frequency,
    fit_severity,
    stop_loss_expectation,
)
from .pricing import Greeks, MicroInsuranceContract, call_greeks, call_price, monte_carlo_call
from .volatility import GBMParams, estimate_gbm, estimate_gbm_from_frame, simulate_gbm

__version__ = "0.1.0"
