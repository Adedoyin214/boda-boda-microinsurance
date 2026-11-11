import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from micro_insurance import comparison, credibility, data, distributions, volatility


@pytest.fixture(scope="module")
def portfolio():
    return data.generate_synthetic_portfolio(seed=7)


# ---------------------------------------------------------------- data
def test_synthetic_portfolio_shape(portfolio):
    c = portfolio.claims
    assert len(c) == 8 * 5
    assert len(portfolio.individual_claims) == c["claim_count"].sum()
    assert np.isclose(portfolio.individual_claims["amount"].sum(), c["claims_amount"].sum(), rtol=1e-6)


def test_validation_errors(portfolio):
    bad = portfolio.claims.copy()
    bad.loc[0, "policyholders"] = 0
    with pytest.raises(ValueError):
        data.validate_claims(bad)
    with pytest.raises(ValueError):
        data.to_matrices(portfolio.claims.iloc[1:])  # unbalanced panel


def test_roundtrip(tmp_path, portfolio):
    data.save_portfolio(portfolio, tmp_path)
    back = data.load_portfolio(tmp_path)
    pd.testing.assert_frame_equal(back.claims, data.validate_claims(portfolio.claims))


# ---------------------------------------------------------------- distributions
def test_poisson_rate():
    fit = distributions.fit_poisson_frequency([2, 3, 5], [10, 10, 20])
    assert fit.rate == pytest.approx(10 / 40)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_stop_loss_matches_integration(family):
    rng = np.random.default_rng(0)
    x = rng.lognormal(9, 0.8, 500)
    fit = distributions.fit_severity(x)[family]
    k = 10_000.0
    numeric, _ = integrate.quad(lambda t: fit.dist.sf(t), k, np.inf, limit=200)
    assert fit.stop_loss(k) == pytest.approx(numeric, rel=1e-5)
    assert fit.stop_loss(0) == pytest.approx(fit.mean, rel=1e-6)


def test_lognormal_preferred_for_lognormal_data():
    x = np.random.default_rng(1).lognormal(9, 1.0, 2000)
    assert distributions.best_fit(distributions.fit_severity(x)).family == "lognormal"


def test_bootstrap_ci_covers_mean():
    x = np.random.default_rng(2).normal(10, 2, 200)
    b = distributions.bootstrap(x, n_boot=1000)
    assert b.ci_low < 10 < b.ci_high


# ---------------------------------------------------------------- credibility
def test_model_2_with_equal_volume_equals_model_1():
    rng = np.random.default_rng(3)
    x = rng.gamma(2, 50, size=(6, 5)) + np.arange(6)[:, None] * 30
    p = np.full_like(x, 1.0)
    m1, m2 = credibility.ebct_model_1(x), credibility.ebct_model_2(x, p)
    np.testing.assert_allclose(m1.premium_per_unit, m2.premium_per_unit)
    assert m1.between_variance == pytest.approx(m2.between_variance)


def test_model_1_hand_example():
    x = np.array([[1.0, 3.0], [5.0, 7.0]])
    r = credibility.ebct_model_1(x)
    # means 2, 6; grand 4; within = 2; between = 8 - 2/2 = 7; Z = 2 / (2 + 2/7) = 0.875
    assert r.z[0] == pytest.approx(0.875)
    np.testing.assert_allclose(r.premium_per_unit, [2.25, 5.75])


def test_no_heterogeneity_gives_zero_credibility():
    x = np.array([[1.0, 9.0], [9.0, 1.0]])
    r = credibility.ebct_model_1(x)
    assert np.all(r.z == 0) and np.allclose(r.premium_per_unit, 5.0)


def test_forecast_volume(portfolio):
    r = credibility.ebct_from_claims(portfolio.claims, model=2)
    f = r.forecast(next_volume=100)
    np.testing.assert_allclose(f["total_premium"], 100 * r.premium_per_unit)
    assert np.all((r.z > 0) & (r.z < 1))


# ---------------------------------------------------------------- volatility
def test_gbm_recovers_sigma():
    path = volatility.simulate_gbm(100, 0.08, 0.25, t=20, n_steps=20 * 365, n_paths=1, seed=4)[0]
    est = volatility.estimate_gbm(path, 365)
    assert est.sigma == pytest.approx(0.25, rel=0.03)


def test_gbm_from_frame(portfolio):
    d = volatility.estimate_gbm_from_frame(portfolio.cost_index, "daily")
    m = volatility.estimate_gbm_from_frame(portfolio.cost_index, "monthly")
    assert 0.15 < d.sigma < 0.35 and 0.1 < m.sigma < 0.45


# ---------------------------------------------------------------- comparison
def test_compare_premiums_zero_deductible_equals_ebct(portfolio):
    cred = credibility.ebct_from_claims(portfolio.claims)
    sev = distributions.fit_lognormal(portfolio.individual_claims["amount"])
    t = comparison.compare_premiums(cred, 0.0, 0.3, 0.0, frequency_by_fleet=comparison.frequency_by_fleet(portfolio.claims),
                                    severity=sev)
    np.testing.assert_allclose(t["bs_premium"], t["ebct_premium"])
    assert {"stop_loss_premium", "delta", "vega"} <= set(t.columns)


def test_sensitivity_and_stress():
    g = comparison.sensitivity_grid(2000, [0, 1000, 2000], [0.1, 0.3], 0.1)
    assert len(g) == 6
    s = comparison.stress_test(2000, 1500, 0.3, 0.1)
    base = s[(s.factor == "cost") & (s.shock == 0)]["change"].iloc[0]
    assert base == pytest.approx(0)
    assert s[(s.factor == "cost") & (s.shock == 0.2)]["change"].iloc[0] > 0
