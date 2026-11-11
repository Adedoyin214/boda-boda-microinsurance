import numpy as np
import pytest

from micro_insurance.pricing import MicroInsuranceContract, call_greeks, call_price, monte_carlo_call


def test_textbook_value():
    # Hull: S=100, K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert call_price(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity_and_bounds():
    s, k, t, r, v = 80.0, 95.0, 1.0, 0.07, 0.35
    c = call_price(s, k, t, r, v)
    assert max(s - k * np.exp(-r * t), 0) <= c <= s


@pytest.mark.parametrize("s,k,t,r,v", [(100, 90, 1, 0.05, 0.3), (50, 70, 0.5, 0.1, 0.6), (20, 5, 2, 0.0, 0.15)])
def test_greeks_match_finite_differences(s, k, t, r, v):
    g = call_greeks(s, k, t, r, v)
    h = 1e-4
    f = lambda **kw: call_price(**{"s": s, "k": k, "tau": t, "r": r, "sigma": v, **kw})
    assert g.delta == pytest.approx((f(s=s + h) - f(s=s - h)) / (2 * h), rel=1e-5, abs=1e-6)
    assert g.gamma == pytest.approx((f(s=s + h) - 2 * f() + f(s=s - h)) / h**2, rel=1e-3, abs=1e-5)
    assert g.vega == pytest.approx((f(sigma=v + h) - f(sigma=v - h)) / (2 * h), rel=1e-5, abs=1e-6)
    assert g.rho == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-5, abs=1e-6)
    assert g.theta == pytest.approx(-(f(tau=t + h) - f(tau=t - h)) / (2 * h), rel=1e-5, abs=1e-6)


def test_degenerate_cases():
    assert call_price(100, 0, 1, 0.05, 0.2) == pytest.approx(100)  # zero deductible = full cover
    assert call_price(100, 90, 0, 0.05, 0.2) == pytest.approx(10)  # at expiry
    assert call_price(100, 90, 1, 0.05, 0.0) == pytest.approx(100 - 90 * np.exp(-0.05))
    g = call_greeks(100, 90, 1, 0.05, 0.0)
    assert g.gamma == 0 and g.vega == 0 and g.delta == 1


def test_vectorised_and_monotone_in_deductible():
    k = np.linspace(0, 200, 21)
    p = call_price(100, k, 1, 0.05, 0.3)
    assert p.shape == k.shape
    assert np.all(np.diff(p) < 0)


def test_monte_carlo_agrees():
    price, se = monte_carlo_call(100, 110, 1, 0.05, 0.4, n_paths=400_000, seed=1)
    assert abs(price - call_price(100, 110, 1, 0.05, 0.4)) < 4 * se


def test_invalid_inputs():
    with pytest.raises(ValueError):
        call_price(-1, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        call_price(100, 100, 1, 0.05, -0.2)


def test_contract_loading():
    c = MicroInsuranceContract(2000, 500, 0.3, 0.1)
    assert c.loaded_premium(0.2, 0.1) == pytest.approx(c.premium * 1.1 / 0.8)
    with pytest.raises(ValueError):
        c.loaded_premium(1.0)
