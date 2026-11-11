"""Write Python reference outputs to tests/fixtures/ for the R parity test.

    python tests/make_fixtures.py
"""

import json
from pathlib import Path

import numpy as np

from micro_insurance import credibility, data, distributions, pricing, volatility

OUT = Path(__file__).parent / "fixtures"

BS_CASES = [  # s, k, tau, r, sigma
    (100, 100, 1, 0.05, 0.2), (3000, 1000, 1, 0.08, 0.25), (2000, 2500, 1, 0.1, 0.6),
    (50, 70, 0.5, 0.1, 0.6), (100, 0, 1, 0.05, 0.2), (100, 90, 1, 0.05, 0.0),
]


def main():
    pf = data.generate_synthetic_portfolio(seed=11)
    data.save_portfolio(pf, OUT)
    claims = data.standardise_claims(pf.claims)
    amounts = pf.individual_claims["amount"]

    def cred(r):
        return {"grand_mean": r.grand_mean, "within_variance": r.within_variance,
                "between_variance": r.between_variance, "z": r.z.tolist(),
                "premium_per_unit": r.premium_per_unit.tolist()}

    sev = distributions.fit_severity(amounts)
    s, k, t, r, v = (np.array(c, float) for c in zip(*BS_CASES))
    g = pricing.call_greeks(s, k, t, r, v)
    gbm = volatility.estimate_gbm_from_frame(pf.cost_index, "daily")
    gbm_m = volatility.estimate_gbm_from_frame(pf.cost_index, "monthly")
    freq = distributions.fit_poisson_frequency(claims["claim_count"], claims["policyholders"])

    expected = {
        "ebct1": cred(credibility.ebct_from_claims(claims, 1)),
        "ebct2": cred(credibility.ebct_from_claims(claims, 2)),
        "bs_cases": [list(c) for c in BS_CASES],
        "greeks": {name: np.asarray(val).tolist() for name, val in g.as_dict().items()},
        "severity": {name: {"params": f.params, "log_likelihood": f.log_likelihood,
                            "stop_loss_2000": f.stop_loss(2000.0)} for name, f in sev.items()},
        "poisson_rate": freq.rate,
        "gbm_daily_sigma": gbm.sigma, "gbm_daily_mu": gbm.mu, "gbm_monthly_sigma": gbm_m.sigma,
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
