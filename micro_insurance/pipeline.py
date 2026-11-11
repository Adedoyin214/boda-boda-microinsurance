"""End-to-end run of the four roadmap weeks, writing all results to a folder.

    python -m micro_insurance.pipeline --out results
    python -m micro_insurance.pipeline --data path/to/folder --deductible 1000 --rate 0.09

``--data`` expects ``claims.csv``, ``individual_claims.csv`` and
``treatment_cost_index.csv`` (see :mod:`micro_insurance.data`); without it a
synthetic portfolio is generated. All parameters are illustrative defaults,
not calibrated market values.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import comparison, credibility, data, distributions, volatility

DEFAULT_DEDUCTIBLE = 1_000.0  # KES per operator per year (illustrative)
DEFAULT_RATE = 0.08  # continuously-compounded short rate (illustrative)


@dataclass
class PipelineConfig:
    data_dir: str | None = None
    out_dir: str = "results"
    deductible: float = DEFAULT_DEDUCTIBLE
    risk_free_rate: float = DEFAULT_RATE
    term: float = 1.0
    vol_frequency: str = "daily"  # or "monthly"
    seed: int = 2026
    n_boot: int = 2000


@dataclass
class PipelineResult:
    portfolio: data.SyntheticPortfolio
    claims: pd.DataFrame
    frequency: distributions.PoissonFit
    severity: dict
    best_severity: distributions.SeverityFit
    bootstrap: distributions.BootstrapResult
    ebct1: credibility.CredibilityResult
    ebct2: credibility.CredibilityResult
    forecast: pd.DataFrame
    gbm: volatility.GBMParams
    gbm_alt: volatility.GBMParams
    comparison: pd.DataFrame
    sensitivity: pd.DataFrame
    stress: pd.DataFrame
    config: PipelineConfig


def run(config: PipelineConfig | None = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    portfolio = (
        data.load_portfolio(cfg.data_dir) if cfg.data_dir
        else data.generate_synthetic_portfolio(seed=cfg.seed)
    )

    # Week 1 - preprocessing and distribution fitting
    claims = data.standardise_claims(portfolio.claims)
    if "claim_count" in claims:
        freq = distributions.fit_poisson_frequency(claims["claim_count"], claims["policyholders"])
    else:
        raise ValueError("claims table needs claim_count for frequency fitting")
    sev = distributions.fit_severity(portfolio.individual_claims["amount"])
    best = distributions.best_fit(sev)
    boot = distributions.bootstrap(claims["claims_per_unit"], n_boot=cfg.n_boot, seed=cfg.seed)

    # Week 2 - EBCT
    ebct1 = credibility.ebct_from_claims(claims, model=1)
    ebct2 = credibility.ebct_from_claims(claims, model=2)
    last_year = claims["year"].max()
    next_vol = claims[claims["year"] == last_year].set_index("fleet")["policyholders"].to_dict()
    forecast = ebct2.forecast(next_volume=next_vol)

    # Week 3 - volatility and option pricing
    alt = "monthly" if cfg.vol_frequency == "daily" else "daily"
    gbm = volatility.estimate_gbm_from_frame(portfolio.cost_index, cfg.vol_frequency)
    gbm_alt = volatility.estimate_gbm_from_frame(portfolio.cost_index, alt)
    comp = comparison.compare_premiums(
        ebct2, cfg.deductible, gbm.sigma, cfg.risk_free_rate, cfg.term,
        frequency_by_fleet=comparison.frequency_by_fleet(claims), severity=best,
    )
    comp.insert(3, "ebct1_premium", ebct1.premium_per_unit)

    # Week 4 - sensitivities and stress testing on the portfolio-average contract
    s_avg = float(ebct2.grand_mean)
    sens = comparison.sensitivity_grid(
        s_avg, np.linspace(0, 2 * s_avg, 41), np.linspace(0.05, 0.80, 16), cfg.risk_free_rate, cfg.term
    )
    stress = comparison.stress_test(s_avg, cfg.deductible, gbm.sigma, cfg.risk_free_rate, cfg.term)

    return PipelineResult(portfolio, claims, freq, sev, best, boot, ebct1, ebct2, forecast,
                          gbm, gbm_alt, comp, sens, stress, cfg)


def summary(res: PipelineResult) -> dict:
    def cred(c: credibility.CredibilityResult) -> dict:
        return {
            "grand_mean": c.grand_mean, "within_variance": c.within_variance,
            "between_variance": c.between_variance, "between_variance_raw": c.between_variance_raw,
            "k": c.k if np.isfinite(c.k) else None,
        }

    comp = res.comparison
    return {
        "config": asdict(res.config),
        "n_fleets": int(res.claims["fleet"].nunique()),
        "years": [int(res.claims["year"].min()), int(res.claims["year"].max())],
        "n_individual_claims": int(len(res.portfolio.individual_claims)),
        "frequency": {**asdict(res.frequency), "dispersion_ratio": res.frequency.dispersion_ratio},
        "severity": {
            name: {"params": f.params, "aic": f.aic, "ks_statistic": f.ks_statistic,
                   "ks_p_value": f.ks_p_value, "mean": f.mean}
            for name, f in res.severity.items()
        },
        "best_severity": res.best_severity.family,
        "bootstrap_mean_claims_per_unit": {
            k: getattr(res.bootstrap, k) for k in ("estimate", "std_error", "ci_low", "ci_high", "confidence")
        },
        "ebct1": cred(res.ebct1),
        "ebct2": cred(res.ebct2),
        "gbm": {res.config.vol_frequency: asdict(res.gbm),
                ("monthly" if res.config.vol_frequency == "daily" else "daily"): asdict(res.gbm_alt)},
        "portfolio_average": {
            "ebct_premium": float(comp["ebct_premium"].mean()),
            "bs_premium": float(comp["bs_premium"].mean()),
            "stop_loss_premium": float(comp["stop_loss_premium"].mean()),
            "bs_to_ebct": float(comp["bs_to_ebct"].mean()),
        },
    }


def write(res: PipelineResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.claims.to_csv(out / "claims_standardised.csv", index=False)
    res.ebct1.table().to_csv(out / "ebct_model_1.csv", index=False)
    res.ebct2.table().to_csv(out / "ebct_model_2.csv", index=False)
    res.forecast.to_csv(out / "ebct_forecast.csv", index=False)
    res.comparison.to_csv(out / "premium_comparison.csv", index=False)
    res.sensitivity.to_csv(out / "sensitivity_grid.csv", index=False)
    res.stress.to_csv(out / "stress_test.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary(res), indent=2, default=float))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", dest="data_dir", help="folder with claims CSVs (default: synthetic)")
    ap.add_argument("--out", dest="out_dir", default="results")
    ap.add_argument("--deductible", type=float, default=DEFAULT_DEDUCTIBLE)
    ap.add_argument("--rate", dest="risk_free_rate", type=float, default=DEFAULT_RATE)
    ap.add_argument("--term", type=float, default=1.0)
    ap.add_argument("--vol-frequency", choices=["daily", "monthly"], default="daily")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    res = run(PipelineConfig(**vars(args)))
    out = write(res, args.out_dir)

    pd.set_option("display.width", 140)
    print(f"sigma* ({res.config.vol_frequency}) = {res.gbm.sigma:.4f}   "
          f"best severity = {res.best_severity.family}   "
          f"Poisson rate = {res.frequency.rate:.4f} (dispersion ratio {res.frequency.dispersion_ratio:.2f})")
    cols = ["fleet", "Z", "ebct1_premium", "ebct_premium", "bs_premium", "stop_loss_premium", "delta", "vega"]
    print(res.comparison[cols].round(3).to_string(index=False))
    print(f"\nresults written to {out.resolve()}")


if __name__ == "__main__":
    main()
