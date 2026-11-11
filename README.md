# micro-insurance

This library prices micro-health and accident insurance for informal-sector workers, using boda-boda motorcycle-taxi operators in East Africa as the case study.

It implements and compares two pricing paradigms:

1. **Empirical Bayes Credibility Theory (EBCT).** Model I (Bühlmann) and Model II (Bühlmann-Straub, volume-weighted).
2. **Black-Scholes call-option replication.** The annual policy is treated as a European call on the accumulated treatment cost, with the deductible as the strike.

**Live dashboard: https://adedoyin214.github.io/boda-boda-microinsurance/**. It runs in the browser with no install: move the deductible, volatility and rate sliders to re-price every fleet. See [Publishing the dashboard](#publishing-the-dashboard-github-pages).

Deliverables (MScFE June roadmap):

- a Python library with an R port that matches it
- an interactive Dash sensitivity dashboard
- a technical white paper ([docs/white_paper.md](docs/white_paper.md))

## Option ↔ insurance mapping

| Option variable | Insurance parameter | Meaning |
|---|---|---|
| Underlying S | Accumulated treatment cost | Expected annual claim cost per insured (the EBCT premium) |
| Strike K | Deductible | Threshold above which the insurer pays |
| T − t | Policy term | Usually 1 year |
| r | Benchmark yield | Short-term sovereign risk-free rate |
| σ | Claim-cost volatility | Annualised std. dev. of log changes in the treatment-cost index |
| C(S, t) | Pure premium | `S·N(d1) − K·e^(−r(T−t))·N(d2)` |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dashboard,dev]"
```

## Quickstart

```python
import micro_insurance as mi

pf = mi.generate_synthetic_portfolio()                 # or mi.load_portfolio("my_data/")
claims = mi.standardise_claims(pf.claims)              # adds X_ij = Y_ij / P_ij

cred = mi.ebct_from_claims(claims, model=2)            # EBCT Model II
print(cred.table())                                    # risk means, Z_i, credibility premiums

gbm = mi.estimate_gbm_from_frame(pf.cost_index, "daily")
table = mi.compare_premiums(cred, deductible=1000, volatility=gbm.sigma, risk_free_rate=0.08)
print(table[["fleet", "ebct_premium", "bs_premium", "delta", "vega"]])

g = mi.call_greeks(s=3000, k=1000, tau=1, r=0.08, sigma=0.25)   # vectorised
```

## Command line

| Command | What it does |
|---|---|
| `python -m micro_insurance.pipeline --out results` | Runs Weeks 1–4 end to end and writes CSVs plus `summary.json` |
| `python -m micro_insurance.pipeline --data my_data --deductible 1500 --rate 0.09` | Same run on your own data |
| `python -m micro_insurance.dashboard` | Starts the dashboard at http://127.0.0.1:8050 |
| `python -m micro_insurance.dashboard --data my_data --port 8051` | Dashboard on your own data |
| `python -m micro_insurance.static_dashboard` | Rebuilds the server-free HTML dashboard at `docs/index.html` (for GitHub Pages) |

All defaults (K = KES 1,000, r = 8%) are **illustrative**, not calibrated market values.

### Input data format (`--data` folder)

| File | Columns |
|---|---|
| `claims.csv` | `fleet, year, claims_amount (Y_ij), policyholders (P_ij), claim_count (N_ij)`. Must be a balanced panel. |
| `individual_claims.csv` | `fleet, year, amount`. One row per claim. |
| `treatment_cost_index.csv` | `date, cost_index`. A daily or monthly series. |

## Publishing the dashboard (GitHub Pages)

[docs/index.html](docs/index.html) is a single, self-contained page. The model results are embedded as JSON, and Black-Scholes and its Greeks are re-computed in JavaScript (they match the Python library to within 3×10⁻⁷), so it needs no server.

1. Rebuild the page after any change to the data or models:
   `python -m micro_insurance.static_dashboard` (add `--data my_data` for your own CSVs).
2. Commit `docs/index.html` and push.
3. On GitHub, go to **Settings → Pages → Build and deployment**. Set the source to **Deploy from a branch**, the branch to `main`, and the folder to `/docs`, then save.
4. After a minute or two the dashboard is live at `https://<your-username>.github.io/<repo-name>/`. Put that link at the top of this README.

The white paper is also served, at `.../white_paper.html`.

To open it locally, double-click `docs/index.html`. Charts load Plotly from cdnjs, so you need an internet connection.

## R library

R 4.x, base R only (`jsonlite` is used only by the parity test).

```r
source("R/microinsurance.R")
claims <- read.csv("tests/fixtures/claims.csv")
cred <- ebct_from_claims(claims, model = 2); print(cred)
bs_greeks(cred$premium_per_unit, k = 1000, tau = 1, r = 0.08, sigma = 0.25)
```

On Windows, R may not be on PATH. In that case call it by its full path, e.g. `"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"`.

## Tests

```bash
python -m pytest                       # Python: 26 tests
python tests/make_fixtures.py          # writes Python reference outputs
Rscript R/tests/test_parity.R          # R: 33 checks, incl. parity with Python to 1e-8
```

The tests check against:

- the Hull textbook option value
- finite-difference Greeks
- Monte Carlo
- numerical integration of the stop-loss
- a worked EBCT example
- the Model II = Model I identity under equal volumes
- recovery of the GBM σ

## Module map

| Week | Python | R | Content |
|---|---|---|---|
| 1 | `data.py`, `distributions.py` | `distributions.R` | Validation, X_ij, Poisson/Gamma/LogNormal fits, stop-loss, bootstrap |
| 2 | `credibility.py` | `ebct.R` | EBCT Models I & II, forecasts |
| 3 | `volatility.py`, `pricing.py` | `volatility.R`, `pricing.R` | GBM σ*, Black-Scholes price and Greeks, Monte Carlo |
| 4 | `comparison.py`, `dashboard.py`, `pipeline.py` | – | EBCT vs option vs stop-loss, K × σ grids, stress tests, dashboard |
