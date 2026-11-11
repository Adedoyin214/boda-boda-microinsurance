# Pricing Micro-Insurance for Informal Workers: Empirical Bayes Credibility versus Option Replication

*Technical white paper, MScFE industrial project (June roadmap). All figures are from the synthetic portfolio produced by `python -m micro_insurance.pipeline` (seed 2026). Monetary amounts are in KES per operator-year.*

## Abstract

Motorcycle-taxi (boda-boda) operators face high accident risk, yet insurance rarely reaches them. Bundled commercial cover is expensive, and underwriters lack the data to price small, heterogeneous groups.

We build an open-source pricing library in Python and R and use it to compare two ways of pricing stand-alone accident/health cover for such groups.

- **Empirical Bayes Credibility Theory (EBCT).** It blends each group's own claims history with the portfolio experience, in proportion to how much the data can be trusted.
- **Black-Scholes replication.** It treats the policy as a European call on the accumulated treatment cost, with the deductible as the strike.

On a synthetic portfolio of 8 operator fleets over 5 years, EBCT Model II gives credibility factors of 0.74–0.92 and premiums of KES 2,100–5,500. With a KES 1,000 deductible, the option premiums are 56–83% of the EBCT premium.

The option framework is valuable mainly as a *sensitivity* tool. It gives closed-form Greeks for deductible design and for cost-inflation and rate shocks. For low deductibles the option is deep in the money, so its premium reduces to the discounted excess over the deductible, and volatility barely matters. A per-claim stop-loss benchmark shows that the choice of where the deductible applies (per claim or per year) moves premiums more than the volatility assumption does.

## 1. Motivation

More than 80% of the workforce in many developing economies is informal. Boda-boda operators provide essential transport but suffer frequent road-traffic injuries. A single hospitalisation can be a catastrophic financial shock for the household. Kenyan insurance penetration is around 3% of GDP.

Three barriers dominate:

- cost, because health/accident cover is bundled into expensive motor packages
- the lack of credit history and individual claims experience
- prohibitive risk loadings when insurers cannot measure group risk

A credible micro-insurance product needs three properties. It must be **(i)** priced from the limited data that exist, **(ii)** fair across groups with genuinely different risk, and **(iii)** cheap enough that it can be decomposed: cover sold separately, with a deductible the buyer can choose.

## 2. Data

### 2.1 Structure

The library works on a fleet-year panel: Y_ij is the aggregate claims of fleet *i* in year *j*, P_ij is the number of insured operators, and N_ij is the claim count. It also uses individual claim amounts and a treatment-cost index. Real data can be supplied as CSVs (see README). This paper uses a reproducible synthetic portfolio, because no anonymised claims data were available.

| Feature | Synthetic design |
|---|---|
| Fleets × years | 8 × 5 (2020–2024), 10,050 operator-years, 1,189 claims |
| Heterogeneity | Fleet risk multiplier ~ LogNormal(0, 0.35²) on claim frequency |
| Volume growth | 5–40% a year per fleet (this motivates EBCT Model II) |
| Frequency | N_ij ~ Poisson(0.12 × multiplier × P_ij) |
| Severity | LogNormal, median KES 18,000, σ_log = 0.9 |
| Cost index | Daily GBM, drift 8%, volatility 25%, 3 years |

### 2.2 Distribution fitting (Week 1)

**Frequency.** The pooled Poisson rate is **0.118 claims per operator-year**. The Pearson dispersion ratio is **3.15** (p < 10⁻⁹). A single Poisson rate for the whole portfolio is rejected because the fleets really do differ, which is exactly the heterogeneity that credibility theory is built to handle.

**Severity.** Both families are fitted by maximum likelihood:

| Family | Parameters | AIC | KS statistic | KS p-value |
|---|---|---|---|---|
| **LogNormal** | μ = 9.80, σ = 0.91 | **26,462** | 0.018 | 0.81 |
| Gamma | shape 1.34, scale 20,493 | 26,628 | 0.071 | < 0.001 |

The LogNormal is preferred on every criterion; the Gamma's tail is too thin. Its mean severity is KES 27,300.

**Bootstrap.** The mean claims per operator-year X̄ = 3,323 has a bootstrap standard error of 256. The 95% interval is [2,832, 3,849], so with only 40 fleet-years the portfolio mean is uncertain by about ±15%.

## 3. Empirical Bayes Credibility (Week 2)

### 3.1 Models

For N risks over n years, **Model I** estimates:

- E[m(θ)] = X̄
- E[s²(θ)] = mean within-risk variance
- Var[m(θ)] = variance of the risk means − E[s²(θ)]/n

It sets Z = n / (n + E[s²]/Var[m]).

**Model II** works with X_ij = Y_ij/P_ij and weights every term by volume. It uses:

- P* = (Nn − 1)⁻¹ Σᵢ Pᵢ(1 − Pᵢ/P)
- Var[m] = [(Nn − 1)⁻¹ Σ P_ij (X_ij − X̄)² − E[s²]] / P*
- Zᵢ = Pᵢ / (Pᵢ + E[s²]/Var[m])

In both models the premium is Zᵢ X̄ᵢ + (1 − Zᵢ) X̄. A negative Var[m] estimate is set to 0, which gives Z = 0.

### 3.2 Results

| Fleet | Volume Σ P_ij | Model I premium (Z = 0.834) | Model II Z | Model II premium |
|---|---|---|---|---|
| A | 1,886 | 3,008 | 0.919 | 2,965 |
| B | 1,642 | 3,535 | 0.908 | 3,518 |
| C | 474 | 2,426 | 0.739 | 2,324 |
| D | 995 | 5,755 | 0.856 | 5,489 |
| E | 558 | 3,407 | 0.770 | 3,486 |
| F | 1,700 | 2,962 | 0.911 | 2,948 |
| G | 1,288 | 2,194 | 0.885 | 2,108 |
| H | 1,507 | 3,300 | 0.900 | 3,321 |

Model I gives every fleet the same credibility. Model II gives the small fleets C and E less (0.74, 0.77) and the large fleets more (≈ 0.92), and so pulls the small fleets further toward the portfolio mean of 3,248. For Fleet D, the riskiest fleet (raw mean 5,866), Model II charges 5,489. That is 1.7 times the portfolio average, but it still holds back about 6% of the experience as a hedge against noise.

This is the fairness property micro-insurance needs: risk-reflective pricing that does not overreact to a small group's bad year.

## 4. Option replication (Week 3)

### 4.1 Mapping and volatility

Each fleet's EBCT Model II premium is the expected annual treatment cost S, and the deductible is K. The policy pays (cost − K)⁺, so its value is the Black-Scholes call:

C = S N(d₁) − K e^(−r(T−t)) N(d₂),   d₁ = [ln(S/K) + (r + σ²/2)(T−t)] / (σ√(T−t)),   d₂ = d₁ − σ√(T−t).

σ* is estimated from the log changes of the treatment-cost index. The daily estimate is **σ* = 24.9%** and the monthly one 23.0%, against a true value of 25%. The drift estimate (−23%, against a true +8%) shows that three years of data cannot identify the drift. That is harmless here, because risk-neutral pricing uses r and not the drift.

### 4.2 Premiums (K = 1,000, r = 8%, σ = 24.9%, T = 1)

| Fleet | EBCT II | Black-Scholes | BS / EBCT | Stop-loss (per claim) | Δ |
|---|---|---|---|---|---|
| A | 2,965 | 2,042 | 0.69 | 2,346 | 1.00 |
| B | 3,518 | 2,594 | 0.74 | 3,480 | 1.00 |
| C | 2,324 | 1,401 | 0.60 | 1,833 | 1.00 |
| D | 5,489 | 4,566 | 0.83 | 5,267 | 1.00 |
| E | 3,486 | 2,563 | 0.74 | 3,398 | 1.00 |
| F | 2,948 | 2,025 | 0.69 | 2,990 | 1.00 |
| G | 2,108 | 1,185 | 0.56 | 2,331 | 1.00 |
| H | 3,321 | 2,398 | 0.72 | 3,372 | 1.00 |
| **Average** | **3,270** | **2,347** | **0.70** | **3,127** | |

Three observations:

1. **Zero deductible means EBCT.** With K = 0 and r = 0 the option premium equals S exactly (this is verified in the tests), so the two frameworks agree on full cover.
2. **Deep in the money.** At K = 1,000 every fleet has S/K > 2, so Δ ≈ 1 and C ≈ S − K e^(−r). The deductible removes almost exactly K·e^(−r) ≈ 923 from every premium, and the volatility assumption hardly matters (Vega ≈ 0).
3. **Where the deductible applies.** The stop-loss benchmark λᵢ·E[(X − K)⁺] applies K to each claim, not to the yearly total. Since a claim averages KES 27,300, a 1,000 per-claim deductible removes only about 4% of the cost. The aggregate deductible used by the option removes about 28%. Product designers must specify which one they mean: it moves premiums more than any volatility assumption.

## 5. Sensitivities and stress testing (Week 4)

### 5.1 Deductible × volatility

Option premium for the portfolio-average contract (S = 3,248, r = 8%):

| K \ σ | 5% | 25% | 50% | 80% |
|---|---|---|---|---|
| 0 | 3,248 | 3,248 | 3,248 | 3,248 |
| 1,624 (½ S) | 1,749 | 1,749 | 1,777 | 1,898 |
| 3,248 (= S) | 253 | 451 | 749 | 1,100 |
| 4,871 (1½ S) | 0 | 43 | 291 | 671 |
| 6,495 (2 S) | 0 | 3 | 115 | 430 |

Volatility matters only once the deductible gets close to the expected cost. There, Vega reaches about 1,170 per unit of σ, so a 10-point rise in vol adds about KES 117. That band (K ≈ S) is where a catastrophe-style, high-deductible product would sit. Such a product would be affordable (KES 450 at σ = 25%, against 3,248 for full cover), but its price depends heavily on the volatility estimate.

### 5.2 Greeks as design tools

| Greek | Meaning for an insurer |
|---|---|
| Delta ∂C/∂S | Premium change per KES of expected cost. It is ≈ 1 for low deductibles, so medical inflation passes straight through to the premium. |
| Gamma ∂²C/∂S² | How fast that pass-through changes. It peaks around K ≈ S, where repricing is least stable. |
| Vega ∂C/∂σ | Exposure to uncertainty in cost growth. It is negligible for low-deductible products. |
| Rho ∂C/∂r | Higher rates *raise* the call value, because the deductible is discounted more. It is small: about +18 per 200 bp. |
| Theta | Time decay: how fast the value of the unexpired cover falls during the term (≈ −74 a year here). |

### 5.3 Stress test (portfolio-average contract S = 3,248, K = 1,000; base premium 2,325)

| Shock | Premium change |
|---|---|
| Treatment cost +10% / +20% / +50% | +14% / +28% / +70% |
| Volatility +10 / +25 vol points | ≈ 0% / +0.1% |
| Rate +200 / +500 bp | +0.8% / +1.9% |

Medical-cost inflation is the dominant risk. Because of the deductible, the premium moves *more* than one-for-one in percentage terms (Δ·S/C ≈ 1.4). Reviewing the cost index every year matters far more than refining σ or r.

## 6. Discussion

**Complementary roles.** EBCT answers *what does each group's cover cost?* It uses the claims data directly, rewards groups with good records, and is transparent to regulators. The option framework answers *how does that cost change with product design and the economy?* Its closed-form Greeks show at a glance what a higher deductible, cost inflation or a rate rise does to the premium. A practical pipeline uses EBCT for S and Black-Scholes for the deductible and sensitivity layer, which is how this library is organised.

**Affordability.** The deductible is the main affordability lever. At about KES 3,270 a year (roughly KES 9 a day before loadings), full cover is within reach of an operator who earns daily cash income. Replacing it with a catastrophe layer at K = S cuts the pure premium by about 86%, while still protecting against the large hospital bills that cause financial ruin. Gross premiums add expense and risk margins: `MicroInsuranceContract.loaded_premium`.

**Limitations.**

- *No replication argument.* Treatment costs are not traded, so the no-arbitrage derivation of Black-Scholes does not hold. The formula is a lognormal pricing kernel with r as the discount rate, not an arbitrage-free price.
- *S as expected cost.* Using the EBCT premium as the spot means the GBM describes uncertainty *around* the credibility estimate, calibrated from a cost index rather than from fleet claims. The aggregate annual claim distribution (compound Poisson–LogNormal) is not lognormal for small fleets.
- *Synthetic data.* The magnitudes show how the method behaves, not market prices. The defaults K = 1,000 and r = 8% are illustrative.
- *Balanced panel.* EBCT here requires every fleet to be observed in every year.

**Extensions.**

- Fit the volatility directly from fleet claim ratios.
- Price the aggregate deductible by Monte Carlo on the compound distribution, as an alternative to the GBM.
- Use hierarchical credibility (fleet within region).
- Use parametric (index-triggered) cover to cut claims-handling costs.

## 7. Conclusion

Credibility theory and option pricing together give a practical, low-data pricing engine for informal-sector micro-insurance. EBCT Model II produces fair, volume-aware group premiums from short histories. The Black-Scholes layer turns those premiums into a menu of deductible options with closed-form sensitivities. Medical-cost inflation and the definition of the deductible drive prices far more than volatility does.

The accompanying Python/R library, dashboard and tests make the framework reproducible and ready to calibrate on real boda-boda claims data.

## Reproducibility

```bash
python -m micro_insurance.pipeline --out results    # every number in this paper
python -m micro_insurance.dashboard                 # interactive version of Sections 4–5
python -m pytest && Rscript R/tests/test_parity.R
```

*References: Bühlmann, H. & Straub, E. (1970), "Glaubwürdigkeit für Schadensätze", Mitteilungen der Vereinigung Schweizerischer Versicherungsmathematiker 70; Gray & Pitts (2012), Risk Modelling in General Insurance; Institute and Faculty of Actuaries, CS2 Core Reading; Black & Scholes (1973), J. Political Economy 81(3); Hull, Options, Futures and Other Derivatives.*
