# R library checks + parity with the Python reference implementation.
#   python tests/make_fixtures.py
#   Rscript R/tests/test_parity.R          (run from the project root)

source("R/microinsurance.R")
fx <- "tests/fixtures"
passed <- 0
check <- function(desc, got, want, tol = 1e-8) {
  got <- unlist(got); want <- unlist(want)
  ok <- length(got) == length(want) && isTRUE(all.equal(unname(got), unname(want), tolerance = tol))
  if (!ok) stop(sprintf("FAIL: %s\n  got:  %s\n  want: %s", desc,
                        paste(signif(got, 10), collapse = ", "), paste(signif(want, 10), collapse = ", ")))
  passed <<- passed + 1
}

# ---- standalone checks
check("Hull textbook call", bs_call(100, 100, 1, 0.05, 0.2), 10.450583572185565)
r1 <- ebct_model_1(matrix(c(1, 5, 3, 7), 2))
check("EBCT I hand example Z", r1$z, c(0.875, 0.875))
check("EBCT I hand example premium", r1$premium_per_unit, c(2.25, 5.75))
check("no heterogeneity -> Z = 0", ebct_model_1(matrix(c(1, 9, 9, 1), 2))$z, c(0, 0))
x <- matrix(c(110, 95, 130, 240, 260, 230, 80, 90, 70), 3, byrow = TRUE)
check("Model II with unit volume equals Model I", ebct_model_2(x, x * 0 + 1)$premium_per_unit,
      ebct_model_1(x)$premium_per_unit)
h <- 1e-4
check("delta vs finite difference", bs_greeks(90, 100, 1, 0.05, 0.3)$delta,
      (bs_call(90 + h, 100, 1, 0.05, 0.3) - bs_call(90 - h, 100, 1, 0.05, 0.3)) / (2 * h), 1e-6)
mc <- bs_monte_carlo(100, 110, 1, 0.05, 0.4, n_paths = 400000, seed = 1)
stopifnot(abs(mc[["price"]] - bs_call(100, 110, 1, 0.05, 0.4)) < 4 * mc[["std_error"]]); passed <- passed + 1

# ---- parity with Python
if (!file.exists(file.path(fx, "expected.json"))) stop("run `python tests/make_fixtures.py` first")
exp_ <- jsonlite::fromJSON(file.path(fx, "expected.json"), simplifyVector = TRUE)
claims <- read.csv(file.path(fx, "claims.csv"))
ind <- read.csv(file.path(fx, "individual_claims.csv"))
cost <- read.csv(file.path(fx, "treatment_cost_index.csv"))

for (m in 1:2) {
  got <- ebct_from_claims(claims, m)
  want <- exp_[[paste0("ebct", m)]]
  for (f in c("grand_mean", "within_variance", "between_variance", "z", "premium_per_unit"))
    check(sprintf("EBCT %d %s", m, f), got[[f]], want[[f]])
}

cs <- exp_$bs_cases
g <- bs_greeks(cs[, 1], cs[, 2], cs[, 3], cs[, 4], cs[, 5])
for (f in names(exp_$greeks)) check(paste("greeks", f), g[[f]], exp_$greeks[[f]])

sev <- fit_severity(ind$amount)
for (fam in c("lognormal", "gamma")) {
  check(paste(fam, "params"), sev[[fam]]$params, exp_$severity[[fam]]$params, 1e-6)
  check(paste(fam, "log-likelihood"), sev[[fam]]$log_likelihood, exp_$severity[[fam]]$log_likelihood, 1e-8)
  check(paste(fam, "stop-loss K=2000"), stop_loss(sev[[fam]], 2000), exp_$severity[[fam]]$stop_loss_2000, 1e-6)
}

check("Poisson rate", fit_poisson_frequency(claims$claim_count, claims$policyholders)$rate, exp_$poisson_rate)
gd <- estimate_gbm_from_frame(cost, "daily")
check("GBM daily sigma", gd$sigma, exp_$gbm_daily_sigma)
check("GBM daily mu", gd$mu, exp_$gbm_daily_mu)
check("GBM monthly sigma", estimate_gbm_from_frame(cost, "monthly")$sigma, exp_$gbm_monthly_sigma)

cat(sprintf("All %d R checks passed (R %s)\n", passed, getRversion()))
