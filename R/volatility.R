# GBM parameter estimation from a treatment-cost series (base R).
# Mirrors micro_insurance/volatility.py.

estimate_gbm <- function(values, periods_per_year) {
  v <- as.double(values)
  if (length(v) < 3) stop("need at least 3 observations to estimate volatility")
  if (any(!is.finite(v)) || any(v <= 0)) stop("cost series must be finite and strictly positive")
  if (periods_per_year <= 0) stop("periods_per_year must be positive")
  r <- diff(log(v))
  sigma <- sd(r) * sqrt(periods_per_year)
  list(mu = mean(r) * periods_per_year + sigma^2 / 2, sigma = sigma, s0 = v[length(v)],
       n_returns = length(r), periods_per_year = periods_per_year)
}

estimate_gbm_from_frame <- function(df, freq = c("daily", "monthly"),
                                    value_col = "cost_index", date_col = "date") {
  freq <- match.arg(freq)
  d <- as.Date(df[[date_col]])
  o <- order(d); d <- d[o]; v <- df[[value_col]][o]
  if (freq == "daily") {
    spacing <- median(as.numeric(diff(d)))
    return(estimate_gbm(v, 365.25 / spacing))
  }
  month <- format(d, "%Y-%m")
  last <- tapply(seq_along(v), month, max)  # month-end level
  estimate_gbm(v[last], 12)
}
