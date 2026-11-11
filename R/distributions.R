# Frequency / severity fitting, stop-loss and bootstrap (base R).
# Mirrors micro_insurance/distributions.py.

fit_poisson_frequency <- function(counts, exposure = rep(1, length(counts))) {
  if (length(counts) != length(exposure)) stop("counts and exposure must have the same length")
  if (any(counts < 0) || any(exposure <= 0)) stop("counts must be >= 0 and exposure > 0")
  rate <- sum(counts) / sum(exposure)
  expected <- rate * exposure
  df <- length(counts) - 1
  stat <- if (rate == 0) 0 else sum((counts - expected)^2 / expected)
  list(rate = rate, dispersion_statistic = stat, dispersion_df = df,
       dispersion_p_value = if (df > 0) pchisq(stat, df, lower.tail = FALSE) else NA,
       dispersion_ratio = stat / df)
}

.clean_positive <- function(x) {
  x <- as.double(x)
  if (length(x) < 2) stop("need at least 2 claim amounts")
  if (any(!is.finite(x)) || any(x <= 0)) stop("claim amounts must be finite and strictly positive")
  x
}

fit_lognormal <- function(x) {
  x <- .clean_positive(x)
  lx <- log(x)
  mu <- mean(lx); sigma <- sqrt(mean((lx - mu)^2))  # MLE (ddof = 0)
  ll <- sum(dlnorm(x, mu, sigma, log = TRUE))
  ks <- suppressWarnings(ks.test(x, "plnorm", mu, sigma))
  list(family = "lognormal", params = list(mu = mu, sigma = sigma), log_likelihood = ll,
       aic = 4 - 2 * ll, ks_statistic = unname(ks$statistic), ks_p_value = ks$p.value,
       mean = exp(mu + sigma^2 / 2))
}

fit_gamma <- function(x) {
  x <- .clean_positive(x)
  # Profile MLE: scale = mean / shape, solve log(a) - digamma(a) = log(mean) - mean(log x)
  s <- log(mean(x)) - mean(log(x))
  if (s <= 0) stop("claim amounts are all identical; cannot fit a gamma")
  a <- uniroot(function(a) log(a) - digamma(a) - s, c(1e-8, 1e8), tol = 1e-14)$root
  theta <- mean(x) / a
  ll <- sum(dgamma(x, shape = a, scale = theta, log = TRUE))
  ks <- suppressWarnings(ks.test(x, "pgamma", shape = a, scale = theta))
  list(family = "gamma", params = list(shape = a, scale = theta), log_likelihood = ll,
       aic = 4 - 2 * ll, ks_statistic = unname(ks$statistic), ks_p_value = ks$p.value,
       mean = a * theta)
}

fit_severity <- function(x) list(gamma = fit_gamma(x), lognormal = fit_lognormal(x))

best_fit <- function(fits) fits[[which.min(vapply(fits, `[[`, 0, "aic"))]]

stop_loss <- function(fit, k) {
  if (any(k < 0)) stop("deductible must be non-negative")
  p <- fit$params
  if (fit$family == "lognormal") {
    m <- exp(p$mu + p$sigma^2 / 2)
    out <- rep(m, length(k))
    j <- k > 0
    d1 <- (p$mu + p$sigma^2 - log(k[j])) / p$sigma
    out[j] <- m * pnorm(d1) - k[j] * pnorm(d1 - p$sigma)
    return(out)
  }
  if (fit$family == "gamma") {
    a <- p$shape; th <- p$scale
    return(a * th * pgamma(k, a + 1, scale = th, lower.tail = FALSE) -
             k * pgamma(k, a, scale = th, lower.tail = FALSE))
  }
  stop("unknown family")
}

bootstrap_stat <- function(x, statistic = mean, n_boot = 2000, confidence = 0.95, seed = 0) {
  set.seed(seed)
  reps <- replicate(n_boot, statistic(sample(x, replace = TRUE)))
  a <- (1 - confidence) / 2
  list(estimate = statistic(x), std_error = sd(reps),
       ci_low = unname(quantile(reps, a)), ci_high = unname(quantile(reps, 1 - a)))
}
