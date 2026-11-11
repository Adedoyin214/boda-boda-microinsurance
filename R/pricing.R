# Black-Scholes call replication of a micro-insurance policy (base R).
# Mirrors micro_insurance/pricing.py. All functions are vectorised (recycling).
#   s = expected annual treatment cost, k = deductible, tau = term (years),
#   r = risk-free rate, sigma = annualised claim-cost volatility.

.bs_inputs <- function(s, k, tau, r, sigma) {
  len <- max(length(s), length(k), length(tau), length(r), length(sigma))
  v <- lapply(list(s = s, k = k, tau = tau, r = r, sigma = sigma), function(a) rep_len(as.double(a), len))
  if (any(v$s <= 0)) stop("underlying S must be strictly positive")
  if (any(v$k < 0)) stop("deductible K must be non-negative")
  if (any(v$tau < 0)) stop("time to expiry must be non-negative")
  if (any(v$sigma < 0)) stop("volatility must be non-negative")
  v$regular <- v$k > 0 & v$sigma * sqrt(v$tau) > 0
  v
}

bs_call <- function(s, k, tau, r, sigma) {
  v <- .bs_inputs(s, k, tau, r, sigma)
  disc_k <- v$k * exp(-v$r * v$tau)
  out <- pmax(v$s - disc_k, 0)
  i <- v$regular
  if (any(i)) {
    vst <- v$sigma[i] * sqrt(v$tau[i])
    d1 <- (log(v$s[i] / v$k[i]) + (v$r[i] + 0.5 * v$sigma[i]^2) * v$tau[i]) / vst
    out[i] <- v$s[i] * pnorm(d1) - disc_k[i] * pnorm(d1 - vst)
  }
  out
}

bs_greeks <- function(s, k, tau, r, sigma) {
  v <- .bs_inputs(s, k, tau, r, sigma)
  disc <- exp(-v$r * v$tau)
  itm <- as.double(v$s > v$k * disc)
  res <- data.frame(
    price = bs_call(s, k, tau, r, sigma), delta = itm, gamma = 0, vega = 0,
    rho = v$k * v$tau * disc * itm, theta = -v$r * v$k * disc * itm
  )
  i <- v$regular
  if (any(i)) {
    st <- sqrt(v$tau[i]); vst <- v$sigma[i] * st
    d1 <- (log(v$s[i] / v$k[i]) + (v$r[i] + 0.5 * v$sigma[i]^2) * v$tau[i]) / vst
    d2 <- d1 - vst
    res$delta[i] <- pnorm(d1)
    res$gamma[i] <- dnorm(d1) / (v$s[i] * vst)
    res$vega[i] <- v$s[i] * dnorm(d1) * st
    res$rho[i] <- v$k[i] * v$tau[i] * disc[i] * pnorm(d2)
    res$theta[i] <- -v$s[i] * dnorm(d1) * v$sigma[i] / (2 * st) - v$r[i] * v$k[i] * disc[i] * pnorm(d2)
  }
  res
}

bs_monte_carlo <- function(s, k, tau, r, sigma, n_paths = 200000, seed = 0) {
  set.seed(seed)
  st <- s * exp((r - 0.5 * sigma^2) * tau + sigma * sqrt(tau) * rnorm(n_paths))
  pay <- exp(-r * tau) * pmax(st - k, 0)
  c(price = mean(pay), std_error = sd(pay) / sqrt(n_paths))
}

loaded_premium <- function(pure_premium, expense_loading = 0, risk_margin = 0) {
  if (expense_loading < 0 || expense_loading >= 1 || risk_margin < 0)
    stop("expense_loading must be in [0, 1) and risk_margin >= 0")
  pure_premium * (1 + risk_margin) / (1 - expense_loading)
}
