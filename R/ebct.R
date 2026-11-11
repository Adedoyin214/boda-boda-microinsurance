# Empirical Bayes Credibility Theory - Models I and II (base R).
# Mirrors micro_insurance/credibility.py. x, y, p are matrices: rows = risks, cols = years.

.check_matrix <- function(x, name) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) < 2) stop("EBCT needs at least 2 risks")
  if (ncol(x) < 2) stop("EBCT needs at least 2 years of data")
  if (any(!is.finite(x))) stop(name, " contains non-finite values")
  x
}

.risk_names <- function(x) {
  if (is.null(rownames(x))) paste0("risk_", seq_len(nrow(x))) else rownames(x)
}

ebct_model_1 <- function(x) {
  x <- .check_matrix(x, "x")
  n <- ncol(x)
  risk_means <- rowMeans(x)
  grand_mean <- mean(risk_means)
  within <- mean(apply(x, 1, var))
  between_raw <- var(risk_means) - within / n
  between <- max(between_raw, 0)
  if (between > 0) {
    k <- within / between
    z <- n / (n + k)
  } else {
    k <- Inf
    z <- 0
  }
  z <- rep(z, nrow(x))
  structure(list(
    model = "EBCT Model I", fleets = .risk_names(x), grand_mean = grand_mean,
    within_variance = within, between_variance = between, between_variance_raw = between_raw,
    k = k, risk_means = unname(risk_means), z = z,
    premium_per_unit = unname(z * risk_means + (1 - z) * grand_mean)
  ), class = "ebct_result")
}

ebct_model_2 <- function(y, p) {
  y <- .check_matrix(y, "y")
  p <- .check_matrix(p, "p")
  if (!all(dim(y) == dim(p))) stop("y and p must have the same shape")
  if (any(p <= 0)) stop("volumes P_ij must be strictly positive")
  N <- nrow(y); n <- ncol(y)
  x <- y / p
  p_i <- rowSums(p)
  p_tot <- sum(p_i)
  p_star <- sum(p_i * (1 - p_i / p_tot)) / (N * n - 1)
  risk_means <- rowSums(p * x) / p_i
  grand_mean <- sum(p * x) / p_tot
  within <- mean(rowSums(p * (x - risk_means)^2) / (n - 1))
  total_disp <- sum(p * (x - grand_mean)^2) / (N * n - 1)
  between_raw <- (total_disp - within) / p_star
  between <- max(between_raw, 0)
  if (between > 0) {
    k <- within / between
    z <- p_i / (p_i + k)
  } else {
    k <- Inf
    z <- rep(0, N)
  }
  structure(list(
    model = "EBCT Model II", fleets = .risk_names(y), grand_mean = grand_mean,
    within_variance = within, between_variance = between, between_variance_raw = between_raw,
    k = k, risk_means = unname(risk_means), z = unname(z),
    premium_per_unit = unname(z * risk_means + (1 - z) * grand_mean), volume = unname(p_i)
  ), class = "ebct_result")
}

# Long fleet-year data.frame (fleet, year, claims_amount, policyholders) -> list(y, p)
claims_to_matrices <- function(df) {
  need <- c("fleet", "year", "claims_amount", "policyholders")
  miss <- setdiff(need, names(df))
  if (length(miss)) stop("claims table is missing required columns: ", paste(miss, collapse = ", "))
  y <- tapply(df$claims_amount, list(df$fleet, df$year), sum)
  p <- tapply(df$policyholders, list(df$fleet, df$year), sum)
  if (anyNA(y) || anyNA(p)) stop("EBCT needs a balanced panel: every fleet must have every year")
  list(y = y, p = p)
}

ebct_from_claims <- function(df, model = 2) {
  m <- claims_to_matrices(df)
  if (model == 1) return(ebct_model_1(m$y / m$p))
  if (model == 2) return(ebct_model_2(m$y, m$p))
  stop("model must be 1 or 2")
}

as.data.frame.ebct_result <- function(x, ...) {
  out <- data.frame(fleet = x$fleets, risk_mean = x$risk_means, Z = x$z,
                    credibility_premium = x$premium_per_unit)
  if (!is.null(x$volume)) out <- cbind(out[1], total_volume = x$volume, out[-1])
  out
}

print.ebct_result <- function(x, ...) {
  cat(x$model, "\n")
  cat(sprintf("  E[m(theta)] = %.4f  E[s2(theta)] = %.4f  Var[m(theta)] = %.4f  k = %.4f\n",
              x$grand_mean, x$within_variance, x$between_variance, x$k))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
