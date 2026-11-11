# Entry point: source("R/microinsurance.R") loads the whole R library.
#   claims <- read.csv("results/claims_standardised.csv")
#   cred   <- ebct_from_claims(claims, model = 2)
#   bs_greeks(cred$premium_per_unit, k = 1000, tau = 1, r = 0.08, sigma = 0.25)

local({
  here <- if (!is.null(sys.frame(1)$ofile)) dirname(sys.frame(1)$ofile) else "R"
  for (f in c("ebct.R", "pricing.R", "distributions.R", "volatility.R"))
    sys.source(file.path(here, f), envir = globalenv())
})
