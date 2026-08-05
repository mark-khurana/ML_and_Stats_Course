# =============================================================================
# Chapter 17c, Exercise 1: Decompose a Known Mediation Effect
# Simulate exposure -> mediator -> outcome and recover NDE, NIE, total, prop. med.
# =============================================================================
# We first implement the regression-based / product-of-coefficients estimator by
# hand with base R lm(), so that every step is visible. For a continuous mediator
# and continuous outcome with NO exposure-mediator interaction this is exactly
# what the packages compute. A bootstrap gives the CI for the indirect effect.
# Part (d) then cross-checks the answer against the regmedint package.
#
# Libraries -------------------------------------------------------------------
# Base R only for parts (a)-(c). Part (d) is optional and needs:
#   install.packages("regmedint")
# (Note: install.packages("CMAverse") does NOT work -- that package is on GitHub
#  only. See the chapter callout, or use remotes::install_github("BS1125/CMAverse").)

set.seed(42)

# --- Simulate exposure -> mediator -> outcome with a KNOWN decomposition ---
# Data-generating coefficients (the "truth"):
#   exposure -> mediator (a):        1.5
#   mediator -> outcome (b):         0.5
#   direct exposure -> outcome (c'): 1.0
a_true  <- 1.5   # effect of exposure on mediator
b_true  <- 0.5   # effect of mediator on outcome
cp_true <- 1.0   # direct effect of exposure on outcome

n <- 5000
exposure <- rbinom(n, 1, 0.5)                         # randomized-like exposure
mediator <- a_true * exposure + rnorm(n)              # continuous mediator
outcome  <- cp_true * exposure + b_true * mediator + rnorm(n)  # continuous outcome

dat <- data.frame(exposure, mediator, outcome)

# -----------------------------------------------------------------------------
# (a) TRUE natural direct/indirect effects (by construction)
# -----------------------------------------------------------------------------
# For a linear model with no exposure-mediator interaction:
#   NIE  = a * b       (path exposure -> mediator -> outcome)
#   NDE  = c'          (direct path)
#   Total = NDE + NIE
#   Proportion mediated = NIE / Total
nie_true   <- a_true * b_true          # 1.5 * 0.5 = 0.75
nde_true   <- cp_true                  # 1.0
total_true <- nde_true + nie_true      # 1.75
prop_true  <- nie_true / total_true    # 0.75 / 1.75 = 0.4286

# -----------------------------------------------------------------------------
# (b) ESTIMATE the decomposition from the data
# -----------------------------------------------------------------------------
# Mediator model: M ~ X  -> coefficient on exposure is the 'a' path
m_model <- lm(mediator ~ exposure, data = dat)
a_hat   <- coef(m_model)["exposure"]

# Outcome model: Y ~ X + M -> exposure coef is NDE (c'), mediator coef is 'b'
y_model <- lm(outcome ~ exposure + mediator, data = dat)
nde_hat <- coef(y_model)["exposure"]   # natural direct effect
b_hat   <- coef(y_model)["mediator"]   # mediator -> outcome

nie_hat   <- a_hat * b_hat             # natural indirect effect (product of coefs)
total_hat <- nde_hat + nie_hat
prop_hat  <- nie_hat / total_hat

# Bootstrap 95% CI for the indirect effect (NIE)
n_boot <- 1000
boot_nie <- numeric(n_boot)
for (i in seq_len(n_boot)) {
  idx <- sample(seq_len(n), n, replace = TRUE)
  d   <- dat[idx, ]
  bm  <- coef(lm(mediator ~ exposure, data = d))["exposure"]
  by  <- coef(lm(outcome ~ exposure + mediator, data = d))["mediator"]
  boot_nie[i] <- bm * by
}
nie_ci <- quantile(boot_nie, c(0.025, 0.975))

# -----------------------------------------------------------------------------
# Print true vs estimated
# -----------------------------------------------------------------------------
cat("=== Exercise 1: Mediation decomposition (true vs estimated) ===\n\n")
res <- data.frame(
  Quantity   = c("NDE (direct)", "NIE (indirect)", "Total effect", "Prop. mediated"),
  True       = c(nde_true, nie_true, total_true, prop_true),
  Estimated  = c(nde_hat,  nie_hat,  total_hat,  prop_hat)
)
res$True      <- round(res$True, 4)
res$Estimated <- round(res$Estimated, 4)
print(res, row.names = FALSE)

cat(sprintf("\nNIE 95%% bootstrap CI: (%.3f, %.3f)\n", nie_ci[1], nie_ci[2]))
cat(sprintf("Truth NIE = 0.75 lies inside CI: %s\n",
            nie_ci[1] <= 0.75 && 0.75 <= nie_ci[2]))

# -----------------------------------------------------------------------------
# (c) Clinician interpretation of the proportion mediated
# -----------------------------------------------------------------------------
cat("\n=== (c) One-sentence interpretation for a clinician ===\n")
cat(sprintf(
  "\"About %.0f%% of the treatment's total benefit travels through the mediator,\n",
  100 * prop_hat
))
cat("so a cheaper intervention that moved the mediator by the same amount would\n")
cat("capture roughly that share of the benefit -- but the majority of the effect\n")
cat("works by some other route, and would be lost.\"\n")

# -----------------------------------------------------------------------------
# (d) OPTIONAL cross-check against regmedint
# -----------------------------------------------------------------------------
# Never trust a hand-rolled estimator you have not checked against a package.
# regmedint is on CRAN, so this really does install and run.
if (requireNamespace("regmedint", quietly = TRUE)) {
  library(regmedint)
  fit <- regmedint(
    data = dat,
    yvar = "outcome", avar = "exposure", mvar = "mediator",
    cvar = NULL,
    a0 = 0, a1 = 1, m_cde = 0, c_cond = NULL,
    mreg = "linear", yreg = "linear",
    interaction = FALSE, casecontrol = FALSE
  )
  # summary_myreg holds the mediation decomposition: rows cde, pnde, tnie,
  # tnde, pnie, te, pm; columns est, se, Z, p, lower, upper.
  decomp <- summary(fit)$summary_myreg
  cat("\n=== (d) Cross-check with regmedint ===\n")
  print(round(decomp[c("pnde", "tnie", "te", "pm"), c("est", "lower", "upper")], 4))
  cat("\nCompare with the hand calculation above:\n")
  cat(sprintf("  pnde (= NDE) %.4f | tnie (= NIE) %.4f | te %.4f | pm %.4f\n",
              nde_hat, nie_hat, total_hat, prop_hat))
  cat("They agree, as they must: with no exposure-mediator interaction, the\n")
  cat("product-of-coefficients estimator IS the regression-based causal estimator.\n")
} else {
  cat("\n(d) Skipped: install.packages(\"regmedint\") to run the cross-check.\n")
}
