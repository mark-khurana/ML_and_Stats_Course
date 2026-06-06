# =============================================================================
# Chapter 17 - Exercise 3: IPTW in R
# (Exercise specifies Python, but we provide an R version as well)
# Beta-blocker use and 1-year mortality using IPTW
# =============================================================================

library(tidyverse)
library(WeightIt)
library(cobalt)
library(survey)
library(broom)

# --- Simulate the dataset (same DGP as Exercise 2) ---
set.seed(123)
n <- 1500

exercise_dat <- tibble(
  age = rnorm(n, 70, 8),
  creatinine = rnorm(n, 1.2, 0.4),
  heart_failure = rbinom(n, 1, 0.35),
  prior_mi = rbinom(n, 1, 0.20),
  treatment = rbinom(n, 1, plogis(-1 + 0.02*age + 0.3*heart_failure +
                                     0.5*prior_mi - 0.8*creatinine)),
  death_1yr = rbinom(n, 1, plogis(-2 + 0.05*age + 0.4*heart_failure +
                                     0.6*prior_mi + 1.0*creatinine -
                                     0.7*treatment))
)

# --- Part (a): Fit PS model and compute stabilised IPTW weights ---
w_out <- weightit(treatment ~ age + creatinine + heart_failure + prior_mi,
                  data = exercise_dat,
                  method = "ps",
                  estimand = "ATE")

cat("Weight summary:\n")
summary(w_out)

# Check for extreme weights
cat("\nWeight distribution:\n")
cat("  Min:", min(w_out$weights), "\n")
cat("  Max:", max(w_out$weights), "\n")
cat("  Mean:", mean(w_out$weights), "\n")
cat("  SD:", sd(w_out$weights), "\n")

# --- Part (b): Assess covariate balance using weighted SMDs ---
cat("\nBalance table:\n")
bt <- bal.tab(w_out, thresholds = c(m = 0.1))
print(bt)

# Love plot
love.plot(w_out,
          thresholds = c(m = 0.1),
          binary = "std",
          var.order = "unadjusted",
          title = "Covariate Balance: Before and After IPTW",
          colors = c("#D55E00", "#0072B2"))

# --- Part (c): Estimate ATE using weighted regression ---
d_weighted <- svydesign(ids = ~1, weights = w_out$weights, data = exercise_dat)

# Logistic regression (odds ratio)
ate_model <- svyglm(death_1yr ~ treatment, design = d_weighted,
                    family = quasibinomial)
cat("\nIPTW ATE estimate (logistic):\n")
print(tidy(ate_model, conf.int = TRUE, exponentiate = TRUE))

# Linear probability model (risk difference)
rd_model <- svyglm(death_1yr ~ treatment, design = d_weighted,
                   family = gaussian())
cat("\nIPTW ATE estimate (risk difference):\n")
print(tidy(rd_model, conf.int = TRUE))

# --- Part (d): Sensitivity analysis with simulated unmeasured confounder ---
cat("\n--- Sensitivity Analysis: Unmeasured Confounder ---\n")

# Simulate an unmeasured confounder U that affects both treatment and outcome
set.seed(456)
U <- rnorm(n, 0, 1)

# Re-simulate treatment and outcome with U included
exercise_dat_u <- tibble(
  age = exercise_dat$age,
  creatinine = exercise_dat$creatinine,
  heart_failure = exercise_dat$heart_failure,
  prior_mi = exercise_dat$prior_mi,
  U = U,
  treatment = rbinom(n, 1, plogis(-1 + 0.02*age + 0.3*heart_failure +
                                     0.5*prior_mi - 0.8*creatinine +
                                     0.6*U)),  # U affects treatment
  death_1yr = rbinom(n, 1, plogis(-2 + 0.05*age + 0.4*heart_failure +
                                     0.6*prior_mi + 1.0*creatinine -
                                     0.7*treatment +
                                     0.8*U))   # U affects outcome
)

# Analysis WITHOUT adjusting for U (biased)
w_no_u <- weightit(treatment ~ age + creatinine + heart_failure + prior_mi,
                   data = exercise_dat_u,
                   method = "ps", estimand = "ATE")

d_no_u <- svydesign(ids = ~1, weights = w_no_u$weights, data = exercise_dat_u)
model_no_u <- svyglm(death_1yr ~ treatment, design = d_no_u,
                     family = quasibinomial)
res_no_u <- tidy(model_no_u, conf.int = TRUE, exponentiate = TRUE)

cat("\nWithout adjusting for U:\n")
print(res_no_u)

# Analysis WITH adjusting for U (unbiased)
w_with_u <- weightit(treatment ~ age + creatinine + heart_failure + prior_mi + U,
                     data = exercise_dat_u,
                     method = "ps", estimand = "ATE")

d_with_u <- svydesign(ids = ~1, weights = w_with_u$weights, data = exercise_dat_u)
model_with_u <- svyglm(death_1yr ~ treatment, design = d_with_u,
                       family = quasibinomial)
res_with_u <- tidy(model_with_u, conf.int = TRUE, exponentiate = TRUE)

cat("\nWith adjusting for U:\n")
print(res_with_u)

cat("\n--- Interpretation ---\n")
cat("The unmeasured confounder U (with associations of 0.6 with treatment\n")
cat("and 0.8 with outcome on the log-odds scale) shifts the treatment effect\n")
cat("estimate when not adjusted for. This demonstrates:\n")
cat("  1. IPTW cannot correct for unmeasured confounding.\n")
cat("  2. Sensitivity analysis helps quantify how strong a confounder\n")
cat("     would need to be to change conclusions.\n")
cat("  3. The E-value provides a formal framework for this assessment.\n")
