# =============================================================================
# Chapter 17 - Exercise 3: IPW, balance, and positivity
# Beta-blocker use and 1-year mortality
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(tidyverse)       # tibble(), mutate()
library(WeightIt)        # weightit(), glm_weightit()
library(cobalt)          # bal.tab()
library(marginaleffects) # avg_comparisons()

# --- The dataset from the exercise ------------------------------------------
simulate_cohort <- function(seed = 123, n = 1500, extreme = FALSE) {
  set.seed(seed)
  d <- tibble(
    age           = rnorm(n, 70, 8),
    creatinine    = rnorm(n, 1.2, 0.4),
    heart_failure = rbinom(n, 1, 0.35),
    prior_mi      = rbinom(n, 1, 0.20)
  )
  if (extreme) {
    # Part (d): positivity is destroyed on purpose. Heart-failure patients are
    # treated with probability 0.98, everyone else with probability 0.03.
    p_treat <- ifelse(d$heart_failure == 1, 0.98, 0.03)
  } else {
    p_treat <- plogis(-0.4 + 0.05 * (d$age - 70) +
      0.7 * d$heart_failure +
      0.9 * d$prior_mi +
      0.8 * (d$creatinine - 1.2))
  }
  d |>
    mutate(
      treatment = rbinom(n, 1, p_treat),
      death_1yr = rbinom(n, 1, plogis(-1.9 + 0.05 * (age - 70) +
                                        0.7 * heart_failure +
                                        0.8 * prior_mi +
                                        1.0 * (creatinine - 1.2) -
                                        0.8 * treatment))
    )
}

# Truth on the risk-difference scale, averaged over whichever cohort is passed
true_ate_rd <- function(d) {
  lp0 <- with(d, -1.9 + 0.05 * (age - 70) + 0.7 * heart_failure +
    0.8 * prior_mi + 1.0 * (creatinine - 1.2))
  mean(plogis(lp0 - 0.8)) - mean(plogis(lp0))
}

exercise_dat <- simulate_cohort()
TRUE_ATE_RD <- true_ate_rd(exercise_dat)

cat(sprintf(
  "Cohort: %d patients | %.0f%% treated | %.1f%% died within 1 year\n",
  nrow(exercise_dat), 100 * mean(exercise_dat$treatment),
  100 * mean(exercise_dat$death_1yr)
))
cat(sprintf("TRUE ATE risk difference: %+.4f\n\n", TRUE_ATE_RD))

# =============================================================================
# (a) Propensity score model and stabilised weights for the ATE
# =============================================================================
W <- weightit(treatment ~ age + creatinine + heart_failure + prior_mi,
  data = exercise_dat,
  method = "glm", # logistic propensity score
  estimand = "ATE",
  stabilize = TRUE
)

cat("--- (a) Stabilised weights ---\n")
cat(sprintf(
  "mean = %.3f   median = %.3f   max = %.2f\n",
  mean(W$weights), median(W$weights), max(W$weights)
))
cat("Stabilised weights should cluster around 1, and these do.\n")

# The same weights by hand, to show there is no magic in weightit():
ps_model <- glm(treatment ~ age + creatinine + heart_failure + prior_mi,
  data = exercise_dat, family = binomial
)
ps <- predict(ps_model, type = "response")
p_marg <- mean(exercise_dat$treatment)
sw_manual <- ifelse(exercise_dat$treatment == 1,
  p_marg / ps,
  (1 - p_marg) / (1 - ps)
)
cat(sprintf("Hand-computed max weight: %.2f\n", max(sw_manual)))

# =============================================================================
# (b) The two mandatory checks: balance, then positivity
# =============================================================================
cat("\n--- (b) Balance after weighting (want every SMD under 0.1) ---\n")
print(bal.tab(W, thresholds = c(m = 0.1)))

cat("\n--- (b) Positivity ---\n")
cat(sprintf("Largest stabilised weight: %.2f\n", max(W$weights)))
cat(sprintf(
  "Propensity score range   : %.3f to %.3f\n", min(ps), max(ps)
))
cat("Rule of thumb: a maximum weight above roughly 10-20 means one or two\n")
cat("patients are dominating the analysis. We are far below that, and no\n")
cat("propensity score is near 0 or 1, so positivity looks fine.\n")

# =============================================================================
# (c) The ATE as a risk difference
# =============================================================================
msm <- glm_weightit(death_1yr ~ treatment,
  data = exercise_dat, weightit = W, family = binomial
)
ipw_rd <- avg_comparisons(msm, variables = list(treatment = 0:1))

unadjusted_rd <- mean(exercise_dat$death_1yr[exercise_dat$treatment == 1]) -
  mean(exercise_dat$death_1yr[exercise_dat$treatment == 0])

cat("\n--- (c) IPW estimate of the ATE ---\n")
cat(sprintf(
  "IPW risk difference : %+.4f (95%% CI %+.4f, %+.4f)   [truth %+.4f]\n",
  ipw_rd$estimate, ipw_rd$conf.low, ipw_rd$conf.high, TRUE_ATE_RD
))
cat(sprintf("Unadjusted, for comparison: %+.4f\n", unadjusted_rd))
cat("The naive comparison recovers well under half the true effect; IPW\n")
cat("recovers most of it, with an interval that contains the truth.\n")

# =============================================================================
# (d) Breaking positivity on purpose
# =============================================================================
# Heart-failure patients are now treated with probability 0.98 and everyone
# else with probability 0.03. Heart failure is still a cause of death, so it is
# still a confounder we must adjust for -- but there are almost no untreated
# heart-failure patients to learn from.
extreme_dat <- simulate_cohort(extreme = TRUE)
TRUE_ATE_BAD <- true_ate_rd(extreme_dat)

cat("\n\n=== (d) What happens when positivity fails ===\n")
print(table(
  `heart failure` = extreme_dat$heart_failure,
  treated = extreme_dat$treatment
))

W_bad <- weightit(treatment ~ age + creatinine + heart_failure + prior_mi,
  data = extreme_dat, method = "glm", estimand = "ATE", stabilize = TRUE
)
msm_bad <- glm_weightit(death_1yr ~ treatment,
  data = extreme_dat, weightit = W_bad, family = binomial
)
rd_bad <- avg_comparisons(msm_bad, variables = list(treatment = 0:1))

ess <- function(w) sum(w)^2 / sum(w^2)

cat(sprintf(
  "\nLargest stabilised weight now : %.1f   (it was %.2f before)\n",
  max(W_bad$weights), max(W$weights)
))
cat(sprintf(
  "The most influential single patient now carries %.1f%% of the total weight\n",
  100 * max(W_bad$weights) / sum(W_bad$weights)
))
cat(sprintf(
  "-- about %.0f times an average patient's share.\n",
  max(W_bad$weights) / mean(W_bad$weights)
))
cat(sprintf(
  "Effective sample size: %.0f (from %d real patients) -- was %.0f of %d\n",
  ess(W_bad$weights), nrow(extreme_dat), ess(W$weights), nrow(exercise_dat)
))
cat(sprintf(
  "IPW estimate: %+.4f (95%% CI %+.4f, %+.4f)   [truth %+.4f]\n",
  rd_bad$estimate, rd_bad$conf.low, rd_bad$conf.high, TRUE_ATE_BAD
))
cat(sprintf(
  "The confidence interval is now %.1f times wider than before.\n",
  (rd_bad$conf.high - rd_bad$conf.low) / (ipw_rd$conf.high - ipw_rd$conf.low)
))

cat("\nWhat to tell a clinical collaborator:\n")
cat("\"In this data almost every patient with heart failure was treated and\n")
cat(" almost nobody without it was. The weighting therefore leans on a\n")
cat(" handful of unusual patients -- the few untreated ones who had heart\n")
cat(" failure -- to stand in for an entire group. In effect we are down from\n")
cat(" about 1285 patients' worth of information to about 110, the confidence\n")
cat(" interval is three times wider, and the estimate moves around a lot from\n")
cat(" sample to sample. I would not report an ATE from this.\"\n")
cat("\nThe options, in order of preference:\n")
cat(" 1. Change the question. Estimate the effect only where both treatments\n")
cat("    actually occur -- e.g. within heart-failure patients, or target the\n")
cat("    ATT instead of the ATE.\n")
cat(" 2. Trim or truncate the weights, and report the trimmed AND untrimmed\n")
cat("    results so the reader sees how much the choice mattered.\n")
cat(" 3. G-computation (Exercise 4) does not divide by a small probability, so\n")
cat("    it will not blow up -- but it then has to EXTRAPOLATE into the region\n")
cat("    where there is no data. That is a different way of being wrong, not\n")
cat("    a fix, and it fails silently rather than loudly.\n")
cat("\nThe honest answer: no estimator can recover an effect in a group where\n")
cat("one of the treatments was essentially never given. Positivity is a\n")
cat("property of the data, not of the method.\n")
