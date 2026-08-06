# =============================================================================
# Chapter 6c, Exercise 4: MNAR sensitivity analysis (conceptual + code sketch)
# BMI suspected MNAR (high-BMI patients less likely to be weighed): reason
# about the bias and sketch a delta-adjustment sensitivity analysis.
# =============================================================================

library(tidyverse)
library(mice)

# -----------------------------------------------------------------------------
# (a) Why standard MI (which assumes MAR) may UNDERESTIMATE the association.
#
#     Standard MI fills the gaps using the observed data under a MAR model, so
#     imputed BMIs are drawn towards the observed (lower) range. If the truly
#     missing patients had systematically HIGHER BMI, the imputations are too
#     low and the upper tail of BMI -- the part most strongly linked to the
#     outcome -- is under-represented, so the fitted BMI-outcome association is
#     biased towards zero (attenuated).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (b) Delta-adjustment: direction and magnitude.
#
#     A delta (pattern-mixture) adjustment adds a fixed offset delta to the
#     imputed BMIs to represent "the unweighed patients were heavier than MAR
#     predicts". Here delta should be POSITIVE (shift imputed BMIs UP), because
#     the suspected mechanism removes high values. The magnitude spans a
#     clinically plausible range, e.g. 0 to +5 BMI units (0, +1, +2, +3, +5),
#     ideally anchored by external knowledge of how much heavier the missing
#     group is thought to be.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (c) Reporting across a range of delta values (illustrative code).
#     For each delta: impute under MAR, add delta to the imputed BMIs only,
#     refit the logistic model on each completed set, pool with Rubin's rules,
#     and tabulate the pooled BMI coefficient (+CI) as a function of delta.
#     A finding that stays clearly non-null across the range is robust to MNAR.
# -----------------------------------------------------------------------------

set.seed(42)
n <- 800
full <- tibble(
  age = rnorm(n, 60, 12),
  bmi = rnorm(n, 28, 5),
  sbp = 100 + 0.4 * age + 0.6 * bmi + rnorm(n, 0, 10),
  event = rbinom(n, 1, plogis(-6 + 0.04 * age + 0.03 * bmi + 0.02 * sbp))
)
# MNAR-style deletion: higher BMI => more likely missing (for illustration).
p_missing_bmi <- plogis(-2 + 0.15 * (full$bmi - 28))
miss_bmi      <- rbinom(n, 1, p_missing_bmi) == 1
missing_data  <- full
missing_data$bmi[miss_bmi] <- NA

cat("=== Exercise 4: MNAR delta-adjustment sensitivity analysis ===\n\n")
cat(sprintf("BMI missing (MNAR mechanism): %d (%.1f%%)\n\n",
            sum(miss_bmi), 100 * mean(miss_bmi)))

# Impute ONCE (m = 20) under MAR, then re-use the imputations with each shift.
imp   <- mice(missing_data, m = 20, method = "pmm", seed = 123, printFlag = FALSE)
where_bmi <- is.na(missing_data$bmi)

deltas <- c(0, 1, 2, 3, 5)   # positive shifts on the BMI scale
results <- lapply(deltas, function(d) {
  shifted <- lapply(seq_len(imp$m), function(k) {
    dat <- complete(imp, k)
    dat$bmi[where_bmi] <- dat$bmi[where_bmi] + d   # delta adjustment
    dat
  })
  ests <- sapply(shifted, function(dat) coef(glm(event ~ age + bmi + sbp,
                                                 data = dat, family = binomial))["bmi"])
  vars <- sapply(shifted, function(dat) {
    s <- summary(glm(event ~ age + bmi + sbp, data = dat, family = binomial))
    s$coefficients["bmi", "Std. Error"]^2
  })
  qbar <- mean(ests)
  ubar <- mean(vars)
  b    <- var(ests)
  se   <- sqrt(ubar + (1 + 1 / imp$m) * b)         # Rubin's total variance
  c(delta = d, bmi_coef = qbar, se = se,
    lo = qbar - 1.96 * se, hi = qbar + 1.96 * se)
})
tab <- as.data.frame(do.call(rbind, results))

cat("Pooled BMI coefficient across delta (truth = 0.03):\n")
print(format(tab, digits = 4), row.names = FALSE)
cat("\nReading the table: as delta increases (assuming heavier unweighed\n")
cat("patients), the pooled BMI coefficient moves away from the attenuated\n")
cat("MAR value (delta = 0). If the coefficient and CI stay clearly positive\n")
cat("across the plausible delta range, the BMI-outcome association is robust\n")
cat("to this MNAR concern; if it collapses under a mild shift, interpret with\n")
cat("caution.\n")
