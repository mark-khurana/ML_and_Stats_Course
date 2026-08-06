# =============================================================================
# Chapter 6c, Exercise 2: Quantify the cost of complete-case analysis
# Add 20% MAR missingness in sbp on top of missing BMI, then compare the
# complete-case age coefficient/SE to the full-data model.
# =============================================================================

library(tidyverse)   # data simulation and wrangling

set.seed(42)
n <- 800

# --- Simulate the chapter's complete clinical cohort ---
full <- tibble(
  age = rnorm(n, 60, 12),
  bmi = rnorm(n, 28, 5),
  sbp = 100 + 0.4 * age + 0.6 * bmi + rnorm(n, 0, 10),   # systolic BP
  event = rbinom(n, 1, plogis(-6 + 0.04 * age + 0.03 * bmi + 0.02 * sbp))
)

# --- Induce MAR missingness in BMI (older patients more likely missing) ---
p_missing_bmi <- plogis(-2 + 0.05 * (full$age - 60))
miss_bmi      <- rbinom(n, 1, p_missing_bmi) == 1

# --- ADD ~20% MAR missingness in sbp, also depending on observed age ---
# Intercept chosen so the marginal missing fraction is about 20%.
p_missing_sbp <- plogis(-1.4 + 0.03 * (full$age - 60))
miss_sbp      <- rbinom(n, 1, p_missing_sbp) == 1

missing_data <- full
missing_data$bmi[miss_bmi] <- NA
missing_data$sbp[miss_sbp] <- NA

# --- (a) How many complete cases remain? ---
n_complete <- sum(complete.cases(missing_data))
cat("=== Exercise 2: Cost of complete-case analysis ===\n\n")
cat(sprintf("BMI missing:            %d (%.1f%%)\n",
            sum(miss_bmi), 100 * mean(miss_bmi)))
cat(sprintf("SBP missing:            %d (%.1f%%)\n",
            sum(miss_sbp), 100 * mean(miss_sbp)))
cat(sprintf("Complete cases (a):     %d of %d (%.1f%%)\n\n",
            n_complete, n, 100 * n_complete / n))

# --- (b) Full-data model vs complete-case model: age coefficient & SE ---
fit_full <- glm(event ~ age + bmi + sbp, data = full, family = binomial)
fit_cc   <- glm(event ~ age + bmi + sbp, data = missing_data, family = binomial)

sm_full <- summary(fit_full)$coefficients
sm_cc   <- summary(fit_cc)$coefficients

cat("--- (b) age coefficient (data-generating truth = 0.04) ---\n")
cat(sprintf("Full-data:      coef = %+.4f   SE = %.4f   (n = %d)\n",
            sm_full["age", "Estimate"], sm_full["age", "Std. Error"],
            length(fit_full$y)))
cat(sprintf("Complete-case:  coef = %+.4f   SE = %.4f   (n = %d)\n\n",
            sm_cc["age", "Estimate"], sm_cc["age", "Std. Error"],
            length(fit_cc$y)))
cat(sprintf("SE inflation (complete-case / full-data): %.2fx\n\n",
            sm_cc["age", "Std. Error"] / sm_full["age", "Std. Error"]))

# --- (c) Why dropping rows became much more costly ---
cat("--- (c) Comment ---\n")
cat("Requiring BOTH bmi and sbp to be present removes any patient missing\n")
cat("either one, so the two missing-data fractions compound -- the surviving\n")
cat("subset shrinks far more than either variable alone, and because both\n")
cat("gaps are age-driven the remainder is increasingly younger-skewed,\n")
cat("giving a smaller, less representative sample with larger standard errors.\n")
