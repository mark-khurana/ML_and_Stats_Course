# =============================================================================
# Chapter 6c, Exercise 3: Multiple imputation end to end
# m = 30 imputations with mice on the cohort with missing BMI and SBP; pool
# with Rubin's rules and compare pooled / complete-case / full-data estimates.
# =============================================================================

library(tidyverse)   # data simulation and wrangling
library(mice)        # multiple imputation by chained equations

set.seed(42)
n <- 800

# --- Simulate the chapter's complete clinical cohort ---
full <- tibble(
  age = rnorm(n, 60, 12),
  bmi = rnorm(n, 28, 5),
  sbp = 100 + 0.4 * age + 0.6 * bmi + rnorm(n, 0, 10),
  event = rbinom(n, 1, plogis(-6 + 0.04 * age + 0.03 * bmi + 0.02 * sbp))
)

# --- Induce MAR missingness in BMI and SBP (both depend on observed age) ---
p_missing_bmi <- plogis(-2 + 0.05 * (full$age - 60))
p_missing_sbp <- plogis(-1.4 + 0.03 * (full$age - 60))
missing_data  <- full
missing_data$bmi[rbinom(n, 1, p_missing_bmi) == 1] <- NA
missing_data$sbp[rbinom(n, 1, p_missing_sbp) == 1] <- NA

cat("=== Exercise 3: Multiple imputation end to end ===\n\n")
cat(sprintf("Complete cases: %d of %d\n\n",
            sum(complete.cases(missing_data)), n))

# --- Reference models: full data and complete-case ---
fit_full <- glm(event ~ age + bmi + sbp, data = full,         family = binomial)
fit_cc   <- glm(event ~ age + bmi + sbp, data = missing_data, family = binomial)

# --- (a) Multiple imputation with m = 30 ---
# Percentage of incomplete cases is high, so we follow the "m >= % missing"
# rule of thumb. The imputation model uses ALL columns, INCLUDING the outcome.
imp <- mice(missing_data, m = 30, method = "pmm", seed = 123, printFlag = FALSE)

# --- (b) Fit the logistic model on each imputed set and pool (Rubin's rules) ---
fits   <- with(imp, glm(event ~ age + bmi + sbp, family = binomial))
pooled <- summary(pool(fits), conf.int = TRUE)
rownames(pooled) <- pooled$term

# --- (c) Compare pooled vs complete-case vs full-data (focus: age coef) ---
get_age <- function(fit) {
  s <- summary(fit)$coefficients
  c(coef = s["age", "Estimate"], se = s["age", "Std. Error"])
}
full_age <- get_age(fit_full)
cc_age   <- get_age(fit_cc)
pool_age <- c(coef = pooled["age", "estimate"], se = pooled["age", "std.error"])

cat("--- (c) age coefficient (data-generating truth = 0.04) ---\n")
cmp <- data.frame(
  method = c("Full data (truth)", "Complete-case", "MI pooled (m=30)"),
  coef   = c(full_age["coef"], cc_age["coef"], pool_age["coef"]),
  se     = c(full_age["se"],   cc_age["se"],   pool_age["se"])
)
print(format(cmp, digits = 4), row.names = FALSE)

cat("\nThe MI pooled estimate uses all 800 patients and should sit between the\n")
cat("complete-case value and the full-data truth, recovering the truth best\n")
cat("while giving honest standard errors (wider than a naive single imputation).\n\n")

# Also show the full pooled table (all coefficients).
cat("--- Full pooled summary (Rubin's rules) ---\n")
print(pooled[, c("term", "estimate", "std.error", "conf.low", "conf.high", "p.value")],
      row.names = FALSE, digits = 4)

# --- (d) Imputation diagnostics ---
cat("\n--- (d) Imputation diagnostics ---\n")
cat("Observed vs imputed summaries (mean [sd]) across the 30 imputations:\n")
diag_var <- function(var) {
  obs <- missing_data[[var]][!is.na(missing_data[[var]])]
  imp_vals <- unlist(lapply(seq_len(imp$m),
                            function(k) complete(imp, k)[[var]][is.na(missing_data[[var]])]))
  cat(sprintf("  %-4s observed: %6.2f [%.2f]   imputed: %6.2f [%.2f]\n",
              var, mean(obs), sd(obs), mean(imp_vals), sd(imp_vals)))
}
diag_var("bmi")
diag_var("sbp")
cat("Imputed means/spreads that track the observed ones indicate plausible\n")
cat("imputations. For interactive checks use plot(imp) (convergence) and\n")
cat("densityplot(imp) (imputed vs observed distributions).\n")
