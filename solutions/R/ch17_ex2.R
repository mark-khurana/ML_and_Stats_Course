# =============================================================================
# Chapter 17 - Exercise 2: Propensity Score Matching in R
# Beta-blocker use and 1-year mortality
# =============================================================================

library(tidyverse)
library(MatchIt)
library(cobalt)
library(broom)

# --- Simulate the dataset (from the exercise) ---
set.seed(123)
n <- 1500
exercise_dat <- tibble(
  age = rnorm(n, 70, 8),
  creatinine = rnorm(n, 1.2, 0.4),
  heart_failure = rbinom(n, 1, 0.35),
  prior_mi = rbinom(n, 1, 0.20),
  # Confounded treatment (beta-blocker use)
  treatment = rbinom(n, 1, plogis(-1 + 0.02*age + 0.3*heart_failure +
                                     0.5*prior_mi - 0.8*creatinine)),
  # Outcome: 1-year mortality
  death_1yr = rbinom(n, 1, plogis(-2 + 0.05*age + 0.4*heart_failure +
                                     0.6*prior_mi + 1.0*creatinine -
                                     0.7*treatment))
)

cat("Dataset dimensions:", nrow(exercise_dat), "patients\n")
cat("Treatment prevalence:", mean(exercise_dat$treatment), "\n")
cat("Outcome prevalence:", mean(exercise_dat$death_1yr), "\n\n")

# --- Part (a): Estimate propensity scores using logistic regression ---
ps_model <- glm(treatment ~ age + creatinine + heart_failure + prior_mi,
                data = exercise_dat, family = binomial)

exercise_dat$ps <- predict(ps_model, type = "response")

cat("Propensity score summary:\n")
print(summary(exercise_dat$ps))

# Visualise propensity score overlap
ggplot(exercise_dat, aes(x = ps, fill = factor(treatment))) +
  geom_density(alpha = 0.5) +
  labs(x = "Propensity Score", fill = "Beta-blocker",
       title = "Propensity Score Distribution by Treatment Group") +
  scale_fill_manual(values = c("#0072B2", "#D55E00"),
                    labels = c("No", "Yes")) +
  theme_minimal()

# --- Part (b): 1:1 nearest-neighbour matching with caliper of 0.2 SD ---
m_out <- matchit(treatment ~ age + creatinine + heart_failure + prior_mi,
                 data = exercise_dat,
                 method = "nearest",
                 distance = "glm",        # logistic regression PS
                 caliper = 0.2,           # 0.2 SD of logit PS
                 ratio = 1)               # 1:1 matching

cat("\nMatching summary:\n")
summary(m_out)

# --- Part (c): Love plot to assess balance ---
love.plot(m_out,
          thresholds = c(m = 0.1),    # SMD threshold of 0.1
          binary = "std",
          var.order = "unadjusted",
          title = "Covariate Balance: Before and After Matching",
          colors = c("#D55E00", "#0072B2"))

# Also check numeric balance
cat("\nBalance table:\n")
print(bal.tab(m_out, thresholds = c(m = 0.1)))

# --- Part (d): Estimate ATT for beta-blocker use on 1-year mortality ---
m_data <- match.data(m_out)

cat("\nMatched sample size:", nrow(m_data), "patients\n")
cat("Treated in matched sample:", sum(m_data$treatment == 1), "\n")
cat("Control in matched sample:", sum(m_data$treatment == 0), "\n")

# Outcome model in matched sample
outcome_model <- glm(death_1yr ~ treatment,
                     data = m_data,
                     family = binomial,
                     weights = weights)

cat("\nATT estimate (logistic regression in matched sample):\n")
result <- tidy(outcome_model, conf.int = TRUE, exponentiate = TRUE)
print(result)

# Risk difference (linear probability model)
rd_model <- lm(death_1yr ~ treatment, data = m_data, weights = weights)
cat("\nRisk difference estimate:\n")
print(tidy(rd_model, conf.int = TRUE))

# Mortality rates in matched sample
cat("\nMortality in matched sample:\n")
cat("  Treated:", mean(m_data$death_1yr[m_data$treatment == 1]), "\n")
cat("  Control:", mean(m_data$death_1yr[m_data$treatment == 0]), "\n")

# --- Part (e): E-value calculation ---
# Extract the odds ratio from the matched analysis
or_est <- result$estimate[result$term == "treatment"]
or_lo <- result$conf.low[result$term == "treatment"]
or_hi <- result$conf.high[result$term == "treatment"]

cat("\nOdds Ratio:", round(or_est, 3),
    "(95% CI:", round(or_lo, 3), "-", round(or_hi, 3), ")\n")

# Convert OR to RR approximation (for rare outcomes, OR ~ RR)
# For the E-value, we need RR on the above-null scale
# Since treatment is protective (OR < 1), convert: RR_above_null = 1/OR
rr_est <- 1 / or_est
rr_lo_bound <- 1 / or_hi  # Note: CI bounds flip when inverting

# E-value formula: E = RR + sqrt(RR * (RR - 1))
e_value <- function(rr) {
  if (rr < 1) rr <- 1 / rr  # Convert to above-null
  return(rr + sqrt(rr * (rr - 1)))
}

e_val_point <- e_value(rr_est)

# E-value for the CI bound closest to null
# If protective, the upper bound of RR (lower bound of 1/OR) is closest to null
e_val_ci <- e_value(rr_lo_bound)

cat("\nE-value for point estimate:", round(e_val_point, 2), "\n")
cat("E-value for CI bound closest to null:", round(e_val_ci, 2), "\n")

# Interpretation
cat("\n--- E-value Interpretation ---\n")
cat("The E-value of", round(e_val_point, 2), "means that an unmeasured\n")
cat("confounder would need to be associated with BOTH beta-blocker use AND\n")
cat("1-year mortality by a risk ratio of at least", round(e_val_point, 2), "\n")
cat("(above and beyond measured confounders) to fully explain away the\n")
cat("observed protective association.\n")
cat("\nThe E-value for the confidence interval bound is", round(e_val_ci, 2), ",\n")
cat("meaning a confounder of that strength could shift the CI to include the null.\n")
cat("If these values are larger than plausible confounders, the result is robust.\n")

# Optional: use EValue package if available
# library(EValue)
# evalues.OR(or_est, lo = or_lo, hi = or_hi, rare = FALSE)
