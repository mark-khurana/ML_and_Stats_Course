# =============================================================================
# Chapter 6, Exercise 2: Build and Evaluate a Cox Model
# PBC dataset: Cox PH with age, log(bili), albumin, protime, edema
# =============================================================================

library(survival)
library(survminer)
library(tidyverse)

# --- Load and prepare data ---
data(pbc, package = "survival")

pbc_clean <- pbc %>%
  filter(!is.na(trt)) %>%
  mutate(
    status_binary = ifelse(status == 2, 1, 0),
    time_years = time / 365.25,
    trt_label = factor(trt, labels = c("D-penicillamine", "Placebo")),
    log_bili = log(bili)
  ) %>%
  select(time_years, status_binary, age, log_bili, albumin, protime, edema) %>%
  drop_na()

cat("Complete cases:", nrow(pbc_clean), "\n")
cat("Events:", sum(pbc_clean$status_binary), "\n\n")

# --- Task 1: Fit the Cox model and report hazard ratios ---
cox_model <- coxph(
  Surv(time_years, status_binary) ~ age + log_bili + albumin + protime + edema,
  data = pbc_clean
)

cat("=== Task 1: Cox Model Summary ===\n")
print(summary(cox_model))

# Extract and display HRs with 95% CIs
cat("\n=== Hazard Ratios and 95% CIs ===\n")
hr_table <- data.frame(
  Variable = names(coef(cox_model)),
  HR = exp(coef(cox_model)),
  Lower_95 = exp(confint(cox_model))[, 1],
  Upper_95 = exp(confint(cox_model))[, 2],
  p_value = summary(cox_model)$coefficients[, 5]
)
print(round(hr_table, 4))

# Forest plot
ggforest(cox_model, data = pbc_clean)

# --- Task 2: Check the proportional hazards assumption ---
cat("\n=== Task 2: Proportional Hazards Assumption ===\n")
ph_test <- cox.zph(cox_model)
print(ph_test)

cat("\nInterpretation of PH test:\n")
cat("- A significant p-value (< 0.05) indicates violation of the PH assumption.\n")
cat("- Look at the GLOBAL test and individual covariate tests.\n")

# Which covariates violate PH?
ph_df <- data.frame(
  Variable = rownames(ph_test$table)[-nrow(ph_test$table)],
  p_value = ph_test$table[-nrow(ph_test$table), "p"]
)
violators <- ph_df$Variable[ph_df$p_value < 0.05]
if (length(violators) > 0) {
  cat("Covariates violating PH assumption (p < 0.05):",
      paste(violators, collapse = ", "), "\n")
} else {
  cat("No covariates significantly violate the PH assumption.\n")
}

# Plot Schoenfeld residuals
par(mfrow = c(2, 3))
plot(ph_test)
par(mfrow = c(1, 1))

# --- Task 3: Concordance index ---
cat("\n=== Task 3: Concordance Index ===\n")
c_index <- summary(cox_model)$concordance[1]
c_se <- summary(cox_model)$concordance[2]
cat(sprintf("C-index: %.3f (SE: %.3f)\n", c_index, c_se))
cat(sprintf("95%% CI: %.3f - %.3f\n", c_index - 1.96 * c_se, c_index + 1.96 * c_se))

cat("\nInterpretation:\n")
cat("- C-index = 0.5: model has no discriminative ability (random)\n")
cat("- C-index = 1.0: perfect discrimination\n")
cat("- C-index > 0.7: generally considered acceptable\n")
cat("- C-index > 0.8: considered strong discrimination\n")
cat(sprintf("This model's C-index of %.3f indicates %s discrimination.\n",
            c_index,
            ifelse(c_index > 0.8, "strong",
                   ifelse(c_index > 0.7, "acceptable", "modest"))))

# --- Task 4: Predict 5-year survival for a specific patient ---
cat("\n=== Task 4: 5-Year Survival Prediction ===\n")
new_patient <- data.frame(
  age = 55,
  log_bili = log(3),       # bilirubin = 3 mg/dL
  albumin = 3.2,
  protime = 11,
  edema = 0                # no edema
)

cat("Patient profile:\n")
cat("  Age: 55 years\n")
cat("  Bilirubin: 3 mg/dL (log = ", round(log(3), 2), ")\n")
cat("  Albumin: 3.2 g/dL\n")
cat("  Prothrombin time: 11 seconds\n")
cat("  Edema: none\n\n")

# Get predicted survival function
pred_surv <- survfit(cox_model, newdata = new_patient)

# Extract 5-year survival probability
surv_summary <- summary(pred_surv, times = 5)
cat(sprintf("Predicted 5-year survival probability: %.3f (%.1f%%)\n",
            surv_summary$surv, surv_summary$surv * 100))
cat(sprintf("95%% CI: %.3f - %.3f\n",
            surv_summary$lower, surv_summary$upper))

# Plot the predicted survival curve
plot(pred_surv,
     xlab = "Time (years)",
     ylab = "Survival probability",
     main = "Predicted Survival for Patient Profile",
     lwd = 2, col = "blue")
abline(h = 0.5, lty = 2, col = "grey")
abline(v = 5, lty = 2, col = "red")
legend("topright", c("Predicted survival", "5-year mark"),
       col = c("blue", "red"), lty = c(1, 2), lwd = c(2, 1))
