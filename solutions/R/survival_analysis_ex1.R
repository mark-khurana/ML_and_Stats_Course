# =============================================================================
# Chapter 6, Exercise 1: Kaplan-Meier Curves by Disease Stage
# PBC dataset: KM curves stratified by stage, log-rank test, median survival
# =============================================================================

library(survival)
library(survminer)
library(tidyverse)

# --- Load and prepare the PBC dataset ---
data(pbc, package = "survival")

pbc_clean <- pbc %>%
  filter(!is.na(trt)) %>%
  mutate(
    status_binary = ifelse(status == 2, 1, 0),
    time_years = time / 365.25,
    trt_label = factor(trt, labels = c("D-penicillamine", "Placebo"))
  ) %>%
  filter(!is.na(stage))  # Remove missing stage values

cat("N =", nrow(pbc_clean), "\n")
cat("Events (deaths):", sum(pbc_clean$status_binary), "\n")
cat("Stages:", paste(sort(unique(pbc_clean$stage)), collapse = ", "), "\n\n")

# --- Fit Kaplan-Meier curves stratified by disease stage ---
km_stage <- survfit(Surv(time_years, status_binary) ~ stage, data = pbc_clean)
print(km_stage)

# --- Report median survival time for each stage ---
cat("\n=== Median Survival Time by Stage ===\n")
median_surv <- summary(km_stage)$table
print(median_surv)

# Extract and display more clearly
cat("\nMedian survival (years) by stage:\n")
for (s in sort(unique(pbc_clean$stage))) {
  km_s <- survfit(Surv(time_years, status_binary) ~ 1,
                   data = pbc_clean[pbc_clean$stage == s, ])
  med <- surv_median(km_s)
  cat(sprintf("  Stage %d: %.2f years (95%% CI: %.2f - %.2f)\n",
              s, med$median, med$lower, med$upper))
}

# --- Perform the log-rank test ---
cat("\n=== Log-Rank Test ===\n")
logrank <- survdiff(Surv(time_years, status_binary) ~ stage, data = pbc_clean)
print(logrank)

# Extract p-value
pvalue <- 1 - pchisq(logrank$chisq, length(logrank$n) - 1)
cat(sprintf("\nLog-rank test p-value: %.6f\n", pvalue))

if (pvalue < 0.05) {
  cat("Result: Significant difference in survival across disease stages.\n")
} else {
  cat("Result: No significant difference detected.\n")
}

# --- Plot the Kaplan-Meier curves ---
p <- ggsurvplot(
  km_stage,
  data = pbc_clean,
  pval = TRUE,               # Show log-rank p-value
  conf.int = TRUE,            # Show confidence intervals
  risk.table = TRUE,          # Show number at risk table
  xlab = "Time (years)",
  ylab = "Survival probability",
  title = "Kaplan-Meier Curves by Disease Stage (PBC)",
  legend.title = "Stage",
  palette = c("#2ecc71", "#3498db", "#e67e22", "#e74c3c"),
  ggtheme = theme_minimal()
)

print(p)

# --- Interpretation ---
cat("\n=== Interpretation ===\n")
cat("The KM curves should show a clear separation by stage, with higher\n")
cat("stages having worse survival. The log-rank test formally confirms\n")
cat("whether these differences are statistically significant.\n")
cat("As expected in PBC, patients with Stage 4 disease have the worst\n")
cat("prognosis, while Stage 1-2 patients have substantially better survival.\n")
