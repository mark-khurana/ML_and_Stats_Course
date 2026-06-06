# =============================================================================
# Chapter 9, Exercise 3: Evaluating Fairness
# Simulate data with two demographic groups, evaluate whether the model
# performs equitably across groups.
# =============================================================================

library(tidyverse)
library(pROC)

# --- Simulate data with two demographic groups ---
set.seed(42)
n <- 2000

group <- sample(c("A", "B"), n, replace = TRUE)

# Group B has slightly different disease prevalence and predictor distribution
true_outcome <- ifelse(group == "A",
                       rbinom(n, 1, 0.10),
                       rbinom(n, 1, 0.15))

# Model performs slightly worse for Group B (weaker signal)
pred_prob <- ifelse(group == "A",
                    plogis(rnorm(n, -1.5 + 2.5 * true_outcome, 1.0)),
                    plogis(rnorm(n, -1.5 + 1.8 * true_outcome, 1.2)))

df <- tibble(group = group, true_outcome = true_outcome, pred_prob = pred_prob)

# --- 1. Calculate AUC by group ---
cat("=== Discrimination (AUC) by Group ===\n")
for (g in c("A", "B")) {
  idx <- df$group == g
  roc_g <- roc(df$true_outcome[idx], df$pred_prob[idx], quiet = TRUE)
  ci_g <- ci.auc(roc_g)
  cat(sprintf("Group %s: AUC = %.3f (95%% CI: %.3f - %.3f), Prevalence = %.1f%%\n",
              g, auc(roc_g), ci_g[1], ci_g[3],
              mean(df$true_outcome[idx]) * 100))
}

# --- 2. Sensitivity and specificity at various thresholds ---
cat("\n=== Performance at Different Thresholds ===\n")
thresholds <- c(0.15, 0.20, 0.30, 0.40, 0.50)

for (threshold in thresholds) {
  cat(sprintf("\nThreshold = %.2f:\n", threshold))
  for (g in c("A", "B")) {
    idx <- df$group == g
    pred_class <- ifelse(df$pred_prob[idx] >= threshold, 1, 0)
    actual <- df$true_outcome[idx]

    tp <- sum(pred_class == 1 & actual == 1)
    fn <- sum(pred_class == 0 & actual == 1)
    tn <- sum(pred_class == 0 & actual == 0)
    fp <- sum(pred_class == 1 & actual == 0)

    sens <- ifelse(tp + fn > 0, tp / (tp + fn), NA)
    spec <- ifelse(tn + fp > 0, tn / (tn + fp), NA)
    ppv  <- ifelse(tp + fp > 0, tp / (tp + fp), NA)

    cat(sprintf("  Group %s: Sens=%.3f  Spec=%.3f  PPV=%.3f  (TP=%d FP=%d FN=%d TN=%d)\n",
                g, sens, spec, ppv, tp, fp, fn, tn))
  }
}

# --- 3. Positive prediction rate (demographic parity) ---
cat("\n=== Positive Prediction Rate (Demographic Parity) ===\n")
threshold <- 0.30
for (g in c("A", "B")) {
  idx <- df$group == g
  pred_class <- ifelse(df$pred_prob[idx] >= threshold, 1, 0)
  pos_rate <- mean(pred_class)
  cat(sprintf("Group %s: %.1f%% predicted positive at threshold %.2f\n",
              g, pos_rate * 100, threshold))
}

# --- 4. ROC curves overlaid ---
roc_a <- roc(df$true_outcome[df$group == "A"],
             df$pred_prob[df$group == "A"], quiet = TRUE)
roc_b <- roc(df$true_outcome[df$group == "B"],
             df$pred_prob[df$group == "B"], quiet = TRUE)

plot(roc_a, col = "steelblue", lwd = 2, legacy.axes = TRUE,
     main = "ROC Curves by Demographic Group")
plot(roc_b, col = "darkorange", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste("Group A (AUC =", round(auc(roc_a), 3), ")"),
                  paste("Group B (AUC =", round(auc(roc_b), 3), ")")),
       col = c("steelblue", "darkorange"), lwd = 2)

# --- Interpretation ---
cat("\n=== Interpretation ===\n")
cat("1. The model shows different AUC values across groups, indicating\n")
cat("   unequal discrimination. Group B (higher prevalence, noisier data)\n")
cat("   has lower AUC than Group A.\n\n")
cat("2. At any fixed threshold, sensitivity and specificity differ between\n")
cat("   groups. This means a single threshold does not provide equitable\n")
cat("   performance. Group-specific thresholds could equalize sensitivity\n")
cat("   but would raise questions about fairness.\n\n")
cat("3. Different positive prediction rates reflect both different prevalence\n")
cat("   and different model performance. Demographic parity (equal positive\n")
cat("   rates) may conflict with calibration if true prevalence differs.\n\n")
cat("4. CLINICAL IMPLICATION: Before deploying a model, always evaluate\n")
cat("   performance across demographic subgroups. A model that works well\n")
cat("   'on average' may perform poorly for specific populations, potentially\n")
cat("   widening health disparities. Report subgroup-specific metrics.\n")
