# =============================================================================
# Chapter 9, Exercise 2: Comparing ROC and Precision-Recall Curves
# Simulate an imbalanced dataset (2% prevalence), plot ROC and PR curves,
# and discuss why PR curves are more informative for rare outcomes.
# =============================================================================

library(tidyverse)
library(pROC)
library(PRROC)

# --- Simulate an imbalanced dataset (2% prevalence) ---
set.seed(123)
n <- 5000
true_outcome <- rbinom(n, 1, 0.02)
# Simulate a moderately good model
pred_prob <- plogis(rnorm(n, mean = -2 + 3 * true_outcome, sd = 1.5))

cat("Number of observations:", n, "\n")
cat("Number of events:", sum(true_outcome), "\n")
cat("Prevalence:", mean(true_outcome), "\n")

# --- ROC Curve ---
roc_obj <- roc(true_outcome, pred_prob, quiet = TRUE)
auroc <- auc(roc_obj)

par(mfrow = c(1, 2))

# Plot ROC
plot(roc_obj,
     main = paste("ROC Curve\nAUROC =", round(auroc, 3)),
     col = "steelblue", lwd = 2,
     legacy.axes = TRUE)
abline(0, 1, lty = 2, col = "grey50")

# --- Precision-Recall Curve ---
pr_obj <- pr.curve(
  scores.class0 = pred_prob[true_outcome == 1],
  scores.class1 = pred_prob[true_outcome == 0],
  curve = TRUE
)
auprc <- pr_obj$auc.integral

# Plot PR curve
plot(pr_obj,
     main = paste("Precision-Recall Curve\nAUPRC =", round(auprc, 3)),
     color = "darkorange", lwd = 2)
abline(h = mean(true_outcome), lty = 2, col = "grey50")

par(mfrow = c(1, 1))

# --- Report metrics ---
cat("\n=== Summary ===\n")
cat("AUROC:", round(auroc, 3), "\n")
cat("AUPRC:", round(auprc, 3), "\n")
cat("Baseline AUPRC (prevalence):", round(mean(true_outcome), 3), "\n")

# --- Find optimal threshold (Youden's J) ---
coords_best <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
cat("\nOptimal threshold (Youden's J):", round(coords_best$threshold, 3), "\n")
cat("Sensitivity:", round(coords_best$sensitivity, 3), "\n")
cat("Specificity:", round(coords_best$specificity, 3), "\n")

# PPV at this threshold
pred_class <- ifelse(pred_prob >= coords_best$threshold, 1, 0)
tp <- sum(pred_class == 1 & true_outcome == 1)
fp <- sum(pred_class == 1 & true_outcome == 0)
ppv_at_optimal <- tp / (tp + fp)
cat("PPV at optimal threshold:", round(ppv_at_optimal, 3), "\n")

# --- Interpretation ---
cat("\n=== Interpretation ===\n")
cat("The AUROC looks excellent (", round(auroc, 3), "), suggesting the model\n")
cat("discriminates well. However, the AUPRC (", round(auprc, 3), ") reveals\n")
cat("the real challenge: achieving high recall while maintaining reasonable\n")
cat("precision is difficult with a 2% prevalence rate.\n\n")
cat("At the Youden-optimal threshold, the PPV is only", round(ppv_at_optimal * 100, 1), "%.\n")
cat("This means that even at the 'best' threshold, most positive predictions\n")
cat("are false positives.\n\n")
cat("KEY LESSON: For rare outcomes, always examine the PR curve alongside\n")
cat("the ROC curve. AUROC can paint an overly optimistic picture because\n")
cat("specificity is calculated over the large majority class.\n")
