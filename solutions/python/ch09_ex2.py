# =============================================================================
# Chapter 9, Exercise 2: Comparing ROC and Precision-Recall Curves
# Simulate an imbalanced dataset (2% prevalence), plot ROC and PR curves,
# and discuss why PR curves are more informative for rare outcomes.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score,
                              precision_recall_curve, average_precision_score)
from scipy.special import expit

# --- Simulate an imbalanced dataset (2% prevalence) ---
np.random.seed(123)
n = 5000
true_outcome = np.random.binomial(1, 0.02, n)
# Simulate a moderately good model
pred_prob = expit(np.random.normal(-2 + 3 * true_outcome, 1.5))

print(f"Number of observations: {n}")
print(f"Number of events: {true_outcome.sum()}")
print(f"Prevalence: {true_outcome.mean():.3f}")

# --- ROC Curve ---
fpr, tpr, roc_thresholds = roc_curve(true_outcome, pred_prob)
auroc = roc_auc_score(true_outcome, pred_prob)

# --- Precision-Recall Curve ---
precision, recall, pr_thresholds = precision_recall_curve(true_outcome, pred_prob)
auprc = average_precision_score(true_outcome, pred_prob)
prevalence = true_outcome.mean()

# --- Plot side by side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC curve
axes[0].plot(fpr, tpr, color="steelblue", lw=2)
axes[0].plot([0, 1], [0, 1], "--", color="grey")
axes[0].set_title(f"ROC Curve (AUROC = {auroc:.3f})")
axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
axes[0].set_ylabel("True Positive Rate (Sensitivity)")

# PR curve
axes[1].plot(recall, precision, color="darkorange", lw=2)
axes[1].axhline(y=prevalence, linestyle="--", color="grey",
                label=f"Baseline (prevalence={prevalence:.3f})")
axes[1].set_title(f"Precision-Recall Curve (AUPRC = {auprc:.3f})")
axes[1].set_xlabel("Recall (Sensitivity)")
axes[1].set_ylabel("Precision (PPV)")
axes[1].legend()

plt.tight_layout()
plt.show()

# --- Report metrics ---
print(f"\n=== Summary ===")
print(f"AUROC: {auroc:.3f}")
print(f"AUPRC: {auprc:.3f}")
print(f"Baseline AUPRC (prevalence): {prevalence:.3f}")

# --- Find optimal threshold (Youden's J) ---
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
optimal_threshold = roc_thresholds[best_idx]
print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.3f}")
print(f"Sensitivity: {tpr[best_idx]:.3f}")
print(f"Specificity: {1 - fpr[best_idx]:.3f}")

# PPV at this threshold
pred_class = (pred_prob >= optimal_threshold).astype(int)
tp = ((pred_class == 1) & (true_outcome == 1)).sum()
fp = ((pred_class == 1) & (true_outcome == 0)).sum()
ppv_at_optimal = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"PPV at optimal threshold: {ppv_at_optimal:.3f}")

# --- Interpretation ---
print(f"\n=== Interpretation ===")
print(f"The AUROC looks excellent ({auroc:.3f}), suggesting the model")
print(f"discriminates well. However, the AUPRC ({auprc:.3f}) reveals")
print(f"the real challenge: achieving high recall while maintaining")
print(f"reasonable precision is difficult with a 2% prevalence rate.")
print(f"\nAt the Youden-optimal threshold, the PPV is only {ppv_at_optimal*100:.1f}%.")
print(f"This means that even at the 'best' threshold, most positive")
print(f"predictions are false positives.")
print(f"\nKEY LESSON: For rare outcomes, always examine the PR curve")
print(f"alongside the ROC curve. AUROC can paint an overly optimistic")
print(f"picture because specificity is calculated over the large")
print(f"majority class.")
