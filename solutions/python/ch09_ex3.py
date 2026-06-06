# =============================================================================
# Chapter 9, Exercise 3: Evaluating Fairness
# Simulate data with two demographic groups, evaluate whether the model
# performs equitably across groups.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.special import expit

# --- Simulate data with two demographic groups ---
np.random.seed(42)
n = 2000

group = np.random.choice(["A", "B"], n)

# Group B has slightly different disease prevalence
true_outcome = np.where(group == "A",
                        np.random.binomial(1, 0.10, n),
                        np.random.binomial(1, 0.15, n))

# Model performs slightly worse for Group B (weaker signal, more noise)
pred_prob = np.where(group == "A",
                     expit(np.random.normal(-1.5 + 2.5 * true_outcome, 1.0)),
                     expit(np.random.normal(-1.5 + 1.8 * true_outcome, 1.2)))

# --- 1. Calculate AUC by group ---
print("=== Discrimination (AUC) by Group ===")
for g in ["A", "B"]:
    mask = group == g
    auc = roc_auc_score(true_outcome[mask], pred_prob[mask])
    prev = true_outcome[mask].mean()
    print(f"Group {g}: AUC = {auc:.3f}, Prevalence = {prev*100:.1f}%")

# --- 2. Sensitivity and specificity at various thresholds ---
print("\n=== Performance at Different Thresholds ===")
thresholds = [0.15, 0.20, 0.30, 0.40, 0.50]

for threshold in thresholds:
    print(f"\nThreshold = {threshold:.2f}:")
    for g in ["A", "B"]:
        mask = group == g
        pred_class = (pred_prob[mask] >= threshold).astype(int)
        actual = true_outcome[mask]

        tp = ((pred_class == 1) & (actual == 1)).sum()
        fn = ((pred_class == 0) & (actual == 1)).sum()
        tn = ((pred_class == 0) & (actual == 0)).sum()
        fp = ((pred_class == 1) & (actual == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"  Group {g}: Sens={sens:.3f}  Spec={spec:.3f}  "
              f"PPV={ppv:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")

# --- 3. Positive prediction rate (demographic parity) ---
print("\n=== Positive Prediction Rate (Demographic Parity) ===")
threshold = 0.30
for g in ["A", "B"]:
    mask = group == g
    pred_class = (pred_prob[mask] >= threshold).astype(int)
    pos_rate = pred_class.mean()
    print(f"Group {g}: {pos_rate*100:.1f}% predicted positive "
          f"at threshold {threshold:.2f}")

# --- 4. ROC curves overlaid ---
fig, ax = plt.subplots(figsize=(7, 7))

for g, color, label_prefix in [("A", "steelblue", "Group A"),
                                ("B", "darkorange", "Group B")]:
    mask = group == g
    fpr, tpr, _ = roc_curve(true_outcome[mask], pred_prob[mask])
    auc = roc_auc_score(true_outcome[mask], pred_prob[mask])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label_prefix} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "--", color="grey")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curves by Demographic Group")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --- Interpretation ---
print("\n=== Interpretation ===")
print("1. The model shows different AUC values across groups, indicating")
print("   unequal discrimination. Group B (higher prevalence, noisier data)")
print("   has lower AUC than Group A.")
print("\n2. At any fixed threshold, sensitivity and specificity differ between")
print("   groups. A single threshold does not provide equitable performance.")
print("   Group-specific thresholds could equalize sensitivity but raise")
print("   questions about fairness.")
print("\n3. Different positive prediction rates reflect both different prevalence")
print("   and different model performance. Demographic parity (equal positive")
print("   rates) may conflict with calibration if true prevalence differs.")
print("\n4. CLINICAL IMPLICATION: Before deploying a model, always evaluate")
print("   performance across demographic subgroups. A model that works well")
print("   'on average' may perform poorly for specific populations, potentially")
print("   widening health disparities. Report subgroup-specific metrics.")
