# Chapter 11, Exercise 1: Calibration Assessment
# Using the stroke mortality model from the chapter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.special import expit, logit

# ---- Simulate the stroke data (same as chapter) ----
np.random.seed(2024)
n = 1500

stroke_data = pd.DataFrame({
    "age": np.round(np.random.normal(72, 12, n)),
    "nihss": np.round(np.maximum(0, np.random.normal(8, 6, n))),
    "glucose": np.round(np.random.normal(140, 50, n)),
    "afib": np.random.binomial(1, 0.25, n),
    "thrombolysis": np.random.binomial(1, 0.30, n)
})

lp = (-5 + 0.04 * stroke_data["age"] +
      0.12 * stroke_data["nihss"] +
      0.003 * stroke_data["glucose"] +
      0.3 * stroke_data["afib"] -
      0.5 * stroke_data["thrombolysis"])

stroke_data["death_30d"] = np.random.binomial(1, expit(lp))

predictors = ["age", "nihss", "glucose", "afib", "thrombolysis"]
X = stroke_data[predictors].values
y = stroke_data["death_30d"].values

model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X, y)
pred_prob = model.predict_proba(X)[:, 1]

# ---- (a) Calibration plot using deciles ----
print("=== Part (a): Calibration Plot ===")
prob_true, prob_pred = calibration_curve(y, pred_prob, n_bins=10, strategy="quantile")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, "o-", color="steelblue", lw=2, markersize=8, label="Model")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Proportion")
ax.set_title("Calibration Plot (Deciles): 30-Day Stroke Mortality")
ax.legend()
plt.tight_layout()
plt.savefig("ch11_ex1_calibration.png", dpi=150)
plt.show()
# The points cluster near the diagonal, suggesting good apparent calibration.

# ---- (b) O:E ratio ----
print("\n=== Part (b): O:E Ratio ===")
obs_rate = y.mean()
mean_pred = pred_prob.mean()
oe_ratio = obs_rate / mean_pred

print(f"Observed event rate: {obs_rate:.3f}")
print(f"Mean predicted probability: {mean_pred:.3f}")
print(f"O:E ratio: {oe_ratio:.3f}")
# An O:E ratio near 1.0 indicates good calibration-in-the-large.

# ---- (c) Calibration slope ----
print("\n=== Part (c): Calibration Slope ===")
lp_pred = logit(np.clip(pred_prob, 1e-8, 1 - 1e-8))
cal_model = LogisticRegression(max_iter=5000, penalty=None)
cal_model.fit(lp_pred.reshape(-1, 1), y)

print(f"Calibration slope: {cal_model.coef_[0][0]:.3f}")
print(f"Calibration intercept: {cal_model.intercept_[0]:.3f}")
# A slope near 1.0 on training data is expected. Values < 1 on new data
# indicate overfitting.

# ---- (d) Bootstrap optimism correction ----
print("\n=== Part (d): Bootstrap Optimism Correction ===")
np.random.seed(42)

# Apparent performance
apparent_auc = roc_auc_score(y, pred_prob)
apparent_brier = brier_score_loss(y, pred_prob)

n_boot = 200
optimism_auc = []
optimism_brier = []

for b in range(n_boot):
    boot_idx = np.random.choice(n, n, replace=True)
    X_boot, y_boot = X[boot_idx], y[boot_idx]

    boot_model = LogisticRegression(max_iter=5000, random_state=42)
    boot_model.fit(X_boot, y_boot)

    # Apparent on bootstrap sample
    pred_boot = boot_model.predict_proba(X_boot)[:, 1]
    auc_boot = roc_auc_score(y_boot, pred_boot)

    # Test on original data
    pred_orig = boot_model.predict_proba(X)[:, 1]
    auc_orig = roc_auc_score(y, pred_orig)

    optimism_auc.append(auc_boot - auc_orig)

mean_opt_auc = np.mean(optimism_auc)
corrected_auc = apparent_auc - mean_opt_auc

print(f"Apparent C-statistic:          {apparent_auc:.4f}")
print(f"Mean optimism (AUC):           {mean_opt_auc:.4f}")
print(f"Optimism-corrected C-statistic:{corrected_auc:.4f}")
print(f"C-statistic decrease:          {mean_opt_auc:.4f}")

# The C-statistic decreases slightly after correction, reflecting mild
# optimism in the apparent performance. With 1500 observations and 5
# predictors, overfitting is modest.
