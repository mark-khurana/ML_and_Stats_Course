# Chapter 11, Exercise 2: External Validation Simulation
# Create three external validation populations and assess model performance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.special import expit, logit

# ---- Simulate development data and fit model (same as chapter) ----
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


# ---- Helper function ----
def evaluate_ext(ext_df, model, predictors, label):
    """Evaluate model on external data and return predictions."""
    X_ext = ext_df[predictors].values
    y_ext = ext_df["death_30d"].values
    pred = model.predict_proba(X_ext)[:, 1]

    obs_rate = y_ext.mean()
    mean_pred = pred.mean()
    oe = obs_rate / mean_pred
    auc = roc_auc_score(y_ext, pred)

    # Calibration slope
    lp_ext = logit(np.clip(pred, 1e-8, 1 - 1e-8))
    cal = LogisticRegression(max_iter=5000, penalty=None)
    cal.fit(lp_ext.reshape(-1, 1), y_ext)

    print(f"\n=== {label} ===")
    print(f"  N = {len(y_ext)}, Observed mortality: {obs_rate:.3f}")
    print(f"  Mean predicted: {mean_pred:.3f}")
    print(f"  C-statistic: {auc:.3f}")
    print(f"  O:E ratio: {oe:.3f}")
    print(f"  Calibration slope: {cal.coef_[0][0]:.3f}")
    print(f"  Calibration intercept: {cal.intercept_[0]:.3f}")

    return pred


# ---- Population A: Temporal validation (3 years later) ----
np.random.seed(101)
n_a = 800
pop_a = pd.DataFrame({
    "age": np.round(np.random.normal(73, 12, n_a)),
    "nihss": np.round(np.maximum(0, np.random.normal(8, 6, n_a))),
    "glucose": np.round(np.random.normal(138, 48, n_a)),
    "afib": np.random.binomial(1, 0.27, n_a),
    "thrombolysis": np.random.binomial(1, 0.40, n_a)  # More thrombolysis
})
lp_a = (-5 + 0.04 * pop_a["age"] + 0.12 * pop_a["nihss"] +
        0.003 * pop_a["glucose"] + 0.3 * pop_a["afib"] -
        0.5 * pop_a["thrombolysis"])
pop_a["death_30d"] = np.random.binomial(1, expit(lp_a))
pred_a = evaluate_ext(pop_a, model, predictors, "Population A: Temporal")

# ---- Population B: Geographical (younger patients) ----
np.random.seed(202)
n_b = 600
pop_b = pd.DataFrame({
    "age": np.round(np.random.normal(62, 10, n_b)),
    "nihss": np.round(np.maximum(0, np.random.normal(6, 5, n_b))),
    "glucose": np.round(np.random.normal(130, 45, n_b)),
    "afib": np.random.binomial(1, 0.15, n_b),
    "thrombolysis": np.random.binomial(1, 0.35, n_b)
})
lp_b = (-5 + 0.04 * pop_b["age"] + 0.12 * pop_b["nihss"] +
        0.003 * pop_b["glucose"] + 0.3 * pop_b["afib"] -
        0.5 * pop_b["thrombolysis"])
pop_b["death_30d"] = np.random.binomial(1, expit(lp_b))
pred_b = evaluate_ext(pop_b, model, predictors, "Population B: Geographical")

# ---- Population C: Domain (primary care, lower severity) ----
np.random.seed(303)
n_c = 500
pop_c = pd.DataFrame({
    "age": np.round(np.random.normal(68, 14, n_c)),
    "nihss": np.round(np.maximum(0, np.random.normal(3, 3, n_c))),
    "glucose": np.round(np.random.normal(120, 35, n_c)),
    "afib": np.random.binomial(1, 0.20, n_c),
    "thrombolysis": np.random.binomial(1, 0.10, n_c)
})
# Different outcome model in primary care
lp_c = (-6 + 0.03 * pop_c["age"] + 0.10 * pop_c["nihss"] +
        0.002 * pop_c["glucose"] + 0.2 * pop_c["afib"] -
        0.3 * pop_c["thrombolysis"])
pop_c["death_30d"] = np.random.binomial(1, expit(lp_c))
pred_c = evaluate_ext(pop_c, model, predictors, "Population C: Domain")

# ---- (b) Calibration plots ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pops = [(pop_a, pred_a, "A: Temporal"), (pop_b, pred_b, "B: Geographical"),
        (pop_c, pred_c, "C: Domain")]

for ax, (pop, pred, title) in zip(axes, pops):
    y_ext = pop["death_30d"].values
    prob_true, prob_pred = calibration_curve(y_ext, pred, n_bins=10,
                                             strategy="quantile")
    ax.plot(prob_pred, prob_true, "o-", color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Proportion")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("ch11_ex2_calibration_plots.png", dpi=150)
plt.show()

# ---- (c) Worst calibration ----
print("\n=== Part (c): Worst Calibration ===")
print("Population C (primary care / domain validation) shows worst calibration.")
print("The outcome model differs from the development setting:")
print("  - Different baseline risk and predictor-outcome relationships")
print("  - Much lower severity patients (NIHSS ~ 3 vs 8)")
print("  - Domain validation is the most stringent test of transportability.")

# ---- (d) Logistic recalibration ----
print("\n=== Part (d): Logistic Recalibration ===")

for pop, pred, label in pops:
    y_ext = pop["death_30d"].values
    lp_ext = logit(np.clip(pred, 1e-8, 1 - 1e-8))
    recal = LogisticRegression(max_iter=5000, penalty=None)
    recal.fit(lp_ext.reshape(-1, 1), y_ext)
    pred_recal = recal.predict_proba(lp_ext.reshape(-1, 1))[:, 1]

    oe_before = y_ext.mean() / pred.mean()
    oe_after = y_ext.mean() / pred_recal.mean()

    print(f"\n{label}:")
    print(f"  O:E before: {oe_before:.3f}")
    print(f"  O:E after:  {oe_after:.3f}")

print("\nLogistic recalibration adjusts intercept and slope, bringing")
print("O:E ratios closer to 1.0 in each external population.")
