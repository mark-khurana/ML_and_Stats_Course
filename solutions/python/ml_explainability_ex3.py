# =============================================================================
# Chapter 9b, Exercise 3: Explaining one patient to a patient
# SHAP waterfall plots for one high-risk and one low-risk patient.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; save figures to file
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import tempfile
import os

# --- Re-create data and fit the model ----------------------------------------
np.random.seed(42)
rng = np.random.default_rng(42)
n = 1500
X = pd.DataFrame({
    "age":               rng.normal(68, 12, n),
    "length_of_stay":    rng.poisson(5, n) + 1,
    "num_comorbidities": rng.poisson(3, n),
    "prior_admissions":  rng.poisson(1, n),
    "discharge_hb":      rng.normal(11, 2, n),
    "discharge_creat":   rng.lognormal(0.2, 0.5, n),
})
lin = (-3 + 0.45 * X["prior_admissions"] + 0.20 * X["num_comorbidities"]
       + 0.015 * (X["age"] - 68) - 0.05 * X["discharge_hb"])
y = rng.binomial(1, 1 / (1 + np.exp(-lin)))

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss"
)
model.fit(X, y)

explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
sv = explainer(X)

# --- Pick one high-risk and one low-risk patient -----------------------------
preds = model.predict_proba(X)[:, 1]
hi = int(np.argmax(preds))
lo = int(np.argmin(preds))

print(f"High-risk patient: row {hi}, predicted risk = {100*preds[hi]:.1f}%")
print(X.iloc[[hi]].to_string())
print(f"\nLow-risk patient:  row {lo}, predicted risk = {100*preds[lo]:.1f}%")
print(X.iloc[[lo]].to_string())

# --- Waterfall plots (saved to temp files) -----------------------------------
plt.figure()
shap.plots.waterfall(sv[hi], show=False)
out_hi = os.path.join(tempfile.gettempdir(), "ch09b_ex3_waterfall_high.png")
plt.savefig(out_hi, dpi=100, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.waterfall(sv[lo], show=False)
out_lo = os.path.join(tempfile.gettempdir(), "ch09b_ex3_waterfall_low.png")
plt.savefig(out_lo, dpi=100, bbox_inches="tight")
plt.close()
print("\nWaterfalls saved to:\n  ", out_hi, "\n  ", out_lo)

# --- Which characteristic contributed most for the high-risk patient? --------
shap_hi = pd.Series(sv.values[hi], index=X.columns)
top_feat = shap_hi.abs().idxmax()
print(f"\nLargest contributor for the high-risk patient: "
      f"{top_feat} (SHAP = {shap_hi[top_feat]:+.3f})")

# =============================================================================
# INTERPRETATION
#
# 1) Plain-language explanations from each waterfall:
#
#    HIGH-RISK patient (to the patient):
#    "Our tool starts everyone at the average readmission risk. For you it moved
#     UP mainly because you have had several previous hospital admissions and a
#     number of ongoing health conditions, which together point to a higher
#     chance of coming back within 30 days."
#
#    LOW-RISK patient (to the patient):
#    "Starting from the average, your estimate moved DOWN because you have had
#     few or no previous admissions and few ongoing conditions, which is why the
#     tool puts your 30-day readmission risk below average."
#
# 2) Which characteristic contributed most for the high-risk patient, and would
#    intervening on it necessarily reduce risk?
#    Read the printed top contributor above. It is tempting to conclude
#    "reduce that variable and the risk falls" -- but that is a CAUSAL claim
#    SHAP does NOT support. SHAP only reports how the MODEL used this patient's
#    data; a comorbidity count (like prior admissions) is a MARKER of underlying
#    frailty, not plausibly the direct cause of the next readmission, and you
#    cannot meaningfully "intervene" on the count itself. Establishing whether
#    any action lowers risk needs a genuine causal-inference study, not an
#    explanation plot. Explanations describe association learned by the model,
#    never an intervention effect.
# =============================================================================
