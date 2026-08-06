# =============================================================================
# Chapter 9b, Exercise 2: Reading a SHAP beeswarm
# Fit a model, compute SHAP, draw the beeswarm, and read it clinically.
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

# --- Compute SHAP values once (TreeExplainer = fast exact TreeSHAP) ----------
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
sv = explainer(X)

# --- Global SHAP importance ordering (mean |SHAP|) ---------------------------
mean_abs = np.abs(sv.values).mean(axis=0)
imp = (pd.Series(mean_abs, index=X.columns)
       .sort_values(ascending=False))
print("=== Global SHAP importance (mean |SHAP value|) ===")
print(imp.round(4).to_string())
print("\nTop two variables:", ", ".join(imp.index[:2]))

# --- Beeswarm summary plot (saved to a temp file) ----------------------------
plt.figure()
shap.plots.beeswarm(sv, show=False)
plt.tight_layout()
out = os.path.join(tempfile.gettempdir(), "ch09b_ex2_beeswarm.png")
plt.savefig(out, dpi=100, bbox_inches="tight")
print("Beeswarm saved to:", out)

# =============================================================================
# INTERPRETATION
#
# 1) Which two variables are globally most important?
#    Read them off the printed ranking above. prior_admissions is clearly the
#    single most important variable; num_comorbidities takes the second slot
#    here (with discharge_hb and age close behind). All of the top variables are
#    ones that actually enter the simulated true risk, while the two that do NOT
#    (length_of_stay and discharge_creat) fall to the bottom. Reassuring: the
#    model relies on clinically sensible signals. (The exact 2nd place can
#    differ by backend -- e.g. discharge_hb in the R/XGBoost version -- because
#    the two variables have different spreads; trust the printout.)
#
# 2) For the TOP variable (prior_admissions), do high values push the
#    prediction UP or DOWN?
#    HIGH values (red dots) sit on the RIGHT -- more prior admissions push the
#    predicted readmission risk UP; few prior admissions (blue) push it down.
#    This is clinically sensible: a history of admissions marks frailty and
#    unstable disease, so higher predicted risk is expected.
#
# 3) A variable with NO clear left-right colour pattern:
#    discharge_creat (and, to a lesser extent, length_of_stay) shows red and
#    blue dots mixed on both sides with SHAP values clustered near zero. We
#    built creatinine as pure noise, so the model found no consistent signal.
#    A scrambled colour pattern with small SHAP values means the variable is
#    essentially unused; the same pattern with LARGE SHAP values would instead
#    flag a complex, non-monotonic or interaction-driven effect to investigate.
# =============================================================================
