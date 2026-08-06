# =============================================================================
# Chapter 9b, Exercise 1: Permutation importance and the wrong-reason model
# Add a leaky variable (discharge_disposition) and see it dominate the ranking.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; save figures to file
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import tempfile
import os

# --- Re-create the readmission data ------------------------------------------
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

# --- Add a LEAKY variable: discharge_disposition -----------------------------
# Where the patient was discharged TO is recorded AFTER the clinical course has
# played out. Patients who go on to be readmitted were far more likely to have
# been sent to a skilled-nursing facility / rehab, whereas those who did well
# went home. The disposition is a CONSEQUENCE of the underlying risk, not a
# cause of readmission -- and it is not truly available at prediction time.
disp_codes = np.array(["Home", "Rehab", "SNF"])  # encoded 0, 1, 2
disp = np.where(
    y == 1,
    rng.choice([0, 1, 2], size=n, p=[0.15, 0.30, 0.55]),
    rng.choice([0, 1, 2], size=n, p=[0.80, 0.12, 0.08]),
)
X["discharge_disposition"] = disp  # integer encoding (0=Home,1=Rehab,2=SNF)

# --- Re-fit the random forest with the leaky variable included ---------------
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

# --- Permutation importance (shuffle each column, watch AUC fall) ------------
result = permutation_importance(rf, X, y, scoring="roc_auc",
                                n_repeats=10, random_state=1)
imp = (pd.DataFrame({"Variable": X.columns,
                     "Importance": result.importances_mean})
       .sort_values("Importance", ascending=False)
       .reset_index(drop=True))

print("=== Permutation importance (drop in AUC when shuffled) ===")
print(imp.to_string(index=False))

leak_rank = imp.index[imp["Variable"] == "discharge_disposition"][0] + 1
print(f"\ndischarge_disposition ranks #{leak_rank} of {len(imp)} predictors.")

# --- Plot (saved to a temp file, no display needed) --------------------------
order = result.importances_mean.argsort()
plt.figure(figsize=(7, 4))
plt.barh(X.columns[order], result.importances_mean[order], color="#2E86AB")
plt.xlabel("Mean drop in AUC when the variable is shuffled")
plt.title("Permutation importance with a leaky variable")
plt.tight_layout()
out = os.path.join(tempfile.gettempdir(), "ch09b_ex1_importance.png")
plt.savefig(out, dpi=100)
print("Plot saved to:", out)

# =============================================================================
# INTERPRETATION
#
# 1) Where does the leaky variable rank?
#    discharge_disposition jumps to the TOP of the ranking -- shuffling it
#    collapses the AUC more than shuffling any genuine clinical predictor.
#
# 2) Why is a high rank here a WARNING, not a discovery?
#    Permutation importance measures how much the MODEL leans on a variable to
#    reproduce the observed outcome -- not whether the variable is usable or
#    causal. discharge_disposition is decided at/after the event we predict and
#    is a downstream MARKER of the risk the team already perceived. A model that
#    leans on it looks brilliant in development and fails in deployment, because
#    at true prediction time the value is unavailable or not yet meaningful. A
#    variable that dominates for no plausible clinical reason should trigger a
#    hunt for leakage, not a celebration.
#
# 3) Real-world variables that behave like this leak:
#    - Discharge destination / disposition codes (as here).
#    - Palliative-care or hospice referral flags; late DNR orders.
#    - Counts of consults, ICU transfers, or rapid-response calls in the stay.
#    - Medications started for complications (vasopressors, broad-spectrum
#      antibiotics) that postdate the predictor cut-off.
#    - Billing/DRG codes finalised after the outcome is known.
#    - Timestamps or ward names that proxy for how sick a patient was.
#    Each is associated with the outcome because it is a CONSEQUENCE of illness,
#    not a baseline predictor available when the model must act.
# =============================================================================
