# =============================================================================
# Chapter 9b, Exercise 4: PDP versus ALE with correlated predictors
# Make creatinine rise with age, then compare PDP and a (manual) ALE.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; save figures to file
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
import tempfile
import os

# --- Re-create data, but make discharge_creat CORRELATED with age ------------
np.random.seed(42)
rng = np.random.default_rng(42)
n = 1500
X = pd.DataFrame({
    "age":               rng.normal(68, 12, n),
    "length_of_stay":    rng.poisson(5, n) + 1,
    "num_comorbidities": rng.poisson(3, n),
    "prior_admissions":  rng.poisson(1, n),
    "discharge_hb":      rng.normal(11, 2, n),
})
# Creatinine now RISES WITH AGE (very strong correlation, corr ~ 0.98) plus a
# little noise. Crucially it is NOT part of the true risk -- it is a proxy for
# age. The tighter the correlation, the more the PDP is forced to extrapolate.
X["discharge_creat"] = 0.4 + 0.02 * X["age"] + rng.normal(0, 0.05, n)
print(f"Correlation(age, discharge_creat) = "
      f"{X['age'].corr(X['discharge_creat']):.2f}")

# True risk depends on age (and prior admissions, comorbidities), NOT creatinine
lin = (-3 + 0.45 * X["prior_admissions"] + 0.20 * X["num_comorbidities"]
       + 0.05 * (X["age"] - 68) - 0.05 * X["discharge_hb"])
y = rng.binomial(1, 1 / (1 + np.exp(-lin)))

rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

feat = "discharge_creat"
j = list(X.columns).index(feat)

# --- Partial dependence (PDP) with scikit-learn ------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
disp = PartialDependenceDisplay.from_estimator(rf, X, features=[feat], ax=ax)
ax.set_title("PDP: predicted risk vs discharge creatinine")
plt.tight_layout()
out_pdp = os.path.join(tempfile.gettempdir(), "ch09b_ex4_pdp.png")
plt.savefig(out_pdp, dpi=100)
plt.close()

# Extract PDP grid + values for a numeric slope comparison
pdp_res = disp.pd_results[0]
pdp_x = pdp_res["grid_values"][0] if "grid_values" in pdp_res else pdp_res["values"][0]
pdp_y = np.asarray(pdp_res["average"]).ravel()

# --- Manual 1D ALE for the same variable -------------------------------------
def ale_1d(model, X, feature, n_bins=20):
    """Simple binned 1D ALE for the positive-class probability."""
    x = X[feature].values
    # quantile bin edges (unique to avoid empty bins)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    centers, local = [], []
    for k in range(1, len(edges)):
        lo, hi = edges[k - 1], edges[k]
        mask = (x >= lo) & (x <= hi) if k == 1 else (x > lo) & (x <= hi)
        if mask.sum() == 0:
            continue
        Xlo = X.loc[mask].copy(); Xlo[feature] = lo
        Xhi = X.loc[mask].copy(); Xhi[feature] = hi
        diff = (model.predict_proba(Xhi)[:, 1]
                - model.predict_proba(Xlo)[:, 1])
        local.append(diff.mean())
        centers.append((lo + hi) / 2)
    ale = np.cumsum(local)
    ale = ale - ale.mean()  # centre the curve
    return np.array(centers), ale

ale_x, ale_y = ale_1d(rf, X, feat, n_bins=20)

plt.figure(figsize=(7, 4))
plt.plot(ale_x, ale_y, lw=1.5, color="#A23B72")
plt.axhline(0, ls="--", color="grey", alpha=0.6)
plt.xlabel("Discharge creatinine (mg/dL)")
plt.ylabel("ALE (centred effect on risk)")
plt.title("ALE: predicted risk vs discharge creatinine")
plt.tight_layout()
out_ale = os.path.join(tempfile.gettempdir(), "ch09b_ex4_ale.png")
plt.savefig(out_ale, dpi=100)
plt.close()

print("PDP saved to:", out_pdp)
print("ALE saved to:", out_ale)

# --- Quantify the slope of each curve for comparison -------------------------
pdp_slope = np.polyfit(pdp_x, pdp_y, 1)[0]
ale_slope = np.polyfit(ale_x, ale_y, 1)[0]
print(f"\nPDP slope (risk per mg/dL creatinine): {pdp_slope:+.4f}")
print(f"ALE slope (centred effect per mg/dL):  {ale_slope:+.4f}")

# =============================================================================
# INTERPRETATION
#
# 1) Do the two curves agree?
#    No. The PDP shows a steep UPWARD slope, making creatinine look strongly
#    risk-increasing. The ALE curve rises too but is SUBSTANTIALLY FLATTER (its
#    slope is roughly a third to a half of the PDP's -- see the printed slopes),
#    indicating that once age is accounted for, creatinine's own conditional
#    effect is much smaller than the PDP suggests.
#
# 2) Why is the PDP misleading here?
#    Creatinine was built as a pure PROXY for age: they are strongly correlated
#    and only AGE truly drives risk. A PDP works by forcing EVERY patient to a
#    given creatinine value while leaving their real age untouched -- so to draw
#    the point at "high creatinine" it averages predictions for IMPOSSIBLE
#    patients (young people with an old person's creatinine). Because age and
#    creatinine travel together in the real data, the model reads creatinine
#    partly as a stand-in for age; when the PDP breaks that link it smears age's
#    genuine effect onto the creatinine axis, giving a spurious upward slope.
#    ALE avoids this by measuring how the prediction CHANGES within small,
#    realistic windows of creatinine (where age is roughly constant) and
#    accumulating those local changes -- it never evaluates impossible
#    combinations and reports creatinine's much smaller independent effect. With
#    correlated clinical predictors, trust the ALE.
# =============================================================================
