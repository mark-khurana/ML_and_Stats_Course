# =============================================================================
# Chapter 6b, Exercise 2: Fit and interpret a random-intercept model
# Recreate the BP data, add a binary treatment, fit a random-intercept MixedLM,
# compute the ICC, and compare the treatment SE against ordinary OLS.
# =============================================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# --- Recreate the chapter's longitudinal, clustered BP dataset (long format) -
n_clinic, n_patient, n_visit = 8, 25, 5
clinic_effect = np.random.normal(0, 6, n_clinic)   # clinic baselines (SD 6)

rows = []
pid = 0
for c in range(n_clinic):
    for p in range(n_patient):
        patient_effect = np.random.normal(0, 8)        # patient baseline (SD 8)
        # Treatment assigned at the PATIENT level (constant across the patient's
        # visits). True built-in effect: -5 mmHg.
        treatment = np.random.binomial(1, 0.5)
        for v in range(n_visit):
            sbp = (135 + clinic_effect[c] + patient_effect
                   - 1.5 * v            # downward drift per visit
                   - 5.0 * treatment    # true treatment effect: -5 mmHg
                   + np.random.normal(0, 5))
            rows.append({"clinic": c, "patient_id": pid, "visit": v,
                         "treatment": treatment, "sbp": sbp})
        pid += 1

bp = pd.DataFrame(rows)
print(f"Dataset: {len(bp)} rows | {n_clinic} clinics x {n_patient} patients "
      f"x {n_visit} visits\n")

# --- Fit the random-intercept model -----------------------------------------
# statsmodels MixedLM groups by ONE level; we use a random intercept per
# patient (the level at which treatment varies), which absorbs the repeated-
# visit correlation. Fixed effects: visit + treatment.
m_ri = smf.mixedlm("sbp ~ visit + treatment", data=bp, groups=bp["patient_id"])
res_ri = m_ri.fit()

print("=== Random-intercept model summary ===")
print(res_ri.summary())

# --- (a) Treatment fixed effect and its interpretation ----------------------
beta_trt = res_ri.params["treatment"]
se_trt = res_ri.bse["treatment"]
p_trt = res_ri.pvalues["treatment"]

print("\n=== (a) Treatment fixed effect ===")
print(f"Estimate: {beta_trt:.3f} mmHg   SE: {se_trt:.3f}   p-value: {p_trt:.4g}")
print("Interpretation: after accounting for visit and the fact that repeated")
print("  visits are clustered within patients, being on treatment is")
print(f"  associated with a {beta_trt:.2f} mmHg change in systolic BP (a")
print("  reduction), holding visit number constant.")

# --- (b) Intraclass correlation ---------------------------------------------
# ICC at the patient level = between-patient variance / total variance.
var_group = float(res_ri.cov_re.iloc[0, 0])   # between-patient variance
var_resid = float(res_ri.scale)               # residual (within-patient) var
icc = var_group / (var_group + var_resid)

print("\n=== (b) Variance components and ICC ===")
print(f"Between-patient variance: {var_group:.2f}")
print(f"Residual variance       : {var_resid:.2f}")
print(f"Patient-level ICC       : {icc:.3f}")
print(f"Plain English: about {100*icc:.0f}% of the total variation in blood")
print("  pressure is due to persistent differences between patients rather than")
print("  visit-to-visit fluctuation within a patient.")

# --- (c) Compare with ordinary OLS that ignores clustering ------------------
m_ols = smf.ols("sbp ~ visit + treatment", data=bp).fit()
se_trt_ols = m_ols.bse["treatment"]

print("\n=== (c) Ordinary OLS ignoring clustering ===")
print(f"Treatment estimate (OLS): {m_ols.params['treatment']:.3f}")
print(f"Treatment SE (OLS)      : {se_trt_ols:.3f}")
print(f"Treatment SE (mixed)    : {se_trt:.3f}")
print(f"Ratio mixed/OLS         : {se_trt / se_trt_ols:.2f}")
print("\nWhich is larger, and why it matters:")
print("  The MIXED-model SE is larger. Treatment is a BETWEEN-PATIENT variable")
print("  (constant across a patient's 5 visits), so the real amount of")
print("  independent information about treatment is roughly the number of")
print("  patients, not the 1000 rows. Ordinary OLS pretends all 1000 rows are")
print("  independent, so it reports an SE that is too small -- an over-")
print("  optimistic confidence interval and p-value. Ignoring the clustering")
print("  would make the treatment effect look more certain than the data")
print("  can support.")
