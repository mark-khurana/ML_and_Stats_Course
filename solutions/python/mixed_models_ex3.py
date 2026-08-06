# =============================================================================
# Chapter 6b, Exercise 3: Random intercept vs random slope
# Extend the Exercise-2 model with a clinic-specific visit slope and compare
# the random-intercept and random-slope models with a likelihood-ratio test.
# =============================================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

np.random.seed(42)

# --- Recreate the chapter's BP dataset with a binary treatment --------------
n_clinic, n_patient, n_visit = 8, 25, 5
clinic_effect = np.random.normal(0, 6, n_clinic)

rows = []
pid = 0
for c in range(n_clinic):
    for p in range(n_patient):
        patient_effect = np.random.normal(0, 8)
        treatment = np.random.binomial(1, 0.5)
        for v in range(n_visit):
            # NOTE: simulated with a SINGLE shared visit slope (-1.5), i.e. no
            # genuine clinic-to-clinic variation in the trend.
            sbp = (135 + clinic_effect[c] + patient_effect
                   - 1.5 * v - 5.0 * treatment + np.random.normal(0, 5))
            rows.append({"clinic": c, "patient_id": pid, "visit": v,
                         "treatment": treatment, "sbp": sbp})
        pid += 1

bp = pd.DataFrame(rows)

# statsmodels MixedLM groups by ONE level, so to let the visit trend vary "by
# clinic" we group by clinic here. (R's lme4 can nest clinic/patient in one
# call; in statsmodels we choose the level of interest for this comparison.)

# --- Random-intercept model: random intercept per clinic --------------------
m_ri = smf.mixedlm("sbp ~ visit + treatment", data=bp,
                   groups=bp["clinic"]).fit(reml=True)

# --- Random-slope model: each clinic gets its own visit slope ----------------
m_rs = smf.mixedlm("sbp ~ visit + treatment", data=bp,
                   groups=bp["clinic"], re_formula="~visit").fit(reml=True)

print("=== Random-slope model summary ===")
print(m_rs.summary())

# --- (a) Likelihood-ratio test comparing the two models ---------------------
# The random-slope model adds a slope variance and an intercept-slope
# covariance -> 2 extra parameters. Both models share the same fixed effects,
# so a REML-based LRT on the random structure is valid.
# (Clamp at 0: when the extra random-effect variance collapses to the boundary
# the richer model can land on a marginally lower REML log-likelihood, giving a
# tiny negative statistic that simply means "no improvement".)
lr_stat = max(0.0, 2 * (m_rs.llf - m_ri.llf))
df_diff = 2
p_lrt = stats.chi2.sf(lr_stat, df_diff)

print("\n=== (a) Likelihood-ratio test (random intercept vs random slope) ===")
print(f"logLik (random intercept): {m_ri.llf:.2f}")
print(f"logLik (random slope)    : {m_rs.llf:.2f}")
print(f"LR statistic: {lr_stat:.3f} on {df_diff} df")
print(f"LRT p-value : {p_lrt:.4g}")

# --- (b) Interpretation ------------------------------------------------------
print("\n=== (b) Does a clinic-specific trend improve the model? ===")
if p_lrt < 0.05:
    print("The random slope significantly improves fit: the visit trend")
    print("genuinely differs between clinics.")
else:
    print("The random slope does NOT significantly improve fit (p > 0.05).")
    print("Clinically: there is no evidence that blood pressure changes at")
    print("different rates across clinics -- the single shared downward trend")
    print("is adequate. This is expected, because the data were simulated with")
    print("one common visit slope. The lesson: add random slopes only when the")
    print("data (and clinical sense) support them; needless complexity can also")
    print("trigger convergence warnings.")
