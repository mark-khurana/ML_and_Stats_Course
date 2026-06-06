# =============================================================================
# Chapter 4, Exercise 1: Recognising the Problem
# Categorisation of systolic blood pressure and stroke risk
# =============================================================================
# This exercise is primarily conceptual. Answers are provided as comments,
# with supporting code to illustrate the points.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix
from scipy.special import expit

# --- Question (a) ---
# What assumptions does categorising SBP into Normal (<120), Elevated (120-129),
# Stage 1 HTN (130-139), Stage 2 HTN (>=140) impose?
#
# ANSWER:
# 1. Within each category, the relationship between SBP and stroke risk is FLAT.
#    A patient with SBP=121 is treated identically to SBP=129.
# 2. At category boundaries, there is a SUDDEN JUMP in risk.
#    SBP=129 and SBP=130 are treated as fundamentally different despite only
#    1 mmHg difference.
# 3. The cut-points (120, 130, 140) are assumed to be biologically meaningful
#    boundaries, which may not reflect the true continuous biology.

# --- Question (b) ---
# A patient with SBP 131 is classified the same as SBP 139. Why is this problematic?
#
# ANSWER:
# The difference of 8 mmHg (131 vs 139) is clinically substantial. In reality,
# stroke risk increases with higher SBP. By lumping these patients together,
# we lose the ability to distinguish their different risk levels. The patient
# at 139 mmHg (close to Stage 2) likely has meaningfully higher stroke risk
# than the patient at 131 mmHg, but the categorised model assigns them
# identical predicted risk. This information loss reduces statistical power
# and can bias effect estimates.

# --- Question (c) ---
# Suggest a better modelling approach.
#
# ANSWER:
# Use a restricted cubic spline (RCS) with 3-5 knots to model SBP as a
# continuous predictor. This:
#   - Preserves the full information in SBP
#   - Allows the relationship to be non-linear (capturing any steepening
#     at higher pressures)
#   - Constrains the curve to be linear in the tails (sensible extrapolation)
#   - Uses only k-1 degrees of freedom for k knots (parsimonious)

# --- Supporting demonstration ---
# Simulate data to visually compare categorised vs spline approaches

np.random.seed(2025)
n = 1000
sbp = np.random.normal(130, 15, size=n)
sbp = np.clip(sbp, 90, 200)

# True relationship: risk accelerates at higher SBP
logit_p = -6 + 0.02 * sbp + 0.0002 * (sbp - 130)**2
y = np.random.binomial(1, expit(logit_p))

df = pd.DataFrame({'sbp': sbp, 'stroke': y})

# --- Model 1: Categorised ---
df['sbp_cat'] = pd.cut(df['sbp'],
                         bins=[-np.inf, 120, 130, 140, np.inf],
                         labels=['Normal', 'Elevated', 'Stage1', 'Stage2'])
X_cat = pd.get_dummies(df['sbp_cat'], drop_first=True).astype(float)
X_cat = sm.add_constant(X_cat)
fit_cat = sm.GLM(df['stroke'], X_cat, family=sm.families.Binomial()).fit()

# --- Model 2: Linear ---
X_lin = sm.add_constant(df[['sbp']])
fit_lin = sm.GLM(df['stroke'], X_lin, family=sm.families.Binomial()).fit()

# --- Model 3: RCS (natural cubic spline with df=3, i.e., 4 knots) ---
X_rcs = dmatrix("cr(sbp, df=3)", data=df, return_type='dataframe')
fit_rcs = sm.GLM(df['stroke'], X_rcs, family=sm.families.Binomial()).fit()

# Compare AIC
print("AIC Comparison:")
print(f"  Categorised: {fit_cat.aic:.1f}")
print(f"  Linear:      {fit_lin.aic:.1f}")
print(f"  RCS (4 knots): {fit_rcs.aic:.1f}")
print("\nLower AIC = better fit. The RCS model captures the true non-linear")
print("relationship without imposing arbitrary cut-points.")

# --- Plot comparison ---
sbp_range = np.linspace(df['sbp'].min(), df['sbp'].max(), 200)

# True probability
true_logit = -6 + 0.02 * sbp_range + 0.0002 * (sbp_range - 130)**2
true_prob = expit(true_logit)

# Linear predictions
X_lin_pred = sm.add_constant(pd.DataFrame({'sbp': sbp_range}))
lin_prob = fit_lin.predict(X_lin_pred)

# RCS predictions
X_rcs_pred = dmatrix("cr(sbp, df=3)",
                      data=pd.DataFrame({'sbp': sbp_range}),
                      return_type='dataframe')
rcs_prob = fit_rcs.predict(X_rcs_pred)

plt.figure(figsize=(10, 6))
plt.plot(sbp_range, true_prob, 'k--', linewidth=2, label='True relationship')
plt.plot(sbp_range, lin_prob, color='#3498db', linewidth=1.5, label='Linear')
plt.plot(sbp_range, rcs_prob, color='#e74c3c', linewidth=1.5, label='RCS (4 knots)')
plt.xlabel('Systolic Blood Pressure (mmHg)')
plt.ylabel('Predicted Probability of Stroke')
plt.title('Comparing modelling approaches for SBP-stroke relationship')
plt.legend()
plt.tight_layout()
plt.show()
