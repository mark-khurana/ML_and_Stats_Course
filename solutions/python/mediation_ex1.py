# =============================================================================
# Chapter 17c, Exercise 1: Decompose a Known Mediation Effect
# Simulate exposure -> mediator -> outcome and recover NDE, NIE, total, prop. med.
# =============================================================================
# NOTE: The chapter demonstrates this with R's CMAverse::cmest(). Here we
# implement the regression-based / product-of-coefficients estimator manually
# with statsmodels, which is exactly equivalent to CMAverse's "rb" model for a
# continuous mediator and continuous outcome WITHOUT an exposure-mediator
# interaction. A bootstrap gives the CI for the indirect effect.

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# --- Simulate exposure -> mediator -> outcome with a KNOWN decomposition ---
# Data-generating coefficients (the "truth"):
#   exposure -> mediator (a):        1.5
#   mediator -> outcome (b):         0.5
#   direct exposure -> outcome (c'): 1.0
a_true, b_true, cp_true = 1.5, 0.5, 1.0

n = 5000
exposure = np.random.binomial(1, 0.5, n)                       # randomized-like
mediator = a_true * exposure + np.random.normal(0, 1, n)       # continuous mediator
outcome = cp_true * exposure + b_true * mediator + np.random.normal(0, 1, n)

df = pd.DataFrame(dict(exposure=exposure, mediator=mediator, outcome=outcome))

# -----------------------------------------------------------------------------
# (a) TRUE natural direct/indirect effects (by construction)
# -----------------------------------------------------------------------------
# Linear model, no exposure-mediator interaction:
#   NIE  = a * b, NDE = c', Total = NDE + NIE, Prop. mediated = NIE / Total
nie_true = a_true * b_true            # 1.5 * 0.5 = 0.75
nde_true = cp_true                    # 1.0
total_true = nde_true + nie_true      # 1.75
prop_true = nie_true / total_true     # 0.4286

# -----------------------------------------------------------------------------
# (b) ESTIMATE the decomposition from the data
# -----------------------------------------------------------------------------
# Mediator model M ~ X: exposure coefficient is the 'a' path
m_model = smf.ols("mediator ~ exposure", data=df).fit()
a_hat = m_model.params["exposure"]

# Outcome model Y ~ X + M: exposure coef = NDE (c'), mediator coef = 'b'
y_model = smf.ols("outcome ~ exposure + mediator", data=df).fit()
nde_hat = y_model.params["exposure"]  # natural direct effect
b_hat = y_model.params["mediator"]    # mediator -> outcome

nie_hat = a_hat * b_hat               # natural indirect effect (product of coefs)
total_hat = nde_hat + nie_hat
prop_hat = nie_hat / total_hat


# Bootstrap 95% CI for the indirect effect (NIE)
def nie_boot(data):
    bm = smf.ols("mediator ~ exposure", data=data).fit().params["exposure"]
    by = smf.ols("outcome ~ exposure + mediator", data=data).fit().params["mediator"]
    return bm * by


boot = np.array([nie_boot(df.sample(len(df), replace=True, random_state=i))
                 for i in range(1000)])
lo, hi = np.percentile(boot, [2.5, 97.5])

# -----------------------------------------------------------------------------
# Print true vs estimated
# -----------------------------------------------------------------------------
print("=== Exercise 1: Mediation decomposition (true vs estimated) ===\n")
res = pd.DataFrame({
    "Quantity": ["NDE (direct)", "NIE (indirect)", "Total effect", "Prop. mediated"],
    "True": [nde_true, nie_true, total_true, prop_true],
    "Estimated": [nde_hat, nie_hat, total_hat, prop_hat],
}).round(4)
print(res.to_string(index=False))

print(f"\nNIE 95% bootstrap CI: ({lo:.3f}, {hi:.3f})")
print(f"Truth NIE = 0.75 lies inside CI: {lo <= 0.75 <= hi}")

# -----------------------------------------------------------------------------
# (c) Clinician interpretation of the proportion mediated
# -----------------------------------------------------------------------------
# About 43% of the exposure's total effect on the outcome travels through the
# mediator, so roughly two-fifths of the benefit could in principle be captured
# by acting on the mediator alone, while the majority is a direct effect that a
# mediator-targeting intervention would miss.
