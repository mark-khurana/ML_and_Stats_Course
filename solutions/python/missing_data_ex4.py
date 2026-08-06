# =============================================================================
# Chapter 6c, Exercise 4: MNAR sensitivity analysis (conceptual + code sketch)
# BMI suspected MNAR (high-BMI patients less likely to be weighed): reason
# about the bias and sketch a delta-adjustment sensitivity analysis.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# (a) Why standard MI (which assumes MAR) may UNDERESTIMATE the association.
#
#     Standard MI fills the gaps using the observed data under a MAR model, so
#     imputed BMIs are drawn towards the observed (lower) range. If the truly
#     missing patients had systematically HIGHER BMI, the imputations are too
#     low and the upper tail of BMI -- the part most strongly linked to the
#     outcome -- is under-represented, so the fitted BMI-outcome association is
#     biased towards zero (attenuated).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (b) Delta-adjustment: direction and magnitude.
#
#     A delta (pattern-mixture) adjustment adds a fixed offset delta to the
#     imputed BMIs to represent "the unweighed patients were heavier than MAR
#     predicts". Here delta should be POSITIVE (shift imputed BMIs UP), because
#     the suspected mechanism removes high values. The magnitude spans a
#     clinically plausible range, e.g. 0 to +5 BMI units (0, +1, +2, +3, +5),
#     ideally anchored by external knowledge of how much heavier the missing
#     group is thought to be.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (c) Reporting across a range of delta values (illustrative code).
#     For each delta: impute under MAR, add delta to the imputed BMIs only,
#     refit the logistic model on each completed set, pool with Rubin's rules,
#     and tabulate the pooled BMI coefficient (+CI) as a function of delta.
#     A finding that stays clearly non-null across the range is robust to MNAR.
# -----------------------------------------------------------------------------

rng = np.random.default_rng(42)
n = 800
age = rng.normal(60, 12, n)
bmi = rng.normal(28, 5, n)
sbp = 100 + 0.4 * age + 0.6 * bmi + rng.normal(0, 10, n)
event = rng.binomial(1, 1 / (1 + np.exp(-(-6 + 0.04 * age + 0.03 * bmi + 0.02 * sbp))))
full = pd.DataFrame({"age": age, "bmi": bmi, "sbp": sbp, "event": event})

# MNAR-style deletion: higher BMI => more likely missing (for illustration).
p_missing_bmi = 1 / (1 + np.exp(-(-2 + 0.15 * (bmi - 28))))
mask_bmi = rng.binomial(1, p_missing_bmi) == 1
data = full.copy()
data.loc[mask_bmi, "bmi"] = np.nan

print("=== Exercise 4: MNAR delta-adjustment sensitivity analysis ===\n")
print(f"BMI missing (MNAR mechanism): {mask_bmi.sum()} ({100 * mask_bmi.mean():.1f}%)\n")


def fit_logit(df):
    X = sm.add_constant(df[["age", "bmi", "sbp"]])
    return sm.Logit(df["event"], X).fit(disp=0)


def rubin(coef_list, se_list, m):
    q = np.array(coef_list)
    u = np.array(se_list) ** 2
    q_bar = q.mean()
    total_var = u.mean() + (1 + 1 / m) * q.var(ddof=1)
    return q_bar, np.sqrt(total_var)


# Impute m sets ONCE under MAR, then re-use them with each delta shift.
m = 20
cols = ["age", "bmi", "sbp", "event"]
mask = data["bmi"].isna().to_numpy()
completed_sets = []
for i in range(m):
    imputer = IterativeImputer(sample_posterior=True, random_state=i)
    comp = pd.DataFrame(imputer.fit_transform(data[cols]), columns=cols)
    comp["event"] = data["event"].to_numpy()
    completed_sets.append(comp)

deltas = [0, 1, 2, 3, 5]      # positive shifts on the BMI scale
rows = []
for d in deltas:
    coefs, ses = [], []
    for comp in completed_sets:
        shifted = comp.copy()
        shifted.loc[mask, "bmi"] = shifted.loc[mask, "bmi"] + d   # delta adjustment
        fit = fit_logit(shifted)
        coefs.append(fit.params["bmi"])
        ses.append(fit.bse["bmi"])
    est, se = rubin(coefs, ses, m)
    rows.append({"delta": d, "bmi_coef": est, "se": se,
                 "lo": est - 1.96 * se, "hi": est + 1.96 * se})

tab = pd.DataFrame(rows)
print("Pooled BMI coefficient across delta (truth = 0.03):")
print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nReading the table: as delta increases (assuming heavier unweighed")
print("patients), the pooled BMI coefficient moves away from the attenuated")
print("MAR value (delta = 0). If the coefficient and CI stay clearly positive")
print("across the plausible delta range, the BMI-outcome association is robust")
print("to this MNAR concern; if it collapses under a mild shift, interpret with")
print("caution.")
