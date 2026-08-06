# =============================================================================
# Chapter 6c, Exercise 3: Multiple imputation end to end
# m = 30 imputations with IterativeImputer on the cohort with missing BMI and
# SBP; pool with Rubin's rules and compare pooled / complete-case / full-data.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 800

# --- Simulate the chapter's complete clinical cohort ---
age = rng.normal(60, 12, n)
bmi = rng.normal(28, 5, n)
sbp = 100 + 0.4 * age + 0.6 * bmi + rng.normal(0, 10, n)
event = rng.binomial(1, 1 / (1 + np.exp(-(-6 + 0.04 * age + 0.03 * bmi + 0.02 * sbp))))
full = pd.DataFrame({"age": age, "bmi": bmi, "sbp": sbp, "event": event})

# --- Induce MAR missingness in BMI and SBP (both depend on observed age) ---
p_missing_bmi = 1 / (1 + np.exp(-(-2 + 0.05 * (age - 60))))
p_missing_sbp = 1 / (1 + np.exp(-(-1.4 + 0.03 * (age - 60))))
data = full.copy()
data.loc[rng.binomial(1, p_missing_bmi) == 1, "bmi"] = np.nan
data.loc[rng.binomial(1, p_missing_sbp) == 1, "sbp"] = np.nan

n_complete = int(data.dropna().shape[0])
print("=== Exercise 3: Multiple imputation end to end ===\n")
print(f"Complete cases: {n_complete} of {n}\n")


def fit_logit(df):
    X = sm.add_constant(df[["age", "bmi", "sbp"]])
    return sm.Logit(df["event"], X).fit(disp=0)


# --- Reference models: full data and complete-case ---
fit_full = fit_logit(full)
fit_cc = fit_logit(data.dropna())

# --- (a) Multiple imputation with m = 30 ---
# IterativeImputer is the sklearn counterpart of MICE. sample_posterior=True
# with a different random_state each pass yields genuinely different completed
# datasets (multiple, not single, imputation). The outcome 'event' is INCLUDED
# among the imputation predictors.
m = 30
cols = ["age", "bmi", "sbp", "event"]
coefs, ses = {"age": [], "bmi": [], "sbp": []}, {"age": [], "bmi": [], "sbp": []}
imputed_bmi, imputed_sbp = [], []
mask_bmi = data["bmi"].isna().to_numpy()
mask_sbp = data["sbp"].isna().to_numpy()

for i in range(m):
    imputer = IterativeImputer(sample_posterior=True, random_state=i)
    completed = pd.DataFrame(imputer.fit_transform(data[cols]), columns=cols)
    completed["event"] = data["event"].to_numpy()          # outcome is observed
    fit = fit_logit(completed)
    for v in ("age", "bmi", "sbp"):
        coefs[v].append(fit.params[v])
        ses[v].append(fit.bse[v])
    imputed_bmi.append(completed["bmi"].to_numpy()[mask_bmi])
    imputed_sbp.append(completed["sbp"].to_numpy()[mask_sbp])


# --- (b) Pool with Rubin's rules (implemented by hand) ---
def rubin(coef_list, se_list, m):
    q = np.array(coef_list)
    u = np.array(se_list) ** 2
    q_bar = q.mean()                       # pooled estimate = average of coefs
    u_bar = u.mean()                       # within-imputation variance
    b = q.var(ddof=1)                      # between-imputation variance
    total_var = u_bar + (1 + 1 / m) * b    # Rubin's total variance
    return q_bar, np.sqrt(total_var)


pooled = {v: rubin(coefs[v], ses[v], m) for v in ("age", "bmi", "sbp")}

# --- (c) Compare pooled vs complete-case vs full-data (focus: age coef) ---
print("--- (c) age coefficient (data-generating truth = 0.04) ---")
cmp = pd.DataFrame({
    "method": ["Full data (truth)", "Complete-case", "MI pooled (m=30)"],
    "coef": [fit_full.params["age"], fit_cc.params["age"], pooled["age"][0]],
    "se":   [fit_full.bse["age"],    fit_cc.bse["age"],    pooled["age"][1]],
})
print(cmp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nThe MI pooled estimate uses all 800 patients and should sit between the")
print("complete-case value and the full-data truth, recovering the truth best")
print("while giving honest standard errors (wider than a naive single imputation).\n")

print("--- Full pooled summary (Rubin's rules) ---")
full_tab = pd.DataFrame({
    "term": ["age", "bmi", "sbp"],
    "estimate": [pooled[v][0] for v in ("age", "bmi", "sbp")],
    "std.error": [pooled[v][1] for v in ("age", "bmi", "sbp")],
    "truth": [0.04, 0.03, 0.02],
})
print(full_tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --- (d) Imputation diagnostics ---
print("\n--- (d) Imputation diagnostics ---")
print("Observed vs imputed summaries (mean [sd]) across the 30 imputations:")
for name, mask, imp_vals in (("bmi", mask_bmi, imputed_bmi),
                             ("sbp", mask_sbp, imputed_sbp)):
    obs = data[name].dropna().to_numpy()
    imp_all = np.concatenate(imp_vals)
    print(f"  {name:<4} observed: {obs.mean():6.2f} [{obs.std(ddof=1):.2f}]   "
          f"imputed: {imp_all.mean():6.2f} [{imp_all.std(ddof=1):.2f}]")
print("Imputed means/spreads that track the observed ones indicate plausible")
print("imputations (the sklearn analogue of mice's densityplot / convergence checks).")
