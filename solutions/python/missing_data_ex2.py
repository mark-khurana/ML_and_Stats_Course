# =============================================================================
# Chapter 6c, Exercise 2: Quantify the cost of complete-case analysis
# Add 20% MAR missingness in sbp on top of missing BMI, then compare the
# complete-case age coefficient/SE to the full-data model.
# =============================================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 800

# --- Simulate the chapter's complete clinical cohort ---
age = rng.normal(60, 12, n)
bmi = rng.normal(28, 5, n)
sbp = 100 + 0.4 * age + 0.6 * bmi + rng.normal(0, 10, n)      # systolic BP
lin = -6 + 0.04 * age + 0.03 * bmi + 0.02 * sbp
event = rng.binomial(1, 1 / (1 + np.exp(-lin)))
full = pd.DataFrame({"age": age, "bmi": bmi, "sbp": sbp, "event": event})

# --- Induce MAR missingness in BMI (older patients more likely missing) ---
p_missing_bmi = 1 / (1 + np.exp(-(-2 + 0.05 * (age - 60))))
miss_bmi = rng.binomial(1, p_missing_bmi) == 1

# --- ADD ~20% MAR missingness in sbp, also depending on observed age ---
# Intercept chosen so the marginal missing fraction is about 20%.
p_missing_sbp = 1 / (1 + np.exp(-(-1.4 + 0.03 * (age - 60))))
miss_sbp = rng.binomial(1, p_missing_sbp) == 1

data = full.copy()
data.loc[miss_bmi, "bmi"] = np.nan
data.loc[miss_sbp, "sbp"] = np.nan

# --- (a) How many complete cases remain? ---
n_complete = int(data.dropna().shape[0])
print("=== Exercise 2: Cost of complete-case analysis ===\n")
print(f"BMI missing:            {miss_bmi.sum()} ({100 * miss_bmi.mean():.1f}%)")
print(f"SBP missing:            {miss_sbp.sum()} ({100 * miss_sbp.mean():.1f}%)")
print(f"Complete cases (a):     {n_complete} of {n} "
      f"({100 * n_complete / n:.1f}%)\n")


def fit_logit(df):
    X = sm.add_constant(df[["age", "bmi", "sbp"]])
    return sm.Logit(df["event"], X).fit(disp=0)


# --- (b) Full-data model vs complete-case model: age coefficient & SE ---
fit_full = fit_logit(full)
fit_cc = fit_logit(data.dropna())

print("--- (b) age coefficient (data-generating truth = 0.04) ---")
print(f"Full-data:      coef = {fit_full.params['age']:+.4f}   "
      f"SE = {fit_full.bse['age']:.4f}   (n = {int(fit_full.nobs)})")
print(f"Complete-case:  coef = {fit_cc.params['age']:+.4f}   "
      f"SE = {fit_cc.bse['age']:.4f}   (n = {int(fit_cc.nobs)})\n")
print(f"SE inflation (complete-case / full-data): "
      f"{fit_cc.bse['age'] / fit_full.bse['age']:.2f}x\n")

# --- (c) Why dropping rows became much more costly ---
print("--- (c) Comment ---")
print("Requiring BOTH bmi and sbp to be present removes any patient missing")
print("either one, so the two missing-data fractions compound -- the surviving")
print("subset shrinks far more than either variable alone, and because both")
print("gaps are age-driven the remainder is increasingly younger-skewed,")
print("giving a smaller, less representative sample with larger standard errors.")
