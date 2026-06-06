# Chapter 14, Exercise 1: Bayesian Logistic Regression for ICU Mortality
# 500 ICU admissions with age, APACHE II, ventilation status, 28-day mortality

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.special import expit

# ---- (a) Simulate dataset ----
np.random.seed(101)
n = 500

icu_data = pd.DataFrame({
    'age': np.round(np.random.normal(62, 15, n)),
    'apache': np.round(np.random.normal(18, 7, n)),
    'ventilated': np.random.binomial(1, 0.35, n)
})

# True model
lp = (-4.5 + 0.02 * icu_data['age'] +
      0.12 * icu_data['apache'] +
      0.8 * icu_data['ventilated'])
icu_data['mortality'] = np.random.binomial(1, expit(lp))

print("Dataset summary:")
print(f"  N: {n}")
print(f"  Mortality rate: {icu_data['mortality'].mean():.3f}")
print(f"  Mean age: {icu_data['age'].mean():.1f}")
print(f"  Mean APACHE: {icu_data['apache'].mean():.1f}")
print(f"  % Ventilated: {icu_data['ventilated'].mean()*100:.1f}%")

# ---- (b) Fit Bayesian logistic regression ----
with pm.Model() as icu_model:
    # Weakly informative priors
    intercept = pm.Normal("Intercept", mu=0, sigma=5)
    b_age = pm.Normal("b_age", mu=0, sigma=2.5)
    b_apache = pm.Normal("b_apache", mu=0, sigma=2.5)
    b_vent = pm.Normal("b_ventilated", mu=0, sigma=2.5)

    # Linear predictor
    logit_p = (intercept +
               b_age * icu_data['age'].values +
               b_apache * icu_data['apache'].values +
               b_vent * icu_data['ventilated'].values)

    # Likelihood
    y_obs = pm.Bernoulli("mortality", logit_p=logit_p,
                          observed=icu_data['mortality'].values)

    # Sample
    trace = pm.sample(1000, tune=1000, chains=4, random_seed=42,
                       progressbar=True)

print("\nModel summary:")
print(az.summary(trace, var_names=["Intercept", "b_age", "b_apache",
                                     "b_ventilated"]))

# ---- (c) Prior predictive check ----
print("\n=== Part (c): Prior Predictive Check ===")

# Simulate mortality rates from the priors
np.random.seed(42)
n_sim = 2000
intercepts = np.random.normal(0, 5, n_sim)
b_ages = np.random.normal(0, 2.5, n_sim)
b_apaches = np.random.normal(0, 2.5, n_sim)
b_vents = np.random.normal(0, 2.5, n_sim)

# For a "typical" patient: age=62, apache=18, ventilated=0
sim_lp = intercepts + b_ages * 62 + b_apaches * 18 + b_vents * 0
sim_mort = expit(sim_lp)

print(f"Prior predictive mortality rates (typical patient):")
print(f"  Mean: {sim_mort.mean():.3f}")
print(f"  SD: {sim_mort.std():.3f}")
print(f"  Range: {sim_mort.min():.3f} to {sim_mort.max():.3f}")
print("The priors allow mortality rates from near 0 to near 1,")
print("covering all plausible ICU mortality rates. The priors are")
print("appropriately weakly informative.")

# ---- (d) Posterior odds ratios with 95% credible intervals ----
print("\n=== Part (d): Posterior Odds Ratios ===")

posterior = az.extract(trace)

for var_name, label in [("b_age", "Age"), ("b_apache", "APACHE"),
                          ("b_ventilated", "Ventilated")]:
    or_vals = np.exp(posterior[var_name].values)
    mean_or = or_vals.mean()
    ci_low = np.percentile(or_vals, 2.5)
    ci_high = np.percentile(or_vals, 97.5)
    print(f"OR {label:>12s}: {mean_or:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

# ---- (e) P(OR_APACHE > 1.10 | data) ----
print("\n=== Part (e): P(OR_APACHE > 1.10 | data) ===")

or_apache = np.exp(posterior["b_apache"].values)
prob_gt_110 = (or_apache > 1.10).mean()

print(f"P(OR_APACHE > 1.10 | data) = {prob_gt_110:.3f}")
print(f"\nInterpretation: There is a {prob_gt_110*100:.1f}% posterior probability")
print("that each unit increase in APACHE score increases the odds of")
print("28-day mortality by more than 10%. This is a direct probability")
print("statement about the parameter -- something only Bayesian inference")
print("can provide.")
