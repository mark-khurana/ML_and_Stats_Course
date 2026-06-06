# Chapter 14, Exercise 2: Hierarchical Model for Multi-Site Drug Trial
# 12 hospitals, LDL cholesterol change, new statin vs standard care

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---- (a) Simulate data ----
np.random.seed(789)
n_hospitals = 12
patients = [20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
grand_effect = -25  # mg/dL
tau = 5             # between-hospital SD

hospital_effects = np.random.normal(0, tau, n_hospitals)

rows = []
for j in range(n_hospitals):
    nj = patients[j]
    trt = np.tile([0, 1], nj // 2 + 1)[:nj]
    ldl_change = ((grand_effect + hospital_effects[j]) * trt +
                  np.random.normal(0, 20, nj))
    for i in range(nj):
        rows.append({
            'hospital': j,
            'treatment': trt[i],
            'ldl_change': ldl_change[i]
        })

trial_data = pd.DataFrame(rows)

print("Data summary:")
print(f"  Total patients: {len(trial_data)}")
print(f"  Hospitals: {n_hospitals}")
print(f"  True grand effect: {grand_effect} mg/dL")
print(f"  True between-hospital SD: {tau} mg/dL")

# ---- No-pooling estimates (OLS per hospital) ----
no_pool = []
for j in range(n_hospitals):
    df_j = trial_data[trial_data['hospital'] == j]
    trt_mean = df_j[df_j['treatment'] == 1]['ldl_change'].mean()
    ctl_mean = df_j[df_j['treatment'] == 0]['ldl_change'].mean()
    no_pool.append(trt_mean - ctl_mean)

no_pool = np.array(no_pool)

# ---- (b) & (c) Partial pooling approximation ----
# For a full Bayesian fit, use PyMC. Here we approximate shrinkage
# to demonstrate the concept without requiring MCMC sampling.

# Estimate within-hospital variance from data
within_var = []
for j in range(n_hospitals):
    df_j = trial_data[trial_data['hospital'] == j]
    trt_vals = df_j[df_j['treatment'] == 1]['ldl_change'].values
    ctl_vals = df_j[df_j['treatment'] == 0]['ldl_change'].values
    # Variance of the treatment effect estimate
    se_j = np.sqrt(np.var(trt_vals, ddof=1) / len(trt_vals) +
                   np.var(ctl_vals, ddof=1) / len(ctl_vals))
    within_var.append(se_j**2)

within_var = np.array(within_var)

# Estimate between-hospital variance using method of moments
grand_mean_est = np.mean(no_pool)
tau_est_sq = max(0, np.var(no_pool, ddof=1) - np.mean(within_var))
tau_est = np.sqrt(tau_est_sq)

# Shrinkage factor: B_j = within_var_j / (within_var_j + tau^2)
shrinkage = within_var / (within_var + tau_est_sq)
partial_pool = grand_mean_est + (1 - shrinkage) * (no_pool - grand_mean_est)

print(f"\nEstimated grand mean effect: {grand_mean_est:.1f} mg/dL")
print(f"Estimated between-hospital SD: {tau_est:.1f} mg/dL")

# ---- (c) Shrinkage plot ----
fig, ax = plt.subplots(figsize=(10, 6))
order = np.argsort(patients)
y_pos = np.arange(n_hospitals)

for i, idx in enumerate(order):
    ax.annotate("", xy=(partial_pool[idx], i), xytext=(no_pool[idx], i),
                arrowprops=dict(arrowstyle="->", color="grey"))

ax.scatter(no_pool[order], y_pos, color="steelblue", s=60, zorder=2,
           label="No pooling (OLS)")
ax.scatter(partial_pool[order], y_pos, color="firebrick", s=60, zorder=2,
           label="Partial pooling")
ax.axvline(grand_mean_est, linestyle="--", color="grey", linewidth=1)
ax.text(grand_mean_est + 0.5, n_hospitals - 0.5,
        f"Grand mean = {grand_mean_est:.1f}", fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"Hospital {order[i]+1}\n(n={patients[order[i]]})"
                     for i in range(n_hospitals)], fontsize=9)
ax.set_xlabel("Treatment Effect (mg/dL change in LDL)")
ax.set_title("Shrinkage in Hierarchical Model\n"
             "Arrows show estimates pulled toward the grand mean")
ax.legend()
plt.tight_layout()
plt.savefig("ch14_ex2_shrinkage.png", dpi=150)
plt.show()

# ---- (d) Compare approaches ----
print("\n=== Part (d): Comparison ===\n")
print(f"{'Hospital':>8} | {'N':>4} | {'No-Pool':>10} | {'Partial Pool':>12} | {'Shrinkage':>9}")
print("-" * 55)
for j in range(n_hospitals):
    shrink_amt = abs(no_pool[j] - partial_pool[j])
    print(f"   {j+1:>2}     | {patients[j]:>3}  | {no_pool[j]:>9.1f}  | {partial_pool[j]:>11.1f}  | {shrink_amt:>8.1f}")

print("\nThe hierarchical model is MORE APPROPRIATE because:")
print("1. Small hospitals have noisy OLS estimates that are shrunk")
print("   toward the grand mean, reducing estimation error.")
print("2. Large hospitals retain their individual estimates since")
print("   their data are informative enough.")
print("3. Between-hospital SD (tau) is estimated from data,")
print("   quantifying heterogeneity across sites.")
print("4. Hospital-by-hospital OLS ignores shared structure --")
print("   all hospitals study the same drug. The hierarchical model")
print("   borrows strength across sites while allowing genuine")
print("   between-site variation.")

# NOTE: For a full Bayesian hierarchical model, use PyMC:
#
# import pymc as pm
# with pm.Model() as hier_model:
#     mu_trt = pm.Normal("mu_trt", mu=0, sigma=30)
#     tau = pm.HalfCauchy("tau", beta=5)
#     trt_j = pm.Normal("trt_j", mu=mu_trt, sigma=tau, shape=n_hospitals)
#     sigma = pm.Exponential("sigma", lam=0.05)
#     mu = trt_j[hospital_idx] * treatment
#     y = pm.Normal("y", mu=mu, sigma=sigma, observed=ldl_change)
#     trace = pm.sample(1000, tune=1000, chains=4, random_seed=42)
