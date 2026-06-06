# =============================================================================
# Chapter 18 - Exercise 1: Basic Meta-Analysis in Python
# (Python version of the R Exercise 1 for completeness)
# New anticoagulant vs warfarin for stroke prevention in AF
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# --- Dataset ---
af_trials = pd.DataFrame({
    'study': ['TRAIL-1', 'GUARD-AF', 'SHIELD', 'ORBIT-AF',
              'VENTURE', 'COMPASS-AF', 'PIONEER-2', 'ATLAS-AF'],
    'events_new':  [28, 45, 112, 67, 33, 89, 52, 41],
    'n_new':       [1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100],
    'events_warf': [42, 58, 148, 84, 29, 102, 61, 53],
    'n_warf':      [1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100]
})

# --- Part (a): Compute log RR and variances ---
af_trials['rr'] = (af_trials['events_new'] / af_trials['n_new']) / \
                  (af_trials['events_warf'] / af_trials['n_warf'])
af_trials['log_rr'] = np.log(af_trials['rr'])
af_trials['var_log_rr'] = (
    1/af_trials['events_new'] - 1/af_trials['n_new'] +
    1/af_trials['events_warf'] - 1/af_trials['n_warf']
)
af_trials['se_log_rr'] = np.sqrt(af_trials['var_log_rr'])

# 95% CI for each study
af_trials['ci_lo'] = np.exp(af_trials['log_rr'] - 1.96 * af_trials['se_log_rr'])
af_trials['ci_hi'] = np.exp(af_trials['log_rr'] + 1.96 * af_trials['se_log_rr'])

print("Individual trial risk ratios:")
print(af_trials[['study', 'rr', 'ci_lo', 'ci_hi']].to_string(
    float_format=lambda x: f"{x:.3f}", index=False))

# --- Part (b): Fixed-effect (inverse variance) ---
w_fe = 1 / af_trials['var_log_rr']
pooled_fe = np.sum(w_fe * af_trials['log_rr']) / np.sum(w_fe)
se_fe = np.sqrt(1 / np.sum(w_fe))
ci_fe = (pooled_fe - 1.96*se_fe, pooled_fe + 1.96*se_fe)

print(f"\n--- Fixed-effect meta-analysis ---")
print(f"Pooled log RR: {pooled_fe:.4f}")
print(f"Pooled RR: {np.exp(pooled_fe):.4f} "
      f"(95% CI: {np.exp(ci_fe[0]):.4f} - {np.exp(ci_fe[1]):.4f})")

# --- Part (c): DerSimonian-Laird random-effects ---
k = len(af_trials)
Q = np.sum(w_fe * (af_trials['log_rr'] - pooled_fe)**2)
C = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
tau2 = max(0, (Q - (k - 1)) / C)

w_re = 1 / (af_trials['var_log_rr'] + tau2)
pooled_re = np.sum(w_re * af_trials['log_rr']) / np.sum(w_re)
se_re = np.sqrt(1 / np.sum(w_re))
ci_re = (pooled_re - 1.96*se_re, pooled_re + 1.96*se_re)

# --- Part (d): Q, I-squared, tau-squared ---
I2 = max(0, (Q - (k-1)) / Q) * 100
p_Q = 1 - stats.chi2.cdf(Q, k-1)

print(f"\n--- Random-effects meta-analysis (DerSimonian-Laird) ---")
print(f"Pooled log RR: {pooled_re:.4f}")
print(f"Pooled RR: {np.exp(pooled_re):.4f} "
      f"(95% CI: {np.exp(ci_re[0]):.4f} - {np.exp(ci_re[1]):.4f})")
print(f"\nHeterogeneity:")
print(f"  tau^2: {tau2:.4f}")
print(f"  I^2: {I2:.1f}%")
print(f"  Q: {Q:.2f}, df = {k-1}, p = {p_Q:.4f}")

# Prediction interval
t_crit = stats.t.ppf(0.975, k - 2)
pi_lo = pooled_re - t_crit * np.sqrt(se_re**2 + tau2)
pi_hi = pooled_re + t_crit * np.sqrt(se_re**2 + tau2)
print(f"\nPrediction interval for RR: ({np.exp(pi_lo):.4f}, {np.exp(pi_hi):.4f})")

if np.exp(pi_lo) < 1 and np.exp(pi_hi) > 1:
    print("The prediction interval crosses 1, meaning some future settings")
    print("might not see a benefit from the new anticoagulant.")
else:
    print("The prediction interval does not cross 1.")

# --- Part (e): Forest plot ---
fig, ax = plt.subplots(figsize=(10, 8))

y_pos = list(range(k, 0, -1))
weights = w_re / w_re.max()

# Individual studies
for i, (_, row) in enumerate(af_trials.iterrows()):
    ci_lo_i = row['ci_lo']
    ci_hi_i = row['ci_hi']
    ax.plot([ci_lo_i, ci_hi_i], [y_pos[i], y_pos[i]], 'k-', linewidth=1)
    size = weights.iloc[i] * 200
    ax.scatter(row['rr'], y_pos[i], s=size, c='steelblue',
               zorder=5, edgecolors='darkblue')

# Null line
ax.axvline(x=1, color='black', linestyle='-', linewidth=0.5)

# Pooled estimate line
ax.axvline(x=np.exp(pooled_re), color='steelblue', linestyle='--', alpha=0.5)

# Pooled diamond
diamond_y = 0
dx = [np.exp(ci_re[0]), np.exp(pooled_re), np.exp(ci_re[1]), np.exp(pooled_re)]
dy = [diamond_y, diamond_y + 0.3, diamond_y, diamond_y - 0.3]
ax.fill(dx, dy, color='steelblue', alpha=0.7)

ax.set_yticks(y_pos + [0])
ax.set_yticklabels(list(af_trials['study']) + ['Pooled RE'])
ax.set_xlabel('Risk Ratio')
ax.set_title('Forest Plot: New Anticoagulant vs Warfarin for AF')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('forest_plot_af.png', dpi=300)
plt.show()

# --- Part (f): Funnel plot and Egger's test ---
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(af_trials['log_rr'], af_trials['se_log_rr'],
           c='steelblue', s=80, edgecolors='darkblue', zorder=5)
ax.axvline(x=pooled_re, color='grey', linestyle='--', alpha=0.7)

# Pseudo-confidence region
se_range = np.linspace(0, af_trials['se_log_rr'].max() * 1.1, 100)
ax.plot(pooled_re - 1.96*se_range, se_range, 'k--', alpha=0.3)
ax.plot(pooled_re + 1.96*se_range, se_range, 'k--', alpha=0.3)

# Add study labels
for _, row in af_trials.iterrows():
    ax.annotate(row['study'], (row['log_rr'], row['se_log_rr']),
                textcoords="offset points", xytext=(5, 5), fontsize=7)

ax.invert_yaxis()
ax.set_xlabel('Log Risk Ratio')
ax.set_ylabel('Standard Error')
ax.set_title('Funnel Plot')
plt.tight_layout()
plt.savefig('funnel_plot_af.png', dpi=300)
plt.show()

# Egger's test: weighted regression of effect on SE
import statsmodels.api as sm

X_egger = sm.add_constant(af_trials['se_log_rr'])
egger_model = sm.WLS(af_trials['log_rr'], X_egger, weights=w_fe).fit()

print(f"\n--- Egger's test ---")
print(f"Intercept: {egger_model.params.iloc[0]:.4f}")
print(f"p-value: {egger_model.pvalues.iloc[0]:.4f}")

if egger_model.pvalues.iloc[0] < 0.05:
    print("Egger's test is significant, suggesting potential publication bias.")
else:
    print("Egger's test is not significant.")
    print(f"Note: With only {k} studies, the test has limited power.")

# --- Leave-one-out sensitivity analysis ---
print(f"\n--- Leave-one-out sensitivity analysis ---")
print(f"{'Excluded study':<15} {'Pooled RR':>10} {'95% CI':>25} {'I2':>8}")
print("-" * 60)

for i in range(k):
    # Exclude study i
    mask = af_trials.index != i
    w_i = 1 / af_trials.loc[mask, 'var_log_rr']

    # Fixed-effect pooled
    pooled_i_fe = np.sum(w_i * af_trials.loc[mask, 'log_rr']) / np.sum(w_i)

    # Q and tau2
    Q_i = np.sum(w_i * (af_trials.loc[mask, 'log_rr'] - pooled_i_fe)**2)
    C_i = np.sum(w_i) - np.sum(w_i**2) / np.sum(w_i)
    tau2_i = max(0, (Q_i - (k - 2)) / C_i)

    # Random-effects
    w_re_i = 1 / (af_trials.loc[mask, 'var_log_rr'] + tau2_i)
    pooled_i = np.sum(w_re_i * af_trials.loc[mask, 'log_rr']) / np.sum(w_re_i)
    se_i = np.sqrt(1 / np.sum(w_re_i))
    ci_i = (pooled_i - 1.96*se_i, pooled_i + 1.96*se_i)
    I2_i = max(0, (Q_i - (k-2)) / Q_i) * 100

    print(f"{af_trials.loc[i, 'study']:<15} {np.exp(pooled_i):>10.4f} "
          f"({np.exp(ci_i[0]):.4f}, {np.exp(ci_i[1]):.4f})  {I2_i:>6.1f}%")

print("\nIf removing any single study changes the conclusion (CI crosses 1),")
print("the result is sensitive to that study.")
