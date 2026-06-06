# =============================================================================
# Chapter 18 - Exercise 2: Meta-Analysis from Scratch in Python
# New anticoagulant vs warfarin for stroke prevention in AF
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
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

# --- Part (a): Compute log risk ratios and variances ---
af_trials['rr'] = (af_trials['events_new'] / af_trials['n_new']) / \
                  (af_trials['events_warf'] / af_trials['n_warf'])
af_trials['log_rr'] = np.log(af_trials['rr'])

# Variance of log RR: 1/a - 1/n1 + 1/c - 1/n2
af_trials['var_log_rr'] = (
    1/af_trials['events_new'] - 1/af_trials['n_new'] +
    1/af_trials['events_warf'] - 1/af_trials['n_warf']
)
af_trials['se_log_rr'] = np.sqrt(af_trials['var_log_rr'])

print("Part (a): Log risk ratios and standard errors")
print(af_trials[['study', 'log_rr', 'se_log_rr', 'rr']].to_string(
    float_format=lambda x: f"{x:.4f}", index=False))

# --- Part (b): Fixed-effect inverse-variance method ---
w_fe = 1 / af_trials['var_log_rr']
pooled_fe = np.sum(w_fe * af_trials['log_rr']) / np.sum(w_fe)
se_fe = np.sqrt(1 / np.sum(w_fe))
ci_fe = (pooled_fe - 1.96 * se_fe, pooled_fe + 1.96 * se_fe)
z_fe = pooled_fe / se_fe
p_fe = 2 * (1 - stats.norm.cdf(abs(z_fe)))

print(f"\n--- Part (b): Fixed-effect meta-analysis ---")
print(f"Pooled log RR: {pooled_fe:.4f}")
print(f"Pooled RR: {np.exp(pooled_fe):.4f} "
      f"(95% CI: {np.exp(ci_fe[0]):.4f} - {np.exp(ci_fe[1]):.4f})")
print(f"z = {z_fe:.3f}, p = {p_fe:.4f}")

# Study weights (as percentages)
af_trials['weight_fe_pct'] = (w_fe / w_fe.sum() * 100).round(1)
print(f"\nStudy weights (fixed-effect):")
print(af_trials[['study', 'weight_fe_pct']].to_string(index=False))

# --- Part (c): DerSimonian-Laird random-effects ---
k = len(af_trials)

# Step 1: Cochran's Q statistic
Q = np.sum(w_fe * (af_trials['log_rr'] - pooled_fe)**2)

# Step 2: C constant
C = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)

# Step 3: Between-study variance tau^2
tau2 = max(0, (Q - (k - 1)) / C)

# Step 4: Random-effects weights
w_re = 1 / (af_trials['var_log_rr'] + tau2)
pooled_re = np.sum(w_re * af_trials['log_rr']) / np.sum(w_re)
se_re = np.sqrt(1 / np.sum(w_re))
ci_re = (pooled_re - 1.96 * se_re, pooled_re + 1.96 * se_re)
z_re = pooled_re / se_re
p_re = 2 * (1 - stats.norm.cdf(abs(z_re)))

print(f"\n--- Part (c): Random-effects meta-analysis (DerSimonian-Laird) ---")
print(f"Pooled log RR: {pooled_re:.4f}")
print(f"Pooled RR: {np.exp(pooled_re):.4f} "
      f"(95% CI: {np.exp(ci_re[0]):.4f} - {np.exp(ci_re[1]):.4f})")
print(f"z = {z_re:.3f}, p = {p_re:.4f}")

af_trials['weight_re_pct'] = (w_re / w_re.sum() * 100).round(1)
print(f"\nStudy weights (random-effects):")
print(af_trials[['study', 'weight_fe_pct', 'weight_re_pct']].to_string(index=False))

# --- Part (d): Q, I-squared, tau-squared ---
p_Q = 1 - stats.chi2.cdf(Q, k - 1)
I2 = max(0, (Q - (k - 1)) / Q) * 100

print(f"\n--- Part (d): Heterogeneity statistics ---")
print(f"Q = {Q:.2f}, df = {k-1}, p = {p_Q:.4f}")
print(f"tau^2 = {tau2:.4f}")
print(f"tau = {np.sqrt(tau2):.4f}")
print(f"I^2 = {I2:.1f}%")

if I2 < 25:
    print("Interpretation: LOW heterogeneity")
elif I2 < 50:
    print("Interpretation: MODERATE heterogeneity")
elif I2 < 75:
    print("Interpretation: SUBSTANTIAL heterogeneity")
else:
    print("Interpretation: CONSIDERABLE heterogeneity")

# Prediction interval
t_crit = stats.t.ppf(0.975, k - 2)
pi_lo = pooled_re - t_crit * np.sqrt(se_re**2 + tau2)
pi_hi = pooled_re + t_crit * np.sqrt(se_re**2 + tau2)
print(f"\nPrediction interval for RR: ({np.exp(pi_lo):.4f}, {np.exp(pi_hi):.4f})")

# --- Part (e): Forest plot ---
fig, ax = plt.subplots(figsize=(10, 8))

y_pos = list(range(k, 0, -1))
weights_norm = w_re / w_re.max()

for i, (_, row) in enumerate(af_trials.iterrows()):
    ci_lo_i = np.exp(row['log_rr'] - 1.96 * row['se_log_rr'])
    ci_hi_i = np.exp(row['log_rr'] + 1.96 * row['se_log_rr'])

    # CI line
    ax.plot([ci_lo_i, ci_hi_i], [y_pos[i], y_pos[i]], 'k-', linewidth=1)

    # Study point (size proportional to weight)
    size = weights_norm.iloc[i] * 200
    ax.scatter(row['rr'], y_pos[i], s=size, c='steelblue',
               zorder=5, edgecolors='darkblue')

    # Weight annotation
    ax.text(2.0, y_pos[i], f"{row['weight_re_pct']:.1f}%",
            ha='left', va='center', fontsize=9)

# Null line
ax.axvline(x=1, color='black', linestyle='-', linewidth=0.5)

# Pooled estimate line
ax.axvline(x=np.exp(pooled_re), color='steelblue', linestyle='--', alpha=0.5)

# Pooled diamond
diamond_x = [np.exp(ci_re[0]), np.exp(pooled_re),
             np.exp(ci_re[1]), np.exp(pooled_re)]
diamond_y = [0, 0.3, 0, -0.3]
ax.fill(diamond_x, diamond_y, color='steelblue', alpha=0.7)

# Labels
ax.set_yticks(y_pos + [0])
ax.set_yticklabels(list(af_trials['study']) + ['Pooled RE'])
ax.set_xlabel('Risk Ratio')
ax.set_title('Forest Plot: New Anticoagulant vs Warfarin for AF\n'
             f'RE Model: RR = {np.exp(pooled_re):.3f} '
             f'(95% CI: {np.exp(ci_re[0]):.3f}-{np.exp(ci_re[1]):.3f}), '
             f'I\u00B2 = {I2:.1f}%')
ax.set_xscale('log')

# Add "Favours" labels
ax.text(0.5, -1.2, 'Favours new anticoagulant', ha='center',
        fontsize=9, style='italic')
ax.text(1.8, -1.2, 'Favours warfarin', ha='center',
        fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('forest_plot_af.png', dpi=300)
plt.show()

# --- Part (f): Funnel plot and Egger's test ---
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(af_trials['log_rr'], af_trials['se_log_rr'],
           c='steelblue', s=80, edgecolors='darkblue', zorder=5)
ax.axvline(x=pooled_re, color='grey', linestyle='--', alpha=0.7,
           label='Pooled effect')

# Pseudo-confidence regions
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
ax.set_title('Funnel Plot: Checking for Publication Bias')
ax.legend()
plt.tight_layout()
plt.savefig('funnel_plot_af.png', dpi=300)
plt.show()

# Egger's regression test
# Weighted regression of effect estimate on standard error
X_egger = sm.add_constant(af_trials['se_log_rr'])
egger_model = sm.WLS(af_trials['log_rr'], X_egger,
                     weights=1/af_trials['var_log_rr']).fit()

print(f"\n--- Part (f): Egger's test ---")
print(f"Intercept: {egger_model.params.iloc[0]:.4f}")
print(f"SE of intercept: {egger_model.bse.iloc[0]:.4f}")
print(f"t-statistic: {egger_model.tvalues.iloc[0]:.3f}")
print(f"p-value: {egger_model.pvalues.iloc[0]:.4f}")

if egger_model.pvalues.iloc[0] < 0.05:
    print("Egger's test is significant (p < 0.05).")
    print("This suggests potential publication bias or small-study effects.")
else:
    print("Egger's test is not significant.")
    print(f"Note: With only {k} studies, the test has limited power.")
    print("At least 10 studies are recommended for reliable Egger's test.")
