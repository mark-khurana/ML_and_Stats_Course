# =============================================================================
# Chapter 18 - Exercise 3: Subgroup Analysis and Meta-Regression in Python
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
    'n_warf':      [1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100],
    'region':      ['Europe', 'North America', 'Europe', 'Asia',
                    'North America', 'Europe', 'Asia', 'North America'],
    'mean_age':    [72, 68, 74, 65, 70, 71, 63, 69],
    'pct_female':  [38, 42, 35, 48, 40, 37, 52, 44]
})

# Compute effect sizes
af_trials['rr'] = (af_trials['events_new'] / af_trials['n_new']) / \
                  (af_trials['events_warf'] / af_trials['n_warf'])
af_trials['log_rr'] = np.log(af_trials['rr'])
af_trials['var_log_rr'] = (
    1/af_trials['events_new'] - 1/af_trials['n_new'] +
    1/af_trials['events_warf'] - 1/af_trials['n_warf']
)
af_trials['se_log_rr'] = np.sqrt(af_trials['var_log_rr'])


def dl_random_effects(log_rr, var):
    """DerSimonian-Laird random-effects meta-analysis."""
    k = len(log_rr)
    w_fe = 1 / var
    pooled_fe = np.sum(w_fe * log_rr) / np.sum(w_fe)
    Q = np.sum(w_fe * (log_rr - pooled_fe)**2)
    C = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau2 = max(0, (Q - (k - 1)) / C)
    w_re = 1 / (var + tau2)
    pooled_re = np.sum(w_re * log_rr) / np.sum(w_re)
    se_re = np.sqrt(1 / np.sum(w_re))
    I2 = max(0, (Q - (k-1)) / Q) * 100 if Q > 0 else 0
    return pooled_re, se_re, tau2, Q, I2, w_re


# Overall random-effects analysis
pooled_re, se_re, tau2, Q, I2, w_re = dl_random_effects(
    af_trials['log_rr'].values, af_trials['var_log_rr'].values
)

# --- Part (a): Subgroup analysis by region ---
print("=" * 60)
print("Part (a): Subgroup Analysis by Region")
print("=" * 60)

regions = af_trials['region'].unique()
subgroup_results = []

for region in regions:
    mask = af_trials['region'] == region
    sub = af_trials[mask]

    if len(sub) >= 2:
        p_re, s_re, t2, q, i2, w = dl_random_effects(
            sub['log_rr'].values, sub['var_log_rr'].values
        )
        ci = (p_re - 1.96*s_re, p_re + 1.96*s_re)
    else:
        # Single study: use study estimate directly
        p_re = sub['log_rr'].values[0]
        s_re = sub['se_log_rr'].values[0]
        t2, q, i2 = 0, 0, 0
        ci = (p_re - 1.96*s_re, p_re + 1.96*s_re)

    subgroup_results.append({
        'region': region,
        'k': len(sub),
        'pooled_rr': np.exp(p_re),
        'ci_lo': np.exp(ci[0]),
        'ci_hi': np.exp(ci[1]),
        'I2': i2,
        'tau2': t2
    })

    print(f"\n{region} (k={len(sub)}):")
    print(f"  Pooled RR: {np.exp(p_re):.4f} "
          f"(95% CI: {np.exp(ci[0]):.4f} - {np.exp(ci[1]):.4f})")
    print(f"  I^2: {i2:.1f}%")

# Test for subgroup differences using meta-regression with region dummies
region_dummies = pd.get_dummies(af_trials['region'], drop_first=True).astype(float)
X_sub = sm.add_constant(region_dummies)
wls_sub = sm.WLS(af_trials['log_rr'], X_sub,
                 weights=1/af_trials['var_log_rr']).fit()

# Wald test for region coefficients
from scipy.stats import chi2
f_stat = wls_sub.f_test(np.eye(len(wls_sub.params))[1:])
print(f"\nTest for subgroup differences:")
print(f"  F-statistic: {f_stat.fvalue[0][0]:.3f}")
print(f"  p-value: {f_stat.pvalue:.4f}")

if f_stat.pvalue < 0.05:
    print("  Significant difference between regions.")
else:
    print("  No significant difference between regions.")
    print(f"  (Limited power with only {len(af_trials)} studies)")

# --- Part (b): Meta-regression with mean age ---
print(f"\n{'=' * 60}")
print("Part (b): Meta-Regression with Mean Age")
print("=" * 60)

X_age = sm.add_constant(af_trials['mean_age'])
wls_age = sm.WLS(af_trials['log_rr'], X_age,
                 weights=1/af_trials['var_log_rr']).fit()

print(f"\nMeta-regression: log_rr ~ mean_age")
print(f"  Intercept: {wls_age.params.iloc[0]:.4f} (p = {wls_age.pvalues.iloc[0]:.4f})")
print(f"  Mean age slope: {wls_age.params.iloc[1]:.4f} (p = {wls_age.pvalues.iloc[1]:.4f})")
print(f"  R-squared: {wls_age.rsquared:.3f}")

if wls_age.pvalues.iloc[1] < 0.05:
    print("\n  There IS a significant relationship between mean age and effect size.")
    if wls_age.params.iloc[1] < 0:
        print("  Trials with older populations show larger treatment benefits.")
    else:
        print("  Trials with older populations show smaller treatment benefits.")
else:
    print(f"\n  No significant relationship (p = {wls_age.pvalues.iloc[1]:.3f}).")

# --- Part (c): Meta-regression with percentage female ---
print(f"\n{'=' * 60}")
print("Part (c): Meta-Regression with Percentage Female")
print("=" * 60)

X_fem = sm.add_constant(af_trials['pct_female'])
wls_fem = sm.WLS(af_trials['log_rr'], X_fem,
                 weights=1/af_trials['var_log_rr']).fit()

print(f"\nMeta-regression: log_rr ~ pct_female")
print(f"  Intercept: {wls_fem.params.iloc[0]:.4f} (p = {wls_fem.pvalues.iloc[0]:.4f})")
print(f"  Pct female slope: {wls_fem.params.iloc[1]:.4f} (p = {wls_fem.pvalues.iloc[1]:.4f})")
print(f"  R-squared: {wls_fem.rsquared:.3f}")

print("\n  *** ECOLOGICAL FALLACY WARNING ***")
print("  This analysis examines TRIAL-LEVEL associations between percentage")
print("  female and treatment effect. Even if significant, this does NOT")
print("  prove that individual women respond differently to treatment.")
print("  Trials with more women may differ in other characteristics:")
print("    - Geographic region and healthcare systems")
print("    - Mean age and comorbidity profiles")
print("    - Trial methodology and inclusion criteria")
print("  Only individual participant data (IPD) meta-analysis with a")
print("  treatment-by-sex interaction can properly test effect modification.")

# --- Part (d): Bubble plot ---
print(f"\n{'=' * 60}")
print("Part (d): Bubble Plot")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))

# Study weights for bubble sizes
w_plot = 1 / af_trials['var_log_rr']
sizes = (w_plot / w_plot.max()) * 500

# Scatter: bubbles
ax.scatter(af_trials['mean_age'], af_trials['log_rr'],
           s=sizes, alpha=0.6, c='steelblue', edgecolors='darkblue',
           linewidth=1, zorder=5)

# Add study labels
for _, row in af_trials.iterrows():
    ax.annotate(row['study'],
                (row['mean_age'], row['log_rr']),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color='grey30')

# Meta-regression line
age_range = np.linspace(af_trials['mean_age'].min() - 2,
                        af_trials['mean_age'].max() + 2, 100)
pred = wls_age.params.iloc[0] + wls_age.params.iloc[1] * age_range
ax.plot(age_range, pred, 'steelblue', linewidth=2,
        label=f'Meta-regression line (slope={wls_age.params.iloc[1]:.4f})')

# Confidence band (approximate)
X_pred = sm.add_constant(age_range)
pred_full = wls_age.get_prediction(X_pred)
pred_ci = pred_full.conf_int(alpha=0.05)
ax.fill_between(age_range, pred_ci[:, 0], pred_ci[:, 1],
                alpha=0.15, color='steelblue', label='95% CI')

# Reference line
ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8,
           label='Null (log RR = 0)')

ax.set_xlabel('Mean Age (years)', fontweight='bold', fontsize=12)
ax.set_ylabel('Log Risk Ratio', fontweight='bold', fontsize=12)
ax.set_title('Bubble Plot: Meta-Regression of Treatment Effect on Mean Age\n'
             f'(Slope p = {wls_age.pvalues.iloc[1]:.3f}; '
             f'bubble size = study weight)',
             fontweight='bold', fontsize=13)
ax.legend(loc='best', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig('bubble_plot_age.png', dpi=300)
plt.show()

print("\nBubble plot saved as 'bubble_plot_age.png'")
print("Larger bubbles represent more precise (higher-weight) studies.")
