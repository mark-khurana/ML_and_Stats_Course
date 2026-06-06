# =============================================================================
# Chapter 17 - Exercise 3: IPTW in Python
# Beta-blocker use and 1-year mortality using IPTW
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- Simulate the dataset (same DGP as Exercise 2) ---
np.random.seed(123)
n = 1500

age = np.random.normal(70, 8, n)
creatinine = np.random.normal(1.2, 0.4, n)
heart_failure = np.random.binomial(1, 0.35, n)
prior_mi = np.random.binomial(1, 0.20, n)

ps_true = 1 / (1 + np.exp(-(-1 + 0.02*age + 0.3*heart_failure +
                               0.5*prior_mi - 0.8*creatinine)))
treatment = np.random.binomial(1, ps_true)

mort_prob = 1 / (1 + np.exp(-(-2 + 0.05*age + 0.4*heart_failure +
                                 0.6*prior_mi + 1.0*creatinine -
                                 0.7*treatment)))
death_1yr = np.random.binomial(1, mort_prob)

df = pd.DataFrame({
    'age': age, 'creatinine': creatinine, 'heart_failure': heart_failure,
    'prior_mi': prior_mi, 'treatment': treatment, 'death_1yr': death_1yr
})

print(f"Dataset: {len(df)} patients")
print(f"Treatment prevalence: {df['treatment'].mean():.3f}")
print(f"Outcome prevalence: {df['death_1yr'].mean():.3f}\n")

# --- Part (a): Fit PS model and compute stabilised IPTW weights ---
covariates = ['age', 'creatinine', 'heart_failure', 'prior_mi']

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(df[covariates], df['treatment'])
df['ps'] = ps_model.predict_proba(df[covariates])[:, 1]

# Unstabilised ATE weights: w = A/ps + (1-A)/(1-ps)
df['iptw'] = np.where(
    df['treatment'] == 1,
    1 / df['ps'],
    1 / (1 - df['ps'])
)

# Stabilised weights: replace numerator with marginal treatment probability
p_treat = df['treatment'].mean()
df['sw'] = np.where(
    df['treatment'] == 1,
    p_treat / df['ps'],
    (1 - p_treat) / (1 - df['ps'])
)

print("Stabilised weight summary:")
print(f"  Min: {df['sw'].min():.4f}")
print(f"  Max: {df['sw'].max():.4f}")
print(f"  Mean: {df['sw'].mean():.4f}")
print(f"  SD: {df['sw'].std():.4f}")

# Check for extreme weights
extreme = (df['sw'] > 10).sum()
print(f"  Extreme weights (>10): {extreme}")

# --- Part (b): Assess covariate balance using weighted SMDs ---
def weighted_smd(data, var, treatment_col, weight_col):
    """Compute weighted standardised mean difference."""
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    mean_t = np.average(treated[var], weights=treated[weight_col])
    mean_c = np.average(control[var], weights=control[weight_col])
    # Use unweighted pooled SD for standardisation
    sd_pooled = np.sqrt((treated[var].var() + control[var].var()) / 2)
    if sd_pooled == 0:
        return 0
    return (mean_t - mean_c) / sd_pooled

def unweighted_smd(data, var, treatment_col):
    """Compute unweighted SMD."""
    treated = data[data[treatment_col] == 1][var]
    control = data[data[treatment_col] == 0][var]
    sd_pooled = np.sqrt((treated.var() + control.var()) / 2)
    if sd_pooled == 0:
        return 0
    return (treated.mean() - control.mean()) / sd_pooled

print("\nCovariate balance (weighted SMD):")
print(f"{'Variable':<20} {'Unweighted':>12} {'Weighted':>12} {'Balanced?':>10}")
print("-" * 56)

smd_before = {}
smd_after = {}
for var in covariates:
    smd_b = unweighted_smd(df, var, 'treatment')
    smd_a = weighted_smd(df, var, 'treatment', 'sw')
    smd_before[var] = smd_b
    smd_after[var] = smd_a
    balanced = "Yes" if abs(smd_a) < 0.1 else "No"
    print(f"{var:<20} {smd_b:>12.4f} {smd_a:>12.4f} {balanced:>10}")

# Love plot equivalent
fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(covariates))

ax.scatter([abs(smd_before[v]) for v in covariates], y_pos,
           color='#D55E00', label='Before IPTW', s=80, zorder=5)
ax.scatter([abs(smd_after[v]) for v in covariates], y_pos,
           color='#0072B2', label='After IPTW', s=80, zorder=5)

for i, var in enumerate(covariates):
    ax.plot([abs(smd_before[var]), abs(smd_after[var])], [i, i],
            'k-', alpha=0.3)

ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='SMD = 0.1')
ax.set_yticks(y_pos)
ax.set_yticklabels(covariates)
ax.set_xlabel('Absolute Standardised Mean Difference')
ax.set_title('Covariate Balance: Before and After IPTW')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('iptw_balance.png', dpi=300)
plt.show()

# --- Part (c): Estimate ATE using weighted regression ---
X = sm.add_constant(df['treatment'])

# Weighted least squares (linear probability model for risk difference)
wls_model = sm.WLS(df['death_1yr'], X, weights=df['sw']).fit()
print("\nIPTW ATE Estimate (Risk Difference):")
print(f"  Coefficient: {wls_model.params.iloc[1]:.4f}")
print(f"  95% CI: ({wls_model.conf_int().iloc[1, 0]:.4f}, "
      f"{wls_model.conf_int().iloc[1, 1]:.4f})")
print(f"  p-value: {wls_model.pvalues.iloc[1]:.4f}")

# Weighted logistic regression (for odds ratio)
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial

glm_model = GLM(df['death_1yr'], X,
                family=Binomial(),
                freq_weights=df['sw']).fit()
or_est = np.exp(glm_model.params.iloc[1])
ci = np.exp(glm_model.conf_int().iloc[1])
print(f"\nIPTW ATE Estimate (Odds Ratio):")
print(f"  OR: {or_est:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")

# --- Part (d): Sensitivity analysis with unmeasured confounder ---
print("\n--- Sensitivity Analysis: Unmeasured Confounder ---")
print("Varying confounder strength to show impact on ATE estimate:\n")

# We simulate data with an unmeasured confounder U of varying strength
results = []
gamma_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

for gamma in gamma_values:
    np.random.seed(42)  # Same seed for comparability
    U = np.random.normal(0, 1, n)

    # Re-simulate with confounder U
    ps_u = 1 / (1 + np.exp(-(-1 + 0.02*age + 0.3*heart_failure +
                                0.5*prior_mi - 0.8*creatinine + gamma*U)))
    trt_u = np.random.binomial(1, ps_u)
    mort_u = 1 / (1 + np.exp(-(-2 + 0.05*age + 0.4*heart_failure +
                                  0.6*prior_mi + 1.0*creatinine -
                                  0.7*trt_u + gamma*U)))
    death_u = np.random.binomial(1, mort_u)

    df_u = pd.DataFrame({
        'age': age, 'creatinine': creatinine, 'heart_failure': heart_failure,
        'prior_mi': prior_mi, 'treatment': trt_u, 'death_1yr': death_u
    })

    # IPTW without U
    ps_mod = LogisticRegression(max_iter=1000)
    ps_mod.fit(df_u[covariates], df_u['treatment'])
    ps_est = ps_mod.predict_proba(df_u[covariates])[:, 1]

    p_t = df_u['treatment'].mean()
    sw = np.where(df_u['treatment'] == 1, p_t / ps_est, (1 - p_t) / (1 - ps_est))

    X_u = sm.add_constant(df_u['treatment'])
    wls_u = sm.WLS(df_u['death_1yr'], X_u, weights=sw).fit()

    results.append({
        'gamma': gamma,
        'ate': wls_u.params.iloc[1],
        'ci_lo': wls_u.conf_int().iloc[1, 0],
        'ci_hi': wls_u.conf_int().iloc[1, 1]
    })

results_df = pd.DataFrame(results)
print(f"{'Gamma (U strength)':<20} {'ATE':>10} {'95% CI':>25}")
print("-" * 57)
for _, row in results_df.iterrows():
    print(f"{row['gamma']:<20.1f} {row['ate']:>10.4f} "
          f"({row['ci_lo']:.4f}, {row['ci_hi']:.4f})")

# Plot sensitivity analysis
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(results_df['gamma'], results_df['ate'],
            yerr=[results_df['ate'] - results_df['ci_lo'],
                  results_df['ci_hi'] - results_df['ate']],
            fmt='o-', color='#0072B2', capsize=5, markersize=8)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Null effect')
ax.set_xlabel('Unmeasured Confounder Strength (gamma)')
ax.set_ylabel('ATE Estimate (Risk Difference)')
ax.set_title('Sensitivity Analysis: Impact of Unmeasured Confounding')
ax.legend()
plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=300)
plt.show()

print("\n--- Interpretation ---")
print("As the strength of the unmeasured confounder increases (gamma),")
print("the IPTW estimate of the treatment effect becomes more biased")
print("because IPTW cannot account for unmeasured confounding.")
print("When gamma = 0 (no unmeasured confounder), the estimate is closest")
print("to the true effect. This demonstrates why sensitivity analysis")
print("is essential for observational causal inference.")
