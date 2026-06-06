# =============================================================================
# Chapter 17 - Exercise 2: Propensity Score Matching in Python
# Beta-blocker use and 1-year mortality
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt

# --- Simulate the dataset (same DGP as the R exercise) ---
np.random.seed(123)
n = 1500

age = np.random.normal(70, 8, n)
creatinine = np.random.normal(1.2, 0.4, n)
heart_failure = np.random.binomial(1, 0.35, n)
prior_mi = np.random.binomial(1, 0.20, n)

# Confounded treatment assignment
ps_true = 1 / (1 + np.exp(-(-1 + 0.02*age + 0.3*heart_failure +
                               0.5*prior_mi - 0.8*creatinine)))
treatment = np.random.binomial(1, ps_true)

# Outcome
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

# --- Part (a): Estimate propensity scores ---
covariates = ['age', 'creatinine', 'heart_failure', 'prior_mi']
ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(df[covariates], df['treatment'])
df['ps'] = ps_model.predict_proba(df[covariates])[:, 1]

# Compute logit of PS for caliper matching
df['logit_ps'] = np.log(df['ps'] / (1 - df['ps']))

print("Propensity score summary:")
print(df['ps'].describe())

# Visualise PS overlap
fig, ax = plt.subplots(figsize=(8, 5))
df[df['treatment'] == 0]['ps'].plot.kde(ax=ax, label='Control', color='#0072B2')
df[df['treatment'] == 1]['ps'].plot.kde(ax=ax, label='Treated', color='#D55E00')
ax.set_xlabel('Propensity Score')
ax.set_ylabel('Density')
ax.set_title('Propensity Score Distribution by Treatment Group')
ax.legend()
plt.tight_layout()
plt.savefig('ps_overlap.png', dpi=300)
plt.show()

# --- Part (b): 1:1 nearest-neighbour matching with caliper ---
treated = df[df['treatment'] == 1].copy()
control = df[df['treatment'] == 0].copy()

# Caliper: 0.2 * SD of logit PS
caliper = 0.2 * df['logit_ps'].std()
print(f"\nCaliper (0.2 SD of logit PS): {caliper:.4f}")

# Nearest-neighbour matching on logit PS
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['logit_ps']].values)

distances, indices = nn.kneighbors(treated[['logit_ps']].values)

# Apply caliper: keep only matches within the caliper
matched_treated_idx = []
matched_control_idx = []
used_controls = set()

for i in range(len(treated)):
    dist = distances[i, 0]
    ctrl_idx = indices[i, 0]
    if dist <= caliper and ctrl_idx not in used_controls:
        matched_treated_idx.append(treated.index[i])
        matched_control_idx.append(control.index[ctrl_idx])
        used_controls.add(ctrl_idx)

# Create matched dataset
matched_df = pd.concat([
    df.loc[matched_treated_idx],
    df.loc[matched_control_idx]
])

print(f"Matched sample: {len(matched_df)} patients")
print(f"  Treated: {len(matched_treated_idx)}")
print(f"  Control: {len(matched_control_idx)}")
print(f"  Unmatched treated: {len(treated) - len(matched_treated_idx)}")

# --- Part (c): Assess balance using SMD (Love plot equivalent) ---
def compute_smd(data, var, group_col):
    """Compute standardised mean difference."""
    treated = data[data[group_col] == 1][var]
    control = data[data[group_col] == 0][var]
    pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_sd == 0:
        return 0
    return (treated.mean() - control.mean()) / pooled_sd

print("\nCovariate balance (SMD):")
print(f"{'Variable':<20} {'Before':>10} {'After':>10} {'Balanced?':>10}")
print("-" * 52)

smd_before = {}
smd_after = {}
for var in covariates:
    smd_b = compute_smd(df, var, 'treatment')
    smd_a = compute_smd(matched_df, var, 'treatment')
    smd_before[var] = smd_b
    smd_after[var] = smd_a
    balanced = "Yes" if abs(smd_a) < 0.1 else "No"
    print(f"{var:<20} {smd_b:>10.4f} {smd_a:>10.4f} {balanced:>10}")

# Love plot
fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(covariates))

ax.scatter([abs(smd_before[v]) for v in covariates], y_pos,
           color='#D55E00', label='Before matching', s=80, zorder=5)
ax.scatter([abs(smd_after[v]) for v in covariates], y_pos,
           color='#0072B2', label='After matching', s=80, zorder=5)

# Connect before and after
for i, var in enumerate(covariates):
    ax.plot([abs(smd_before[var]), abs(smd_after[var])], [i, i],
            'k-', alpha=0.3)

ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='SMD = 0.1 threshold')
ax.set_yticks(y_pos)
ax.set_yticklabels(covariates)
ax.set_xlabel('Absolute Standardised Mean Difference')
ax.set_title('Love Plot: Covariate Balance Before and After Matching')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('love_plot.png', dpi=300)
plt.show()

# --- Part (d): Estimate ATT ---
# Mortality rates in matched sample
mort_treated = matched_df[matched_df['treatment'] == 1]['death_1yr'].mean()
mort_control = matched_df[matched_df['treatment'] == 0]['death_1yr'].mean()

print(f"\nMortality in matched sample:")
print(f"  Treated (beta-blocker): {mort_treated:.4f}")
print(f"  Control (no beta-blocker): {mort_control:.4f}")
print(f"  Risk difference (ATT): {mort_treated - mort_control:.4f}")

# Odds ratio
from statsmodels.api import Logit, add_constant

X = add_constant(matched_df['treatment'])
logit_model = Logit(matched_df['death_1yr'], X).fit(disp=0)
print(f"\nLogistic regression in matched sample:")
print(logit_model.summary2().tables[1])

or_est = np.exp(logit_model.params['treatment'])
ci = np.exp(logit_model.conf_int().loc['treatment'])
print(f"\nOdds Ratio: {or_est:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")

# --- Part (e): E-value ---
# Convert protective OR to RR on above-null scale
rr_est = 1 / or_est  # Flip because protective
rr_ci_closest_null = 1 / ci[1]  # Upper CI of OR -> lower bound of 1/OR

def e_value(rr):
    """Compute E-value for a risk ratio."""
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))

e_val_point = e_value(rr_est)
e_val_ci = e_value(rr_ci_closest_null)

print(f"\n--- E-value Analysis ---")
print(f"E-value for point estimate: {e_val_point:.2f}")
print(f"E-value for CI bound closest to null: {e_val_ci:.2f}")

print(f"\nInterpretation:")
print(f"An unmeasured confounder would need to be associated with BOTH")
print(f"beta-blocker use AND 1-year mortality by a risk ratio of at least")
print(f"{e_val_point:.2f} to fully explain away the observed protective effect.")
print(f"For the confidence interval, a confounder with RR >= {e_val_ci:.2f}")
print(f"could shift the CI to include the null.")
