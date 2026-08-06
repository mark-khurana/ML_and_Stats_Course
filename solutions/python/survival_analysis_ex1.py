# =============================================================================
# Chapter 6, Exercise 1: Kaplan-Meier Curves by Disease Stage
# PBC dataset (simulated): KM curves by stage, log-rank test, median survival
# =============================================================================

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import matplotlib.pyplot as plt

# --- Simulate PBC-like data with stage variable ---
# (Since PBC is an R-native dataset, we simulate its structure as done
#  in the chapter's Python code)
np.random.seed(42)
n = 312

# Simulate stage (1-4) with realistic proportions
stage = np.random.choice([1, 2, 3, 4], size=n, p=[0.05, 0.25, 0.40, 0.30])

# Simulate survival times that depend on stage (higher stage = worse survival)
# Base scale varies by stage
scale_by_stage = {1: 14, 2: 10, 3: 7, 4: 4}
time_years = np.array([np.random.exponential(scale=scale_by_stage[s]) for s in stage])
time_years = np.clip(time_years, 0.1, 15)  # Clip to realistic range

# Event probability increases with stage
event_prob = {1: 0.20, 2: 0.35, 3: 0.50, 4: 0.65}
event = np.array([np.random.binomial(1, event_prob[s]) for s in stage])

pbc = pd.DataFrame({
    'time_years': time_years,
    'event': event,
    'stage': stage
})

print(f"N = {len(pbc)}")
print(f"Events (deaths): {pbc['event'].sum()}")
print(f"Stages: {sorted(pbc['stage'].unique())}\n")

# --- Fit KM for each stage and report median survival ---
print("=== Median Survival Time by Stage ===")
kmf_dict = {}
for s in sorted(pbc['stage'].unique()):
    mask = pbc['stage'] == s
    kmf = KaplanMeierFitter()
    kmf.fit(pbc.loc[mask, 'time_years'],
            event_observed=pbc.loc[mask, 'event'],
            label=f'Stage {s}')
    kmf_dict[s] = kmf

    median_surv = kmf.median_survival_time_
    ci = kmf.confidence_interval_survival_function_
    print(f"  Stage {s}: Median survival = {median_surv:.2f} years "
          f"(n={mask.sum()}, events={pbc.loc[mask, 'event'].sum()})")

# --- Perform the log-rank test (overall comparison across all stages) ---
print("\n=== Log-Rank Test (overall) ===")
result = multivariate_logrank_test(
    pbc['time_years'], pbc['stage'], pbc['event']
)
print(f"Test statistic: {result.test_statistic:.3f}")
print(f"p-value: {result.p_value:.6f}")

if result.p_value < 0.05:
    print("Result: Significant difference in survival across disease stages.")
else:
    print("Result: No significant difference detected.")

# --- Pairwise log-rank tests (Stage 1 vs 4, etc.) ---
print("\n=== Pairwise Log-Rank Tests (selected) ===")
for s1, s2 in [(1, 4), (2, 3), (3, 4)]:
    mask1 = pbc['stage'] == s1
    mask2 = pbc['stage'] == s2
    if mask1.sum() > 0 and mask2.sum() > 0:
        lr = logrank_test(
            pbc.loc[mask1, 'time_years'], pbc.loc[mask2, 'time_years'],
            pbc.loc[mask1, 'event'], pbc.loc[mask2, 'event']
        )
        print(f"  Stage {s1} vs {s2}: p = {lr.p_value:.4f}")

# --- Plot KM curves for all stages ---
fig, ax = plt.subplots(figsize=(10, 6))

colors = {1: '#2ecc71', 2: '#3498db', 3: '#e67e22', 4: '#e74c3c'}
for s, kmf in kmf_dict.items():
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[s])

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Survival probability", fontsize=12)
ax.set_title("Kaplan-Meier Curves by Disease Stage (PBC)", fontsize=14)
ax.legend(fontsize=10)

# Add p-value annotation
ax.text(0.7, 0.95, f'Log-rank p = {result.p_value:.4f}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# --- Interpretation ---
print("\n=== Interpretation ===")
print("The KM curves show a clear separation by stage, with higher stages")
print("having worse survival. The log-rank test formally confirms whether")
print("these differences are statistically significant.")
print("Patients with Stage 4 disease have the worst prognosis, while")
print("Stage 1-2 patients have substantially better survival.")
