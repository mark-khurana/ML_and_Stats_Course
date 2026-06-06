# =============================================================================
# Chapter 6, Exercise 2: Build and Evaluate a Cox Model
# PBC-like dataset: Cox PH with age, log(bili), albumin, protime, edema
# =============================================================================

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# --- Simulate PBC-like clinical dataset ---
# (Following the chapter's Python approach with simulated data)
np.random.seed(42)
n = 276

pbc = pd.DataFrame({
    'time_years': np.abs(np.random.exponential(7, n) +
                          np.random.normal(0, 1, n)),
    'event': np.random.binomial(1, 0.45, n),
    'age': np.random.normal(50, 10, n),
    'log_bili': np.random.normal(0.5, 1.0, n),     # log(bilirubin)
    'albumin': np.random.normal(3.5, 0.4, n),
    'protime': np.random.normal(11, 1, n),
    'edema': np.random.choice([0, 0.5, 1], n, p=[0.75, 0.15, 0.10])
})

# Make survival time depend on covariates to create realistic associations
hazard_linear = (0.04 * pbc['age'] + 0.3 * pbc['log_bili'] -
                 0.5 * pbc['albumin'] + 0.2 * pbc['protime'] +
                 0.8 * pbc['edema'])
# Adjust times based on hazard
pbc['time_years'] = pbc['time_years'] * np.exp(-0.1 * hazard_linear)
pbc['time_years'] = np.clip(pbc['time_years'], 0.05, 15)

print(f"N = {len(pbc)}")
print(f"Events: {pbc['event'].sum()}")
print(f"Median follow-up: {pbc['time_years'].median():.2f} years\n")

# --- Task 1: Fit Cox model and report hazard ratios ---
cph = CoxPHFitter()
cph.fit(pbc, duration_col='time_years', event_col='event',
        formula='age + log_bili + albumin + protime + edema')

print("=== Task 1: Cox Model Summary ===")
cph.print_summary()

# Extract HR table
print("\n=== Hazard Ratios and 95% CIs ===")
hr_table = pd.DataFrame({
    'HR': np.exp(cph.params_),
    'Lower_95': np.exp(cph.confidence_intervals_.iloc[:, 0]),
    'Upper_95': np.exp(cph.confidence_intervals_.iloc[:, 1]),
    'p_value': cph.summary['p']
})
print(hr_table.round(4))

# Forest plot
fig, ax = plt.subplots(figsize=(8, 5))
cph.plot(ax=ax)
ax.set_title("Cox PH Model - Hazard Ratios")
plt.tight_layout()
plt.show()

# --- Task 2: Check proportional hazards assumption ---
print("\n=== Task 2: Proportional Hazards Assumption ===")
try:
    cph.check_assumptions(pbc, p_value_threshold=0.05, show_plots=True)
except Exception as e:
    print(f"PH assumption check result: {e}")
    print("If no warnings are raised, the PH assumption is not violated.")

# Interpretation
print("\nInterpretation:")
print("- A significant p-value (< 0.05) indicates violation of PH.")
print("- Look at the Schoenfeld residual plots for trends over time.")
print("- If a covariate violates PH, consider stratifying on it or")
print("  adding a time interaction.")

# --- Task 3: Concordance index ---
print(f"\n=== Task 3: Concordance Index ===")
c_index = cph.concordance_index_
print(f"C-index: {c_index:.3f}")
print(f"\nInterpretation:")
print(f"- C-index = 0.5: no discriminative ability (random)")
print(f"- C-index = 1.0: perfect discrimination")
print(f"- C-index > 0.7: generally considered acceptable")
print(f"- C-index > 0.8: considered strong discrimination")
quality = "strong" if c_index > 0.8 else ("acceptable" if c_index > 0.7 else "modest")
print(f"This model's C-index of {c_index:.3f} indicates {quality} discrimination.")

# --- Task 4: 5-year survival for a specific patient ---
print(f"\n=== Task 4: 5-Year Survival Prediction ===")
print("Patient profile:")
print("  Age: 55 years")
print(f"  Bilirubin: 3 mg/dL (log = {np.log(3):.2f})")
print("  Albumin: 3.2 g/dL")
print("  Prothrombin time: 11 seconds")
print("  Edema: none (0)")

new_patient = pd.DataFrame({
    'age': [55],
    'log_bili': [np.log(3)],    # bilirubin = 3 mg/dL
    'albumin': [3.2],
    'protime': [11],
    'edema': [0]
})

# Get predicted survival function
surv_func = cph.predict_survival_function(new_patient)

# Find 5-year survival
# Get the closest time point to 5 years
times = surv_func.index
closest_5yr = times[np.argmin(np.abs(times - 5))]
surv_5yr = surv_func.loc[closest_5yr].values[0]

print(f"\nPredicted 5-year survival probability: {surv_5yr:.3f} ({surv_5yr*100:.1f}%)")

# Plot predicted survival curve
fig, ax = plt.subplots(figsize=(8, 5))
surv_func.plot(ax=ax, color='blue', linewidth=2, label='Predicted survival')
ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.7, label='50% survival')
ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='5-year mark')
ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Survival probability", fontsize=12)
ax.set_title("Predicted Survival for Patient Profile", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
