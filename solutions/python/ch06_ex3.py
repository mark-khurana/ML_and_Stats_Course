# =============================================================================
# Chapter 6, Exercise 3: Competing Risks Thinking
# Kidney transplant rejection with death as a competing risk
# =============================================================================
# This exercise is primarily conceptual (parts 1-2) with a bonus coding
# component (part 3).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Question 1 ---
# Explain why treating death as censoring would bias the results.
# What direction would the bias go?
#
# ANSWER:
# Treating death as simple censoring VIOLATES the assumption of
# NON-INFORMATIVE censoring. Non-informative censoring means that
# censored patients have the same future risk of the event as those
# who remain under observation. But patients who die are NOT like
# patients who remain alive -- they have ZERO future risk of rejection
# (because dead patients cannot experience rejection).
#
# The bias direction: treating death as censoring INFLATES (overestimates)
# the estimated probability of rejection. This happens because the
# Kaplan-Meier estimator assumes that censored patients would eventually
# experience the event at the same rate as those still at risk. But dead
# patients will NEVER experience rejection, so acting as if they could
# leads to an overestimate of the cumulative incidence of rejection.
#
# In the KM framework, when a patient dies and is "censored," they are
# removed from the risk set, effectively assuming they would have had
# the same rejection rate as surviving patients. Since they actually
# have zero rejection risk, this inflates the estimated rate.

# --- Question 2 ---
# Which approach for estimating probability of rejection within 2 years:
# cause-specific hazards or Fine-Gray?
#
# ANSWER:
# For estimating the PROBABILITY of rejection within 2 years, use the
# FINE-GRAY subdistribution hazard model. Here is why:
#
# - Cause-specific hazards model the RATE of rejection among those
#   currently alive and rejection-free. This answers: "Among patients
#   still alive, what is the instantaneous risk of rejection?" This is
#   useful for understanding ETIOLOGY (what causes rejection).
#
# - The Fine-Gray model directly models the CUMULATIVE INCIDENCE
#   FUNCTION, which gives the probability of rejection by time t,
#   accounting for the fact that some patients will die first. This
#   is the right quantity for PREDICTION and CLINICAL COMMUNICATION.
#
# When your goal is prediction ("What is the probability that this
# patient's transplant will be rejected within 2 years?"), you need
# the cumulative incidence, which properly accounts for the competing
# risk of death. The Fine-Gray model provides this directly.

# --- Question 3 (Bonus): Competing risks analysis in Python ---

from lifelines import KaplanMeierFitter, CoxPHFitter

# Simulate kidney transplant data with competing risks
np.random.seed(2025)
n = 500

# Covariates
age = np.random.normal(50, 12, size=n)
donor_type = np.random.binomial(1, 0.4, size=n)  # 0=living, 1=deceased
hla_mismatch = np.random.poisson(2, size=n)

# Simulate competing event times
# Time to rejection (cause 1)
lambda_reject = np.exp(-4 + 0.02 * age + 0.5 * donor_type + 0.2 * hla_mismatch)
time_reject = np.random.exponential(1 / lambda_reject)

# Time to death without rejection (cause 2)
lambda_death = np.exp(-5 + 0.03 * age + 0.3 * donor_type)
time_death = np.random.exponential(1 / lambda_death)

# Administrative censoring at 5-10 years
time_censor = np.random.uniform(5, 10, size=n)

# Determine observed time and event type
time_obs = np.minimum(np.minimum(time_reject, time_death), time_censor)
event = np.where(time_obs == time_censor, 0,
                 np.where(time_obs == time_reject, 1, 2))
# 0 = censored, 1 = rejection, 2 = death

df = pd.DataFrame({
    'time': time_obs,
    'event': event,
    'age': age,
    'donor_type': donor_type,
    'hla_mismatch': hla_mismatch
})

print("=== Dataset summary ===")
print(f"N: {n}")
print(f"Rejections (event=1): {(event == 1).sum()}")
print(f"Deaths (event=2): {(event == 2).sum()}")
print(f"Censored (event=0): {(event == 0).sum()}\n")

# --- Approach 1: Naive KM (treating death as censoring) ---
# This is WRONG but illustrative
kmf_naive = KaplanMeierFitter()
event_naive = (event == 1).astype(int)  # only rejection = event; death = censored
kmf_naive.fit(time_obs, event_observed=event_naive, label="Naive KM (death = censored)")

# --- Approach 2: Cause-specific Cox model ---
# For rejection: censor deaths and administrative censoring
print("=== Cause-Specific Cox Model (rejection) ===")
df_cs = df.copy()
df_cs['event_rejection'] = (df_cs['event'] == 1).astype(int)

cph_cs = CoxPHFitter()
cph_cs.fit(df_cs, duration_col='time', event_col='event_rejection',
           formula='age + donor_type + hla_mismatch')
cph_cs.print_summary()

# --- Approach 3: Cumulative Incidence Function (Aalen-Johansen estimator) ---
# Compute the CIF manually using the Aalen-Johansen approach
# This properly accounts for competing risks

def compute_cif(times, events, cause=1):
    """
    Compute the cumulative incidence function for a specific cause
    using the Aalen-Johansen estimator.
    """
    # Sort by time
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]

    unique_times = np.unique(t_sorted[e_sorted > 0])
    n_risk = len(times)

    cif = []
    surv = 1.0  # Overall survival (from all causes)
    cif_times = []

    for t in unique_times:
        at_risk = np.sum(t_sorted >= t)
        d_cause = np.sum((t_sorted == t) & (e_sorted == cause))
        d_all = np.sum((t_sorted == t) & (e_sorted > 0))

        if at_risk > 0:
            # CIF increment: S(t-) * d_cause/n_risk
            cif_increment = surv * (d_cause / at_risk)
            # Update overall survival
            surv *= (1 - d_all / at_risk)

            cif.append(cif_increment)
            cif_times.append(t)

    # Cumulative sum
    cif_cumulative = np.cumsum(cif)
    return np.array(cif_times), cif_cumulative

# Compute CIF for rejection (cause 1)
cif_times, cif_values = compute_cif(time_obs, event, cause=1)

# Compute CIF for death (cause 2)
cif_times_d, cif_values_d = compute_cif(time_obs, event, cause=2)

# --- Plot: Naive KM vs CIF ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Naive KM (1 - S(t)) vs CIF for rejection
ax1 = axes[0]
# Naive 1 - KM
naive_times = kmf_naive.survival_function_.index
naive_1_minus_s = 1 - kmf_naive.survival_function_.values.flatten()
ax1.step(naive_times, naive_1_minus_s, color='red', linewidth=2,
         label='Naive 1-KM (BIASED)', where='post')
# Proper CIF
ax1.step(cif_times, cif_values, color='blue', linewidth=2,
         label='Cumulative Incidence (correct)', where='post')
ax1.set_xlabel("Time (years)", fontsize=12)
ax1.set_ylabel("Probability of Rejection", fontsize=12)
ax1.set_title("Naive KM vs Cumulative Incidence", fontsize=13)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 10)

# Right panel: Stacked CIF (rejection + death)
ax2 = axes[1]
# Interpolate to common time grid
from scipy import interpolate

t_grid = np.linspace(0.01, min(cif_times.max(), cif_times_d.max()), 200)
f_reject = interpolate.interp1d(cif_times, cif_values, kind='previous',
                                 bounds_error=False, fill_value=(0, cif_values[-1]))
f_death = interpolate.interp1d(cif_times_d, cif_values_d, kind='previous',
                                bounds_error=False, fill_value=(0, cif_values_d[-1]))

cif_r = f_reject(t_grid)
cif_d = f_death(t_grid)

ax2.fill_between(t_grid, 0, cif_r, alpha=0.4, color='#e74c3c', label='Rejection')
ax2.fill_between(t_grid, cif_r, cif_r + cif_d, alpha=0.4,
                 color='#3498db', label='Death (competing)')
ax2.plot(t_grid, cif_r + cif_d, color='black', linewidth=1)
ax2.set_xlabel("Time (years)", fontsize=12)
ax2.set_ylabel("Cumulative Incidence", fontsize=12)
ax2.set_title("Stacked Cumulative Incidence Functions", fontsize=13)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# --- 2-year cumulative incidence of rejection ---
# Find closest time to 2 years
idx_2yr = np.argmin(np.abs(cif_times - 2))
ci_2yr = cif_values[idx_2yr]
print(f"\n=== 2-Year Cumulative Incidence of Rejection ===")
print(f"Proper CIF estimate: {ci_2yr:.3f} ({ci_2yr*100:.1f}%)")

# Compare with naive KM
naive_2yr = 1 - kmf_naive.predict(2)
print(f"Naive 1-KM estimate: {naive_2yr:.3f} ({naive_2yr*100:.1f}%)")
print(f"\nThe naive estimate is HIGHER (biased upward) because it treats")
print(f"deaths as censoring, overestimating the probability of rejection.")

# --- Summary ---
print("\n=== Summary ===")
print("1. Treating death as censoring overestimates rejection probability.")
print("2. For prediction, use the Fine-Gray model / cumulative incidence.")
print("3. For etiology, use cause-specific Cox models.")
print("4. Always report cumulative incidence functions, not 1-KM,")
print("   when competing risks are present.")
