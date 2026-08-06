# =============================================================================
# Chapter 17 - Exercise 2: Propensity score matching
# Beta-blocker use and 1-year mortality
# =============================================================================

# Libraries -------------------------------------------------------------------
# pip install numpy pandas scikit-learn statsmodels matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expit(x):
    return 1 / (1 + np.exp(-x))


# --- The dataset from the exercise ------------------------------------------
# Note that all covariates are CENTRED in the linear predictors, so the
# intercepts set the baseline rate for an average patient rather than for a
# 0-year-old. Without that, this cohort has a 92% one-year mortality.
rng = np.random.default_rng(123)
n = 1500

age = rng.normal(70, 8, n)
creatinine = rng.normal(1.2, 0.4, n)
heart_failure = rng.binomial(1, 0.35, n)
prior_mi = rng.binomial(1, 0.20, n)

treatment = rng.binomial(1, expit(-0.4 + 0.05 * (age - 70)
                                 + 0.7 * heart_failure
                                 + 0.9 * prior_mi
                                 + 0.8 * (creatinine - 1.2)))
lp_untreated = (-1.9 + 0.05 * (age - 70) + 0.7 * heart_failure
                + 0.8 * prior_mi + 1.0 * (creatinine - 1.2))
death_1yr = rng.binomial(1, expit(lp_untreated - 0.8 * treatment))

df = pd.DataFrame(dict(age=age, creatinine=creatinine,
                       heart_failure=heart_failure, prior_mi=prior_mi,
                       treatment=treatment, death_1yr=death_1yr))

# The truth, available only because we simulated the data.
treated_mask = df.treatment == 1
p1 = expit(lp_untreated[treated_mask] - 0.8).mean()
p0 = expit(lp_untreated[treated_mask]).mean()
TRUE_ATT_RD = p1 - p0
TRUE_ATT_OR = (p1 / (1 - p1)) / (p0 / (1 - p0))

print(f"Cohort: {n} patients | {100 * df.treatment.mean():.0f}% treated "
      f"| {100 * df.death_1yr.mean():.1f}% died within 1 year")
print(f"TRUE ATT: risk difference {TRUE_ATT_RD:+.4f}, "
      f"odds ratio {TRUE_ATT_OR:.3f}\n")

covs = ["age", "creatinine", "heart_failure", "prior_mi"]

# =============================================================================
# (a) Estimate the propensity score and look at OVERLAP
# =============================================================================
ps_model = smf.logit("treatment ~ age + creatinine + heart_failure + prior_mi",
                     data=df).fit(disp=0)
df["ps"] = ps_model.predict(df)
df["logit_ps"] = np.log(df.ps / (1 - df.ps))

print("--- (a) Propensity score distribution ---")
print(df.groupby("treatment")["ps"].agg(["count", "min", "median", "max"]).round(3))

fig, ax = plt.subplots(figsize=(8, 4.5))
for value, label, colour in [(0, "No beta-blocker", "#0072B2"),
                             (1, "Beta-blocker", "#D55E00")]:
    df.loc[df.treatment == value, "ps"].plot.density(
        ax=ax, alpha=0.6, label=label, color=colour)
ax.set_xlabel("Propensity score")
ax.set_ylabel("Density")
ax.set_title("Propensity score overlap by treatment group")
ax.legend()
plt.tight_layout()
plt.show()

print("\nBoth groups span roughly the same range of scores, with no pile-up at")
print("0 or 1, so positivity is not obviously violated and matching is feasible.")

# =============================================================================
# (b) 1:1 nearest-neighbour matching with a caliper of 0.2 SD
# =============================================================================
# Greedy 1:1 nearest-neighbour matching without replacement, with a caliper of
# 0.2 SD of the propensity score (the same convention MatchIt uses in R).
# Treated patients are processed in DESCENDING propensity-score order, so the
# hardest-to-match patients get first pick of the controls.
caliper = 0.2 * df.ps.std()

treated_idx = df.index[df.treatment == 1].to_numpy()
treated_idx = treated_idx[np.argsort(-df.loc[treated_idx, "ps"].to_numpy())]
control_idx = df.index[df.treatment == 0].to_numpy()

nn = NearestNeighbors(n_neighbors=len(control_idx)).fit(
    df.loc[control_idx, ["ps"]].to_numpy())
dist, order = nn.kneighbors(df.loc[treated_idx, ["ps"]].to_numpy())

used = set()
pairs = []
for row, t_i in enumerate(treated_idx):
    for d, c_pos in zip(dist[row], order[row]):
        if d > caliper:
            break                       # nothing else is close enough either
        c_i = control_idx[c_pos]
        if c_i not in used:
            used.add(c_i)
            pairs.append((t_i, c_i))
            break

matched_idx = [i for pair in pairs for i in pair]
m_data = df.loc[matched_idx].copy()

n_treated = len(treated_idx)
n_matched = len(pairs)
print(f"\n--- (b) Matching ---")
print(f"{n_matched} of {n_treated} treated patients found a partner; "
      f"{n_treated - n_matched} did NOT.")
print("""Those unmatched patients are silently DROPPED. They are not a random
subset -- they are the ones with the most extreme propensity scores, i.e. the
patients who were most obviously going to be treated. So the estimate no longer
describes 'all treated patients'; it describes the treated patients for whom a
comparable untreated patient exists. Always report how many were dropped, and
compare their characteristics.""")

# =============================================================================
# (c) Balance before and after (the Love plot)
# =============================================================================
def smd(data, var):
    t = data.loc[data.treatment == 1, var]
    c = data.loc[data.treatment == 0, var]
    pooled_sd = np.sqrt((t.var() + c.var()) / 2)
    return (t.mean() - c.mean()) / pooled_sd


balance = pd.DataFrame({
    "before": [smd(df, v) for v in covs],
    "after": [smd(m_data, v) for v in covs],
}, index=covs)

print("\n--- (c) Standardised mean differences ---")
print(balance.round(3))

fig, ax = plt.subplots(figsize=(7, 3.6))
y = np.arange(len(covs))
ax.scatter(balance["before"].abs(), y, label="Before matching",
           color="#D55E00", s=60)
ax.scatter(balance["after"].abs(), y, label="After matching",
           color="#0072B2", s=60)
ax.axvline(0.1, linestyle="--", color="grey")
ax.set_yticks(y)
ax.set_yticklabels(covs)
ax.set_xlabel("|Standardised mean difference|")
ax.set_title("Covariate balance: before and after matching")
ax.legend()
plt.tight_layout()
plt.show()

worst = balance["after"].abs().idxmax()
worst_value = balance.loc[worst, "after"]
print(f"\nThe worst-balanced covariate after matching is {worst} at "
      f"{abs(worst_value):.3f}")
if abs(worst_value) > 0.1:
    print("-- marginally OUTSIDE the conventional 0.1 threshold. That is a real")
    print("(if mild) failure, and greedy 1:1 matching often ends up here when a")
    print("lot of treated patients go unmatched. The fixes, in order: allow more")
    print("controls per treated patient, match on the Mahalanobis distance rather")
    print("than the propensity score alone, or abandon matching for weighting")
    print("(Exercise 3), which uses every patient and usually balances better.")
else:
    print("-- inside the conventional 0.1 threshold, so the matched groups have a")
    print("comparable mix of patients.")

# =============================================================================
# (d) The ATT, as an odds ratio and as a risk difference
# =============================================================================
or_fit = smf.glm("death_1yr ~ treatment", data=m_data,
                 family=sm.families.Binomial()).fit()
log_or = or_fit.params["treatment"]
or_ci = or_fit.conf_int().loc["treatment"]

rd_fit = smf.ols("death_1yr ~ treatment", data=m_data).fit()
rd = rd_fit.params["treatment"]
rd_ci = rd_fit.conf_int().loc["treatment"]

unadjusted_rd = (df.loc[df.treatment == 1, "death_1yr"].mean()
                 - df.loc[df.treatment == 0, "death_1yr"].mean())

# The estimate describes the MATCHED treated patients, not all treated patients,
# so recompute the truth over exactly that subgroup for a fair comparison.
matched_treated = m_data.index[m_data.treatment == 1]
lp_m = lp_untreated[df.index.get_indexer(matched_treated)]
p1_m = expit(lp_m - 0.8).mean()
p0_m = expit(lp_m).mean()
TRUE_MATCHED_RD = p1_m - p0_m
TRUE_MATCHED_OR = (p1_m / (1 - p1_m)) / (p0_m / (1 - p0_m))

print("\n--- (d) ATT estimates in the matched sample ---")
print(f"Odds ratio      : {np.exp(log_or):.3f} "
      f"(95% CI {np.exp(or_ci[0]):.3f}, {np.exp(or_ci[1]):.3f})"
      f"   [truth {TRUE_ATT_OR:.3f}]")
print(f"Risk difference : {rd:+.4f} "
      f"(95% CI {rd_ci[0]:+.4f}, {rd_ci[1]:+.4f})"
      f"   [truth {TRUE_ATT_RD:+.4f}]")
print(f"\nUnadjusted risk difference in the full cohort: {unadjusted_rd:+.4f}")
print("-- a fraction of the true effect. Matching recovers most of what the")
print("naive comparison hides.")

print(f"""
But compare against the right target. The truth quoted above is the ATT over
ALL treated patients. Our estimate only describes the {len(pairs)} treated patients
who found a match, and for THAT subgroup the true values are:
  risk difference {TRUE_MATCHED_RD:+.4f}   odds ratio {TRUE_MATCHED_OR:.3f}
which is what the estimate should be judged against. The gap between the two
targets is the price of discarding unmatched patients: matching answers a
slightly different question from the one you asked.""")

print("\nMortality in the matched sample:")
print(f"  Treated: {m_data.loc[m_data.treatment == 1, 'death_1yr'].mean():.3f}"
      f"    Control: {m_data.loc[m_data.treatment == 0, 'death_1yr'].mean():.3f}")

# =============================================================================
# (e) E-value: how strong would a hidden confounder have to be?
# =============================================================================
# The E-value works on the risk-ratio scale and is symmetric about the null, so
# a protective estimate is first flipped to the above-null side.
def e_value(rr):
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


# With a ~15% outcome the odds ratio overstates the risk ratio, so compute the
# risk ratio directly rather than reusing the OR.
risk_t = m_data.loc[m_data.treatment == 1, "death_1yr"].mean()
risk_c = m_data.loc[m_data.treatment == 0, "death_1yr"].mean()
rr = risk_t / risk_c

a = m_data.loc[m_data.treatment == 1, "death_1yr"].sum()
b = (m_data.treatment == 1).sum()
c = m_data.loc[m_data.treatment == 0, "death_1yr"].sum()
d = (m_data.treatment == 0).sum()
se_log_rr = np.sqrt(1 / a - 1 / b + 1 / c - 1 / d)
rr_lo = np.exp(np.log(rr) - 1.96 * se_log_rr)
rr_hi = np.exp(np.log(rr) + 1.96 * se_log_rr)

e_point = e_value(rr)
e_ci = 1.0 if rr_hi >= 1 else e_value(rr_hi)

print("\n--- (e) E-value ---")
print(f"Risk ratio: {rr:.3f} (95% CI {rr_lo:.3f}, {rr_hi:.3f})")
print(f"E-value for the point estimate       : {e_point:.2f}")
print(f"E-value for the CI bound nearest null: {e_ci:.2f}")
print(f"""
Interpretation in one sentence: an unmeasured confounder would have to be
associated with BOTH beta-blocker use and death by a risk ratio of at least
{e_point:.2f} -- over and above age, creatinine, heart failure and prior MI -- to
explain away this result entirely.

Whether that is plausible is a clinical judgement, not a statistical one.
Compare it with the strength of the confounders you DID measure: if none of them
reaches that magnitude, a hidden one probably does not either.""")
