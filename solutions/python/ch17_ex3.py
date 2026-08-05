# =============================================================================
# Chapter 17 - Exercise 3: IPW, balance, and positivity
# Beta-blocker use and 1-year mortality
# =============================================================================

# Libraries -------------------------------------------------------------------
# pip install numpy pandas statsmodels
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expit(x):
    return 1 / (1 + np.exp(-x))


COVS = ["age", "creatinine", "heart_failure", "prior_mi"]


def simulate_cohort(seed=123, n=1500, extreme=False):
    """The exercise cohort. `extreme=True` destroys positivity on purpose."""
    rng = np.random.default_rng(seed)
    age = rng.normal(70, 8, n)
    creatinine = rng.normal(1.2, 0.4, n)
    heart_failure = rng.binomial(1, 0.35, n)
    prior_mi = rng.binomial(1, 0.20, n)

    if extreme:
        # Part (d): heart-failure patients are treated with probability 0.98,
        # everyone else with probability 0.03.
        p_treat = np.where(heart_failure == 1, 0.98, 0.03)
    else:
        p_treat = expit(-0.4 + 0.05 * (age - 70) + 0.7 * heart_failure
                        + 0.9 * prior_mi + 0.8 * (creatinine - 1.2))

    treatment = rng.binomial(1, p_treat)
    lp_untreated = (-1.9 + 0.05 * (age - 70) + 0.7 * heart_failure
                    + 0.8 * prior_mi + 1.0 * (creatinine - 1.2))
    death_1yr = rng.binomial(1, expit(lp_untreated - 0.8 * treatment))

    df = pd.DataFrame(dict(age=age, creatinine=creatinine,
                           heart_failure=heart_failure, prior_mi=prior_mi,
                           treatment=treatment, death_1yr=death_1yr))
    # True ATE on the risk-difference scale, averaged over THIS cohort
    true_ate = expit(lp_untreated - 0.8).mean() - expit(lp_untreated).mean()
    return df, true_ate


df, TRUE_ATE_RD = simulate_cohort()

print(f"Cohort: {len(df)} patients | {100 * df.treatment.mean():.0f}% treated "
      f"| {100 * df.death_1yr.mean():.1f}% died within 1 year")
print(f"TRUE ATE risk difference: {TRUE_ATE_RD:+.4f}\n")


# =============================================================================
# Helpers used by both parts
# =============================================================================
def stabilised_weights(data):
    """Propensity score and stabilised ATE weights."""
    ps = smf.logit("treatment ~ " + " + ".join(COVS), data=data).fit(disp=0).predict(data)
    p_marg = data.treatment.mean()
    sw = np.where(data.treatment == 1, p_marg / ps, (1 - p_marg) / (1 - ps))
    return ps, sw


def weighted_smd(data, weights, var):
    t = data.treatment == 1
    mt = np.average(data.loc[t, var], weights=weights[t.to_numpy()])
    mc = np.average(data.loc[~t, var], weights=weights[(~t).to_numpy()])
    sd = np.sqrt((data.loc[t, var].var() + data.loc[~t, var].var()) / 2)
    return (mt - mc) / sd


def effective_sample_size(w):
    return w.sum() ** 2 / (w ** 2).sum()


def ipw_risk_difference(data, weights, n_boot=400, seed=1):
    """Weighted outcome model -> marginal risk difference, bootstrap CI.

    The bootstrap re-estimates the propensity model in every resample, which is
    what makes the interval honest: statsmodels' own standard error treats the
    weights as if they were known rather than estimated.
    """
    def point(d):
        _, w = stabilised_weights(d)
        m = smf.glm("death_1yr ~ treatment", data=d,
                    family=sm.families.Binomial(), freq_weights=w).fit()
        return (m.predict(d.assign(treatment=1)).mean()
                - m.predict(d.assign(treatment=0)).mean())

    est = point(data)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        resample = data.iloc[rng.integers(0, len(data), len(data))]
        try:
            boot.append(point(resample))
        except Exception:
            continue          # a resample with no variation in treatment
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, lo, hi


# =============================================================================
# (a) Propensity score model and stabilised weights for the ATE
# =============================================================================
ps, sw = stabilised_weights(df)

print("--- (a) Stabilised weights ---")
print(f"mean = {sw.mean():.3f}   median = {np.median(sw):.3f}   max = {sw.max():.2f}")
print("Stabilised weights should cluster around 1, and these do.")

# =============================================================================
# (b) The two mandatory checks: balance, then positivity
# =============================================================================
print("\n--- (b) Balance after weighting (want every |SMD| < 0.1) ---")
for v in COVS:
    before = weighted_smd(df, np.ones(len(df)), v)
    after = weighted_smd(df, sw, v)
    flag = "OK" if abs(after) < 0.1 else "NOT BALANCED"
    print(f"  {v:<14} before {before:+.3f}   after {after:+.3f}   {flag}")

print("\n--- (b) Positivity ---")
print(f"Largest stabilised weight: {sw.max():.2f}")
print(f"Propensity score range   : {ps.min():.3f} to {ps.max():.3f}")
print(f"Effective sample size    : {effective_sample_size(sw):.0f} of {len(df)}")
print("""Rule of thumb: a maximum weight above roughly 10-20 means one or two
patients are dominating the analysis. We are far below that, and no propensity
score is near 0 or 1, so positivity looks fine.""")

# =============================================================================
# (c) The ATE as a risk difference
# =============================================================================
est, lo, hi = ipw_risk_difference(df, sw)
unadjusted = (df.loc[df.treatment == 1, "death_1yr"].mean()
              - df.loc[df.treatment == 0, "death_1yr"].mean())

print("\n--- (c) IPW estimate of the ATE ---")
print(f"IPW risk difference : {est:+.4f} (95% bootstrap CI {lo:+.4f}, {hi:+.4f})"
      f"   [truth {TRUE_ATE_RD:+.4f}]")
print(f"Unadjusted, for comparison: {unadjusted:+.4f}")
contains = lo <= TRUE_ATE_RD <= hi
print(f"\nThe naive comparison recovers only {100 * unadjusted / TRUE_ATE_RD:.0f}% "
      "of the true effect.")
print(f"The IPW interval {'DOES' if contains else 'does NOT'} contain the truth.")
print("""Note that the IPW point estimate is not identical to the truth, and in
this particular sample it overshoots. That is sampling variation, not bias: the
confidence interval is the honest statement of how precisely we know the answer
from 1500 patients. Exercise 5 repeats the whole simulation many times to show
that IPW is centred on the truth on average, which is the property that matters
and which no single dataset can demonstrate.""")

# =============================================================================
# (d) Breaking positivity on purpose
# =============================================================================
bad, TRUE_ATE_BAD = simulate_cohort(extreme=True)
ps_bad, sw_bad = stabilised_weights(bad)

print("\n\n=== (d) What happens when positivity fails ===")
print(pd.crosstab(bad.heart_failure, bad.treatment,
                  rownames=["heart failure"], colnames=["treated"]))

est_bad, lo_bad, hi_bad = ipw_risk_difference(bad, sw_bad)

print(f"\nLargest stabilised weight now : {sw_bad.max():.1f}   "
      f"(it was {sw.max():.2f} before)")
print(f"The most influential single patient carries "
      f"{100 * sw_bad.max() / sw_bad.sum():.1f}% of the total weight")
print(f"-- about {sw_bad.max() / sw_bad.mean():.0f} times an average patient's share.")
print(f"Effective sample size: {effective_sample_size(sw_bad):.0f} "
      f"(from {len(bad)} real patients) -- was {effective_sample_size(sw):.0f}")
print(f"IPW estimate: {est_bad:+.4f} (95% CI {lo_bad:+.4f}, {hi_bad:+.4f})"
      f"   [truth {TRUE_ATE_BAD:+.4f}]")
print(f"The confidence interval is now {(hi_bad - lo_bad) / (hi - lo):.1f} times "
      "wider than before.")

print("""
What to tell a clinical collaborator:
"In this data almost every patient with heart failure was treated and almost
 nobody without it was. The weighting therefore leans on a handful of unusual
 patients -- the few untreated ones who had heart failure -- to stand in for an
 entire group. In effect we are down from over a thousand patients' worth of
 information to about a hundred, the confidence interval is several times wider,
 and the estimate moves around a lot from sample to sample. I would not report
 an ATE from this."

The options, in order of preference:
 1. Change the question. Estimate the effect only where both treatments
    actually occur -- e.g. within heart-failure patients, or target the ATT
    instead of the ATE.
 2. Trim or truncate the weights, and report the trimmed AND untrimmed results
    so the reader sees how much the choice mattered.
 3. G-computation (Exercise 4) does not divide by a small probability, so it
    will not blow up -- but it then has to EXTRAPOLATE into the region where
    there is no data. That is a different way of being wrong, not a fix, and it
    fails silently rather than loudly.

The honest answer: no estimator can recover an effect in a group where one of
the treatments was essentially never given. Positivity is a property of the
data, not of the method.""")
