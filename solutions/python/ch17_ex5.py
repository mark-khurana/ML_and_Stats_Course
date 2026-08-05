# =============================================================================
# Chapter 17 - Exercise 5: Do the adjusted methods really recover the truth?
# One dataset is not evidence of unbiasedness. Repeat the whole simulation.
# =============================================================================

# Libraries -------------------------------------------------------------------
# pip install numpy pandas statsmodels matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


def expit(x):
    return 1 / (1 + np.exp(-x))


# --- The data-generating process --------------------------------------------
# ONE binary confounder, `frail`, which raises BOTH the chance of treatment and
# the risk of death. The treatment has a known protective effect.
TRUTH_LOG_ODDS = -0.8


def simulate_cohort(rng, n=2000):
    frail = rng.binomial(1, 0.4, n)
    # Frail patients are much more likely to be treated (confounding by indication)
    treat = rng.binomial(1, expit(-0.4 + 1.6 * frail))
    death = rng.binomial(1, expit(-0.7 + 1.0 * frail + TRUTH_LOG_ODDS * treat))
    return pd.DataFrame(dict(frail=frail, treat=treat, death=death))


# The TRUE marginal risk difference, by brute force on a huge cohort.
big_frail = np.random.default_rng(99).binomial(1, 0.4, 2_000_000)
TRUE_RD = (expit(-0.7 + 1.0 * big_frail + TRUTH_LOG_ODDS).mean()
           - expit(-0.7 + 1.0 * big_frail).mean())
print(f"TRUE marginal risk difference: {TRUE_RD:+.4f}")
print(f"(built from a conditional log-odds of {TRUTH_LOG_ODDS:+.2f})\n")


# --- The three estimators, each returning a marginal risk difference --------
def contrast(model, d):
    return (model.predict(d.assign(treat=1)).mean()
            - model.predict(d.assign(treat=0)).mean())


def est_naive(d):
    m = smf.glm("death ~ treat", data=d, family=sm.families.Binomial()).fit(disp=0)
    return contrast(m, d)


def est_ipw(d):
    ps = smf.logit("treat ~ frail", data=d).fit(disp=0).predict(d)
    p_marg = d.treat.mean()
    sw = np.where(d.treat == 1, p_marg / ps, (1 - p_marg) / (1 - ps))
    m = smf.glm("death ~ treat", data=d, family=sm.families.Binomial(),
                freq_weights=sw).fit()
    return contrast(m, d)


def est_gcomp(d):
    m = smf.glm("death ~ treat * frail", data=d,
                family=sm.families.Binomial()).fit(disp=0)
    return contrast(m, d)


ESTIMATORS = {"Naive": est_naive, "IPW": est_ipw, "G-computation": est_gcomp}

# =============================================================================
# (a) One dataset, three estimates
# =============================================================================
dat = simulate_cohort(np.random.default_rng(42))

print("--- (a) A single dataset (n = 2000) ---")
print(f"Treatment rate: {100 * dat.loc[dat.frail == 0, 'treat'].mean():.1f}% of "
      f"non-frail, {100 * dat.loc[dat.frail == 1, 'treat'].mean():.1f}% of frail "
      "patients")
for name, fn in ESTIMATORS.items():
    value = fn(dat)
    print(f"  {name:<14} {value:+.4f}   (error vs truth: {value - TRUE_RD:+.4f})")
print("""
On this one sample the adjusted estimates look good -- but a single sample cannot
distinguish an unbiased estimator from a lucky one.""")

# =============================================================================
# (b) Repeat the whole simulation 500 times
# =============================================================================
R = 500
rng = np.random.default_rng(2024)
rows = []
for _ in range(R):
    d = simulate_cohort(rng)
    rows.append({name: fn(d) for name, fn in ESTIMATORS.items()})
sims = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
colours = {"Naive": "#D55E00", "IPW": "#0072B2", "G-computation": "#009E73"}
for name in ESTIMATORS:
    sims[name].plot.density(ax=ax, label=name, color=colours[name], lw=2)
ax.axvline(TRUE_RD, linestyle="--", color="#b02a2a", lw=1.6)
ax.text(TRUE_RD, ax.get_ylim()[1] * 0.95, f"  truth = {TRUE_RD:+.3f}",
        color="#b02a2a", fontweight="bold", va="top")
ax.set_xlabel("Estimated marginal risk difference")
ax.set_ylabel("Density")
ax.set_title(f"Sampling distribution over {R} simulated cohorts")
ax.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# (c) Bias
# =============================================================================
summary = pd.DataFrame({
    "mean_estimate": sims.mean(),
    "bias": sims.mean() - TRUE_RD,
    "sd": sims.std(ddof=1),
    "rmse": np.sqrt(((sims - TRUE_RD) ** 2).mean()),
})

print(f"\n--- (c) Over {R} replications ---")
print(summary.round(4))
print("""
Read the `bias` column: it is the average error, and averaging over 500 cohorts
is what lets us see it. The naive estimator's bias is large and in a consistent
direction -- it is not noise, it is a systematic failure. IPW and g-computation
have bias close to zero: they are centred on the truth, which is what 'unbiased'
means and what one dataset could never have shown us.""")

# =============================================================================
# (d) Spread, and why the tightest estimator is not automatically the best
# =============================================================================
print("\n--- (d) Spread ---")
print(f"Smallest standard deviation: {summary['sd'].idxmin()}")
print(f"Smallest RMSE              : {summary['rmse'].idxmin()}")
print("""
The naive estimator is typically the TIGHTEST of the three, and it is also the
only one that is wrong. That is the whole point: precision measures how
consistently an estimator returns the same answer, not whether that answer is
right. A biased estimator can be beautifully precise -- reliably wrong.

The quantity that combines both is the root mean squared error (RMSE) in the
table above, which penalises bias and variance together. On RMSE the adjusted
methods win comfortably despite being noisier.

One striking detail: IPW and g-computation give IDENTICAL numbers here, to every
decimal place, in every replication. That is not a coincidence and not a bug.
With a single binary confounder, both models are SATURATED -- `treat * frail` has
one parameter for each of the four treatment-by-frailty cells, and the propensity
model likewise reproduces the observed treatment rate in each cell exactly. Both
estimators then reduce to the same non-parametric calculation: take the observed
death rate in each of the four cells and re-average it over the frailty
distribution. There is nothing left for them to disagree about.

They come apart as soon as a model has to make an assumption -- with continuous
confounders, non-linear effects, or omitted interactions. Then IPW is at risk
from a wrong TREATMENT model and extreme weights, and g-computation from a wrong
OUTCOME model. Neither is universally better; they fail in different
circumstances, which is why agreement between them is informative and why doubly
robust estimators combine the two.

Try it: change `frail` to a continuous variable, or fit the outcome model without
the interaction, and the two columns will separate.""")
