"""Exercise 1: Sample size calculation.

Pre-eclampsia model: prevalence 4%, 12 candidate predictors, an anticipated
C-statistic of 0.72 from a published model in a comparable population.

There is no Python port of pmsampsize, so both halves of what the R package
does are implemented here: the C-statistic to Cox-Snell R-squared conversion,
and the three Riley criteria. The numbers are checked against pmsampsize at
the bottom.
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

C_STATISTIC = 0.72
PARAMETERS = 12
PREVALENCE = 0.04


def cstat_to_cs_rsquared(cstatistic, prevalence, n=1_000_000, seed=123456):
    """Convert an anticipated C-statistic into a Cox-Snell R-squared.

    This mirrors what pmsampsize does internally. There is no closed form, so
    it simulates a large population whose linear predictor separates events
    from non-events by exactly the amount implied by the C-statistic, fits a
    logistic regression to it, and reads off the Cox-Snell R-squared.

    The separation is mu = sqrt(2) * qnorm(C): under two unit-variance normals
    one mu apart, the probability that a randomly chosen event scores above a
    randomly chosen non-event is exactly the C-statistic.
    """
    rng = np.random.default_rng(seed)
    mu = np.sqrt(2) * norm.ppf(cstatistic)

    n0 = int(prevalence * n)
    n1 = int((1 - prevalence) * n)
    lp = np.concatenate([rng.normal(0.0, 1.0, n0), rng.normal(mu, 1.0, n1)])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)

    # Cox-Snell R-squared = 1 - exp(-(null deviance - model deviance) / n).
    # statsmodels reports log-likelihoods; deviance = -2 * log-likelihood.
    lr_stat = 2 * (fit.llf - fit.llnull)
    return 1 - np.exp(-lr_stat / len(y))


def max_cs_rsquared(prevalence):
    """The largest Cox-Snell R-squared attainable at this prevalence.

    Cox-Snell cannot reach 1 for a binary outcome; the ceiling depends only on
    how common the outcome is. Nagelkerke R-squared is Cox-Snell divided by it.
    """
    phi = prevalence
    return 1 - (phi**phi * (1 - phi) ** (1 - phi)) ** 2


def riley_sample_size(cs_rsquared, parameters, prevalence,
                      shrinkage=0.9, delta=0.05, moe=0.05):
    """The three Riley et al. criteria for a binary-outcome model."""
    phi = prevalence
    max_r2 = max_cs_rsquared(phi)

    # Criterion 1: expected shrinkage of at least `shrinkage` (0.9 = at most
    # 10% overfitting)
    n1 = parameters / ((shrinkage - 1) * np.log(1 - cs_rsquared / shrinkage))

    # Criterion 2: apparent and adjusted Nagelkerke R-squared differ by <= delta
    s2 = cs_rsquared / (cs_rsquared + delta * max_r2)
    n2 = parameters / ((s2 - 1) * np.log(1 - cs_rsquared / s2))

    # Criterion 3: the overall risk itself estimated to within +/- moe
    n3 = (1.96 / moe) ** 2 * phi * (1 - phi)

    return {
        "Criterion 1 (shrinkage >= 0.9)": int(np.ceil(n1)),
        "Criterion 2 (R-squared gap <= 0.05)": int(np.ceil(n2)),
        "Criterion 3 (risk within +/- 0.05)": int(np.ceil(n3)),
    }


# (a) Minimum sample size ----------------------------------------------------
r2cs = cstat_to_cs_rsquared(C_STATISTIC, PREVALENCE)
max_r2 = max_cs_rsquared(PREVALENCE)

print(f"C-statistic {C_STATISTIC} at {PREVALENCE:.0%} prevalence")
print(f"  Cox-Snell R-squared:      {r2cs:.4f}")
print(f"  Maximum possible:         {max_r2:.4f}")
print(f"  Nagelkerke equivalent:    {r2cs / max_r2:.4f}\n")

criteria = riley_sample_size(r2cs, PARAMETERS, PREVALENCE)
for name, n in criteria.items():
    print(f"  {name:38s} n = {n}")

n_min = max(criteria.values())
events = int(np.ceil(n_min * PREVALENCE))
print(f"\nMinimum sample size: {n_min} pregnancies")
print(f"Minimum number of events: {events}")
print(f"Events per parameter: {events / PARAMETERS:.2f}")

# (b) Which criterion is binding? -------------------------------------------
binding = max(criteria, key=criteria.get)
print(f"\nBinding criterion: {binding}")
print("Criterion 1 almost always binds for a binary outcome: it is the one "
      "that limits overfitting,\nwhile criterion 3 only asks that the average "
      "risk be pinned down, which needs far fewer patients.")

# (c) Comparison with the 10-EPV rule of thumb -----------------------------
epv_events = 10 * PARAMETERS
epv_n = int(np.ceil(epv_events / PREVALENCE))

print("\n--- 10 events per variable rule ---")
print(f"Events required: {epv_events}")
print(f"Implied sample size at {PREVALENCE:.0%} prevalence: {epv_n}")
print("\n--- Riley criteria ---")
print(f"Events required: {events}")
print(f"Sample size: {n_min}")
print(f"\nRiley / EPV ratio: {n_min / epv_n:.2f} times the EPV recommendation")

# Check against pmsampsize -------------------------------------------------
# pmsampsize(type = "b", cstatistic = 0.72, parameters = 12, prevalence = 0.04)
# reports Cox-Snell R-squared 0.0251 and n = 4243 / 825 / 60 for the three
# criteria. The conversion above is simulation-based and Python's random
# numbers are not R's, so agreement to the nearest few patients is what to
# expect, not an exact match.
print("\n--- agreement with pmsampsize (R) ---")
for label, ours, theirs in [
    ("Cox-Snell R-squared", round(r2cs, 4), 0.0251),
    ("Criterion 1", criteria["Criterion 1 (shrinkage >= 0.9)"], 4243),
    ("Criterion 2", criteria["Criterion 2 (R-squared gap <= 0.05)"], 825),
    ("Criterion 3", criteria["Criterion 3 (risk within +/- 0.05)"], 60),
]:
    print(f"  {label:22s} python {ours:<10} pmsampsize {theirs}")
