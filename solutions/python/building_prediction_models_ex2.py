"""Exercise 2: Missing data simulation.

Compare complete case analysis, single mean imputation, and multiple
imputation on the simulated Framingham cohort from the chapter.

Python's random numbers are not R's, so the numbers here differ from the R
solution in detail. The pattern -- which approach fails, and how -- is the same.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

PREDICTORS = ["age", "male", "sbp", "total_chol", "hdl_chol",
              "smoking", "diabetes", "bp_treatment"]
# The true coefficients: we simulated the data, so we know them.
TRUTH = pd.Series({"const": -7.5, "age": 0.06, "male": 0.4, "sbp": 0.012,
                   "total_chol": 0.005, "hdl_chol": -0.02, "smoking": 0.5,
                   "diabetes": 0.7, "bp_treatment": 0.3})

# --- The cohort, as in the chapter -----------------------------------------
rng = np.random.default_rng(2024)
n = 2000

framingham = pd.DataFrame({
    "age": rng.integers(30, 75, n),
    "male": rng.binomial(1, 0.48, n),
    "sbp": np.round(rng.normal(130, 18, n)),
    "total_chol": np.round(rng.normal(210, 38, n)),
    "hdl_chol": np.round(rng.normal(52, 15, n)),
    "smoking": rng.binomial(1, 0.22, n),
    "diabetes": rng.binomial(1, 0.08, n),
    "bp_treatment": rng.binomial(1, 0.15, n),
})
lp = (-7.5 + 0.06 * framingham["age"] + 0.4 * framingham["male"]
      + 0.012 * framingham["sbp"] + 0.005 * framingham["total_chol"]
      - 0.02 * framingham["hdl_chol"] + 0.5 * framingham["smoking"]
      + 0.7 * framingham["diabetes"] + 0.3 * framingham["bp_treatment"])
framingham["cvd_10yr"] = rng.binomial(1, expit(lp))
print(f"Cohort: {len(framingham)} patients, "
      f"{framingham['cvd_10yr'].sum()} events")


def fit_logit(df):
    """Unpenalised logistic regression, returning coefficients and SEs."""
    X = sm.add_constant(df[PREDICTORS].astype(float))
    res = sm.Logit(df["cvd_10yr"], X).fit(disp=0)
    return res.params, res.bse


def punch_holes(df, intercept, seed=7):
    """Delete values of two predictors under a MAR mechanism.

    Missingness in total_chol depends on the OUTCOME and on age, both of which
    are recorded -- so this is MAR, not MNAR. That is deliberate: if
    missingness depended only on predictors already in the model, complete case
    analysis would still be unbiased for the coefficients and there would be
    nothing to demonstrate.
    """
    out = df.copy()
    r = np.random.default_rng(seed)
    p_chol = expit(intercept + 1.0 * out["cvd_10yr"] + 0.03 * (out["age"] - 52))
    p_hdl = expit(intercept + 0.2 + 0.9 * out["smoking"]
                  + 0.02 * (out["sbp"] - 130))
    out.loc[r.uniform(size=len(out)) < p_chol, "total_chol"] = np.nan
    out.loc[r.uniform(size=len(out)) < p_hdl, "hdl_chol"] = np.nan
    return out


def mean_impute(df):
    out = df.copy()
    for v in ["total_chol", "hdl_chol"]:
        out[v] = out[v].fillna(out[v].mean())
    return out


def impute_once(df, rng, iterations=5):
    """Create ONE filled-in dataset by chained equations.

    This is written out rather than delegated to a package because it is short
    enough to read, and reading it is the point: each missing value is drawn
    from a regression on the other variables, with fresh randomness every time,
    which is what makes the m datasets differ from each other.

    Two details decide whether it works at all:

    * The draw includes both parameter uncertainty (a draw of sigma^2, then of
      beta given sigma^2) and residual noise. Using the fitted value alone
      would make every dataset identical -- single imputation in disguise.
    * The OUTCOME is one of the predictors in the imputation model. Leaving it
      out imputes predictors as though they were unrelated to the outcome,
      which biases their coefficients towards zero. R's `mice` includes every
      column by default, which is why the R solution just hands it the whole
      data frame.

    In practice use `mice` in R or `miceforest` in Python. Note that
    scikit-learn's `IterativeImputer` is not a drop-in substitute: even with
    `sample_posterior=True` its imputations are under-dispersed, which
    understates the between-imputation variance this exercise is about.
    """
    columns = PREDICTORS + ["cvd_10yr"]
    work = df[columns].astype(float).copy()
    incomplete = [v for v in columns if work[v].isna().any()]
    missing = {v: work[v].isna().to_numpy() for v in incomplete}

    for v in incomplete:  # start each variable off at its observed mean
        work.loc[missing[v], v] = work[v].mean()

    for _ in range(iterations):
        for v in incomplete:
            others = [c for c in columns if c != v]
            observed = ~missing[v]
            X = np.column_stack([np.ones(observed.sum()),
                                 work.loc[observed, others].to_numpy()])
            y = work.loc[observed, v].to_numpy()

            XtX_inv = np.linalg.pinv(X.T @ X)
            beta_hat = XtX_inv @ X.T @ y
            residuals = y - X @ beta_hat
            dof = max(len(y) - X.shape[1], 1)

            # Posterior draws, in the order mice's "norm" method uses them
            sigma2 = residuals @ residuals / rng.chisquare(dof)
            chol = np.linalg.cholesky(sigma2 * XtX_inv
                                      + 1e-12 * np.eye(len(beta_hat)))
            beta = beta_hat + chol @ rng.standard_normal(len(beta_hat))

            X_missing = np.column_stack([
                np.ones(missing[v].sum()),
                work.loc[missing[v], others].to_numpy(),
            ])
            work.loc[missing[v], v] = (
                X_missing @ beta
                + rng.normal(0, np.sqrt(sigma2), missing[v].sum())
            )

    work["cvd_10yr"] = df["cvd_10yr"].to_numpy()  # never imputed
    return work


def multiple_impute(df, m=20, seed=42):
    """Impute m times, fit the model in each, pool with Rubin's rules."""
    rng = np.random.default_rng(seed)
    estimates, variances = [], []
    for _ in range(m):
        params, ses = fit_logit(impute_once(df, rng))
        estimates.append(params)
        variances.append(ses**2)

    estimates = pd.DataFrame(estimates)
    variances = pd.DataFrame(variances)

    # Rubin's rules: the pooled variance is the average within-imputation
    # variance plus the between-imputation variance, inflated by 1 + 1/m.
    within = variances.mean()
    between = estimates.var(ddof=1)
    total = within + (1 + 1 / m) * between
    return estimates.mean(), np.sqrt(total), within, between


incomplete = punch_holes(framingham, intercept=-2.2)
complete_rows = incomplete[PREDICTORS].notna().all(axis=1)

print(f"Missing total_chol: {incomplete['total_chol'].isna().sum()}")
print(f"Missing hdl_chol:  {incomplete['hdl_chol'].isna().sum()}")
print(f"Complete rows: {complete_rows.sum()} "
      f"({100 * (1 - complete_rows.mean()):.0f}% of the cohort discarded by a "
      "complete case analysis)")

# --- The three approaches, plus the full data as a benchmark -------------
b_full, se_full = fit_logit(framingham)
b_cca, se_cca = fit_logit(incomplete[complete_rows])
b_mean, se_mean = fit_logit(mean_impute(incomplete))
b_mi, se_mi, within, between = multiple_impute(incomplete)

comparison = pd.DataFrame({
    "truth": TRUTH,
    "full_data": b_full,
    "complete_case": b_cca,
    "mean_imputation": b_mean,
    "multiple_imputation": b_mi,
})[["truth", "full_data", "complete_case", "mean_imputation",
    "multiple_imputation"]]

print("\n--- Coefficient estimates ---")
print(comparison.round(4))

# --- (a) Which approach lands closest to the truth? ---------------------
# Two error measures, and the difference between them is the point. Distance
# from the truth mixes up two things: damage done by the missing data, and the
# sampling noise already present in this cohort of 2000. Distance from the
# full-data estimates isolates the first.
slopes = comparison.drop(index="const")
errors = pd.DataFrame({
    "vs_truth": (slopes.drop(columns="truth")
                 .sub(slopes["truth"], axis=0).abs().mean()),
    "vs_full_data": (slopes.drop(columns=["truth", "full_data"])
                     .sub(slopes["full_data"], axis=0).abs().mean()),
})
print("\n--- Mean absolute error across the 8 slopes ---")
print(errors.round(5))

print("\nRanked by distance from the full-data estimates:")
print(errors["vs_full_data"].dropna().sort_values().round(5))

# --- (b) What happens to the standard errors? --------------------------
se_table = pd.DataFrame({
    "complete_case": se_cca,
    "mean_imputation": se_mean,
    "multiple_imputation": se_mi,
})
print("\n--- Standard errors ---")
print(se_table.round(5))

print("\nMean imputation SE as a percentage of the multiple-imputation SE,")
print("for the two variables that were actually imputed:")
for v in ["total_chol", "hdl_chol"]:
    print(f"  {v:11s} {100 * se_mean[v] / se_mi[v]:.1f}%")

# What single imputation actually discards, as a number. Rubin's rules split the
# pooled variance into a within-imputation part (ordinary sampling uncertainty)
# and a between-imputation part (uncertainty about the guesses themselves).
# Single imputation sets the second to zero by construction. This share is what
# mice reports as lambda.
print("\nShare of the pooled MI variance that comes from the imputation itself:")
for v in ["total_chol", "hdl_chol"]:
    share = (1 + 1 / 20) * between[v] / (within[v] + (1 + 1 / 20) * between[v])
    print(f"  {v:11s} {100 * share:.1f}%")

# Two opposing effects act on the mean-imputation standard error, which is why
# the percentages above sit close to 100 at this fraction of missing data:
#   1. it ignores the imputation uncertainty just quantified  -> SE too small
#   2. it flattens the variable's spread, and less spread in a
#      predictor means less information about its coefficient  -> SE too large
# The first grows with the fraction missing; the second is why they can briefly
# cancel, and one of the two variables below may even come out above 100%.
print(f"\nSD of total_chol: {incomplete['total_chol'].std():.1f} observed"
      f" -> {mean_impute(incomplete)['total_chol'].std():.1f}"
      " after mean imputation")

# --- The same comparison with far more missing data -------------------
heavy = punch_holes(framingham, intercept=-0.4)
heavy_rows = heavy[PREDICTORS].notna().all(axis=1)
b_cca_h, se_cca_h = fit_logit(heavy[heavy_rows])
b_mean_h, se_mean_h = fit_logit(mean_impute(heavy))
b_mi_h, se_mi_h, within_h, between_h = multiple_impute(heavy)

print(f"\n--- With {100 * heavy['total_chol'].isna().mean():.0f}% of total_chol"
      f" and {100 * heavy['hdl_chol'].isna().mean():.0f}% of hdl_chol missing ---")
for v in ["total_chol", "hdl_chol"]:
    print(f"{v:11s} truth {TRUTH[v]:+.4f}")
    print(f"  complete case   {b_cca_h[v]:+.4f} (SE {se_cca_h[v]:.5f})")
    print(f"  mean imputation {b_mean_h[v]:+.4f} (SE {se_mean_h[v]:.5f})")
    print(f"  MI              {b_mi_h[v]:+.4f} (SE {se_mi_h[v]:.5f})")
    share_now = ((1 + 1 / 20) * between_h[v]
                 / (within_h[v] + (1 + 1 / 20) * between_h[v]))
    share_was = ((1 + 1 / 20) * between[v]
                 / (within[v] + (1 + 1 / 20) * between[v]))
    print(f"  -> mean imputation's SE is "
          f"{100 * se_mean_h[v] / se_mi_h[v]:.0f}% of MI's, and "
          f"{100 * share_now:.0f}% of MI's variance")
    print(f"     for this coefficient now comes from the imputation "
          f"(was {100 * share_was:.0f}%)")

print("""
Conclusions
-----------
(a) Complete case analysis is the clear loser. It sits furthest from the
    full-data estimates, and it is biased by construction here: because
    missingness depends on the outcome, the retained rows under-represent
    patients who had an event. It also discards a quarter of the cohort, so
    its standard errors are the widest of the three.

    Mean imputation and multiple imputation give similar point estimates at
    this fraction of missing data. Measuring against the true values alone is
    misleading, though: the diabetes coefficient is far from its true 0.7 in
    every column, including the full-data one, because 2000 patients and
    roughly 200 events cannot pin it down. That error is sampling noise, not
    missing-data handling -- which is why the comparison against the full-data
    estimates is the informative one.

(b) Filling every gap with the mean asserts those values were measured rather
    than guessed, so nothing in the model widens to reflect the guessing.
    Rubin's rules make visible exactly what is discarded: the printed share
    says how much of the pooled uncertainty comes from the imputation itself,
    and single imputation sets that share to zero.

    That does not always surface as a smaller standard error, and the output
    above is a good reminder to check rather than assume. Two effects pull in
    opposite directions -- ignoring the imputation uncertainty makes the
    standard error too small, while flattening the variable's spread makes it
    too large -- so at this fraction of missing data they nearly cancel, and
    with half the values missing one variable still comes out slightly above
    the multiple-imputation standard error while the other falls to about two
    thirds of it.

    Watch the share rather than the ratio, because the share is the part that
    behaves predictably: it climbs from a fifth or a quarter to a half or more
    as the missingness grows. The fair statement is not that single imputation always
    looks more precise, but that its uncertainty is unaccounted for, and the
    size of what it ignores grows with the amount you imputed.

    Which method lands nearest the truth in any one dataset is luck; all of
    them sit within a standard error of each other. The missing variance
    component is systematic.

    One caveat: multiple imputation assumes MAR, which holds here by
    construction. If the highest cholesterol values were missing precisely
    because they were high (MNAR), no method here would recover them, and the
    honest response would be a sensitivity analysis.
""")
