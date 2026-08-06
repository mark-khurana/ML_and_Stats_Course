"""Exercise 3: Variable selection.

Compare a pre-specified model, forward stepwise selection, and LASSO on the
simulated Framingham cohort, then check how stable each selection is across
100 bootstrap samples.

Python's random numbers are not R's, so the numbers differ from the R solution
in detail; the pattern is the same. Runtime is a minute or two, because the
bootstrap refits both selection procedures 100 times.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)

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

REAL = [c for c in framingham.columns if c != "cvd_10yr"]
print(f"Events: {framingham['cvd_10yr'].sum()} of {n}")


# --- Helpers ---------------------------------------------------------------
def log_likelihood(X, y):
    """Maximised log-likelihood of an unpenalised logistic fit."""
    if X.shape[1] == 0:
        p = np.full(len(y), y.mean())
    else:
        p = (LogisticRegression(penalty=None, max_iter=1000)
             .fit(X, y).predict_proba(X)[:, 1])
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def forward_stepwise(df, candidates, outcome="cvd_10yr"):
    """Forward selection on AIC, the equivalent of R's stepAIC(direction='forward').

    The candidates are standardised first purely for speed: the likelihood, and
    therefore AIC and every selection decision, is unchanged by rescaling a
    predictor, but the optimiser converges far faster when age (30-74) and
    total cholesterol (around 210) are on a common scale.
    """
    y = df[outcome].to_numpy()
    Z = pd.DataFrame(
        StandardScaler().fit_transform(df[candidates].astype(float)),
        columns=candidates,
    )
    chosen, remaining = [], list(candidates)
    current_aic = 2 * 1 - 2 * log_likelihood(np.empty((len(df), 0)), y)

    while remaining:
        best, best_aic = None, current_aic
        for v in remaining:
            X = Z[chosen + [v]].to_numpy()
            aic = 2 * (X.shape[1] + 1) - 2 * log_likelihood(X, y)
            if aic < best_aic:
                best, best_aic = v, aic
        if best is None:  # nothing improves AIC any further
            break
        chosen.append(best)
        remaining.remove(best)
        current_aic = best_aic
    return chosen


def lasso_selected(df, candidates, outcome="cvd_10yr", cv=10, one_se=False,
                   seed=1):
    """Variables kept by a cross-validated LASSO.

    Two settings matter and are easy to get wrong (both are flagged in the
    chapter): predictors must be standardised, because the penalty acts on the
    coefficient scale, and scoring must be a probability score -- the default
    accuracy is maximised by predicting the majority class, which would choose
    maximum shrinkage and drop everything.
    """
    X = StandardScaler().fit_transform(df[candidates].astype(float))
    y = df[outcome].to_numpy()
    grid = np.logspace(-4, 1, 25)
    model = LogisticRegressionCV(
        Cs=grid, penalty="l1", solver="liblinear", cv=cv,
        scoring="neg_log_loss", max_iter=5000, random_state=seed,
    ).fit(X, y)

    if one_se:
        # The equivalent of glmnet's lambda.1se: the strongest penalty whose
        # mean CV score is within one standard error of the best.
        scores = model.scores_[1]                  # folds x Cs
        mean, se = scores.mean(axis=0), scores.std(axis=0) / np.sqrt(cv)
        best = mean.argmax()
        ok = np.where(mean >= mean[best] - se[best])[0]
        chosen_C = model.Cs_[ok.min()]             # smallest C = strongest penalty
        model = LogisticRegression(C=chosen_C, penalty="l1", solver="liblinear",
                                   max_iter=5000).fit(X, y)

    return [c for c, b in zip(candidates, model.coef_[0]) if b != 0]


def describe(label, selected):
    n_real = sum(v in REAL for v in selected)
    n_noise = sum("noise" in v for v in selected)
    print(f"  {label:18s} kept {len(selected):2d}  "
          f"({n_real} of 8 real, {n_noise} of 12 noise)")


# --- (a) The three approaches on the pre-specified predictors -------------
print("\n=== Part (a): candidates are the 8 pre-specified predictors ===")

X_all = sm.add_constant(framingham[REAL].astype(float))
fit_all = sm.Logit(framingham["cvd_10yr"], X_all).fit(disp=0)
print("\nPre-specified model (all 8 predictors):")
print(pd.DataFrame({"coef": fit_all.params, "se": fit_all.bse,
                    "p": fit_all.pvalues}).round(4))

step_a = forward_stepwise(framingham, REAL)
lasso_min_a = lasso_selected(framingham, REAL)
lasso_1se_a = lasso_selected(framingham, REAL, one_se=True)

print("\nSelected sets:")
print(f"  pre-specified      (8): {', '.join(REAL)}")
print(f"  forward stepwise   ({len(step_a)}): {', '.join(sorted(step_a))}")
print(f"  LASSO best lambda  ({len(lasso_min_a)}): {', '.join(sorted(lasso_min_a))}")
print(f"  LASSO 1-SE lambda  ({len(lasso_1se_a)}): {', '.join(sorted(lasso_1se_a))}")

print("""
Every one of these 8 variables belongs in the model -- we put them all in the
simulation -- so a perfect selection method would keep all 8. Seven are clearly
significant here. bp_treatment is not (p = 0.34 in this draw, its true effect
being the smallest of the eight), and forward stepwise duly drops it.

That is the first lesson, and it is easy to miss because dropping a variable
feels like tidying up: stepwise did not remove a useless predictor, it removed
a real one that this sample could not resolve. The LASSO at its best lambda
keeps all 8. Selection cannot improve on a well-chosen pre-specified list; it
can only lose parts of it.""")

# --- (b) The realistic case: candidates that do not belong ---------------
# Selection only becomes interesting when the candidate list contains variables
# with no relationship to the outcome, which is the usual situation when the
# list is drawn up from everything that happened to be measured.
noise_rng = np.random.default_rng(99)
wide = framingham.copy()
for i in range(1, 13):
    wide[f"noise{i}"] = noise_rng.normal(size=n)
CANDIDATES = REAL + [f"noise{i}" for i in range(1, 13)]

print("\n=== Part (b): 8 real predictors + 12 pure-noise candidates ===\n")
step_b = forward_stepwise(wide, CANDIDATES)
lasso_min_b = lasso_selected(wide, CANDIDATES)
lasso_1se_b = lasso_selected(wide, CANDIDATES, one_se=True)

describe("forward stepwise", step_b)
describe("LASSO best lambda", lasso_min_b)
describe("LASSO 1-SE lambda", lasso_1se_b)
print("\n  stepwise kept these noise variables:",
      ", ".join(sorted(v for v in step_b if "noise" in v)) or "none")

# --- (c) Stability across 100 bootstrap samples -------------------------
print("\n=== Part (c): selection frequency across 100 bootstrap samples ===")

B = 100
boot_rng = np.random.default_rng(2025)
counts = {m: dict.fromkeys(CANDIDATES, 0) for m in ("stepwise", "lasso")}
sizes = {m: [] for m in counts}
signatures = {m: [] for m in counts}

for b in range(B):
    idx = boot_rng.integers(0, n, n)
    boot = wide.iloc[idx].reset_index(drop=True)
    picks = {
        "stepwise": forward_stepwise(boot, CANDIDATES),
        # cv=5 inside the loop purely to keep the runtime reasonable
        "lasso": lasso_selected(boot, CANDIDATES, cv=5),
    }
    for m, chosen in picks.items():
        for v in chosen:
            counts[m][v] += 1
        sizes[m].append(len(chosen))
        signatures[m].append("|".join(sorted(chosen)))

freq = pd.DataFrame({
    "variable": CANDIDATES,
    "truth": ["real" if v in REAL else "noise" for v in CANDIDATES],
    "stepwise_pct": [100 * counts["stepwise"][v] / B for v in CANDIDATES],
    "lasso_pct": [100 * counts["lasso"][v] / B for v in CANDIDATES],
}).sort_values(["truth", "stepwise_pct"], ascending=[True, False])

print("\nHow often each candidate was selected (%):")
print(freq.to_string(index=False))

print("\nSummary across the 100 bootstrap samples:")
for m in ("stepwise", "lasso"):
    real_pct = np.mean([counts[m][v] for v in REAL]) / B * 100
    noise_pct = np.mean([counts[m][v] for v in CANDIDATES
                         if "noise" in v]) / B * 100
    print(f"  {m:9s} median size {np.median(sizes[m]):.0f} | real predictors "
          f"kept {real_pct:.0f}% of the time | noise kept {noise_pct:.0f}% | "
          f"{len(set(signatures[m]))} distinct models")

print("""
Conclusions
-----------
(a) Every one of the 8 pre-specified predictors belongs in the model, so the
    only thing selection can do is lose one -- which is what happens: stepwise
    drops bp_treatment, the weakest real effect. The best-lambda LASSO keeps
    all 8. Selection cannot improve on a well-chosen pre-specified list.

(b) Add candidates that do not belong and every method starts letting them in.
    Forward stepwise admits noise variables because a variable enters on
    whether it improves AIC in this particular sample, and with 12 chances some
    noise variable always looks helpful. The best-lambda LASSO is the worst
    offender, keeping 9 of the 12: that lambda minimises prediction error, not
    the number of wrong variables, so it prefers to retain noise with small
    coefficients. The 1-SE lambda is more parsimonious but not clean either --
    it still admits noise, and it still misses bp_treatment.

    So none of the three recovers the true model, and the two that look tidiest
    are tidy for the wrong reason: they dropped a real predictor along with
    some of the noise.

(c) The bootstrap frequencies are the real finding. Stepwise produced 99
    different models in 100 resamples of the same patients, and the LASSO 70.
    Real predictors were kept 84% of the time by stepwise and 97% by the LASSO;
    noise variables 40% and 83%.

    The instability sits exactly where you would want the method's advice.
    age and hdl_chol are chosen every single time, while bp_treatment -- a
    genuine predictor -- is chosen by stepwise in under a third of samples. A
    reader of any one such model would conclude that blood-pressure treatment
    does not predict risk, and the next sample would tell them otherwise.

    One subtlety worth naming, because it cuts against the obvious reading.
    Two of the noise variables are selected in about 79% of bootstrap samples
    by stepwise and 97% by the LASSO, which looks like reliability. It is not:
    they happen to correlate with the outcome in this particular cohort, and
    every bootstrap sample is drawn from that same cohort, so the fluke is
    faithfully reproduced. Bootstrap stability measures robustness to
    resampling these patients, not to collecting new ones -- it is a lower
    bound on the instability a fresh dataset would reveal.

    That is what makes a paper reporting one stepwise-selected model
    misleading. The list of retained variables is presented as a finding --
    these are the predictors of risk -- when a different sample of the same
    patients would have produced a different list, and a genuinely new sample
    a different one again. The p-values and confidence intervals are also
    wrong, because they take no account of the searching that preceded them.
    The defensible options are to pre-specify the predictors on clinical
    grounds and keep them all, or to use a penalised model and report the whole
    procedure rather than the variables that happened to survive it.
""")
