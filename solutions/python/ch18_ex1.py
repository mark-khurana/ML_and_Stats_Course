# =============================================================================
# Chapter 18 - Exercise 1: Fixed versus random effects, and why it mattered
# Intravenous magnesium after acute myocardial infarction (16 trials)
# =============================================================================

# Libraries -------------------------------------------------------------------
# pip install numpy pandas scipy
import numpy as np
import pandas as pd
from scipy import stats

# The 16 magnesium trials (dat.egger2001 in metafor / metadat).
# ai / n1 = deaths / patients on magnesium; ci / n2 = deaths / patients on control.
STUDY = ["Morton 1984", "Rasmussen 1986", "Smith 1986", "Abraham 1987",
         "Feldstedt 1988", "Shechter 1989", "Ceremuzynski 1989", "Bertschat 1989",
         "Singh 1990", "Pereira 1990", "Shechter 1991", "Golf 1991",
         "Thogersen 1991", "LIMIT-2 1992", "Shechter 1995", "ISIS-4 1995"]
ai = np.array([1, 9, 2, 1, 10, 1, 1, 0, 6, 1, 2, 5, 4, 90, 4, 2216])
n1 = np.array([40, 135, 200, 48, 150, 59, 25, 22, 76, 27, 89, 23, 130, 1159, 107, 29011])
ci = np.array([2, 23, 7, 1, 8, 9, 3, 1, 11, 7, 12, 13, 8, 118, 17, 2103])
n2 = np.array([36, 135, 200, 46, 148, 56, 23, 21, 75, 27, 80, 33, 122, 1157, 108, 29039])


def effect_sizes(ai, n1, ci, n2):
    """Log risk ratios and variances; 0.5 added only to trials with a zero cell."""
    incr = np.where((ai == 0) | (ci == 0), 0.5, 0.0)
    a, c = ai + incr, ci + incr
    b, d = n1 - ai + incr, n2 - ci + incr
    log_rr = np.log((a / (a + b)) / (c / (c + d)))
    var = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)
    return log_rr, var


def pool(log_rr, var):
    """Fixed-effect and DerSimonian-Laird random effects, with an HKSJ interval."""
    k = len(log_rr)
    w_fe = 1 / var
    te_fe = np.sum(w_fe * log_rr) / np.sum(w_fe)
    se_fe = np.sqrt(1 / np.sum(w_fe))

    Q = np.sum(w_fe * (log_rr - te_fe) ** 2)
    C = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0.0, (Q - (k - 1)) / C)
    I2 = max(0.0, (Q - (k - 1)) / Q)

    w_re = 1 / (var + tau2)
    te_re = np.sum(w_re * log_rr) / np.sum(w_re)
    q_hk = np.sum(w_re * (log_rr - te_re) ** 2) / (k - 1)
    se_hk = np.sqrt(q_hk / np.sum(w_re))

    return dict(k=k, w_fe=w_fe, w_re=w_re, te_fe=te_fe, se_fe=se_fe, te_re=te_re,
                se_hk=se_hk, tau2=tau2, I2=I2, Q=Q,
                Q_p=1 - stats.chi2.cdf(Q, k - 1))


def prediction_interval(res):
    se = np.sqrt(res["se_hk"] ** 2 + res["tau2"])
    t = stats.t.ppf(0.975, res["k"] - 2)
    return np.exp(res["te_re"] - t * se), np.exp(res["te_re"] + t * se)


log_rr, var = effect_sizes(ai, n1, ci, n2)
res = pool(log_rr, var)
t_crit = stats.t.ppf(0.975, res["k"] - 1)

# -----------------------------------------------------------------------------
# (a) Both models
# -----------------------------------------------------------------------------
print("=== (a) Fixed-effect vs random-effects ===")
print(f"Fixed effect   RR = {np.exp(res['te_fe']):.3f} "
      f"(95% CI {np.exp(res['te_fe'] - 1.96 * res['se_fe']):.3f} to "
      f"{np.exp(res['te_fe'] + 1.96 * res['se_fe']):.3f})")
print(f"Random effects RR = {np.exp(res['te_re']):.3f} "
      f"(95% CI {np.exp(res['te_re'] - t_crit * res['se_hk']):.3f} to "
      f"{np.exp(res['te_re'] + t_crit * res['se_hk']):.3f})")
print("\nSame 16 trials, same outcome. One model says magnesium does nothing;")
print("the other says it roughly halves mortality.")
print("\n(These differ slightly from the R solution, which uses Mantel-Haenszel for")
print("the fixed-effect estimate and REML for tau-squared. See the chapter note on")
print("reconciling the two.)")

# -----------------------------------------------------------------------------
# (b) Where did the weight go?
# -----------------------------------------------------------------------------
weights = pd.DataFrame({
    "trial": STUDY,
    "n": n1 + n2,
    "fixed_pct": 100 * res["w_fe"] / res["w_fe"].sum(),
    "random_pct": 100 * res["w_re"] / res["w_re"].sum(),
}).sort_values("n", ascending=False)

print("\n=== (b) Percentage weight under each model ===")
print(weights.round(1).to_string(index=False))

isis = weights.iloc[0]
print(f"\nISIS-4 holds {100 * isis['n'] / weights['n'].sum():.0f}% of all patients, "
      f"{isis['fixed_pct']:.1f}% of the fixed-effect weight,")
print(f"and only {isis['random_pct']:.1f}% of the random-effects weight.")
print(f"\nWhy: random-effects weights are 1 / (within-study variance + tau^2).")
print(f"ISIS-4's within-study variance is tiny, so adding tau^2 = {res['tau2']:.3f}")
print("swamps it and its weight collapses. A small trial's variance is already")
print("large, so the same addition barely changes it. The effect is to level the")
print("weights -- which hands the analysis to the 13 small trials.")

# -----------------------------------------------------------------------------
# (c) Heterogeneity
# -----------------------------------------------------------------------------
pi_lo, pi_hi = prediction_interval(res)
print("\n=== (c) Heterogeneity ===")
print(f"tau^2 = {res['tau2']:.3f} (tau = {np.sqrt(res['tau2']):.3f} on the log-RR scale)")
print(f"I^2 = {100 * res['I2']:.1f}%   Q = {res['Q']:.1f}, p = {res['Q_p']:.4f}")
print(f"Prediction interval: {pi_lo:.3f} to {pi_hi:.3f}")
print("\nThe PREDICTION INTERVAL answers 'how much does the effect vary between")
print("settings'. I^2 is a ratio -- the share of observed scatter that is real")
print("rather than sampling noise -- and would rise if you simply ran the same")
print("trials with more patients each. Note that the prediction interval includes")
print("1 while the confidence interval does not.")

# -----------------------------------------------------------------------------
# (d) What you would tell a guideline committee
# -----------------------------------------------------------------------------
print("\n=== (d) Two one-sentence summaries ===")
print('(i)  Fixed effect only: "Pooling 62,607 patients across 16 randomised')
print('     trials, intravenous magnesium had no effect on mortality after')
print('     myocardial infarction."')
print('(ii) Random effects only: "Pooling 16 randomised trials, intravenous')
print('     magnesium reduced mortality after myocardial infarction by almost')
print('     half."')
print("\nBoth are defensible from the same data, which is why you must report both")
print("models when they disagree, and why the prediction interval and the funnel")
print("plot are not optional extras.")
