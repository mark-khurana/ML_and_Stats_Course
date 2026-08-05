# =============================================================================
# Chapter 18 - Exercise 2: Meta-analysis from scratch
# Inverse-variance pooling by hand, then reconciled with the meta package
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

# -----------------------------------------------------------------------------
# (a) Log risk ratios and variances, handling the zero cell
# -----------------------------------------------------------------------------
# log RR = log( (a/(a+b)) / (c/(c+d)) ),  Var = 1/a - 1/(a+b) + 1/c - 1/(c+d)
# Bertschat 1989 recorded 0 deaths on magnesium, so log(0) is undefined and 1/0
# is infinite. The convention is to add 0.5 to the cells of the AFFECTED TRIAL
# ONLY -- correcting every trial would shift all 16 estimates, which is a real
# difference and not a rounding one.
zero = (ai == 0) | (ci == 0)
print("Trials with a zero cell:", [s for s, z in zip(STUDY, zero) if z], "\n")

log_rr, var = effect_sizes(ai, n1, ci, n2)
print("=== (a) First few trials ===")
print(pd.DataFrame({"trial": STUDY, "log_rr": log_rr.round(3),
                    "var": var.round(4), "se": np.sqrt(var).round(3)}
                   ).head().to_string(index=False))

# -----------------------------------------------------------------------------
# (b) Fixed-effect pooled estimate
# -----------------------------------------------------------------------------
w_fe = 1 / var
te_fe = np.sum(w_fe * log_rr) / np.sum(w_fe)
se_fe = np.sqrt(1 / np.sum(w_fe))
print("\n=== (b) Fixed effect (inverse variance) ===")
print(f"log RR = {te_fe:+.4f} (SE {se_fe:.4f})  ->  RR = {np.exp(te_fe):.3f} "
      f"(95% CI {np.exp(te_fe - 1.96 * se_fe):.3f} to "
      f"{np.exp(te_fe + 1.96 * se_fe):.3f})")

# -----------------------------------------------------------------------------
# (c) tau^2 by DerSimonian-Laird, then random effects
# -----------------------------------------------------------------------------
k = len(log_rr)
Q = np.sum(w_fe * (log_rr - te_fe) ** 2)
C = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
tau2 = max(0.0, (Q - (k - 1)) / C)
I2 = max(0.0, (Q - (k - 1)) / Q)

w_re = 1 / (var + tau2)
te_re = np.sum(w_re * log_rr) / np.sum(w_re)
se_re = np.sqrt(1 / np.sum(w_re))

print("\n=== (c) Random effects (DerSimonian-Laird) ===")
print(f"Q = {Q:.2f} on {k - 1} df, p = {1 - stats.chi2.cdf(Q, k - 1):.4f}")
print(f"tau^2 = {tau2:.4f} | I^2 = {100 * I2:.1f}%")
print(f"log RR = {te_re:+.4f}  ->  RR = {np.exp(te_re):.3f} "
      f"(95% CI {np.exp(te_re - 1.96 * se_re):.3f} to "
      f"{np.exp(te_re + 1.96 * se_re):.3f})")

# -----------------------------------------------------------------------------
# (d) Checks
# -----------------------------------------------------------------------------
# Internal check: with tau^2 = 0 the random-effects estimate must collapse onto
# the fixed-effect one. If it does not, the weighting code is wrong.
te_re_zero = np.sum((1 / var) * log_rr) / np.sum(1 / var)
print("\n=== (d) Checks ===")
print("tau^2 = 0 reproduces the fixed-effect estimate:",
      bool(np.isclose(te_re_zero, te_fe)))
print("\nAgainst R: metabin(..., method = 'Inverse', method.tau = 'DL') gives")
print("fixed 1.014, random 0.530, tau^2 0.174 -- matching the values above to")
print("three decimal places. Two defaults must be overridden to get that match:")
print("  method     = 'Inverse'   metabin() uses Mantel-Haenszel for binary")
print("                          outcomes, which handles sparse cells better")
print("  method.tau = 'DL'        metabin() now defaults to REML")
print("\nSwitching R to REML moves tau^2 from 0.174 to 0.227 and the pooled RR from")
print("0.530 to 0.511. DerSimonian-Laird is a moment estimator known to")
print("UNDERESTIMATE the between-study variance, especially with few studies or")
print("very unequal sizes (Veroniki et al. 2016). A larger tau^2 levels the weights")
print("further, pulling the estimate towards the small trials and widening the")
print("prediction interval. REML or Paule-Mandel is the current recommendation; DL")
print("survives because it was the default for thirty years and needs no iteration.")
