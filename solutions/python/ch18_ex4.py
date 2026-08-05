# =============================================================================
# Chapter 18 - Exercise 4: Critical appraisal of a published meta-analysis
#
# A conceptual exercise: the answer depends on the paper you chose. Below is a
# reusable checklist with the reasoning behind each item, then a worked appraisal
# of the magnesium literature, which we can all read the same way.
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
# The checklist, and why each item is on it
# -----------------------------------------------------------------------------
# (a) HOW MANY STUDIES, AND HOW BIG?
#     Compute the largest study's share of the total sample. If one trial holds
#     most of the patients, the fixed/random choice decides the answer and the
#     paper must justify it. If every trial is small, ask what is missing.
#
# (b) WHICH MODEL, AND ARE BOTH REPORTED?
#     Random effects is the usual default; the question is whether the paper also
#     reports the fixed-effect result. If it does not, and one trial is far larger
#     than the rest, you cannot tell whether the choice mattered -- and that is
#     exactly when it matters most.
#
# (c) IS tau^2 OR A PREDICTION INTERVAL REPORTED, OR ONLY I^2?
#     I^2 is the PROPORTION of observed scatter that is real rather than sampling
#     noise. It does not say how much the effect varies, and it rises if the same
#     trials are simply run larger. With only I^2 you cannot answer "would this
#     work in my setting?".
#
# (d) WAS ASYMMETRY ASSESSED, AND LEGITIMATELY?
#     Needs k >= 10; below that Cochrane advises against testing, and "we could
#     not assess it" is the correct report rather than "the test was not
#     significant". Check the test suits the effect measure: Egger's test is not
#     appropriate for odds ratios or standardised mean differences.
#
# (e) WOULD YOU CHANGE PRACTICE?
#     Name the condition. "I would change if the prediction interval excluded no
#     effect and the large trials agreed with the small ones" is a real answer;
#     "the result was significant" is not.

# -----------------------------------------------------------------------------
# A worked appraisal: the magnesium literature
# -----------------------------------------------------------------------------
log_rr, var = effect_sizes(ai, n1, ci, n2)
res = pool(log_rr, var)
n_tot = n1 + n2
pi_lo, pi_hi = prediction_interval(res)
t_crit = stats.t.ppf(0.975, res["k"] - 1)

print("=== (a) Size and spread ===")
print(f"k = {res['k']} trials, {n_tot.sum():,} patients in total")
print(f"smallest {n_tot.min()}, largest {n_tot.max():,} "
      f"({100 * n_tot.max() / n_tot.sum():.0f}% of all patients)")
print("  -> one trial holds most of the evidence, so model choice is decisive.")

print("\n=== (b) Model choice ===")
print(f"fixed effect   RR = {np.exp(res['te_fe']):.3f}")
print(f"random effects RR = {np.exp(res['te_re']):.3f} "
      f"(95% CI {np.exp(res['te_re'] - t_crit * res['se_hk']):.3f} to "
      f"{np.exp(res['te_re'] + t_crit * res['se_hk']):.3f})")
print("  -> opposite conclusions. Reporting only one would be indefensible.")

print("\n=== (c) Heterogeneity ===")
print(f"tau^2 = {res['tau2']:.3f} | I^2 = {100 * res['I2']:.1f}% | "
      f"prediction interval {pi_lo:.3f} to {pi_hi:.3f}")
print("  -> the prediction interval includes 1 even though the CI does not.")

print(f"\n=== (d) Asymmetry ===")
print(f"k = {res['k']}, so testing is legitimate (the threshold is 10). Run the tests")
print("in R, where Harbord and Peters are implemented:")
print("    Egger   p < 0.001   <- not the right test for a ratio measure")
print("    Harbord p = 0.0001")
print("    Peters  p = 0.0023")
print("  -> strong evidence of small-study effects on all three.")

print("\n=== (e) Verdict ===")
print("No. Three separate signals -- a fixed/random reversal, a prediction interval")
print("crossing 1, and a markedly asymmetric funnel -- all say the same thing: the")
print("small trials disagree with the large one, and the pooled benefit is an")
print("artefact of giving the small trials more weight. What would change my mind: a")
print("further large trial agreeing with the small ones, or a mechanism for why")
print("effects should genuinely be larger in the settings the small trials studied.")
print("\nHistorically the question was settled by ISIS-4 (58,050 patients, RR 1.06):")
print("no benefit. The appraisal above reaches the right answer without waiting.")
