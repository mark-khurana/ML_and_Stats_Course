# =============================================================================
# Chapter 18 - Exercise 3: Detecting the problem before the mega-trial
# What the magnesium evidence looked like in 1993, before ISIS-4 reported

# Libraries -------------------------------------------------------------------
# pip install numpy pandas scipy matplotlib statsmodels
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

# The 15 trials available before ISIS-4 reported in 1995
pre = np.array([s != "ISIS-4 1995" for s in STUDY])
print(f"Trials available pre-ISIS-4: {pre.sum()} | "
      f"total patients: {(n1[pre] + n2[pre]).sum():,}")
print(f"Largest of them: {np.array(STUDY)[pre][np.argmax((n1 + n2)[pre])]} "
      f"with {(n1 + n2)[pre].max():,} patients\n")

log_rr_all, var_all = effect_sizes(ai, n1, ci, n2)
log_rr, var = log_rr_all[pre], var_all[pre]
res = pool(log_rr, var)
t_crit = stats.t.ppf(0.975, res["k"] - 1)
pi_lo, pi_hi = prediction_interval(res)

# -----------------------------------------------------------------------------
# (a) What you would have concluded in 1993
# -----------------------------------------------------------------------------
print("=== (a) The 15 trials, random effects ===")
print(f"RR = {np.exp(res['te_re']):.3f} "
      f"(95% CI {np.exp(res['te_re'] - t_crit * res['se_hk']):.3f} to "
      f"{np.exp(res['te_re'] + t_crit * res['se_hk']):.3f})")
print(f"tau^2 = {res['tau2']:.3f} | I^2 = {100 * res['I2']:.1f}%")
print(f"Prediction interval: {pi_lo:.3f} to {pi_hi:.3f}")
print(f"Fixed effect for comparison: RR = {np.exp(res['te_fe']):.3f}")
print("\nYou would have concluded that magnesium roughly halves mortality, and the")
print("fixed and random models AGREE -- because without ISIS-4 there is no dominant")
print("large trial to disagree with the small ones. That agreement is falsely")
print("reassuring.")

# -----------------------------------------------------------------------------
# (b) Funnel plot on the 15 trials
# -----------------------------------------------------------------------------
se = np.sqrt(var)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
grid = np.linspace(0.001, se.max() * 1.05, 100)
for z, shade in [(1.96, "0.85"), (2.58, "0.92")]:
    ax.fill_betweenx(grid, res["te_fe"] - z * grid, res["te_fe"] + z * grid,
                     color=shade, zorder=0)
ax.scatter(log_rr, se, s=30, color="#2c3e50", zorder=3)
ax.axvline(res["te_fe"], color="grey", ls="--", lw=1.2, zorder=2)
ax.axvline(0, color="#b02a2a", lw=1.0, zorder=2)
ax.invert_yaxis()
ax.set_xlabel("log risk ratio")
ax.set_ylabel("Standard error (precision increases upwards)")
ax.set_title("Magnesium trials available before ISIS-4 (k = 15)")
plt.tight_layout()
plt.show()

print("\n=== (b) Funnel plot ===")
print("Yes -- the asymmetry is visible without the mega-trial. The lower LEFT")
print("(small trials showing benefit) is populated; the lower RIGHT (small trials")
print("showing no benefit) is close to empty. The warning sign was available years")
print("before ISIS-4 reported.")

# -----------------------------------------------------------------------------
# (c) Egger's test, and why it is not the right one here
# -----------------------------------------------------------------------------
# Egger's test is a weighted regression of the effect estimate on its standard
# error; the intercept is what the test looks at.
X = sm.add_constant(se)
egger = sm.WLS(log_rr, X, weights=1 / var).fit()
print("\n=== (c) Egger's test ===")
print(f"slope on SE = {egger.params[1]:+.3f}, p = {egger.pvalues[1]:.4f}")
print("\nFor a ratio measure the standard error is mathematically linked to the size")
print("of the effect, which manufactures asymmetry. Cochrane therefore recommends")
print("the Harbord or Peters tests for binary outcomes. Neither has a maintained")
print("Python implementation, so run them in R:")
print("    metabias(m_pre, method.bias = 'Harbord')   # p = 0.017")
print("    metabias(m_pre, method.bias = 'Peters')    # p = 0.043")
print("All three agree here, which is the easy case. When they disagree, believe")
print("the one appropriate to your effect measure, not the smallest p-value.")

# -----------------------------------------------------------------------------
# (d) The limitations paragraph
# -----------------------------------------------------------------------------
print("\n=== (d) A two-sentence limitations paragraph ===")
print(f'"Fifteen trials totalling only {(n1[pre] + n2[pre]).sum():,} patients (the largest')
print(f' randomising {(n1 + n2)[pre].max():,}) suggest that intravenous magnesium substantially')
print(f' reduces mortality after myocardial infarction (RR {np.exp(res["te_re"]):.2f}); however the')
print(' funnel plot is markedly asymmetric and the effect size falls as trial size')
print(' rises, so we cannot exclude that small trials with null results are missing')
print(f' from the literature. The prediction interval spans {pi_lo:.2f} to {pi_hi:.2f} and')
print(' therefore includes no effect, so a large pragmatic trial is needed before')
print(' magnesium is adopted into routine practice."')
print("\nThat trial was ISIS-4: 58,050 patients, RR 1.06, no benefit.")
print(f"\nOne last thing worth noticing: I^2 was only {100 * res['I2']:.1f}% -- 'low")
print("heterogeneity' by the conventional bands -- and the two models agreed. Both of")
print("the reassurances people usually look for were present. The two that were NOT")
print("reassuring were the prediction interval crossing 1 and the asymmetric funnel.")
