# =============================================================================
# Chapter 17 - Exercise 4: G-computation and interactions
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


# --- The dataset from Exercise 2 --------------------------------------------
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

TRUE_ATE_RD = expit(lp_untreated - 0.8).mean() - expit(lp_untreated).mean()
print(f"TRUE ATE risk difference: {TRUE_ATE_RD:+.4f}\n")

INTERACTION = "death_1yr ~ treatment * (age + creatinine + heart_failure + prior_mi)"
ADDITIVE = "death_1yr ~ treatment + age + creatinine + heart_failure + prior_mi"

# =============================================================================
# (a) G-computation, spelled out
# =============================================================================
# Step 1: ONE outcome model, including treatment-covariate interactions.
out_model = smf.glm(INTERACTION, data=df, family=sm.families.Binomial()).fit(disp=0)

# Step 2: predict EVERY patient twice -- once as if treated, once as if not.
#         Their real covariates are left untouched; only treatment changes.
p1 = out_model.predict(df.assign(treatment=1))
p0 = out_model.predict(df.assign(treatment=0))

# Step 3: average each set and contrast.
gcomp_rd = p1.mean() - p0.mean()

print("--- (a) G-computation ---")
print(f"Average predicted risk if EVERYONE treated  : {p1.mean():.4f}")
print(f"Average predicted risk if NOBODY treated    : {p0.mean():.4f}")
print(f"Risk difference (their contrast)            : {gcomp_rd:+.4f}"
      f"   [truth {TRUE_ATE_RD:+.4f}]")
print("""
That is the whole method: fit once, predict twice, average, subtract. Note that
we never read a coefficient -- which is exactly the point, because a single
coefficient on the log-odds scale is not the marginal risk difference.""")

# =============================================================================
# (b) Confidence interval: bootstrap the WHOLE procedure
# =============================================================================
# The uncertainty does not come out of the outcome model directly, because we
# fit, predict, average, and contrast. So resample patients and repeat all of it.
def gcomp_once(data, formula=INTERACTION):
    m = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(disp=0)
    return (m.predict(data.assign(treatment=1)).mean()
            - m.predict(data.assign(treatment=0)).mean())


boot_rng = np.random.default_rng(1)
boot = np.array([
    gcomp_once(df.iloc[boot_rng.integers(0, len(df), len(df))])
    for _ in range(500)
])
lo, hi = np.percentile(boot, [2.5, 97.5])

print("\n--- (b) Bootstrap confidence interval ---")
print(f"G-computation risk difference: {gcomp_rd:+.4f} "
      f"(95% percentile CI {lo:+.4f}, {hi:+.4f})")
print(f"The interval {'DOES' if lo <= TRUE_ATE_RD <= hi else 'does NOT'} "
      "contain the truth.")
print(f"Bootstrap SE: {boot.std(ddof=1):.4f}")

# =============================================================================
# (c) What the interaction terms allow
# =============================================================================
print("""
--- (c) What do the interactions do? ---
Writing `treatment * (age + creatinine + ...)` rather than
`treatment + age + creatinine + ...` allows the drug's effect to be DIFFERENT
for different kinds of patient. Without the interactions, the model is forced to
say 'the beta-blocker shifts the log-odds of death by the same amount for a fit
55-year-old and a frail 85-year-old with heart failure'. With them, the model can
say 'it helps the sicker patients more' (or less) and let the data decide.

That matters because the ATE is an AVERAGE of individual effects. If the effect
genuinely varies, the average we want is the average of the patient-specific
effects across our actual patient mix -- which is precisely what predicting each
patient twice and then averaging gives us. A single coefficient cannot represent
that.""")

individual_rd = (p1 - p0).to_numpy()
print(f"\nIndividual risk differences implied by the model range from "
      f"{individual_rd.min():+.3f} to {individual_rd.max():+.3f}")
print(f"with a mean of {individual_rd.mean():+.3f} (the ATE) and an SD of "
      f"{individual_rd.std(ddof=1):.3f}.")
print("""Even with no true interaction, the effect on the RISK scale varies across
patients, because a constant shift in log-odds produces a bigger change in risk
for a patient near 50% risk than for one near 2%.""")
if individual_rd.max() > 0:
    print("""
Note also that the range creeps slightly ABOVE zero for a few patients, implying
the drug harms them. It does not -- we built it to be protective for everyone.
That is the interaction model overfitting a handful of sparsely populated corners
of covariate space, and it is a good reminder not to read individual predicted
effects as real subgroup findings.""")

# =============================================================================
# (d) Drop the interactions -- does it matter?
# =============================================================================
add_model = smf.glm(ADDITIVE, data=df, family=sm.families.Binomial()).fit(disp=0)
gcomp_add = (add_model.predict(df.assign(treatment=1)).mean()
             - add_model.predict(df.assign(treatment=0)).mean())

print("\n--- (d) With and without interactions ---")
print(f"With interactions   : {gcomp_rd:+.4f}")
print(f"Without interactions: {gcomp_add:+.4f}")
print(f"Truth               : {TRUE_ATE_RD:+.4f}")
print(f"\nRaw `treatment` coefficient from the additive model: "
      f"{add_model.params['treatment']:+.3f}")
print("(the data-generating value was -0.800, on the log-odds scale)")

print("""
Why the two g-computation estimates barely differ here: we SIMULATED the data
with no treatment-covariate interaction, so the extra terms have nothing to find
and only add a little noise. In real data you do not know that, so including them
is the safer default -- the cost is a few degrees of freedom, and the benefit is
not silently assuming a constant effect.

Why standardisation still recovers a sensible average even if you omit
interactions that DO exist: g-computation averages predicted RISKS over the real
distribution of patient characteristics. Even a misspecified model that gets the
average risk in each arm roughly right will get the contrast roughly right. What
you lose is the ability to say anything about WHICH patients benefit -- and if
the misspecification is severe enough to distort the average risks themselves,
the estimate does become biased. That is exactly the vulnerability doubly robust
estimators (AIPW, TMLE) are designed to insure against.""")
