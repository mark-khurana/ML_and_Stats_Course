# =============================================================================
# Chapter 17c, Exercise 4: Sensitivity Analysis - the E-value (Conceptual + code)
# How strong must unmeasured M-Y confounding be to explain away the indirect effect?
# =============================================================================
# The chapter obtains this "free" via R's CMAverse::cmsens(object, sens = "uc").
# Here we compute the E-value with the closed-form formula (VanderWeele & Ding
# 2017), which is exactly what cmsens() reports for an effect on the RR scale.

import numpy as np

np.random.seed(42)


# -----------------------------------------------------------------------------
# E-value formula (risk-ratio scale)
# -----------------------------------------------------------------------------
# For a point estimate RR (with RR >= 1):
#     E-value = RR + sqrt(RR * (RR - 1))
# It is the minimum strength of association (on the RR scale) that an unmeasured
# confounder would need with BOTH the mediator and the outcome, above and beyond
# measured covariates, to fully explain away the observed indirect effect.
# For a protective effect (RR < 1), first transform: RR <- 1 / RR.
def evalue(rr):
    if rr < 1:
        rr = 1 / rr                 # put protective effects on the >=1 scale
    return rr + np.sqrt(rr * (rr - 1))


# -----------------------------------------------------------------------------
# (a) Reason about the strength needed, with a worked calculation
# -----------------------------------------------------------------------------
# Suppose the indirect (mediated) effect from Exercise 1, re-expressed on a
# risk-ratio scale for a binary version of the outcome, corresponds to an
# indirect-effect RR of about 1.50. We compute its E-value:
rr_indirect = 1.50
ev = evalue(rr_indirect)

print("=== Exercise 4: E-value for the indirect effect ===\n")
print(f"Indirect-effect risk ratio (example): RR = {rr_indirect:.2f}")
print(f"E-value = RR + sqrt(RR*(RR-1)) = {ev:.3f}\n")
print("Interpretation: an unmeasured mediator-outcome confounder would need to be")
print("associated with BOTH the mediator and the outcome by a risk ratio of at")
print(f"least {ev:.2f} each (beyond measured covariates) to reduce the indirect")
print("effect to the null. Weaker confounding could shift but not erase it.\n")

# For reference, the E-value at a range of indirect-effect RRs:
print("E-value as the indirect effect grows:")
for rr in (1.1, 1.3, 1.5, 2.0, 3.0):
    print(f"  RR = {rr:.1f}  ->  E-value = {evalue(rr):.2f}")

# -----------------------------------------------------------------------------
# (b) How to temper the conclusion if the E-value were 1.3
# -----------------------------------------------------------------------------
# An E-value of 1.3 is SMALL. It says a fairly modest unmeasured mediator-outcome
# confounder - one associated with both the mediator and the outcome by only
# about a 1.3-fold risk ratio each - would be enough to explain away the entire
# indirect effect. Confounders of that magnitude are common and plausible in
# observational health data (e.g. an unmeasured lifestyle or comorbidity factor).
# So we would report the mechanistic claim cautiously: "the data are CONSISTENT
# with partial mediation, but the indirect effect is NOT robust - a mild
# unmeasured confounder of the mediator-outcome relationship could account for
# it." We would avoid strong statements that the treatment 'works through' the
# mediator, call for measuring/adjusting more M-Y confounders, and treat the
# proportion mediated as fragile rather than established.
