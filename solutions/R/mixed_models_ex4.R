# =============================================================================
# Chapter 6b, Exercise 4: GLMM and the marginal/conditional distinction
# Conceptual: LMM vs GLMM, why GEE and mixed-model estimates differ, and which
# to report for a hospital-specific question.
# =============================================================================

# This is a conceptual exercise. The answer is written as structured comments.
# Setting: a binary outcome (30-day readmission) for patients clustered within
# hospitals.

# -----------------------------------------------------------------------------
# (a) LMM or GLMM? Why?
# -----------------------------------------------------------------------------
#   Fit a GLMM (a generalised linear mixed model -- here a logistic mixed
#   model). The outcome is BINARY (readmitted / not), not continuous and
#   normally distributed, so a linear mixed model (LMM) is inappropriate: an
#   LMM assumes a continuous, roughly normal outcome with constant variance and
#   can predict probabilities outside 0-1. A GLMM keeps the random-effects idea
#   (e.g. a random intercept for hospital) but uses the logit link and a
#   binomial error, exactly as ordinary logistic regression extends linear
#   regression. In R: glmer(readmit ~ intervention + (1 | hospital),
#   family = binomial).

# -----------------------------------------------------------------------------
# (b) Why do the population-average (GEE) and subject-specific (mixed) estimates
#     differ for a logistic model, and why is the GEE coefficient smaller?
# -----------------------------------------------------------------------------
#   The two answer different questions:
#     * The mixed model is CONDITIONAL / SUBJECT-SPECIFIC. Its coefficient is
#       the effect of the intervention FOR A GIVEN HOSPITAL (holding that
#       hospital's random intercept fixed).
#     * GEE is MARGINAL / POPULATION-AVERAGE. Its coefficient is the effect
#       AVERAGED OVER the whole population of hospitals.
#   For a linear (identity-link) model the two coincide. For a logistic model
#   they genuinely differ because the logit link is NON-LINEAR: the average of
#   the individual (hospital-specific) log-odds effects is not equal to the
#   effect on the population-averaged probability. Averaging a curved (S-shaped)
#   relationship over the spread of hospital random effects flattens it, pulling
#   the population-average effect TOWARD THE NULL. So the marginal (GEE)
#   coefficient is smaller in magnitude than the conditional (mixed-model) one.
#   (This is the well-known non-collapsibility of the odds ratio.) The larger
#   the between-hospital variance, the bigger the gap.
#   Plain language: "the average of the individual effects is not the same as
#   the effect on the average patient", and the logit link's curvature shrinks
#   the averaged effect toward zero.

# -----------------------------------------------------------------------------
# (c) Which would you report if a hospital manager asks: "what happens to THIS
#     hospital's readmission rate if we adopt the intervention?"
# -----------------------------------------------------------------------------
#   Report the SUBJECT-SPECIFIC / CONDITIONAL estimate from the MIXED MODEL
#   (GLMM). The manager is asking a cluster-specific question -- the change for
#   one particular hospital -- which is exactly what the conditional coefficient
#   describes. The marginal (GEE) estimate answers a different question: the
#   average shift in readmission across the whole population of hospitals, which
#   is what you would report to a health-system regulator planning a
#   population-wide rollout. General good practice when reporting a mixed model:
#   state the fixed-effect estimates with confidence intervals (on the odds-
#   ratio scale for a logistic GLMM), report the random-effects variance /
#   between-hospital SD (or the ICC) to convey how much hospitals differ, and be
#   explicit that the estimates are conditional (subject-specific).

cat("Exercise 4 is conceptual -- see the commented answers above.\n")
