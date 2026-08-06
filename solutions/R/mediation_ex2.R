# =============================================================================
# Chapter 17c, Exercise 2: Why Baron and Kenny Can Mislead (Conceptual)
# Limits of the coefficient-shrinkage / product-of-coefficients approach.
# =============================================================================
# This exercise is conceptual; the answers are written as structured comments.
# A tiny simulation illustrates part (a) so the point is concrete rather than
# asserted. (The chapter shows the CMAverse equivalent that handles these cases.)

set.seed(42)

# -----------------------------------------------------------------------------
# (a) Coefficient shrinkage misleads under exposure-mediator INTERACTION
# -----------------------------------------------------------------------------
# The Baron-Kenny logic: fit Y ~ X (total effect c), then Y ~ X + M and read the
# shrunk exposure coefficient (c') as the "direct" effect, with c - c' the
# "mediated" effect. This assumes ONE number describes the direct effect for
# everyone.
#
# When exposure and mediator INTERACT, the exposure's direct effect is not a
# single number: it depends on the value of the mediator. The true outcome model
# is  Y = c'*X + b*M + d*(X*M) + e , so the effect of switching X on is c' + d*M,
# which changes from patient to patient. A model that omits the X:M term forces a
# single average coefficient, so:
#   - the reported "direct effect" is an ill-defined blend that matches no
#     specific patient, and
#   - the natural direct and indirect effects (which are properly defined even
#     WITH interaction) are not recovered by the simple shrinkage.
# The modern causal estimators keep the X:M term and define NDE/NIE as contrasts
# of nested potential outcomes, which stay meaningful under interaction.
#
# Illustration: data WITH an exposure-mediator interaction.

n <- 5000
X <- rbinom(n, 1, 0.5)
M <- 1.5 * X + rnorm(n)                       # exposure moves the mediator
# Outcome with a strong X:M interaction (d = 0.8)
Y <- 1.0 * X + 0.5 * M + 0.8 * (X * M) + rnorm(n)

# Naive Baron-Kenny: no interaction term
naive <- lm(Y ~ X + M)
cat("Naive Y ~ X + M (ignores interaction):\n")
cat(sprintf("  'direct' exposure coefficient = %.3f  (a single blended number)\n",
            coef(naive)["X"]))

# Correct model: include the interaction
correct <- lm(Y ~ X * M)
cat("\nCorrect Y ~ X * M (models interaction):\n")
cat(sprintf("  exposure main effect  = %.3f\n", coef(correct)["X"]))
cat(sprintf("  X:M interaction       = %.3f  (direct effect depends on M!)\n",
            coef(correct)["X:M"]))
cat("\n=> The naive single coefficient hides that the direct effect grows with M.\n")

# -----------------------------------------------------------------------------
# (b) Product-of-coefficients is unreliable for a BINARY outcome (logistic)
# -----------------------------------------------------------------------------
# For a linear outcome, effects add on the same (natural) scale, so
# NIE = a * b and NDE = c' decompose the total effect cleanly.
#
# For a binary outcome fitted by logistic regression, the coefficients live on
# the LOG-ODDS scale, and log-odds are NON-COLLAPSIBLE and NON-LINEAR:
#   1. Odds ratios do not add or multiply to reproduce the total-effect odds
#      ratio, so "a * b" and "c'" no longer sum to the total effect.
#   2. The exposure coefficient changes when the mediator is added even with NO
#      mediation and NO confounding, purely because of non-collapsibility of the
#      odds ratio - so the shrinkage is not a mediated effect at all.
#   3. The estimand implied by the product depends on the (arbitrary) outcome
#      scale, so the "proportion mediated" is not stable or interpretable.
# The fix is the causal, counterfactual definition of NDE/NIE (e.g. via the
# Valeri-VanderWeele regression formulas or simulation-based estimators), which
# are defined on the probability/risk scale and remain valid for logistic,
# Poisson, Cox, and AFT outcome models. This is exactly what CMAverse and
# regmedint implement.
