# =============================================================================
# Chapter 17c, Exercise 3: Assumptions and a DAG (Conceptual)
# Mediation triangle for exercise -> weight loss -> blood pressure.
# =============================================================================
# Conceptual exercise; answers as structured comments. The script prints the
# mediation triangle as text so the reasoning is anchored to a concrete DAG.

print("Mediation triangle for the clinical question:\n")
print("      Exercise (X) ------------------> Blood pressure (Y)   [direct path]")
print("           |                               ^")
print("           |                               |")
print("           v                               |")
print("      Weight loss (M) -------------------- +               [indirect path]\n")
print("  Indirect: Exercise -> Weight loss -> Blood pressure")
print("  Direct:   Exercise -> Blood pressure (via mechanisms other than weight)\n")

# -----------------------------------------------------------------------------
# (a) One exposure-mediator confounder and one mediator-outcome confounder
# -----------------------------------------------------------------------------
# Exposure(exercise)-mediator(weight loss) confounder:
#   BASELINE DIET / caloric intake. People who eat a healthier, lower-calorie
#   diet tend to both exercise more AND lose more weight, creating a spurious
#   exercise-weight-loss association not due to exercise itself.
#   (Other valid answers: baseline motivation, socioeconomic status, age.)
#
# Mediator(weight loss)-outcome(blood pressure) confounder:
#   DIETARY SODIUM INTAKE. High salt intake both impedes weight loss (fluid
#   retention, dietary pattern) AND directly raises blood pressure, confounding
#   the weight-loss -> blood-pressure arrow.
#   (Other valid answers: alcohol intake, antihypertensive medication use.)

# -----------------------------------------------------------------------------
# (b) Hardest no-unmeasured-confounding assumption, and why
# -----------------------------------------------------------------------------
# The four assumptions are: no unmeasured confounding of (1) X->Y, (2) X->M,
# (3) M->Y, and (4) no exposure-affected M->Y confounder.
#
# HARDEST here: assumption (3), no unmeasured mediator(weight loss)-outcome
# (blood pressure) confounding. Even in a randomized exercise trial, weight loss
# is NOT randomized - it arises naturally within each arm. Whatever drives how
# much weight a person loses (diet, sodium, alcohol, adherence, metabolic
# health) also tends to affect blood pressure directly. Randomizing exercise
# fixes assumptions (1) and (2) but does nothing for the M->Y arrow, so the
# indirect effect leans on adjusting for confounders we can rarely measure fully.

# -----------------------------------------------------------------------------
# (c) Why an exposure-CAUSED mediator-outcome confounder threatens the NIE
# -----------------------------------------------------------------------------
# Suppose exercise causes better sleep, and sleep both further influences weight
# loss and directly lowers blood pressure. Then sleep is a mediator-outcome
# confounder that is ITSELF caused by the exposure (assumption 4 violated).
#   - It sits on the causal path, so we cannot simply "adjust" for it: adjusting
#     for a variable on the exercise -> ... -> blood pressure path BLOCKS part of
#     exercise's real effect (over-adjustment / mediator-of-a-mediator problem).
#   - Yet NOT adjusting leaves the weight-loss -> blood-pressure arrow confounded.
# There is no way to condition our way out: the natural indirect effect is no
# longer identified by standard regression, because any choice biases it in one
# direction or the other. Handling it requires more advanced methods
# (interventional/randomized-interventional effects, or g-methods), not the
# simple product-of-coefficients.
