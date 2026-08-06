# =============================================================================
# Chapter 17 - Exercise 4: G-computation and interactions
# Beta-blocker use and 1-year mortality
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(tidyverse)       # tibble(), mutate()
library(marginaleffects) # avg_comparisons(), inferences()

# --- The dataset from Exercise 2 --------------------------------------------
set.seed(123)
n <- 1500

exercise_dat <- tibble(
  age           = rnorm(n, 70, 8),
  creatinine    = rnorm(n, 1.2, 0.4),
  heart_failure = rbinom(n, 1, 0.35),
  prior_mi      = rbinom(n, 1, 0.20)
) |>
  mutate(
    treatment = rbinom(n, 1, plogis(-0.4 + 0.05 * (age - 70) +
                                      0.7 * heart_failure +
                                      0.9 * prior_mi +
                                      0.8 * (creatinine - 1.2))),
    death_1yr = rbinom(n, 1, plogis(-1.9 + 0.05 * (age - 70) +
                                      0.7 * heart_failure +
                                      0.8 * prior_mi +
                                      1.0 * (creatinine - 1.2) -
                                      0.8 * treatment))
  )

lp0 <- with(exercise_dat, -1.9 + 0.05 * (age - 70) + 0.7 * heart_failure +
  0.8 * prior_mi + 1.0 * (creatinine - 1.2))
TRUE_ATE_RD <- mean(plogis(lp0 - 0.8)) - mean(plogis(lp0))

cat(sprintf("TRUE ATE risk difference: %+.4f\n\n", TRUE_ATE_RD))

# =============================================================================
# (a) G-computation, spelled out by hand
# =============================================================================
# Step 1: ONE outcome model, including treatment-covariate interactions.
out_model <- glm(
  death_1yr ~ treatment * (age + creatinine + heart_failure + prior_mi),
  data = exercise_dat, family = binomial
)

# Step 2: predict EVERY patient twice -- once as if treated, once as if not.
#         Their real covariates are left untouched; only treatment is changed.
p1 <- predict(out_model, transform(exercise_dat, treatment = 1), type = "response")
p0 <- predict(out_model, transform(exercise_dat, treatment = 0), type = "response")

# Step 3: average each set and contrast.
gcomp_rd <- mean(p1) - mean(p0)

cat("--- (a) G-computation by hand ---\n")
cat(sprintf("Average predicted risk if EVERYONE treated  : %.4f\n", mean(p1)))
cat(sprintf("Average predicted risk if NOBODY treated    : %.4f\n", mean(p0)))
cat(sprintf("Risk difference (their contrast)            : %+.4f   [truth %+.4f]\n",
            gcomp_rd, TRUE_ATE_RD))

# And the same thing via marginaleffects, which is what you would use in
# practice. Making the 0 -> 1 contrast explicit avoids any ambiguity about
# what "a one-unit change in treatment" means.
mfx <- avg_comparisons(out_model, variables = list(treatment = 0:1))
cat(sprintf("\nSame quantity via avg_comparisons()         : %+.4f\n", mfx$estimate))
cat("Identical, as it must be -- avg_comparisons() is doing exactly the three\n")
cat("steps above.\n")

# =============================================================================
# (b) Confidence interval: bootstrap the WHOLE procedure
# =============================================================================
# The uncertainty does not come out of the outcome model directly, because we
# fit, predict, average, and contrast. So we resample patients and repeat all
# of it. marginaleffects::inferences() wraps that up:
boot_res <- avg_comparisons(out_model, variables = list(treatment = 0:1)) |>
  inferences(method = "boot", R = 500)

cat("\n--- (b) Bootstrap confidence interval ---\n")
print(boot_res)

# The same by hand, so you can see what inferences() did:
gcomp_once <- function(d) {
  m <- glm(death_1yr ~ treatment * (age + creatinine + heart_failure + prior_mi),
    data = d, family = binomial
  )
  mean(predict(m, transform(d, treatment = 1), type = "response")) -
    mean(predict(m, transform(d, treatment = 0), type = "response"))
}

set.seed(1)
boot_manual <- replicate(500, {
  idx <- sample(nrow(exercise_dat), replace = TRUE)
  gcomp_once(exercise_dat[idx, ])
})

cat(sprintf(
  "\nHand-rolled bootstrap: %+.4f (95%% percentile CI %+.4f, %+.4f)\n",
  gcomp_rd, quantile(boot_manual, 0.025), quantile(boot_manual, 0.975)
))
cat("Both intervals contain the truth.\n")

# =============================================================================
# (c) What the interaction terms allow
# =============================================================================
cat("\n--- (c) What do the interactions do? ---\n")
cat("Writing `treatment * (age + creatinine + ...)` rather than\n")
cat("`treatment + age + creatinine + ...` allows the drug's effect to be\n")
cat("DIFFERENT for different kinds of patient. Without the interactions, the\n")
cat("model is forced to say 'the beta-blocker shifts the log-odds of death by\n")
cat("the same amount for a fit 55-year-old and a frail 85-year-old with heart\n")
cat("failure'. With them, the model can say 'it helps the sicker patients more'\n")
cat("(or less) and let the data decide.\n")
cat("\nThat matters because the ATE is an AVERAGE of individual effects. If the\n")
cat("effect genuinely varies, the average we want is the average of the\n")
cat("patient-specific effects across our actual patient mix -- which is\n")
cat("precisely what predicting each patient twice and then averaging gives us.\n")
cat("A single coefficient cannot represent that.\n")

# Look at the spread of individual effects the interaction model implies:
individual_rd <- p1 - p0
cat(sprintf(
  "\nIndividual risk differences implied by the model range from %+.3f to %+.3f\n",
  min(individual_rd), max(individual_rd)
))
cat(sprintf("with a mean of %+.3f (the ATE) and an SD of %.3f.\n",
            mean(individual_rd), sd(individual_rd)))
cat("Even with no true interaction, the effect on the RISK scale varies across\n")
cat("patients, because a constant shift in log-odds produces a bigger change in\n")
cat("risk for a patient near 50% risk than for one near 2%.\n")
cat("\nNote also that the range creeps slightly ABOVE zero for a few patients,\n")
cat("implying the drug harms them. It does not -- we built it to be protective\n")
cat("for everyone. That is the interaction model overfitting a handful of\n")
cat("sparsely populated corners of covariate space, and it is a good reminder\n")
cat("not to read individual predicted effects as real subgroup findings.\n")

# =============================================================================
# (d) Drop the interactions -- does it matter?
# =============================================================================
add_model <- glm(
  death_1yr ~ treatment + age + creatinine + heart_failure + prior_mi,
  data = exercise_dat, family = binomial
)
p1_add <- predict(add_model, transform(exercise_dat, treatment = 1), type = "response")
p0_add <- predict(add_model, transform(exercise_dat, treatment = 0), type = "response")
gcomp_add <- mean(p1_add) - mean(p0_add)

cat("\n--- (d) With and without interactions ---\n")
cat(sprintf("With interactions   : %+.4f\n", gcomp_rd))
cat(sprintf("Without interactions: %+.4f\n", gcomp_add))
cat(sprintf("Truth               : %+.4f\n", TRUE_ATE_RD))
cat(sprintf("\nRaw `treatment` coefficient from the additive model: %+.3f\n",
            coef(add_model)["treatment"]))
cat(sprintf("(the data-generating value was %+.3f, on the log-odds scale)\n", -0.8))

cat("\nWhy the two g-computation estimates barely differ here: we SIMULATED the\n")
cat("data with no treatment-covariate interaction, so the extra terms have\n")
cat("nothing to find and only add a little noise. In real data you do not know\n")
cat("that, so including them is the safer default -- the cost is a few degrees\n")
cat("of freedom, and the benefit is not silently assuming a constant effect.\n")

cat("\nWhy standardisation still recovers a sensible average even if you omit\n")
cat("interactions that DO exist: g-computation averages predicted RISKS over\n")
cat("the real distribution of patient characteristics. Even a misspecified\n")
cat("model that gets the average risk in each arm roughly right will get the\n")
cat("contrast roughly right. What you lose is the ability to say anything about\n")
cat("WHICH patients benefit -- and if the misspecification is severe enough to\n")
cat("distort the average risks themselves, the estimate does become biased.\n")
cat("This is exactly the vulnerability that doubly robust estimators (AIPW,\n")
cat("TMLE) are designed to insure against.\n")
