# =============================================================================
# Chapter 4, Exercise 1: Recognising the Problem
# Categorisation of systolic blood pressure and stroke risk
# =============================================================================
# This exercise is primarily conceptual. Answers are provided as comments,
# with supporting code to illustrate the points.

library(ggplot2)

# --- Question (a) ---
# What assumptions does categorising SBP into Normal (<120), Elevated (120-129),
# Stage 1 HTN (130-139), Stage 2 HTN (>=140) impose?
#
# ANSWER:
# 1. Within each category, the relationship between SBP and stroke risk is FLAT.
#    A patient with SBP=121 is treated identically to SBP=129.
# 2. At category boundaries, there is a SUDDEN JUMP in risk.
#    SBP=129 and SBP=130 are treated as fundamentally different despite only
#    1 mmHg difference.
# 3. The cut-points (120, 130, 140) are assumed to be biologically meaningful
#    boundaries, which may not reflect the true continuous biology.

# --- Question (b) ---
# A patient with SBP 131 is classified the same as SBP 139. Why is this problematic?
#
# ANSWER:
# The difference of 8 mmHg (131 vs 139) is clinically substantial. In reality,
# stroke risk increases with higher SBP. By lumping these patients together,
# we lose the ability to distinguish their different risk levels. The patient
# at 139 mmHg (close to Stage 2) likely has meaningfully higher stroke risk
# than the patient at 131 mmHg, but the categorised model assigns them
# identical predicted risk. This information loss reduces statistical power
# and can bias effect estimates.

# --- Question (c) ---
# Suggest a better modelling approach.
#
# ANSWER:
# Use a restricted cubic spline (RCS) with 3-5 knots to model SBP as a
# continuous predictor. This:
#   - Preserves the full information in SBP
#   - Allows the relationship to be non-linear (capturing any steepening
#     at higher pressures)
#   - Constrains the curve to be linear in the tails (sensible extrapolation)
#   - Uses only k-1 degrees of freedom for k knots (parsimonious)

# --- Supporting demonstration ---
# Simulate data to visually compare categorised vs spline approaches

library(rms)

set.seed(2025)
n <- 1000
sbp <- rnorm(n, mean = 130, sd = 15)
sbp <- pmax(sbp, 90)
sbp <- pmin(sbp, 200)

# True relationship: risk accelerates at higher SBP
logit_p <- -6 + 0.02 * sbp + 0.0002 * (sbp - 130)^2
y <- rbinom(n, 1, plogis(logit_p))

sim_data <- data.frame(sbp = sbp, stroke = y)

# Categorise as described in the exercise
sim_data$sbp_cat <- cut(sim_data$sbp,
                         breaks = c(-Inf, 120, 130, 140, Inf),
                         labels = c("Normal", "Elevated", "Stage1", "Stage2"))

# Set up datadist for rms
dd <- datadist(sim_data)
options(datadist = "dd")

# Fit three models
fit_cat <- lrm(stroke ~ sbp_cat, data = sim_data)
fit_linear <- lrm(stroke ~ sbp, data = sim_data)
fit_rcs <- lrm(stroke ~ rcs(sbp, 4), data = sim_data)

# Compare AIC
cat("AIC Comparison:\n")
cat("  Categorised:", AIC(fit_cat), "\n")
cat("  Linear:     ", AIC(fit_linear), "\n")
cat("  RCS (4 knots):", AIC(fit_rcs), "\n")
cat("\nLower AIC = better fit. The RCS model captures the true non-linear\n")
cat("relationship without imposing arbitrary cut-points.\n")

# Plot comparison
sbp_seq <- seq(min(sim_data$sbp), max(sim_data$sbp), length.out = 200)

pred_rcs <- Predict(fit_rcs, sbp = sbp_seq)
pred_linear_vals <- predict(fit_linear,
                             newdata = data.frame(sbp = sbp_seq),
                             type = "fitted")

# True probability
true_logit <- -6 + 0.02 * sbp_seq + 0.0002 * (sbp_seq - 130)^2
true_prob <- plogis(true_logit)

plot_df <- data.frame(
  sbp = rep(sbp_seq, 3),
  probability = c(true_prob, pred_linear_vals,
                  plogis(as.data.frame(pred_rcs)$yhat)),
  Model = rep(c("True relationship", "Linear", "RCS (4 knots)"),
              each = length(sbp_seq))
)

p <- ggplot(plot_df, aes(x = sbp, y = probability, colour = Model)) +
  geom_line(linewidth = 1.1) +
  scale_colour_manual(values = c("True relationship" = "black",
                                  "Linear" = "#3498db",
                                  "RCS (4 knots)" = "#e74c3c")) +
  labs(x = "Systolic Blood Pressure (mmHg)",
       y = "Predicted Probability of Stroke",
       title = "Comparing modelling approaches for SBP-stroke relationship") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

print(p)
