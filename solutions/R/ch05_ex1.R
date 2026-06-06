# =============================================================================
# Chapter 5, Exercise 1: Conceptual Questions
# Penalised Regression: Ridge, LASSO, Elastic Net
# =============================================================================
# This exercise is conceptual. Answers are provided as comments.

# --- Question (a) ---
# A colleague says: "I used LASSO and it removed age from my model predicting
# mortality. This means age is not a risk factor for death."
# Explain why this interpretation is incorrect.
#
# ANSWER:
# The LASSO removing age does NOT mean age is unimportant. It means that,
# given the OTHER predictors in the model, age does not provide enough
# ADDITIONAL predictive information to justify its inclusion under the
# penalty. Possible reasons why LASSO dropped age:
#
# 1. Age may be highly CORRELATED with another retained variable (e.g.,
#    number of comorbidities, renal function). The LASSO tends to pick
#    one of a group of correlated predictors and drop the rest arbitrarily.
#
# 2. The sample size may be too small to detect age's effect given the
#    penalty strength.
#
# 3. The LASSO performs variable selection based on PREDICTION, not
#    CAUSAL importance. A variable can be a genuine risk factor but
#    redundant for prediction when other variables are present.
#
# The correct interpretation is: "Age was not selected by the LASSO model,
# possibly because its predictive information overlaps with other retained
# variables. This does not imply age is not a risk factor for death."

# --- Question (b) ---
# Ridge CV-RMSE = 2.1 days. Standard linear regression RMSE = 1.8 days
# (on the same training data). Colleague says standard model is better.
# What is wrong?
#
# ANSWER:
# The comparison is unfair because:
#
# 1. The Ridge RMSE of 2.1 is CROSS-VALIDATED -- it estimates performance
#    on UNSEEN data by training and testing on different folds.
#
# 2. The standard regression RMSE of 1.8 is likely computed on the SAME
#    training data used to fit the model (apparent performance). This is
#    OVERLY OPTIMISTIC because the model has memorised the training data's
#    noise.
#
# 3. If you computed a cross-validated RMSE for the standard regression,
#    it would likely be HIGHER than 2.1 (worse than Ridge), especially
#    with 20 predictors.
#
# The correct comparison requires evaluating both models using the SAME
# methodology -- either both cross-validated, or both on a held-out test
# set. Never compare a cross-validated metric to a training metric.

# --- Question (c) ---
# Explain in one sentence why the LASSO produces exact zeros but Ridge does not.
#
# ANSWER:
# The L1 penalty (LASSO) constrains coefficients to a diamond-shaped region
# whose corners lie on the axes, so the loss function contours are much more
# likely to first touch the constraint region at a corner (where one or more
# coefficients are exactly zero) than the smooth circular boundary of the L2
# penalty (Ridge), which has no corners.

# --- Supporting code to illustrate the geometry ---
library(ggplot2)

theta <- seq(0, 2 * pi, length.out = 200)

# L2 constraint (circle)
l2_x <- cos(theta)
l2_y <- sin(theta)

# L1 constraint (diamond)
l1_x <- c(1, 0, -1, 0, 1)
l1_y <- c(0, 1, 0, -1, 0)

p <- ggplot() +
  # L2 constraint region
  geom_polygon(data = data.frame(x = l2_x, y = l2_y),
               aes(x, y), fill = "#3498db", alpha = 0.15) +
  geom_path(data = data.frame(x = l2_x, y = l2_y),
            aes(x, y), colour = "#3498db", linewidth = 1.2) +
  # L1 constraint region
  geom_polygon(data = data.frame(x = l1_x, y = l1_y),
               aes(x, y), fill = "#e74c3c", alpha = 0.15) +
  geom_path(data = data.frame(x = l1_x, y = l1_y),
            aes(x, y), colour = "#e74c3c", linewidth = 1.2) +
  # Labels
  annotate("text", x = 0.7, y = 0.7, label = "L2 (Ridge)",
           colour = "#3498db", size = 4) +
  annotate("text", x = -0.7, y = -0.3, label = "L1 (LASSO)",
           colour = "#e74c3c", size = 4) +
  annotate("text", x = 0, y = -1.3,
           label = "The diamond has corners on the axes\nwhere coefficients = 0",
           size = 3.5) +
  coord_fixed() +
  labs(x = expression(beta[1]), y = expression(beta[2]),
       title = "L1 vs L2 constraint geometry") +
  theme_minimal(base_size = 14) +
  xlim(-1.5, 1.5) + ylim(-1.5, 1.0)

print(p)

cat("\nThe L1 diamond has corners at the axes, making it more likely that\n")
cat("the optimal solution lies at a corner (coefficient = 0), producing\n")
cat("exact sparsity. The L2 circle has no corners, so coefficients are\n")
cat("shrunk toward zero but never reach it exactly.\n")
