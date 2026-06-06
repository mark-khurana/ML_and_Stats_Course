# =============================================================================
# Chapter 5, Exercise 1: Conceptual Questions
# Penalised Regression: Ridge, LASSO, Elastic Net
# =============================================================================
# This exercise is conceptual. Answers are provided as comments.

import numpy as np
import matplotlib.pyplot as plt

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

theta = np.linspace(0, 2 * np.pi, 200)

# L2 constraint (circle)
l2_x = np.cos(theta)
l2_y = np.sin(theta)

# L1 constraint (diamond)
l1_x = [1, 0, -1, 0, 1]
l1_y = [0, 1, 0, -1, 0]

fig, ax = plt.subplots(figsize=(7, 7))

# L2 region
ax.fill(l2_x, l2_y, alpha=0.15, color='#3498db')
ax.plot(l2_x, l2_y, color='#3498db', linewidth=1.5, label='L2 (Ridge)')

# L1 region
ax.fill(l1_x, l1_y, alpha=0.15, color='#e74c3c')
ax.plot(l1_x, l1_y, color='#e74c3c', linewidth=1.5, label='L1 (LASSO)')

# Mark the corners
ax.plot([1, -1, 0, 0], [0, 0, 1, -1], 'ro', markersize=6)
ax.annotate('Corner (exact zero for $\\beta_2$)', xy=(1, 0), fontsize=9,
            xytext=(1.1, 0.3), arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel(r'$\beta_1$', fontsize=14)
ax.set_ylabel(r'$\beta_2$', fontsize=14)
ax.set_title('L1 vs L2 constraint geometry')
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.set_xlim(-1.5, 1.8)
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
plt.tight_layout()
plt.show()

print("The L1 diamond has corners at the axes, making it more likely that")
print("the optimal solution lies at a corner (coefficient = 0), producing")
print("exact sparsity. The L2 circle has no corners, so coefficients are")
print("shrunk toward zero but never reach it exactly.")
