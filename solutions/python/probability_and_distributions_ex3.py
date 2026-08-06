"""
Chapter 2 (Probability and Distributions) - Exercise 3: Poisson Distribution --- Emergency Department Visits
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

lam = 8

# 1. P(X = 8)
prob_exactly_8 = stats.poisson.pmf(8, lam)
print(f"P(X = 8): {prob_exactly_8:.4f}")
# About 0.1396

# 2. P(X >= 12)
prob_12_or_more = stats.poisson.sf(11, lam)
print(f"P(X >= 12): {prob_12_or_more:.4f}")
# About 0.1121

# 3. Expected days with 0 cases
prob_zero = stats.poisson.pmf(0, lam)
expected_zero_days = 365 * prob_zero
print(f"P(X = 0): {prob_zero:.6f}")
print(f"Expected zero-case days per year: {expected_zero_days:.3f}")
# Very small, about 0.12 days per year

# 4. Plot
k = np.arange(0, 21)
probs = stats.poisson.pmf(k, lam)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(k, probs, color="darkorange", alpha=0.8)
ax.axvline(lam, linestyle="--", color="firebrick")
ax.set_title(f"Poisson Distribution: Trauma Cases per Day (lambda = {lam})")
ax.set_xlabel("Number of Trauma Cases")
ax.set_ylabel("Probability")
ax.set_xticks(k)
plt.tight_layout()
plt.show()
