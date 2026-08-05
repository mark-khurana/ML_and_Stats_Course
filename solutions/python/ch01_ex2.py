"""
Chapter 2 (Probability and Distributions) - Exercise 2: Binomial Distribution --- Vaccine Efficacy
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

n = 25
p = 0.10

# 1. Expected number
expected = n * p
print(f"Expected breakthrough infections: {expected}")
# 2.5

# 2. P(X = 0)
prob_zero = stats.binom.pmf(0, n, p)
print(f"P(X = 0): {prob_zero:.4f}")
# About 0.0718

# 3. P(X >= 5)
prob_five_or_more = stats.binom.sf(4, n, p)  # sf = survival function = 1 - cdf
print(f"P(X >= 5): {prob_five_or_more:.4f}")
# About 0.0980

# 4. Plot
k = np.arange(0, n + 1)
probs = stats.binom.pmf(k, n, p)
colors = ["firebrick" if ki >= 5 else "steelblue" for ki in k]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(k, probs, color=colors, alpha=0.8)
ax.axvline(expected, linestyle="--", color="black")
ax.set_title("Breakthrough Infections in 25 Vaccinated Individuals\nBinomial(25, 0.10)")
ax.set_xlabel("Number of Breakthrough Infections")
ax.set_ylabel("Probability")
ax.set_xticks(range(0, 26, 2))
plt.tight_layout()
plt.show()
