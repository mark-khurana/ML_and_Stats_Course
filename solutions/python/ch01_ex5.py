"""
Chapter 2 (Probability and Distributions) - Exercise 5: Central Limit Theorem --- Hands-On Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(123)
lam = 3
n_sim = 5000

# 1. Raw Poisson data
raw_data = np.random.poisson(lam, 10000)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(raw_data, bins=range(0, 15), color="steelblue", alpha=0.8, edgecolor="white")
ax.set_title("Raw Poisson(3) Data (right-skewed)")
ax.set_xlabel("Value")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# 2. CLT simulation
sample_sizes = [5, 15, 50, 200]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, n in enumerate(sample_sizes):
    means = [np.mean(np.random.poisson(lam, n)) for _ in range(n_sim)]
    axes[i].hist(means, bins=40, density=True, color="steelblue", alpha=0.7)
    axes[i].set_title(f"n = {n}")
    axes[i].set_xlabel("Sample Mean")
    axes[i].set_ylabel("Density")

fig.suptitle("CLT: Distribution of Sample Means from Poisson(3)", fontsize=13)
plt.tight_layout()
plt.show()

# 3. Overlay normal curve on n = 50
n50_means = [np.mean(np.random.poisson(lam, 50)) for _ in range(n_sim)]
theoretical_sd = np.sqrt(lam / 50)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(n50_means, bins=40, density=True, color="steelblue", alpha=0.7, label="Simulated")
x_range = np.linspace(min(n50_means), max(n50_means), 200)
ax.plot(x_range, stats.norm.pdf(x_range, loc=lam, scale=theoretical_sd),
        color="firebrick", linewidth=1.5, label="Normal approx.")
ax.set_title("Sample Means (n=50) with Normal Approximation Overlay")
ax.set_xlabel("Sample Mean")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
