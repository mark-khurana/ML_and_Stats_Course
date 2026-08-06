"""
Chapter 2 (Probability and Distributions) - Exercise 1: Z-Scores and the Normal Distribution
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

mu = 90
sigma = 10
patient_value = 115

# 1. Z-score
z_score = (patient_value - mu) / sigma
print(f"Z-score: {z_score}")
# Z-score: 2.5

# 2. Proportion above 115
prop_above = 1 - stats.norm.cdf(patient_value, loc=mu, scale=sigma)
# Or equivalently: stats.norm.sf(patient_value, loc=mu, scale=sigma)
print(f"Proportion above 115: {prop_above:.4f}")
# About 0.0062 or 0.62%

# 3. 95th percentile
percentile_95 = stats.norm.ppf(0.95, loc=mu, scale=sigma)
print(f"95th percentile: {percentile_95:.1f} mg/dL")
# About 106.4 mg/dL

# 4. Plot
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
y = stats.norm.pdf(x, loc=mu, scale=sigma)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, color="steelblue", linewidth=1.5)
x_fill = x[x >= patient_value]
y_fill = stats.norm.pdf(x_fill, loc=mu, scale=sigma)
ax.fill_between(x_fill, y_fill, color="firebrick", alpha=0.4)
ax.axvline(patient_value, linestyle="--", color="firebrick")
ax.text(117, 0.02, f"Glucose = {patient_value} mg/dL\nZ = {z_score}",
        color="firebrick")
ax.set_title("Distribution of Fasting Blood Glucose")
ax.set_xlabel("Fasting Blood Glucose (mg/dL)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()
