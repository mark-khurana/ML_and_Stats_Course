"""
Chapter 2 (Probability and Distributions) - Exercise 4: Bayes' Theorem --- Interpreting a Cancer Screening Test
"""

import numpy as np
import matplotlib.pyplot as plt

sensitivity = 0.87
specificity = 0.95
prevalence = 0.02

# 1. PPV
ppv = (sensitivity * prevalence) / (
    sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
)
print(f"PPV (prevalence = 2%): {ppv:.4f}")
# About 0.2623 or 26.2%

# 2. NPV
npv = (specificity * (1 - prevalence)) / (
    specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
)
print(f"NPV (prevalence = 2%): {npv:.4f}")
# About 0.9972 or 99.7%

# 3. PPV with high-risk prevalence
prev_high = 0.10
ppv_high = (sensitivity * prev_high) / (
    sensitivity * prev_high + (1 - specificity) * (1 - prev_high)
)
print(f"PPV (prevalence = 10%): {ppv_high:.4f}")
# About 0.6588 or 65.9%

# 4. Plot
prev_range = np.arange(0.001, 0.301, 0.001)
ppv_curve = (sensitivity * prev_range) / (
    sensitivity * prev_range + (1 - specificity) * (1 - prev_range)
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(prev_range * 100, ppv_curve * 100, color="steelblue", linewidth=1.5)
ax.plot([2, 10], [ppv * 100, ppv_high * 100], "o", color="firebrick", markersize=8)
ax.annotate(f"General pop (2%): PPV = {ppv*100:.1f}%",
            xy=(2, ppv*100), xytext=(5, ppv*100 - 5),
            color="firebrick", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="firebrick"))
ax.annotate(f"High risk (10%): PPV = {ppv_high*100:.1f}%",
            xy=(10, ppv_high*100), xytext=(13, ppv_high*100 - 5),
            color="firebrick", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="firebrick"))
ax.set_xlabel("Prevalence (%)")
ax.set_ylabel("Positive Predictive Value (%)")
ax.set_title("Mammography PPV Depends on Prevalence\nSensitivity = 87%, Specificity = 95%")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.show()
