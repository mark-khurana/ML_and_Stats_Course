"""
Chapter 1 (Setup) - Exercise 2: Explore a Clinical Dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
clinical = pd.DataFrame({
    "age": np.round(np.random.normal(55, 12, n)).astype(int),
})
clinical["systolic_bp"] = np.round(100 + 0.8 * clinical["age"] + np.random.normal(0, 10, n))
clinical["bmi"] = np.round(np.random.normal(27, 5, n), 1)

plt.figure(figsize=(8, 5))
sns.regplot(
    data=clinical, x="age", y="systolic_bp",
    scatter_kws={"alpha": 0.5, "color": "darkblue"},
    line_kws={"color": "firebrick"}
)
plt.title("Age vs. Systolic Blood Pressure\nSimulated clinical data (n = 200)")
plt.xlabel("Age (years)")
plt.ylabel("Systolic Blood Pressure (mmHg)")
plt.tight_layout()
plt.show()
