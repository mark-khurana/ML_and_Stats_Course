"""
Chapter 1 (Setup) - Exercise 1: Verify Your Setup
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

plt.figure(figsize=(8, 5))
sns.histplot(iris["sepal width (cm)"], bins=15, kde=True, color="steelblue")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
