# Chapter 16, Exercise 2: Hierarchical Clustering with Different Linkage Methods
# Using the same dataset from Exercise 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ---- Simulate data (same as Exercise 1) ----
np.random.seed(42)

p1 = np.column_stack([
    np.random.normal(115, 12, 200), np.random.normal(28, 5, 200),
    np.random.normal(38.5, 0.8, 200), np.random.normal(80, 12, 200),
    np.random.normal(2.5, 0.8, 200), np.random.normal(5, 2, 200)
])
p2 = np.column_stack([
    np.random.normal(90, 10, 200), np.random.normal(20, 3, 200),
    np.random.normal(37.2, 0.5, 200), np.random.normal(110, 15, 200),
    np.random.normal(1.2, 0.3, 200), np.random.normal(1.5, 0.5, 200)
])
p3 = np.column_stack([
    np.random.normal(100, 10, 200), np.random.normal(22, 4, 200),
    np.random.normal(39.2, 0.7, 200), np.random.normal(120, 10, 200),
    np.random.normal(1.0, 0.2, 200), np.random.normal(2.0, 0.8, 200)
])

X = np.vstack([p1, p2, p3])
X_scaled = StandardScaler().fit_transform(X)

# K-means reference
km3 = KMeans(n_clusters=3, n_init=25, random_state=42).fit(X_scaled)

# ---- (a) Hierarchical clustering with 4 linkage methods ----
print("=== Part (a): Hierarchical Clustering ===\n")

methods = ['single', 'complete', 'average', 'ward']
titles = ['Single', 'Complete', 'Average', "Ward's"]

# Dendrograms (use subset for readability)
np.random.seed(42)
idx = np.random.choice(len(X_scaled), 80, replace=False)
X_sub = X_scaled[idx]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, method, title in zip(axes, methods, titles):
    Z = linkage(X_sub, method=method)
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(title)
    ax.set_xlabel("")

plt.suptitle("Hierarchical Clustering: Linkage Method Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("ch16_ex2_dendrograms.png", dpi=150)
plt.show()

# ---- (b) Cut at k=3 and compare ----
print("=== Part (b): Cluster Assignments ===\n")

hc_labels = {}
for method in methods:
    Z = linkage(X_scaled, method=method)
    hc_labels[method] = fcluster(Z, t=3, criterion='maxclust')

# Crosstabs
print("Crosstab: Ward's vs Complete:")
ct = pd.crosstab(pd.Series(hc_labels['ward'], name='Ward'),
                  pd.Series(hc_labels['complete'], name='Complete'))
print(ct)

print("\nCrosstab: Ward's vs Single:")
ct2 = pd.crosstab(pd.Series(hc_labels['ward'], name='Ward'),
                   pd.Series(hc_labels['single'], name='Single'))
print(ct2)

# ---- (c) ARI comparison with K-means ----
print("\n=== Part (c): Adjusted Rand Index vs K-means ===\n")

for method, title in zip(methods, titles):
    ari = adjusted_rand_score(km3.labels_, hc_labels[method])
    print(f"ARI({title} vs K-means): {ari:.3f}")

print("\nWard's method produces clusters most similar to K-means (highest ARI).")
print("Both minimise within-cluster variance, so they share similar objectives.")

# ---- (d) Recommendation ----
print("\n=== Part (d): Recommendation ===\n")

print("Ward's method is recommended for clinical data because:\n")
print("1. OBJECTIVE: Minimises within-cluster variance, producing")
print("   compact, homogeneous clusters.\n")
print("2. CONSISTENCY: Most similar to K-means, strengthening")
print("   confidence when both methods agree.\n")
print("3. ROBUSTNESS: Avoids chaining (single linkage) and outlier")
print("   sensitivity (complete linkage).\n")
print("4. COMPACT CLUSTERS: Clinical phenotypes are typically compact")
print("   groups, which Ward's naturally produces.\n")
print("5. INTERPRETABILITY: Dendrogram with clear height gaps at")
print("   natural cluster boundaries.\n")
print("CAVEAT: For non-spherical clusters, average linkage may be")
print("more appropriate. But Ward's is the safe default for most")
print("clinical phenotyping applications.")
