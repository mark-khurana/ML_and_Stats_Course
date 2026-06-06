# Chapter 16, Exercise 3: DBSCAN for Outlier Detection
# Add outlier patients to the Exercise 1 dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

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

X_normal = np.vstack([p1, p2, p3])
is_outlier = np.array([False] * 600)

# Add 30 outlier patients
np.random.seed(99)
outliers = np.column_stack([
    np.random.normal(180, 10, 30),     # Extreme HR
    np.random.normal(40, 5, 30),       # Extreme RR
    np.random.normal(40.5, 0.5, 30),   # Very high temp
    np.random.normal(50, 10, 30),      # Very low SBP
    np.random.normal(8, 1.5, 30),      # Very high creatinine
    np.random.normal(15, 3, 30)        # Very high lactate
])

X = np.vstack([X_normal, outliers])
is_outlier = np.concatenate([is_outlier, np.array([True] * 30)])
X_scaled = StandardScaler().fit_transform(X)

print(f"Total patients: {len(X)} (600 normal + 30 outliers)\n")

# ---- (a) K-means with k=3 ----
print("=== Part (a): K-means with k=3 ===")

km3 = KMeans(n_clusters=3, n_init=25, random_state=42).fit(X_scaled)

outlier_clusters = km3.labels_[is_outlier]
print("Outlier distribution across K-means clusters:")
for c in range(3):
    n_out = (outlier_clusters == c).sum()
    n_tot = (km3.labels_ == c).sum()
    print(f"  Cluster {c+1}: {n_tot} patients ({n_out} outliers)")

print("\nK-means assigns outliers to existing clusters, distorting centroids.")

# ---- (b) DBSCAN ----
print("\n=== Part (b): DBSCAN ===")

# kNN distance plot to choose eps
nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
knn_dist = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(knn_dist, color="steelblue")
plt.axhline(y=2.5, color="firebrick", linestyle="--", label="eps = 2.5")
plt.xlabel("Points (sorted)")
plt.ylabel("5-NN Distance")
plt.title("kNN Distance Plot for eps Selection")
plt.legend()
plt.tight_layout()
plt.savefig("ch16_ex3_knn_dist.png", dpi=150)
plt.show()

# Run DBSCAN
db = DBSCAN(eps=2.5, min_samples=10).fit(X_scaled)

n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = (db.labels_ == -1).sum()
outliers_as_noise = ((db.labels_ == -1) & is_outlier).sum()

print(f"Number of clusters: {n_clusters}")
print(f"Noise points (outliers): {n_noise}")
print(f"Actual outliers detected as noise: {outliers_as_noise} out of 30")

# ---- (c) Compare non-outlier assignments ----
print("\n=== Part (c): Comparison for Non-Outlier Patients ===")

normal_mask = ~is_outlier
km_normal = km3.labels_[normal_mask]
db_normal = db.labels_[normal_mask]

db_noise_normal = (db_normal == -1).sum()
print(f"Normal patients misclassified as noise: {db_noise_normal}")

# ARI for non-noise, non-outlier patients
both_assigned = db_normal >= 0
if both_assigned.sum() > 0:
    ari = adjusted_rand_score(km_normal[both_assigned], db_normal[both_assigned])
    print(f"ARI (K-means vs DBSCAN, non-outlier, non-noise): {ari:.3f}")

# ---- Visualise comparison ----
pca2 = PCA(n_components=2).fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-means
colors_km = ["steelblue", "firebrick", "forestgreen"]
for c in range(3):
    mask = (km3.labels_ == c) & ~is_outlier
    axes[0].scatter(pca2[mask, 0], pca2[mask, 1], c=colors_km[c], s=12, alpha=0.6)
    mask_out = (km3.labels_ == c) & is_outlier
    axes[0].scatter(pca2[mask_out, 0], pca2[mask_out, 1], c=colors_km[c],
                    marker="x", s=50, linewidths=2)
axes[0].set_title("K-means (k=3)\n(X = outlier patients)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# DBSCAN
color_map = {-1: "grey", 0: "steelblue", 1: "firebrick", 2: "forestgreen"}
for c_val in sorted(set(db.labels_)):
    mask = db.labels_ == c_val
    marker = "x" if c_val == -1 else "o"
    label = "Noise" if c_val == -1 else f"Cluster {c_val+1}"
    axes[1].scatter(pca2[mask, 0], pca2[mask, 1],
                    c=color_map.get(c_val, "purple"),
                    marker=marker, s=15 if c_val >= 0 else 40,
                    alpha=0.6, label=label)
axes[1].set_title("DBSCAN\n(X/grey = noise)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("ch16_ex3_comparison.png", dpi=150)
plt.show()

# ---- (d) Preferred approach ----
print("\n=== Part (d): Preferred Approach ===\n")
print("DBSCAN is preferred for clinical phenotyping when outliers are expected.\n")
print("1. EXPLICIT OUTLIER IDENTIFICATION: DBSCAN labels outliers as noise,")
print("   making them visible. K-means silently absorbs them.\n")
print("2. CLINICAL RELEVANCE: Outlier patients may represent data errors,")
print("   rare presentations, or patients needing individual assessment.\n")
print("3. CLUSTER INTEGRITY: DBSCAN preserves cluster purity by excluding")
print("   outliers. K-means centroids can be pulled by outliers.\n")
print("4. PRACTICAL WORKFLOW: Use DBSCAN first to identify outliers,")
print("   review them clinically, then apply K-means to non-outliers.\n")
print("5. CAVEAT: DBSCAN requires tuning eps and min_samples, which")
print("   affects how many points are labelled as noise.")
