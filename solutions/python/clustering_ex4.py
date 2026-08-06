# Chapter 16, Exercise 4: Full Pipeline - PCA + Clustering + UMAP Visualisation
# 800 patients, 30 clinical variables, 4 underlying subtypes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

# ---- Simulate data ----
np.random.seed(42)
n = 800
p = 30

# 4 subtypes
true_subtype = np.random.choice(4, n, p=[0.30, 0.30, 0.25, 0.15])
print("True subtype distribution:", {i: (true_subtype == i).sum() for i in range(4)})

X = np.random.normal(0, 1, (n, p))

# Add subtype-specific signals
for s in range(4):
    start = s * 6
    end = min((s + 1) * 6, p)
    X[true_subtype == s, start:end] += 2.5

# ---- (a) Standardise and PCA ----
print("\n=== Part (a): PCA ===")

X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
scores = pca.fit_transform(X_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)
n_80 = np.argmax(cum_var >= 0.80) + 1
print(f"PCs needed for 80% variance: {n_80}")

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(1, 16), pca.explained_variance_ratio_[:15],
            color="steelblue", edgecolor="white")
axes[0].axhline(y=1/p, color="firebrick", linestyle="--", label="1/p")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Proportion of Variance")
axes[0].set_title("Scree Plot")
axes[0].legend()

axes[1].plot(range(1, 16), cum_var[:15], "o-", color="steelblue")
axes[1].axhline(y=0.80, color="firebrick", linestyle="--", label="80%")
axes[1].set_xlabel("Components")
axes[1].set_ylabel("Cumulative Proportion")
axes[1].set_title("Cumulative Variance")
axes[1].legend()

plt.tight_layout()
plt.savefig("ch16_ex4_scree.png", dpi=150)
plt.show()

# Use 10 PCs
n_pcs = 10
pca_scores = scores[:, :n_pcs]

# ---- (b) K-means on PCA scores ----
print("\n=== Part (b): K-means on PCA Scores ===")

sil_list = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(pca_scores)
    sil = silhouette_score(pca_scores, km.labels_)
    sil_list.append(sil)
    print(f"k = {k}: Silhouette = {sil:.3f}")

best_k = list(range(2, 6))[np.argmax(sil_list)]
print(f"Best k by silhouette: {best_k}")

# Fit final clustering
km_final = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(pca_scores)

# ---- (c) UMAP visualisation ----
print("\n=== Part (c): UMAP Visualisation ===")

umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                random_state=42).fit_transform(pca_scores)

colors_4 = ["steelblue", "firebrick", "forestgreen", "goldenrod"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Discovered clusters
for c in range(best_k):
    mask = km_final.labels_ == c
    axes[0].scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors_4[c],
                    s=10, alpha=0.6, label=f"Cluster {c+1}")
axes[0].set_title("K-means Clusters on UMAP")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
axes[0].legend(fontsize=8)

# True subtypes
for s in range(4):
    mask = true_subtype == s
    axes[1].scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors_4[s],
                    s=10, alpha=0.6, label=f"Subtype {s+1}")
axes[1].set_title("True Subtypes on UMAP")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("ch16_ex4_umap.png", dpi=150)
plt.show()

# ---- (d) Cluster profiles ----
print("\n=== Part (d): Cluster Profiles ===\n")

var_names = [f"V{i+1}" for i in range(p)]
df = pd.DataFrame(X, columns=var_names)
df['cluster'] = km_final.labels_

print("Cluster sizes:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster means for signal variables:")
for cl in range(best_k):
    mask = df['cluster'] == cl
    n_cl = mask.sum()
    print(f"\nCluster {cl+1} (n = {n_cl}):")
    for s in range(4):
        start = s * 6
        end = (s + 1) * 6
        signal_vars = [f"V{i+1}" for i in range(start, min(end, p))]
        mean_val = df.loc[mask, signal_vars].mean().mean()
        print(f"  Subtype {s+1} signal vars (V{start+1}-V{min(end,p)}): mean = {mean_val:.2f}")

# ---- (e) Discussion ----
print("\n=== Part (e): Validation Discussion ===\n")

print("To validate these clusters in a real clinical study:\n")
print("1. EXTERNAL OUTCOMES: Compare clusters on outcomes NOT used in")
print("   clustering (mortality, LOS, treatment response). Clusters")
print("   that predict external outcomes are clinically meaningful.\n")
print("2. STABILITY ANALYSIS: Bootstrap resampling + re-clustering.")
print("   Consistent membership = robust; variable = artefact.\n")
print("3. INDEPENDENT REPLICATION: Apply pipeline to an independent")
print("   dataset (different hospital/time). Similar clusters = generalisable.\n")
print("4. CLINICAL EXPERT REVIEW: Do clusters correspond to")
print("   recognisable patient types? Do they suggest different management?\n")
print("5. MULTIPLE ALGORITHMS: Compare K-means, hierarchical, DBSCAN.")
print("   Agreement across methods strengthens confidence.\n")
print("6. AVOID CIRCULAR REASONING: Do NOT validate on the same")
print("   variables used for clustering. That proves nothing.")
