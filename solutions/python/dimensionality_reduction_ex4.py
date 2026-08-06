# Chapter 15, Exercise 4: Full Workflow on Simulated Genomic Data
# 1000 patients, 500 gene expression features, 4 cancer subtypes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# ---- Simulate data ----
np.random.seed(42)
n = 1000
p = 500

# 4 cancer subtypes (one rare at 5%)
subtype = np.random.choice(4, n, p=[0.35, 0.30, 0.30, 0.05])
print("Subtype distribution:", {i: (subtype == i).sum() for i in range(4)})

# Generate base gene expression
X = np.random.normal(0, 1, (n, p))

# Add subtype-specific signals
X[subtype == 0, :30] += 2.0       # Subtype 1: genes 0-29
X[subtype == 1, 30:60] += 2.5     # Subtype 2: genes 30-59
X[subtype == 1, 60:80] -= 1.5     # Subtype 2: downregulated genes 60-79
X[subtype == 2, 80:120] += 2.0    # Subtype 3: genes 80-119
X[subtype == 3, 120:160] += 3.5   # Subtype 4 (rare): genes 120-159

# ---- (a) Scale and PCA ----
print("\n=== Part (a): PCA ===")
X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
scores = pca.fit_transform(X_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)
n_80 = np.argmax(cum_var >= 0.80) + 1
print(f"PCs needed for 80% variance: {n_80}")
print(f"First 10 cumulative variance: {np.round(cum_var[:10], 3)}")

# Scree plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, 31), pca.explained_variance_ratio_[:30],
       color="steelblue", edgecolor="white")
ax.axhline(y=1/p, color="firebrick", linestyle="--", label=f"1/p = {1/p:.4f}")
ax.set_xlabel("Component")
ax.set_ylabel("Proportion of Variance")
ax.set_title("Scree Plot (first 30 PCs)")
ax.legend()
plt.tight_layout()
plt.savefig("ch15_ex4_scree.png", dpi=150)
plt.show()

# ---- (b) UMAP on first 30 PCs ----
print("\n=== Part (b): UMAP on First 30 PCs ===")
pca_30 = scores[:, :30]
umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                random_state=42).fit_transform(pca_30)

subtype_names = ["Subtype 1", "Subtype 2", "Subtype 3", "Subtype 4 (rare)"]
colors_4 = ["steelblue", "firebrick", "forestgreen", "goldenrod"]

fig, ax = plt.subplots(figsize=(8, 6))
for s in range(4):
    mask = subtype == s
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors_4[s],
               s=12, alpha=0.6, label=subtype_names[s])
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("UMAP of Simulated Genomic Data (4 Cancer Subtypes)")
ax.legend()
plt.tight_layout()
plt.savefig("ch15_ex4_umap.png", dpi=150)
plt.show()

print("The four subtypes are visually separable in the UMAP embedding.")

# ---- (c) t-SNE comparison ----
print("\n=== Part (c): t-SNE Comparison ===")
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000).fit_transform(pca_30)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for s in range(4):
    mask = subtype == s
    axes[0].scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors_4[s],
                    s=12, alpha=0.6, label=subtype_names[s])
    axes[1].scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], c=colors_4[s],
                    s=12, alpha=0.6, label=subtype_names[s])

axes[0].set_title("UMAP (n_neighbors=15)")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
axes[0].legend(fontsize=8)

axes[1].set_title("t-SNE (perplexity=30)")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("ch15_ex4_comparison.png", dpi=150)
plt.show()

print("Both UMAP and t-SNE successfully separate the four subtypes.")
print("UMAP preserves more global structure; t-SNE may produce more")
print("compact clusters but with unreliable inter-cluster distances.")

# ---- (d) Identifying the rare subtype ----
print("\n=== Part (d): Identifying the Rare Subtype ===")
n_rare = (subtype == 3).sum()
print(f"Rare subtype (Subtype 4) has {n_rare} patients (5% of total).\n")

print("YES, the rare subtype can be identified in the UMAP plot.")
print("Despite having only ~50 patients, Subtype 4 forms a distinct")
print("cluster (shown in goldenrod). This is because:")
print("  1. The signal is strong (effect size = 3.5 in 40 genes)")
print("  2. UMAP preserves local structure, so small groups remain cohesive")
print("  3. The gene expression pattern is qualitatively different\n")
print("In practice, rare subtypes CAN be identified if:")
print("  - Their molecular profile is sufficiently distinct")
print("  - Sample size is not too small (>20-30 patients)")
print("  - Dimensionality reduction preserves the relevant structure")
print("If the signal were weaker, the rare subtype might merge with")
print("another group or scatter as outliers.")
