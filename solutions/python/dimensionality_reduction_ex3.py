# Chapter 15, Exercise 3: UMAP Parameter Exploration
# Using the three-group simulated dataset from the chapter

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

# ---- Simulate data (same as chapter) ----
np.random.seed(42)
n_per = 100
p = 20

g1 = np.random.normal(0, 1, (n_per, p))
g2 = np.random.normal(2, 1.2, (n_per, p))
g3 = np.random.normal(-1, 0.8, (n_per, p))
g2[:, :5] += 3
g3[:, 10:15] -= 4

X = np.vstack([g1, g2, g3])
labels = np.repeat(["Type A", "Type B", "Type C"], n_per)
colors = {"Type A": "steelblue", "Type B": "firebrick", "Type C": "forestgreen"}

# ---- (a) UMAP with varying n_neighbors, min_dist = 0.1 ----
n_neighbors_vals = [5, 15, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, nn in zip(axes, n_neighbors_vals):
    umap_emb = UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                      random_state=42).fit_transform(X)
    for lab in ["Type A", "Type B", "Type C"]:
        mask = labels == lab
        ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1], c=colors[lab],
                   s=15, alpha=0.7, label=lab)
    ax.set_title(f"n_neighbors = {nn}, min_dist = 0.1")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8)

plt.suptitle("UMAP: Varying n_neighbors", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("ch15_ex3_umap_neighbors.png", dpi=150)
plt.show()

# ---- (b) UMAP with varying min_dist, n_neighbors = 15 ----
min_dist_vals = [0.0, 0.1, 0.5, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, md in zip(axes, min_dist_vals):
    umap_emb = UMAP(n_components=2, n_neighbors=15, min_dist=md,
                      random_state=42).fit_transform(X)
    for lab in ["Type A", "Type B", "Type C"]:
        mask = labels == lab
        ax.scatter(umap_emb[mask, 0], umap_emb[mask, 1], c=colors[lab],
                   s=15, alpha=0.7, label=lab)
    ax.set_title(f"n_neighbors = 15, min_dist = {md}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8)

plt.suptitle("UMAP: Varying min_dist", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("ch15_ex3_umap_mindist.png", dpi=150)
plt.show()

# ---- (d) Description and recommendation ----
print("=== Part (d): How Each Parameter Affects Visual Appearance ===\n")

print("n_neighbors (with min_dist = 0.1 fixed):")
print("  n_neighbors = 5:   Very tight, fragmented clusters. Emphasises")
print("    micro-structure, may split true clusters.")
print("  n_neighbors = 15:  Well-separated, compact clusters. Good balance")
print("    between local and global structure.")
print("  n_neighbors = 50:  Broader clusters, more connected. Groups")
print("    start to merge as wider neighbourhoods are considered.")
print("  n_neighbors = 100: Spread out. Global structure emphasised.\n")

print("min_dist (with n_neighbors = 15 fixed):")
print("  min_dist = 0.0:  Very tight, dense clusters. Maximum separation.")
print("  min_dist = 0.1:  Slightly looser. Good default.")
print("  min_dist = 0.5:  Spread-out embedding. Internal structure visible")
print("    but inter-cluster gaps shrink.")
print("  min_dist = 1.0:  Very spread out, near-uniform. Clusters overlap.\n")

print("Recommendation for this dataset:")
print("  n_neighbors = 15, min_dist = 0.1")
print("  This provides well-separated, compact clusters that clearly")
print("  reveal the three-group structure. It is the default for good")
print("  reason: it balances local detail with global structure.")
