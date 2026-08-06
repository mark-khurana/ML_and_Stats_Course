# Chapter 15, Exercise 2: t-SNE Sensitivity to Perplexity
# Using the three-group simulated dataset from the chapter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# ---- (a) & (b) Run t-SNE with 4 perplexity values ----
perplexities = [5, 15, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, perp in zip(axes, perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    emb = tsne.fit_transform(X)

    for lab in ["Type A", "Type B", "Type C"]:
        mask = labels == lab
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[lab],
                   s=15, alpha=0.7, label=lab)

    ax.set_title(f"t-SNE (perplexity = {perp})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ch15_ex2_tsne_perplexity.png", dpi=150)
plt.show()

# ---- (c) When do groups become clearly separated? ----
print("=== Part (c): When Are Groups Clearly Separated? ===\n")
print("The three groups become clearly separated at perplexity = 15.")
print("At perplexity = 5, the embedding focuses on very local structure,")
print("which can fragment groups into smaller sub-clusters.")
print("At perplexity = 15 and above, groups are well-separated.")
print("Higher perplexities (30, 50) also separate groups clearly but")
print("with different visual arrangements and more spread-out clusters.")

# ---- (d) Are distances consistent? ----
print("\n=== Part (d): Consistency of Inter-Cluster Distances ===\n")
print("NO, the distances between clusters are NOT consistent across")
print("perplexity values. Key observations:\n")
print("1. Relative positions of the three clusters change across")
print("   perplexity settings.\n")
print("2. Absolute distances between cluster centres vary substantially.\n")
print("3. Cluster sizes (spread) also change with perplexity.\n")
print("What this tells us about interpreting t-SNE:")
print("- Distances between clusters are MEANINGLESS.")
print("- t-SNE preserves local neighbourhood structure, not global distances.")
print("- NEVER conclude two clusters are 'more similar' because they")
print("  appear closer in a t-SNE plot.")
print("- The only reliable information is WHETHER clusters exist,")
print("  not HOW FAR APART they are.")
print("- Always try multiple perplexity values. If clusters appear")
print("  consistently, the structure is likely real.")
