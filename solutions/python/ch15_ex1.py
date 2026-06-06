# Chapter 15, Exercise 1: PCA on Clinical Lab Data
# Using the simulated metabolic panel data from the chapter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---- Simulate metabolic data (same as chapter) ----
np.random.seed(42)
n = 300

Sigma = np.array([
    [1.0, 0.7, 0.5, 0.3,-0.2, 0.1, 0.2,-0.1],
    [0.7, 1.0, 0.4, 0.2,-0.2, 0.1, 0.1,-0.1],
    [0.5, 0.4, 1.0, 0.4,-0.3, 0.1, 0.3,-0.1],
    [0.3, 0.2, 0.4, 1.0,-0.5, 0.1, 0.2, 0.0],
    [-0.2,-0.2,-0.3,-0.5, 1.0,-0.1,-0.1, 0.2],
    [0.1, 0.1, 0.1, 0.1,-0.1, 1.0, 0.1,-0.3],
    [0.2, 0.1, 0.3, 0.2,-0.1, 0.1, 1.0,-0.2],
    [-0.1,-0.1,-0.1, 0.0, 0.2,-0.3,-0.2, 1.0]
])

z = np.random.multivariate_normal(np.zeros(8), Sigma, n)
labels = ["glucose", "hba1c", "triglycerides", "ldl",
          "hdl", "creatinine", "alt", "albumin"]

metabolic = pd.DataFrame({
    "glucose": np.round(z[:, 0] * 30 + 100),
    "hba1c": np.round(z[:, 1] * 1.0 + 5.8, 1),
    "triglycerides": np.round(z[:, 2] * 50 + 150),
    "ldl": np.round(z[:, 3] * 30 + 120),
    "hdl": np.round(z[:, 4] * 12 + 55),
    "creatinine": np.round(z[:, 5] * 0.3 + 1.0, 2),
    "alt": np.round(z[:, 6] * 15 + 30),
    "albumin": np.round(z[:, 7] * 0.4 + 4.0, 1)
})

# ---- (a) PCA on standardised data ----
print("=== Part (a): PCA on Standardised Data ===")
X = metabolic.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
scores = pca.fit_transform(X_scaled)

print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))

# ---- (b) Scree plot and component selection ----
print("\n=== Part (b): Scree Plot and Component Selection ===")

var_prop = pca.explained_variance_ratio_
cum_var = np.cumsum(var_prop)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
axes[0].bar(range(1, 9), var_prop, color="steelblue", edgecolor="white")
axes[0].axhline(y=1/8, color="firebrick", linestyle="--", label="Kaiser threshold")
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Proportion of Variance")
axes[0].set_title("Scree Plot")
axes[0].set_xticks(range(1, 9))
axes[0].legend()

# Cumulative variance
axes[1].plot(range(1, 9), cum_var, "o-", color="steelblue", lw=2)
axes[1].axhline(y=0.80, color="firebrick", linestyle="--", label="80% threshold")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Proportion")
axes[1].set_title("Cumulative Variance")
axes[1].set_xticks(range(1, 9))
axes[1].legend()

plt.tight_layout()
plt.savefig("ch15_ex1_scree.png", dpi=150)
plt.show()

# Kaiser criterion: eigenvalues > 1
eigenvalues = pca.explained_variance_
n_kaiser = np.sum(eigenvalues > 1)
n_80 = np.argmax(cum_var >= 0.80) + 1

print(f"\nKaiser criterion (eigenvalue > 1): {n_kaiser} components")
print(f"80% cumulative variance threshold: {n_80} components")
print(f"Eigenvalues: {np.round(eigenvalues, 3)}")
print(f"Cumulative variance: {np.round(cum_var, 3)}")

# ---- (c) Loadings of first two PCs ----
print("\n=== Part (c): Loadings and Interpretation ===")

for pc_idx in [0, 1]:
    loadings = pca.components_[pc_idx]
    order = np.argsort(np.abs(loadings))[::-1]
    print(f"\nPC{pc_idx+1} loadings (sorted by |loading|):")
    for idx in order:
        print(f"  {labels[idx]:>15s}: {loadings[idx]:+.3f}")

print("\nClinical interpretation:")
print("PC1: Dominated by glucose, HbA1c, triglycerides (positive) and")
print("     HDL (negative). This is a 'metabolic syndrome' axis.")
print("PC2: Dominated by creatinine and albumin (opposite signs).")
print("     This captures a 'renal/hepatic function' dimension.")

# ---- (d) Biplot coloured by diabetes status ----
print("\n=== Part (d): Biplot with Diabetes Status ===")

diabetes = np.where(
    (metabolic["glucose"] > 110) & (metabolic["hba1c"] > 6.2),
    "Diabetic", "Non-diabetic"
)
print(f"Diabetes prevalence: {(diabetes == 'Diabetic').mean():.3f}")

fig, ax = plt.subplots(figsize=(9, 7))

# Plot scores
for label, color in [("Non-diabetic", "steelblue"), ("Diabetic", "firebrick")]:
    mask = diabetes == label
    ax.scatter(scores[mask, 0], scores[mask, 1], c=color, s=20,
               alpha=0.6, label=label)

# Add loading arrows
loadings_2d = pca.components_[:2].T
scale_factor = 3
for i, lab in enumerate(labels):
    ax.annotate("", xy=(loadings_2d[i, 0]*scale_factor,
                          loadings_2d[i, 1]*scale_factor),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="grey30", lw=1.5))
    ax.text(loadings_2d[i, 0]*scale_factor*1.15,
             loadings_2d[i, 1]*scale_factor*1.15,
             lab, fontsize=8, color="grey20", ha="center")

ax.set_xlabel(f"PC1 ({var_prop[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({var_prop[1]:.1%} var)")
ax.set_title("PCA Biplot Coloured by Diabetes Status")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("ch15_ex1_biplot.png", dpi=150)
plt.show()

print("\nDiabetic patients tend to cluster toward higher PC1 values,")
print("aligning with the 'metabolic syndrome' interpretation.")
print("PC1 captures the metabolic dimension along which diabetic")
print("patients differ from non-diabetic patients.")
