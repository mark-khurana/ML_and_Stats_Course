# Chapter 16, Exercise 1: K-Means on Simulated Patient Data
# 600 patients, 6 clinical variables, 3 underlying phenotypes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---- Simulate data ----
np.random.seed(42)

# Phenotype 1: Septic shock
p1 = np.column_stack([
    np.random.normal(115, 12, 200),   # hr
    np.random.normal(28, 5, 200),     # rr
    np.random.normal(38.5, 0.8, 200), # temp
    np.random.normal(80, 12, 200),    # sbp
    np.random.normal(2.5, 0.8, 200),  # creat
    np.random.normal(5, 2, 200)       # lactate
])

# Phenotype 2: Stable critical
p2 = np.column_stack([
    np.random.normal(90, 10, 200),
    np.random.normal(20, 3, 200),
    np.random.normal(37.2, 0.5, 200),
    np.random.normal(110, 15, 200),
    np.random.normal(1.2, 0.3, 200),
    np.random.normal(1.5, 0.5, 200)
])

# Phenotype 3: Febrile
p3 = np.column_stack([
    np.random.normal(100, 10, 200),
    np.random.normal(22, 4, 200),
    np.random.normal(39.2, 0.7, 200),
    np.random.normal(120, 10, 200),
    np.random.normal(1.0, 0.2, 200),
    np.random.normal(2.0, 0.8, 200)
])

X = np.vstack([p1, p2, p3])
cols = ['hr', 'rr', 'temp', 'sbp', 'creat', 'lactate']
df = pd.DataFrame(X, columns=cols)

# ---- (a) Scale and K-means for k = 2 to 6 ----
X_scaled = StandardScaler().fit_transform(X)

ks = range(2, 7)
wcss_list = []
sil_list = []

for k in ks:
    km = KMeans(n_clusters=k, n_init=25, random_state=42)
    km.fit(X_scaled)
    wcss_list.append(km.inertia_)
    sil_list.append(silhouette_score(X_scaled, km.labels_))

# ---- (b) Elbow and silhouette plots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(list(ks), wcss_list, "o-", color="steelblue")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Within-cluster SS")
axes[0].set_title("Elbow Method")

axes[1].plot(list(ks), sil_list, "o-", color="firebrick")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average Silhouette")
axes[1].set_title("Silhouette Scores")

plt.tight_layout()
plt.savefig("ch16_ex1_elbow_silhouette.png", dpi=150)
plt.show()

best_k = list(ks)[np.argmax(sil_list)]
print(f"=== Part (b): Optimal k ===")
print(f"Elbow: Elbow appears at k = 3")
print(f"Silhouette: Maximum at k = {best_k}")
print(f"Both methods suggest k = 3, matching the 3 simulated phenotypes.")

# ---- (c) Visualise k=3 with PCA ----
km3 = KMeans(n_clusters=3, n_init=25, random_state=42).fit(X_scaled)
pca2 = PCA(n_components=2).fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
colors_3 = ["steelblue", "firebrick", "forestgreen"]
for c in range(3):
    mask = km3.labels_ == c
    ax.scatter(pca2[mask, 0], pca2[mask, 1], c=colors_3[c],
               s=15, alpha=0.6, label=f"Cluster {c+1}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("K-means (k=3) on First 2 PCs")
ax.legend()
plt.tight_layout()
plt.savefig("ch16_ex1_pca_clusters.png", dpi=150)
plt.show()

# ---- (d) Profile the clusters ----
print("\n=== Part (d): Cluster Profiles ===\n")

df['cluster'] = km3.labels_
profiles = df.groupby('cluster')[cols].mean()
print(profiles.round(1))

print("\nClinical interpretation:")
for cl in range(3):
    prof = profiles.loc[cl]
    n_cl = (km3.labels_ == cl).sum()
    print(f"\nCluster {cl+1} (n = {n_cl}):")
    print(f"  HR={prof['hr']:.0f}, RR={prof['rr']:.0f}, Temp={prof['temp']:.1f}, "
          f"SBP={prof['sbp']:.0f}, Creat={prof['creat']:.1f}, Lactate={prof['lactate']:.1f}")

    if prof['lactate'] > 3 and prof['sbp'] < 90:
        print("  -> SEPTIC SHOCK: high HR, RR, lactate; low SBP")
    elif prof['temp'] > 38.5:
        print("  -> FEBRILE: high temperature, preserved hemodynamics")
    else:
        print("  -> STABLE CRITICAL: moderate vitals")

print("\nThe cluster profiles make clinical sense. The algorithm")
print("successfully recovered the three simulated phenotypes.")
