# =============================================================================
# Chapter 5, Exercise 3: The Effect of Correlation
# How LASSO and Ridge handle correlated predictors differently
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

np.random.seed(2025)
n = 200

# --- Create 4 correlated predictors (correlation ~ 0.9) ---
mean = np.zeros(4)
cov = np.full((4, 4), 0.9)
np.fill_diagonal(cov, 1.0)
X_corr = np.random.multivariate_normal(mean, cov, size=n)

# Add 6 independent noise predictors
X_indep = np.random.randn(n, 6)
X = np.hstack([X_corr, X_indep])

feature_names = ([f"Corr_{i+1}" for i in range(4)] +
                 [f"Noise_{i+1}" for i in range(6)])

# True model: all 4 correlated predictors contribute
true_beta = np.array([0.5]*4 + [0.0]*6)
y = np.random.binomial(1, expit(X @ true_beta))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Task 1: LASSO ---
lasso = LogisticRegressionCV(penalty='l1', solver='saga', Cs=50,
                              cv=10, max_iter=10000, random_state=42)
lasso.fit(X_scaled, y)

print("=== Task 1: LASSO coefficients ===")
for name, coef in zip(feature_names, lasso.coef_[0]):
    marker = " *" if abs(coef) > 1e-6 else ""
    print(f"  {name:10s}: {coef:8.4f}{marker}")

# --- Task 2: Ridge ---
ridge = LogisticRegressionCV(penalty='l2', solver='lbfgs', Cs=50,
                              cv=10, max_iter=10000, random_state=42)
ridge.fit(X_scaled, y)

print("\n=== Task 2: Ridge coefficients ===")
for name, coef in zip(feature_names, ridge.coef_[0]):
    print(f"  {name:10s}: {coef:8.4f}")

# --- Task 3: Elastic Net ---
enet = LogisticRegressionCV(penalty='elasticnet', solver='saga', Cs=50,
                             cv=10, l1_ratios=[0.5], max_iter=10000,
                             random_state=42)
enet.fit(X_scaled, y)

print("\n=== Task 3: Elastic Net coefficients ===")
for name, coef in zip(feature_names, enet.coef_[0]):
    marker = " *" if abs(coef) > 1e-6 else ""
    print(f"  {name:10s}: {coef:8.4f}{marker}")

# --- Question (a): Does LASSO select all 4 correlated predictors? ---
print("\n=== Question (a) ===")
selected_lasso = [name for name, coef in zip(feature_names, lasso.coef_[0])
                  if abs(coef) > 1e-6]
print(f"LASSO selected: {selected_lasso}")
print("LASSO typically selects only 1-2 of the 4 correlated predictors.")
print("This is because the LASSO arbitrarily picks one predictor from a")
print("correlated group and drops the rest. With correlation ~0.9, any")
print("single predictor carries almost the same information as all four.")

# --- Question (b): How does Ridge handle correlated predictors? ---
print("\n=== Question (b) ===")
print("Ridge coefficients for correlated predictors:")
for i in range(4):
    print(f"  {feature_names[i]}: {ridge.coef_[0][i]:.4f}")
print("Ridge distributes the effect EVENLY across all correlated predictors.")
print("All four retain non-zero (and similar) coefficients.")

# --- Question (c): Does Elastic Net improve on LASSO? ---
print("\n=== Question (c) ===")
selected_enet = [name for name, coef in zip(feature_names, enet.coef_[0])
                 if abs(coef) > 1e-6]
print(f"Elastic Net selected: {selected_enet}")
print("The Elastic Net typically selects MORE of the correlated predictors")
print("than LASSO, thanks to the L2 component (grouping effect).")

# --- Question (d): Stability analysis ---
print("\n=== Question (d): Stability analysis (10 bootstrap samples) ===")
print("\nLASSO variable selection:")
for seed in range(10):
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, n, replace=True)
    lasso_boot = LogisticRegressionCV(penalty='l1', solver='saga',
                                       Cs=20, cv=5, max_iter=10000,
                                       random_state=42)
    lasso_boot.fit(X_scaled[idx], y[idx])
    selected = [feature_names[j] for j in range(10)
                if abs(lasso_boot.coef_[0][j]) > 1e-6]
    print(f"  Seed {seed}: {selected}")

print("\nRidge — all predictors retained (non-zero coefficients):")
for seed in range(10):
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, n, replace=True)
    ridge_boot = LogisticRegressionCV(penalty='l2', solver='lbfgs',
                                       Cs=20, cv=5, max_iter=10000,
                                       random_state=42)
    ridge_boot.fit(X_scaled[idx], y[idx])
    # Ridge always keeps all predictors; show which have coef > 0.01
    selected = [feature_names[j] for j in range(10)
                if abs(ridge_boot.coef_[0][j]) > 0.01]
    print(f"  Seed {seed}: {selected}")

print("\nConclusion: LASSO variable selection is UNSTABLE for correlated")
print("predictors -- the selected predictor(s) change across samples.")
print("Ridge is stable because it always retains all predictors.")
print("Elastic Net offers a compromise with better stability than LASSO.")
