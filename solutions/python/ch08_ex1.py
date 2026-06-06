# =============================================================================
# Chapter 8, Exercise 1: Build and Prune a Classification Tree
# 1. Fit trees with different max_depth values and count leaves.
# 2. Cross-validate each to find optimal max_depth.
# 3. Plot the optimal tree.
# 4. Examine feature importances.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# --- Simulate the readmission dataset (same as chapter) ---
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.normal(68, 12, n),
    'length_of_stay': np.random.poisson(5, n) + 1,
    'num_comorbidities': np.random.poisson(3, n),
    'prior_admissions': np.random.poisson(1, n),
    'discharge_hemoglobin': np.random.normal(11, 2, n),
    'discharge_creatinine': np.random.lognormal(0.2, 0.5, n),
    'has_diabetes': np.random.binomial(1, 0.35, n),
    'has_chf': np.random.binomial(1, 0.25, n)
})
y = np.random.binomial(1, 0.18, n)

feature_names = list(df.columns)
print(f"Readmission rate: {y.mean():.3f}")

# --- Part 1: Fit a full, unpruned tree ---
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(df, y)
n_leaves_full = full_tree.get_n_leaves()
print(f"\nFull (unpruned) tree: {n_leaves_full} terminal nodes")
print(f"Full tree depth: {full_tree.get_depth()}")

# --- Part 2: Cross-validation to find optimal max_depth ---
depths = [2, 3, 4, 5, 7, 10, 15, 20, None]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nCross-validation results by max_depth:")
print(f"{'max_depth':>10s}  {'Mean AUC':>10s}  {'Std AUC':>10s}")
print("-" * 35)

cv_results = []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(tree, df, y, cv=cv, scoring='roc_auc')
    label = str(d) if d is not None else "None"
    cv_results.append({'max_depth': d, 'label': label,
                       'mean_auc': scores.mean(), 'std_auc': scores.std()})
    print(f"{label:>10s}  {scores.mean():>10.3f}  {scores.std():>10.3f}")

# Find the best depth
best = max(cv_results, key=lambda x: x['mean_auc'])
print(f"\nBest max_depth: {best['label']} (AUC = {best['mean_auc']:.3f})")

# --- Part 3: Fit and plot the optimal tree ---
best_depth = best['max_depth']
optimal_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
optimal_tree.fit(df, y)

print(f"\nOptimal tree: {optimal_tree.get_n_leaves()} terminal nodes")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(optimal_tree, feature_names=feature_names, class_names=['No', 'Yes'],
          filled=True, rounded=True, ax=ax, fontsize=9,
          max_depth=4)  # show up to depth 4 for readability
ax.set_title(f"Pruned Decision Tree (max_depth={best['label']})")
plt.tight_layout()
plt.show()

# --- Part 4: Top 3 splitting variables ---
importances = pd.Series(optimal_tree.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print("\nFeature Importances:")
print(importances.to_string())

top3 = importances.head(3).index.tolist()
print(f"\nTop 3 splitting variables: {top3}")

# Plot feature importances
fig, ax = plt.subplots(figsize=(8, 5))
importances.sort_values(ascending=True).plot(kind='barh', ax=ax, color='#2E86AB')
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()

print("\nClinical interpretation:")
print("- These variables capture patient acuity and complexity.")
print("- Discharge lab values reflect the patient's clinical status at discharge.")
print("- Age, comorbidity count, and prior admissions reflect disease burden.")
print("- These are well-established readmission risk factors in the literature.")
