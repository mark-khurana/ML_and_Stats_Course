# =============================================================================
# Chapter 7, Exercise 1: Cross-Validation Experiment
# Compare logistic regression and SVM (RBF kernel) using 10-fold stratified CV.
# Report AUC for both models.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Simulate the clinical dataset ---
np.random.seed(123)
n = 600

X = pd.DataFrame({
    'age': np.random.normal(65, 10, n),
    'creatinine': np.random.lognormal(0, 0.5, n),
    'hemoglobin': np.random.normal(12, 2, n),
    'platelets': np.random.normal(250, 70, n),
    'wbc': np.random.lognormal(2, 0.4, n)
})

# Simulate ICU outcome (binary)
prob = 1 / (1 + np.exp(-(-4 + 0.03 * np.random.normal(65, 10, n) +
                          0.5 * np.random.lognormal(0, 0.5, n))))
y = np.random.binomial(1, prob)
print(f"ICU admission rate: {y.mean():.3f}")

# --- 10-fold stratified CV ---
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# --- Logistic Regression ---
lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring='roc_auc')
print(f"\nLogistic Regression AUC: {lr_scores.mean():.3f} (+/- {lr_scores.std():.3f})")

# --- SVM (RBF kernel) ---
svm_pipe = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0, gamma=0.5, probability=True)
)
svm_scores = cross_val_score(svm_pipe, X, y, cv=cv, scoring='roc_auc')
print(f"SVM (RBF) AUC:           {svm_scores.mean():.3f} (+/- {svm_scores.std():.3f})")

# --- Comparison ---
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM (RBF)'],
    'Mean AUC': [lr_scores.mean(), svm_scores.mean()],
    'Std AUC': [lr_scores.std(), svm_scores.std()]
})
print("\n", results.to_string(index=False))

# --- Interpretation ---
# Both models are expected to show similar AUC because the simulated data
# has simple, roughly linear relationships. Logistic regression is well-suited
# for such data. The RBF SVM adds flexibility that is not needed here.
# With real clinical data containing non-linear effects, SVM may outperform
# logistic regression.
