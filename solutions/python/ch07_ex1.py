# =============================================================================
# Chapter 7, Exercise 1: Feature Engineering and Cross-Validation
# Compare logistic regression with raw features vs engineered features
# using 10-fold stratified CV. Report AUC for both models.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Simulate the clinical dataset ---
np.random.seed(123)
n = 600

X = pd.DataFrame({
    'age': np.random.normal(65, 10, n),
    'sex': np.random.binomial(1, 0.5, n),
    'creatinine': np.random.lognormal(0, 0.5, n),
    'hemoglobin': np.random.normal(12, 2, n),
    'platelets': np.random.normal(250, 70, n),
    'wbc': np.random.lognormal(2, 0.4, n)
})

y = np.random.binomial(
    1,
    1 / (1 + np.exp(-(-4 + 0.03 * X['age'] + 0.5 * X['creatinine'])))
)
print(f"ICU admission rate: {y.mean():.3f}")

# --- 10-fold stratified CV ---
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# --- Model A: raw features ---
pipe_raw = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores_raw = cross_val_score(pipe_raw, X, y, cv=cv, scoring='roc_auc')
print(f"\nModel A (raw features) AUC: {scores_raw.mean():.3f} (+/- {scores_raw.std():.3f})")

# --- Model B: engineered features ---
X_eng = X.copy()

# Simplified eGFR (CKD-EPI-inspired, not the full equation)
# Higher creatinine -> lower eGFR; older age -> lower eGFR
X_eng['egfr'] = (
    140 * (np.minimum(X['creatinine'], 0.9) / 0.9) ** (-0.411)
    * (np.maximum(X['creatinine'], 0.9) / 0.9) ** (-1.209)
    * 0.993 ** X['age']
    * np.where(X['sex'] == 1, 1.0, 1.018)
)

# Hemoglobin-to-platelet ratio
X_eng['hb_platelet_ratio'] = X['hemoglobin'] / X['platelets']

# Log-transformed WBC (reduces skew)
X_eng['log_wbc'] = np.log(X['wbc'])

pipe_eng = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores_eng = cross_val_score(pipe_eng, X_eng, y, cv=cv, scoring='roc_auc')
print(f"Model B (engineered)   AUC: {scores_eng.mean():.3f} (+/- {scores_eng.std():.3f})")

# --- Comparison ---
results = pd.DataFrame({
    'Model': ['A (raw)', 'B (engineered)'],
    'Mean AUC': [scores_raw.mean(), scores_eng.mean()],
    'Std AUC': [scores_raw.std(), scores_eng.std()]
})
print("\n", results.to_string(index=False))

# --- Interpretation ---
# In this simulated dataset the true outcome depends on age and creatinine
# via a logistic link. Since logistic regression can already capture that
# linear relationship from the raw features, the engineered features (eGFR,
# ratios, log transforms) may add only a modest improvement — or none at all.
#
# In real clinical data, feature engineering often matters more: eGFR is a
# non-linear transform of creatinine that better reflects kidney function,
# and log-WBC handles the right skew common in lab values. The lesson is
# that engineered features encode domain knowledge the model cannot discover
# on its own from raw inputs — even if the benefit is small in this toy
# example.
