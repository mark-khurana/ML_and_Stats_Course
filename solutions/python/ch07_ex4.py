# =============================================================================
# Chapter 7, Exercise 4: Spot the Data Leakage
# A colleague reports AUC ~ 0.99. Find the leaking features, remove them,
# and re-evaluate.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Simulate the clinical dataset (same as exercise) ---
np.random.seed(42)
n = 800

X = pd.DataFrame({
    'age': np.random.normal(65, 10, n),
    'creatinine': np.random.lognormal(0, 0.5, n),
    'hemoglobin': np.random.normal(12, 2, n),
    'wbc': np.random.lognormal(2, 0.4, n),
})
y = np.random.binomial(
    1,
    1 / (1 + np.exp(-(-4 + 0.03 * X['age'] + 0.5 * X['creatinine'])))
)

# Leaked features (consequences of ICU admission, not causes)
X['ventilator'] = np.where(y == 1, np.random.binomial(1, 0.85, n), 0)
X['sedation_score'] = np.where(y == 1, np.random.randint(1, 11, n), 0)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# --- Step 1: reproduce the colleague's result ---
pipe_leaked = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores_leaked = cross_val_score(pipe_leaked, X, y, cv=cv, scoring='roc_auc')
print(f"With leakage    — AUC: {scores_leaked.mean():.3f} (+/- {scores_leaked.std():.3f})")

# --- Step 2: identify and remove the leaking features ---
# ventilator: only ICU patients receive mechanical ventilation, so it is a
#   *consequence* of ICU admission, not a predictor available before admission.
# sedation_score: recorded only for ICU patients (0 for everyone else), so
#   the value directly encodes the outcome.
X_clean = X.drop(columns=['ventilator', 'sedation_score'])

# --- Step 3: re-evaluate ---
pipe_clean = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores_clean = cross_val_score(pipe_clean, X_clean, y, cv=cv, scoring='roc_auc')
print(f"Without leakage — AUC: {scores_clean.mean():.3f} (+/- {scores_clean.std():.3f})")

# --- Interpretation ---
# The AUC drops dramatically (from ~0.99 to something much more modest).
# The original near-perfect AUC was an artefact: ventilator and sedation_score
# are recorded *after* ICU admission and essentially encode the outcome.
# Including them is the ML equivalent of looking at the answer sheet.
# In clinical ML, always ask: "Would this variable be available at the time
# the prediction needs to be made?" If not, it must be excluded.
