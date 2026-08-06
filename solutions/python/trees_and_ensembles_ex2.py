# =============================================================================
# Chapter 8, Exercise 2: Random Forest vs XGBoost Tuning Challenge
# 1. Split data 80/20. 2. Tune RF and XGBoost with 5-fold CV.
# 3. Select best hyperparameters. 4. Evaluate on test set.
# 5. Create variable importance plots.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                      StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
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

# --- Part 1: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Part 2a: Tune Random Forest ---
rf_param_dist = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2', 3, 5, 7],
    'min_samples_leaf': [5, 10, 15, 20, 30],
    'max_depth': [5, 10, 15, 20, None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_search = RandomizedSearchCV(
    rf, rf_param_dist, n_iter=30, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)

print(f"\n--- Random Forest Tuning ---")
print(f"Best CV AUC: {rf_search.best_score_:.3f}")
print(f"Best params: {rf_search.best_params_}")

# --- Part 2b: Tune XGBoost ---
xgb_param_dist = {
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 5, 10]
}

xgb_model = xgb.XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)

xgb_search = RandomizedSearchCV(
    xgb_model, xgb_param_dist, n_iter=30, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train)

print(f"\n--- XGBoost Tuning ---")
print(f"Best CV AUC: {xgb_search.best_score_:.3f}")
print(f"Best params: {xgb_search.best_params_}")

# --- Part 4: Evaluate on test set ---
rf_best = rf_search.best_estimator_
xgb_best = xgb_search.best_estimator_

rf_test_probs = rf_best.predict_proba(X_test)[:, 1]
xgb_test_probs = xgb_best.predict_proba(X_test)[:, 1]

rf_test_auc = roc_auc_score(y_test, rf_test_probs)
xgb_test_auc = roc_auc_score(y_test, xgb_test_probs)

print(f"\n=== Test Set Performance ===")
print(f"Random Forest AUC: {rf_test_auc:.3f}")
print(f"XGBoost AUC:       {xgb_test_auc:.3f}")

if xgb_test_auc > rf_test_auc:
    print("\nXGBoost performs better on the test set.")
else:
    print("\nRandom Forest performs better on the test set.")

# --- Part 5: Variable importance plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest importance
rf_imp = pd.Series(rf_best.feature_importances_, index=feature_names)
rf_imp = rf_imp.sort_values(ascending=True)
rf_imp.plot(kind='barh', ax=axes[0], color='#2E86AB')
axes[0].set_xlabel("Feature Importance (Gini)")
axes[0].set_title("Random Forest Variable Importance")

# XGBoost importance
xgb_imp = pd.Series(xgb_best.feature_importances_, index=feature_names)
xgb_imp = xgb_imp.sort_values(ascending=True)
xgb_imp.plot(kind='barh', ax=axes[1], color='#D55E00')
axes[1].set_xlabel("Feature Importance (Gain)")
axes[1].set_title("XGBoost Variable Importance")

plt.tight_layout()
plt.show()

# Compare top features
print("\nRF top 3 features:", rf_imp.sort_values(ascending=False).head(3).index.tolist())
print("XGB top 3 features:", xgb_imp.sort_values(ascending=False).head(3).index.tolist())
print("\nBoth models should generally agree on the most important features,")
print("though rankings may differ due to how each algorithm uses features.")
