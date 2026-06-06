# =============================================================================
# Chapter 8b, Exercise 1: Neural Network vs XGBoost on Tabular Data
# Fit both a neural network and XGBoost on the readmission dataset.
# Compare 5-fold cross-validated AUC.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- Simulate readmission data (same as chapter) ---
np.random.seed(42)
n = 1000

age = np.random.normal(68, 12, n)
length_of_stay = np.random.poisson(5, n) + 1
num_comorbidities = np.random.poisson(3, n)
prior_admissions = np.random.poisson(1, n)
discharge_hgb = np.random.normal(11, 2, n)
discharge_creatinine = np.random.lognormal(0.2, 0.5, n)
has_diabetes = np.random.binomial(1, 0.35, n)
has_chf = np.random.binomial(1, 0.25, n)

X = np.column_stack([age, length_of_stay, num_comorbidities,
                     prior_admissions, discharge_hgb,
                     discharge_creatinine, has_diabetes, has_chf])

# Generate outcome with known logistic relationship
prob = 1 / (1 + np.exp(-(-3 + 0.02 * (age - 68) +
                          0.15 * prior_admissions +
                          0.1 * num_comorbidities +
                          0.3 * has_chf -
                          0.1 * discharge_hgb)))
y = np.random.binomial(1, prob)
print(f"Readmission rate: {y.mean():.3f}")

# --- XGBoost with 5-fold CV ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)

xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
print(f"\nXGBoost CV AUC: {xgb_scores.mean():.3f} (+/- {xgb_scores.std():.3f})")

# --- Neural Network with 5-fold CV (manual loop) ---
nn_aucs = []
fold_idx = 0

for train_idx, val_idx in cv.split(X, y):
    fold_idx += 1

    # Split and scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Build neural network (same architecture as chapter)
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )

    # Train with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
        verbose=0
    )

    # Evaluate
    y_pred_prob = model.predict(X_val, verbose=0).ravel()
    fold_auc = roc_auc_score(y_val, y_pred_prob)
    nn_aucs.append(fold_auc)
    print(f"  Fold {fold_idx}: NN AUC = {fold_auc:.3f}")

nn_mean_auc = np.mean(nn_aucs)
nn_std_auc = np.std(nn_aucs)
print(f"\nNeural Network CV AUC: {nn_mean_auc:.3f} (+/- {nn_std_auc:.3f})")

# --- Comparison ---
print(f"\n=== Comparison ===")
print(f"XGBoost CV AUC:        {xgb_scores.mean():.3f}")
print(f"Neural Network CV AUC: {nn_mean_auc:.3f}")

print("\nInterpretation:")
print("XGBoost typically matches or outperforms neural networks on tabular")
print("clinical data. This is expected -- the Grinsztajn et al. (2022)")
print("NeurIPS benchmark showed that tree-based models consistently")
print("outperform neural networks on typical tabular datasets. Deep learning")
print("excels on images, text, and sequences, not spreadsheets.")
