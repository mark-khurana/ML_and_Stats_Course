# =============================================================================
# Chapter 5, Exercise 2: Predicting Diabetes Onset
# Pima Indians Diabetes dataset
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# --- Load the Pima Indians Diabetes dataset ---
url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
       "pima-indians-diabetes.data.csv")
columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
           'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
pima = pd.read_csv(url, names=columns)

# Remove rows with zero values where zero is not meaningful
cols_no_zero = ['glucose', 'blood_pressure', 'skin_thickness',
                'insulin', 'bmi']
pima[cols_no_zero] = pima[cols_no_zero].replace(0, np.nan)
pima = pima.dropna()

print(f"Complete cases: {len(pima)}\n")

X = pima[columns[:-1]].values
y = pima['outcome'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Task 1: Fit LASSO with cross-validation ---
lasso_cv = LogisticRegressionCV(
    penalty='l1', solver='saga', Cs=50, cv=10,
    scoring='neg_log_loss', max_iter=10000, random_state=42
)
lasso_cv.fit(X_scaled, y)

print("=== LASSO Cross-Validation ===")
print(f"Best C (= 1/lambda): {lasso_cv.C_[0]:.4f}")
print(f"Best lambda: {1/lasso_cv.C_[0]:.4f}\n")

print("LASSO selected features:")
n_selected_lasso = 0
for name, coef in zip(columns[:-1], lasso_cv.coef_[0]):
    if abs(coef) > 1e-6:
        print(f"  {name}: {coef:.4f}")
        n_selected_lasso += 1
print(f"\nNumber selected: {n_selected_lasso} of 8\n")

# --- Task 2: Fit Ridge with cross-validation ---
ridge_cv = LogisticRegressionCV(
    penalty='l2', solver='lbfgs', Cs=50, cv=10,
    scoring='neg_log_loss', max_iter=10000, random_state=42
)
ridge_cv.fit(X_scaled, y)

# --- Task 3: Fit Elastic Net with cross-validation ---
enet_cv = LogisticRegressionCV(
    penalty='elasticnet', solver='saga', Cs=50, cv=10,
    l1_ratios=[0.25, 0.5, 0.75], scoring='neg_log_loss',
    max_iter=10000, random_state=42
)
enet_cv.fit(X_scaled, y)

print(f"Elastic Net best l1_ratio (alpha): {enet_cv.l1_ratio_[0]:.2f}")
print(f"Elastic Net best C (1/lambda): {enet_cv.C_[0]:.4f}\n")

# --- Task 4: Compare cross-validated scores ---
print("=== Cross-validated log-loss (lower = better) ===")
for name, model in [('LASSO', lasso_cv), ('Ridge', ridge_cv),
                     ('Elastic Net', enet_cv)]:
    scores = cross_val_score(model, X_scaled, y, cv=10,
                              scoring='neg_log_loss')
    print(f"  {name}: {-scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Task 5: Plot regularisation path for LASSO ---
Cs = np.logspace(-2, 2, 80)
coefs = []
for C in Cs:
    model = LogisticRegression(penalty='l1', C=C, solver='saga',
                                max_iter=10000, random_state=42)
    model.fit(X_scaled, y)
    coefs.append(model.coef_[0])

coefs = np.array(coefs)
lambdas = 1.0 / Cs

plt.figure(figsize=(10, 6))
for j, name in enumerate(columns[:-1]):
    plt.plot(np.log10(lambdas), coefs[:, j], label=name)
plt.axvline(x=np.log10(1/lasso_cv.C_[0]), color='k', linestyle='--',
            label='CV optimal lambda')
plt.xlabel('log10(lambda)')
plt.ylabel('Coefficient')
plt.title('LASSO Regularisation Path - Pima Diabetes')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()

# --- Task 6: Recommendation ---
print("\n=== Recommendation ===")
print("For the Pima diabetes dataset (8 predictors, ~390 complete cases):")
print("- The number of predictors is small relative to sample size,")
print("  so penalisation is not strictly necessary.")
print("- LASSO is useful for identifying the most predictive variables.")
print("- All three methods produce similar cross-validated performance.")
print("- For variable selection, LASSO provides a parsimonious model.")
print("- For best prediction, any method works well here.")

# --- Bonus: Compare coefficients side by side ---
print("\n=== Coefficient comparison ===")
coef_df = pd.DataFrame({
    'Variable': columns[:-1],
    'LASSO': lasso_cv.coef_[0],
    'Ridge': ridge_cv.coef_[0],
    'Elastic Net': enet_cv.coef_[0]
})
print(coef_df.round(4).to_string(index=False))
