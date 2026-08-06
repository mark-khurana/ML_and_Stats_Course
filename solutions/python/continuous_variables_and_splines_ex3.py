# =============================================================================
# Chapter 4, Exercise 3: Categorisation vs Splines Head-to-Head
# Simulate U-shaped relationship and compare approaches
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix
from scipy.special import expit

np.random.seed(2025)
n = 1000
x = np.random.normal(50, 10, size=n)

# True U-shaped relationship
logit_p = -5 + 0.004 * (x - 50)**2
y = np.random.binomial(1, expit(logit_p))

df = pd.DataFrame({'x': x, 'y': y})

# --- Model 1: Categorise into quartiles ---
df['x_cat'] = pd.qcut(df['x'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
X_cat = pd.get_dummies(df['x_cat'], drop_first=True).astype(float)
X_cat = sm.add_constant(X_cat)
fit_cat = sm.GLM(df['y'], X_cat, family=sm.families.Binomial()).fit()

# --- Model 2: Linear ---
X_lin = sm.add_constant(df[['x']])
fit_lin = sm.GLM(df['y'], X_lin, family=sm.families.Binomial()).fit()

# --- Model 3: RCS with df=4 (5 knots) ---
X_rcs = dmatrix("cr(x, df=4)", data=df, return_type='dataframe')
fit_rcs = sm.GLM(df['y'], X_rcs, family=sm.families.Binomial()).fit()

# --- Question (a): Compare AIC ---
print("=== Question (a): AIC Comparison ===")
print(f"Categorised (quartiles): {fit_cat.aic:.1f}")
print(f"Linear:                  {fit_lin.aic:.1f}")
print(f"RCS (5 knots):           {fit_rcs.aic:.1f}")
print("\nThe RCS model should have the lowest AIC, indicating best fit.")
print("The linear model has the worst fit because it cannot capture")
print("the U-shape at all.\n")

# --- Question (b): Plot all three fitted curves with the true function ---
x_range = np.linspace(df['x'].min(), df['x'].max(), 200)

# True curve
true_logit = -5 + 0.004 * (x_range - 50)**2
true_prob = expit(true_logit)

# RCS predictions
X_rcs_pred = dmatrix("cr(x, df=4)",
                      data=pd.DataFrame({'x': x_range}),
                      return_type='dataframe')
rcs_prob = fit_rcs.predict(X_rcs_pred)

# Linear predictions
X_lin_pred = sm.add_constant(pd.DataFrame({'x': x_range}))
lin_prob = fit_lin.predict(X_lin_pred)

# Categorised predictions
# Assign each x_range value to its quartile, then predict
quartile_edges = df['x'].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
quartile_edges[0] = -np.inf
quartile_edges[-1] = np.inf
x_cat_pred = pd.cut(x_range, bins=quartile_edges,
                      labels=['Q1', 'Q2', 'Q3', 'Q4'])
X_cat_pred = pd.get_dummies(x_cat_pred, drop_first=True).astype(float)
X_cat_pred = sm.add_constant(X_cat_pred)
# Ensure columns match
for col in X_cat.columns:
    if col not in X_cat_pred.columns:
        X_cat_pred[col] = 0.0
X_cat_pred = X_cat_pred[X_cat.columns]
cat_prob = fit_cat.predict(X_cat_pred)

plt.figure(figsize=(10, 6))
plt.plot(x_range, true_prob, 'k--', linewidth=2, label='True relationship')
plt.plot(x_range, rcs_prob, color='#27ae60', linewidth=2,
         label='RCS (5 knots)')
plt.plot(x_range, cat_prob, color='#e74c3c', linewidth=1.5,
         label='Categorised (quartiles)')
plt.plot(x_range, lin_prob, color='#3498db', linewidth=1.5, label='Linear')
plt.xlabel('Predictor (x)')
plt.ylabel('Predicted Probability')
plt.title('Comparing modelling strategies for a U-shaped relationship')
plt.legend()
plt.tight_layout()
plt.show()

print("=== Question (b): Interpretation ===")
print("The RCS model best recovers the true U-shaped relationship.")
print("The linear model completely misses the U-shape.")
print("The categorised model captures some pattern but in crude steps.\n")

# --- Question (c): Change quartiles to tertiles ---
print("=== Question (c): Tertile categorisation ===")
df['x_tert'] = pd.qcut(df['x'], q=3, labels=['T1', 'T2', 'T3'])
X_tert = pd.get_dummies(df['x_tert'], drop_first=True).astype(float)
X_tert = sm.add_constant(X_tert)
fit_tert = sm.GLM(df['y'], X_tert, family=sm.families.Binomial()).fit()

print(f"AIC with tertiles:  {fit_tert.aic:.1f}")
print(f"AIC with quartiles: {fit_cat.aic:.1f}")
print(f"AIC with RCS:       {fit_rcs.aic:.1f}")
print("\nChanging from quartiles to tertiles changes the categorised model's")
print("estimates. This demonstrates that categorisation results are ARBITRARY")
print("and depend on cut-point choice. The RCS model does not have this")
print("problem because it models the continuous relationship directly.")

# --- Likelihood ratio test for non-linearity ---
from scipy import stats

lr_stat = fit_lin.deviance - fit_rcs.deviance
df_diff = len(fit_rcs.params) - len(fit_lin.params)
p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
print(f"\n=== Non-linearity test ===")
print(f"LR statistic: {lr_stat:.3f}, df: {df_diff}, p-value: {p_value:.6f}")
if p_value < 0.05:
    print("Significant non-linearity: the U-shaped relationship is confirmed.")
