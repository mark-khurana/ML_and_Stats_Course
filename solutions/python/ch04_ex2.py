# =============================================================================
# Chapter 4, Exercise 2: Fitting and Interpreting Spline Models
# PBC dataset: bilirubin and transplant status
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrix

# --- Simulate data similar to PBC bilirubin-transplant relationship ---
# (Following the chapter's Python approach since PBC is an R-native dataset)
np.random.seed(42)
n = 300
bili = np.random.exponential(scale=2, size=n)
bili = np.clip(bili, 0.3, 25)
logit_p = -3 + 0.1 * bili + 0.005 * bili**2  # Non-linear true relationship
prob = 1 / (1 + np.exp(-logit_p))
transplant = np.random.binomial(1, prob)

df = pd.DataFrame({'bili': bili, 'transplant': transplant})

# --- Task 1: Fit logistic regression with linear bilirubin ---
X_linear = sm.add_constant(df[['bili']])
fit_linear = sm.GLM(df['transplant'], X_linear,
                     family=sm.families.Binomial()).fit()
print("=== Linear Model ===")
print(fit_linear.summary())

# --- Task 2: Create spline basis and fit logistic regression with splines ---
# cr() creates a natural cubic spline (restricted cubic spline)
# df=3 means 4 knots (df = number of knots - 1)
spline_basis = dmatrix("cr(bili, df=3)", data=df, return_type='dataframe')
fit_spline = sm.GLM(df['transplant'], spline_basis,
                      family=sm.families.Binomial()).fit()
print("\n=== Spline Model (RCS, 4 knots) ===")
print(fit_spline.summary())

# --- Task 3: Compare AIC ---
print(f"\n=== AIC Comparison ===")
print(f"Linear model AIC: {fit_linear.aic:.1f}")
print(f"Spline model AIC: {fit_spline.aic:.1f}")
print("Lower AIC = better fit (penalised for complexity)")

# --- Task 4: Test for non-linearity ---
# Compare the linear model to the spline model using a likelihood ratio test.
# The difference in deviance follows a chi-squared distribution with
# df = (spline params - linear params) degrees of freedom.
from scipy import stats

lr_stat = fit_linear.deviance - fit_spline.deviance
df_diff = fit_linear.df_model - fit_spline.df_model  # difference in parameters
# Note: df_model counts differently in statsmodels; compute manually
n_params_linear = len(fit_linear.params)
n_params_spline = len(fit_spline.params)
df_diff = n_params_spline - n_params_linear

p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
print(f"\n=== Likelihood Ratio Test for Non-linearity ===")
print(f"LR statistic: {lr_stat:.3f}")
print(f"Degrees of freedom: {df_diff}")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Significant non-linearity detected. The spline model fits better.")
else:
    print("Result: No significant non-linearity. A linear term may suffice.")

# --- Task 5: Plot the predicted probabilities across bilirubin range ---
bili_range = np.linspace(df['bili'].min(), df['bili'].max(), 200)

# Linear predictions
X_lin_pred = sm.add_constant(pd.DataFrame({'bili': bili_range}))
lin_prob = fit_linear.predict(X_lin_pred)

# Spline predictions
spline_pred = dmatrix("cr(bili, df=3)",
                       data=pd.DataFrame({'bili': bili_range}),
                       return_type='dataframe')
rcs_prob = fit_spline.predict(spline_pred)

plt.figure(figsize=(10, 6))
plt.plot(bili_range, lin_prob, color='#3498db', linewidth=1.5,
         label='Linear model')
plt.plot(bili_range, rcs_prob, color='#2c3e50', linewidth=2,
         label='RCS model (4 knots)')
plt.scatter(df['bili'], df['transplant'], alpha=0.15, color='grey', s=20)
plt.xlabel('Serum Bilirubin (mg/dL)')
plt.ylabel('Predicted Probability of Transplant')
plt.title('Non-linear effect of bilirubin on transplant (RCS)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Task 6: Interpretation ---
print("\n=== Interpretation ===")
print("The plot shows the predicted probability of transplant as a function of")
print("serum bilirubin, comparing the linear and RCS models.")
print("If the RCS curve departs notably from a straight line, the relationship")
print("is non-linear and the linear model is insufficient.")
print("The AIC comparison and likelihood ratio test formally assess this.")
print("The RCS model captures the accelerating risk at higher bilirubin levels")
print("that the linear model cannot represent.")
