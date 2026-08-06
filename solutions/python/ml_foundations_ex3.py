# =============================================================================
# Chapter 7, Exercise 3: The Bias-Variance Tradeoff in Practice
# Fit polynomials of degree 1, 3, 5, 10, and 20 to a training set.
# Plot fitted curves, compute training/test error, identify optimal degree.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Generate data ---
np.random.seed(42)
x_train = np.sort(np.random.uniform(0, 10, 50))
y_train = np.sin(x_train) + np.random.normal(0, 0.3, 50)
x_test = np.sort(np.random.uniform(0, 10, 200))
y_test = np.sin(x_test) + np.random.normal(0, 0.3, 200)

# --- Fit polynomials and compute RMSE ---
degrees = [1, 3, 5, 10, 20]
results = []
predictions = {}

for d in degrees:
    # Create polynomial features and fit
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    X_test_poly = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predictions
    pred_train = model.predict(X_train_poly)
    pred_test = model.predict(X_test_poly)

    # RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    results.append({'degree': d, 'train_rmse': rmse_train, 'test_rmse': rmse_test})

    # Smooth curve for plotting
    x_grid = np.linspace(0, 10, 300).reshape(-1, 1)
    X_grid_poly = poly.transform(x_grid)
    pred_grid = model.predict(X_grid_poly)
    # Clip extreme predictions for high-degree polynomials
    pred_grid = np.clip(pred_grid, -3, 3)
    predictions[d] = (x_grid.ravel(), pred_grid)

# --- Print RMSE results ---
print("Polynomial Regression: Training vs Test RMSE")
print("=" * 50)
print(f"{'Degree':>8s}  {'Train RMSE':>12s}  {'Test RMSE':>12s}")
print("-" * 50)
for r in results:
    print(f"{r['degree']:>8d}  {r['train_rmse']:>12.4f}  {r['test_rmse']:>12.4f}")

# --- Part 2: Plot fitted curves ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_train, y_train, alpha=0.5, s=30, label='Training data', zorder=5)
x_true = np.linspace(0, 10, 300)
ax.plot(x_true, np.sin(x_true), 'k--', linewidth=1.5, label='True function sin(x)')

colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7']
for (d, color) in zip(degrees, colors):
    x_g, pred_g = predictions[d]
    ax.plot(x_g, pred_g, linewidth=1.5, color=color, label=f'Degree {d}')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Polynomial Fits of Varying Complexity')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(-3, 3)
plt.tight_layout()
plt.show()

# --- Part 3: Plot training vs test error ---
degs = [r['degree'] for r in results]
train_rmses = [r['train_rmse'] for r in results]
test_rmses = [r['test_rmse'] for r in results]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(degs, train_rmses, 'o-', linewidth=2, label='Training RMSE', color='steelblue')
ax.plot(degs, test_rmses, 'o-', linewidth=2, label='Test RMSE', color='darkorange')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('RMSE')
ax.set_title('Training vs Test RMSE by Polynomial Degree')
ax.legend()
plt.tight_layout()
plt.show()

# --- Part 4: Interpretation ---
best_idx = np.argmin(test_rmses)
best_degree = degs[best_idx]
print(f"\nBest polynomial degree (lowest test RMSE): {best_degree}")
print("\nInterpretation (Bias-Variance Tradeoff):")
print("- Degree 1 (linear): High bias -- too simple to capture the sine curve.")
print("  Underfits both training and test data.")
print("- Degree 3-5: Good balance. Captures the main curvature of sin(x)")
print("  without fitting noise. Test error is minimized here.")
print("- Degree 10-20: High variance -- the polynomial wiggles to fit")
print("  training noise. Training error drops but test error increases.")
print("  This is classic overfitting.")
print("\nThe optimal degree (~3-5) sits at the sweet spot of the bias-variance")
print("tradeoff, where the model is flexible enough to capture the true")
print("pattern but not so flexible that it memorizes noise.")
