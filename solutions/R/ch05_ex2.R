# =============================================================================
# Chapter 5, Exercise 2: Predicting Diabetes Onset
# Pima Indians Diabetes dataset
# =============================================================================

library(glmnet)
library(mlbench)
library(ggplot2)

# --- Load data ---
data(PimaIndiansDiabetes2, package = "mlbench")
pima <- na.omit(PimaIndiansDiabetes2)
cat("Complete cases:", nrow(pima), "\n\n")

# Prepare data
X <- as.matrix(pima[, 1:8])
y <- ifelse(pima$diabetes == "pos", 1, 0)

# --- Task 1: Fit a LASSO model with 10-fold cross-validation ---
set.seed(123)
cv_lasso <- cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 10)

cat("=== LASSO Cross-Validation ===\n")
cat("Lambda.min:", round(cv_lasso$lambda.min, 4), "\n")
cat("Lambda.1se:", round(cv_lasso$lambda.1se, 4), "\n\n")

# --- Task 2: Plot the cross-validation curve ---
plot(cv_lasso, main = "LASSO Cross-Validation Curve")

# --- Task 3: Which variables are selected at lambda.1se? ---
cat("=== Variables selected at lambda.1se ===\n")
coef_1se <- coef(cv_lasso, s = "lambda.1se")
print(coef_1se)

# Count non-zero coefficients (excluding intercept)
n_selected_lasso <- sum(coef_1se[-1, ] != 0)
cat("\nNumber of variables selected by LASSO at lambda.1se:", n_selected_lasso, "\n")

# --- Task 4: Fit Ridge and Elastic Net (alpha = 0.5) ---
set.seed(123)
cv_ridge <- cv.glmnet(X, y, family = "binomial", alpha = 0, nfolds = 10)
set.seed(123)
cv_enet <- cv.glmnet(X, y, family = "binomial", alpha = 0.5, nfolds = 10)

# --- Task 5: Compare the three methods ---

# (a) How many variables does each select at lambda.1se?
coef_ridge_1se <- coef(cv_ridge, s = "lambda.1se")
coef_enet_1se <- coef(cv_enet, s = "lambda.1se")

n_selected_ridge <- sum(abs(coef_ridge_1se[-1, ]) > 1e-6)
n_selected_enet <- sum(abs(coef_enet_1se[-1, ]) > 1e-6)

cat("\n=== Number of variables selected (lambda.1se) ===\n")
cat("LASSO:       ", n_selected_lasso, "of 8\n")
cat("Elastic Net: ", n_selected_enet, "of 8\n")
cat("Ridge:       ", n_selected_ridge, "of 8 (Ridge never sets to exactly 0)\n")

# (b) Cross-validated deviance for each
cat("\n=== Cross-validated deviance (at lambda.min) ===\n")
cat("LASSO:       ", round(min(cv_lasso$cvm), 4), "\n")
cat("Elastic Net: ", round(min(cv_enet$cvm), 4), "\n")
cat("Ridge:       ", round(min(cv_ridge$cvm), 4), "\n")

# (c) Plot the regularisation path for the LASSO model
fit_lasso <- glmnet(X, y, family = "binomial", alpha = 1)
plot(fit_lasso, xvar = "lambda", label = TRUE,
     main = "LASSO Regularisation Path")
abline(v = log(cv_lasso$lambda.min), lty = 2, col = "blue")
abline(v = log(cv_lasso$lambda.1se), lty = 2, col = "red")
legend("topright", legend = c("lambda.min", "lambda.1se"),
       col = c("blue", "red"), lty = 2, cex = 0.8)

# --- Task 6: Recommendation ---
cat("\n=== Recommendation ===\n")
cat("For the Pima diabetes dataset (8 predictors, ~390 complete cases):\n")
cat("- The number of predictors is small relative to sample size,\n")
cat("  so penalisation is not strictly necessary.\n")
cat("- LASSO is useful here for identifying the most predictive variables.\n")
cat("- All three methods produce similar cross-validated deviance,\n")
cat("  confirming that the choice of penalty type matters less than\n")
cat("  using penalisation vs not.\n")
cat("- For variable selection and a parsimonious model, LASSO at\n")
cat("  lambda.1se is a good choice.\n")
cat("- For best prediction, any of the three at lambda.min would work.\n")

# --- Bonus: Compare coefficients side by side ---
coef_comparison <- data.frame(
  Variable = colnames(X),
  LASSO = as.vector(coef(cv_lasso, s = "lambda.1se"))[-1],
  Ridge = as.vector(coef(cv_ridge, s = "lambda.1se"))[-1],
  ElasticNet = as.vector(coef(cv_enet, s = "lambda.1se"))[-1]
)

cat("\n=== Coefficient comparison (lambda.1se) ===\n")
print(round(coef_comparison, 4))
