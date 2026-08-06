# =============================================================================
# Chapter 5, Exercise 3: The Effect of Correlation
# How LASSO and Ridge handle correlated predictors differently
# =============================================================================

library(glmnet)
library(MASS)

set.seed(2025)
n <- 200

# --- Create 4 correlated predictors (correlation ~ 0.9) ---
Sigma <- matrix(0.9, 4, 4)
diag(Sigma) <- 1
X_corr <- mvrnorm(n, mu = rep(0, 4), Sigma = Sigma)

# Add 6 independent predictors (noise)
X_indep <- matrix(rnorm(n * 6), n, 6)
X <- cbind(X_corr, X_indep)
colnames(X) <- c(paste0("Corr_", 1:4), paste0("Noise_", 1:6))

# True model: all 4 correlated predictors have effect = 0.5
true_beta <- c(rep(0.5, 4), rep(0, 6))
y <- rbinom(n, 1, plogis(X %*% true_beta))

# --- Task 1: Fit LASSO and examine which variables are selected ---
set.seed(42)
cv_lasso <- cv.glmnet(X, y, family = "binomial", alpha = 1)
cat("=== Task 1: LASSO coefficients (lambda.1se) ===\n")
print(round(as.matrix(coef(cv_lasso, s = "lambda.1se")), 3))

# --- Task 2: Fit Ridge and examine coefficients ---
set.seed(42)
cv_ridge <- cv.glmnet(X, y, family = "binomial", alpha = 0)
cat("\n=== Task 2: Ridge coefficients (lambda.1se) ===\n")
print(round(as.matrix(coef(cv_ridge, s = "lambda.1se")), 3))

# --- Task 3: Fit Elastic Net (alpha = 0.5) ---
set.seed(42)
cv_enet <- cv.glmnet(X, y, family = "binomial", alpha = 0.5)
cat("\n=== Task 3: Elastic Net coefficients (lambda.1se) ===\n")
print(round(as.matrix(coef(cv_enet, s = "lambda.1se")), 3))

# --- Question (a) ---
cat("\n=== Question (a): Does LASSO select all 4 correlated predictors? ===\n")
lasso_coefs <- as.vector(coef(cv_lasso, s = "lambda.1se"))[-1]
names(lasso_coefs) <- colnames(X)
selected_lasso <- names(lasso_coefs[abs(lasso_coefs) > 1e-6])
cat("LASSO selected:", paste(selected_lasso, collapse = ", "), "\n")
cat("LASSO typically selects only 1-2 of the 4 correlated predictors.\n")
cat("This is because the LASSO arbitrarily picks one predictor from a\n")
cat("correlated group and drops the rest. With correlation ~0.9, any\n")
cat("single correlated predictor carries almost the same information\n")
cat("as all four combined, so the LASSO keeps just one (or two) and\n")
cat("sets the others to zero.\n")

# --- Question (b) ---
cat("\n=== Question (b): How does Ridge handle correlated predictors? ===\n")
ridge_coefs <- as.vector(coef(cv_ridge, s = "lambda.1se"))[-1]
names(ridge_coefs) <- colnames(X)
cat("Ridge coefficients for correlated predictors:\n")
print(round(ridge_coefs[1:4], 4))
cat("\nRidge distributes the effect EVENLY across all correlated predictors.\n")
cat("All four correlated predictors retain non-zero (and similar) coefficients.\n")
cat("This is the 'grouping effect' -- Ridge shrinks correlated predictors\n")
cat("toward each other rather than picking one arbitrarily.\n")

# --- Question (c) ---
cat("\n=== Question (c): Does Elastic Net improve on LASSO? ===\n")
enet_coefs <- as.vector(coef(cv_enet, s = "lambda.1se"))[-1]
names(enet_coefs) <- colnames(X)
selected_enet <- names(enet_coefs[abs(enet_coefs) > 1e-6])
cat("Elastic Net selected:", paste(selected_enet, collapse = ", "), "\n")
cat("The Elastic Net typically selects MORE of the correlated predictors\n")
cat("than LASSO, thanks to the L2 component (grouping effect).\n")
cat("It provides a middle ground: some sparsity (like LASSO) but\n")
cat("better handling of correlated groups (like Ridge).\n")

# --- Question (d): Stability analysis ---
cat("\n=== Question (d): Stability analysis (10 different seeds) ===\n")
cat("LASSO variable selection across 10 bootstrap samples:\n")
for (seed in 1:10) {
  set.seed(seed)
  idx <- sample(n, n, replace = TRUE)
  cv_boot <- cv.glmnet(X[idx, ], y[idx], family = "binomial", alpha = 1,
                        nfolds = 5)
  boot_coefs <- as.vector(coef(cv_boot, s = "lambda.1se"))[-1]
  selected <- colnames(X)[abs(boot_coefs) > 1e-6]
  cat(sprintf("  Seed %2d: %s\n", seed, paste(selected, collapse = ", ")))
}

cat("\nRidge variable selection (all retained) across 10 bootstrap samples:\n")
for (seed in 1:10) {
  set.seed(seed)
  idx <- sample(n, n, replace = TRUE)
  cv_boot <- cv.glmnet(X[idx, ], y[idx], family = "binomial", alpha = 0,
                        nfolds = 5)
  boot_coefs <- as.vector(coef(cv_boot, s = "lambda.1se"))[-1]
  selected <- colnames(X)[abs(boot_coefs) > 0.01]
  cat(sprintf("  Seed %2d: %s\n", seed, paste(selected, collapse = ", ")))
}

cat("\nConclusion: LASSO variable selection is UNSTABLE for correlated\n")
cat("predictors -- the selected predictor(s) change across samples.\n")
cat("Ridge is stable because it always retains all predictors.\n")
cat("Elastic Net offers a compromise with better stability than LASSO.\n")
