# =============================================================================
# Chapter 7, Exercise 1: Cross-Validation Experiment
# Compare logistic regression and SVM (RBF kernel) using 10-fold stratified CV.
# Report AUC for both models.
# =============================================================================

library(tidyverse)
library(tidymodels)
library(kernlab)

# --- Simulate the clinical dataset ---
set.seed(123)
n <- 600
ex_data <- tibble(
  age = rnorm(n, 65, 10),
  creatinine = rlnorm(n, 0, 0.5),
  hemoglobin = rnorm(n, 12, 2),
  platelets = rnorm(n, 250, 70),
  wbc = rlnorm(n, 2, 0.4),
  icu = factor(rbinom(n, 1, plogis(-4 + 0.03 * rnorm(n, 65, 10) +
                                    0.5 * rlnorm(n, 0, 0.5))),
               labels = c("No", "Yes"))
)

cat("ICU admission rate:", mean(ex_data$icu == "Yes"), "\n")

# --- 10-fold stratified cross-validation ---
set.seed(42)
folds <- vfold_cv(ex_data, v = 10, strata = icu)

# --- Logistic Regression workflow ---
lr_spec <- logistic_reg() %>%
  set_engine("glm")

lr_recipe <- recipe(icu ~ ., data = ex_data) %>%
  step_normalize(all_numeric_predictors())

lr_wf <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(lr_recipe)

lr_results <- fit_resamples(lr_wf, resamples = folds,
                            metrics = metric_set(roc_auc))

# --- SVM (RBF kernel) workflow ---
svm_spec <- svm_rbf(cost = 1, rbf_sigma = 0.5) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_recipe <- recipe(icu ~ ., data = ex_data) %>%
  step_normalize(all_numeric_predictors())

svm_wf <- workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(svm_recipe)

svm_results <- fit_resamples(svm_wf, resamples = folds,
                             metrics = metric_set(roc_auc))

# --- Collect and compare results ---
lr_metrics  <- collect_metrics(lr_results) %>% mutate(model = "Logistic Regression")
svm_metrics <- collect_metrics(svm_results) %>% mutate(model = "SVM (RBF)")

comparison <- bind_rows(lr_metrics, svm_metrics) %>%
  select(model, .metric, mean, std_err)

print(comparison)

# --- Interpretation ---
# Both models are expected to show similar AUC because the simulated data
# has simple, roughly linear relationships between predictors and outcome.
# Logistic regression handles linear relationships well, and an RBF SVM
# adds flexibility that is not needed here.
# With real clinical data that has non-linear or interactive effects,
# SVM may outperform logistic regression.
