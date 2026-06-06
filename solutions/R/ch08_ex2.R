# =============================================================================
# Chapter 8, Exercise 2: Random Forest vs XGBoost Tuning Challenge
# 1. Split data 80/20. 2. Tune RF and XGBoost with 5-fold CV.
# 3. Select best hyperparameters. 4. Evaluate on test set.
# 5. Create variable importance plots.
# =============================================================================

library(tidyverse)
library(tidymodels)
library(vip)

# --- Simulate the readmission dataset (same as chapter) ---
set.seed(42)
n <- 1000

readmit_data <- tibble(
  age = rnorm(n, 68, 12),
  length_of_stay = rpois(n, 5) + 1,
  num_comorbidities = rpois(n, 3),
  prior_admissions = rpois(n, 1),
  discharge_hemoglobin = rnorm(n, 11, 2),
  discharge_creatinine = rlnorm(n, 0.2, 0.5),
  has_diabetes = rbinom(n, 1, 0.35),
  has_chf = rbinom(n, 1, 0.25),
  readmitted = factor(
    rbinom(n, 1, plogis(-3 + 0.02 * (rnorm(n, 68, 12) - 68) +
                          0.15 * rpois(n, 1) +
                          0.1 * rpois(n, 3) +
                          0.3 * rbinom(n, 1, 0.25) -
                          0.1 * rnorm(n, 11, 2))),
    labels = c("No", "Yes")
  )
)

# --- Part 1: Train/test split ---
set.seed(42)
split <- initial_split(readmit_data, prop = 0.8, strata = readmitted)
train <- training(split)
test  <- testing(split)

cat("Training set size:", nrow(train), "\n")
cat("Test set size:", nrow(test), "\n")

# 5-fold CV on training set
folds <- vfold_cv(train, v = 5, strata = readmitted)

# Recipe (shared)
base_recipe <- recipe(readmitted ~ ., data = train)

# --- Part 2a: Tune Random Forest ---
rf_spec <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(base_recipe)

rf_grid <- grid_regular(
  mtry(range = c(2, 7)),
  min_n(range = c(5, 30)),
  levels = 5
)

set.seed(42)
rf_tune <- tune_grid(rf_wf, resamples = folds, grid = rf_grid,
                     metrics = metric_set(roc_auc))

cat("\n--- Random Forest Tuning Results (Top 5) ---\n")
print(show_best(rf_tune, metric = "roc_auc", n = 5))

best_rf <- select_best(rf_tune, metric = "roc_auc")
cat("\nBest RF params - mtry:", best_rf$mtry, "min_n:", best_rf$min_n, "\n")

# --- Part 2b: Tune XGBoost ---
xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(base_recipe)

xgb_grid <- grid_regular(
  tree_depth(range = c(2, 6)),
  learn_rate(range = c(-3, -1)),   # log10 scale: 0.001 to 0.1
  min_n(range = c(5, 20)),
  levels = 4
)

set.seed(42)
xgb_tune <- tune_grid(xgb_wf, resamples = folds, grid = xgb_grid,
                      metrics = metric_set(roc_auc))

cat("\n--- XGBoost Tuning Results (Top 5) ---\n")
print(show_best(xgb_tune, metric = "roc_auc", n = 5))

best_xgb <- select_best(xgb_tune, metric = "roc_auc")
cat("\nBest XGB params - depth:", best_xgb$tree_depth,
    "learn_rate:", best_xgb$learn_rate,
    "min_n:", best_xgb$min_n, "\n")

# --- Part 3: Finalize and fit on full training set ---
final_rf_wf <- finalize_workflow(rf_wf, best_rf)
final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

rf_final_fit <- fit(final_rf_wf, data = train)
xgb_final_fit <- fit(final_xgb_wf, data = train)

# --- Part 4: Evaluate on test set ---
rf_test_pred  <- predict(rf_final_fit, test, type = "prob") %>%
  bind_cols(test %>% select(readmitted))
xgb_test_pred <- predict(xgb_final_fit, test, type = "prob") %>%
  bind_cols(test %>% select(readmitted))

rf_auc  <- roc_auc(rf_test_pred, truth = readmitted, .pred_Yes)
xgb_auc <- roc_auc(xgb_test_pred, truth = readmitted, .pred_Yes)

cat("\n=== Test Set Performance ===\n")
cat("Random Forest AUC:", rf_auc$.estimate, "\n")
cat("XGBoost AUC:      ", xgb_auc$.estimate, "\n")

if (xgb_auc$.estimate > rf_auc$.estimate) {
  cat("\nXGBoost performs better on the test set.\n")
} else {
  cat("\nRandom Forest performs better on the test set.\n")
}

# --- Part 5: Variable importance plots ---
# Random Forest
rf_vi <- rf_final_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 8) +
  ggtitle("Random Forest Variable Importance")
print(rf_vi)

# XGBoost
xgb_vi <- xgb_final_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 8) +
  ggtitle("XGBoost Variable Importance")
print(xgb_vi)

cat("\nBoth models should generally agree on the most important features,\n")
cat("though rankings may differ. Continuous variables with more possible\n")
cat("split points (e.g., age, creatinine) often rank higher in tree-based\n")
cat("importance measures than binary variables (e.g., has_chf).\n")
