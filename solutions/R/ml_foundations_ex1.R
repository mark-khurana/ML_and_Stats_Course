# =============================================================================
# Chapter 7, Exercise 1: Feature Engineering and Cross-Validation
# Compare logistic regression with raw features vs engineered features
# using 10-fold stratified CV. Report AUC for both models.
# =============================================================================

library(tidyverse)
library(tidymodels)

# --- Simulate the clinical dataset ---
set.seed(123)
n <- 600
ex_data <- tibble(
  age = rnorm(n, 65, 10),
  sex = rbinom(n, 1, 0.5),
  creatinine = rlnorm(n, 0, 0.5),
  hemoglobin = rnorm(n, 12, 2),
  platelets = rnorm(n, 250, 70),
  wbc = rlnorm(n, 2, 0.4),
  icu = factor(
    rbinom(n, 1, plogis(-4 + 0.03 * age + 0.5 * creatinine)),
    labels = c("No", "Yes")
  )
)

cat("ICU admission rate:", mean(ex_data$icu == "Yes"), "\n")

# --- 10-fold stratified cross-validation ---
set.seed(42)
folds <- vfold_cv(ex_data, v = 10, strata = icu)

# --- Model A: raw features ---
raw_recipe <- recipe(icu ~ ., data = ex_data) %>%
  step_normalize(all_numeric_predictors())

lr_spec <- logistic_reg() %>%
  set_engine("glm")

raw_wf <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(raw_recipe)

raw_results <- fit_resamples(raw_wf, resamples = folds,
                             metrics = metric_set(roc_auc))

# --- Model B: engineered features ---
eng_recipe <- recipe(icu ~ ., data = ex_data) %>%
  step_mutate(
    # Simplified eGFR (CKD-EPI-inspired, not the full equation)
    egfr = 140 *
      (pmin(creatinine, 0.9) / 0.9)^(-0.411) *
      (pmax(creatinine, 0.9) / 0.9)^(-1.209) *
      0.993^age *
      ifelse(sex == 1, 1.0, 1.018),
    # Hemoglobin-to-platelet ratio
    hb_platelet_ratio = hemoglobin / platelets,
    # Log-transformed WBC (reduces skew)
    log_wbc = log(wbc)
  ) %>%
  step_normalize(all_numeric_predictors())

eng_wf <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(eng_recipe)

eng_results <- fit_resamples(eng_wf, resamples = folds,
                             metrics = metric_set(roc_auc))

# --- Collect and compare results ---
raw_metrics <- collect_metrics(raw_results) %>% mutate(model = "A (raw)")
eng_metrics <- collect_metrics(eng_results) %>% mutate(model = "B (engineered)")

comparison <- bind_rows(raw_metrics, eng_metrics) %>%
  select(model, .metric, mean, std_err)

print(comparison)

# --- Interpretation ---
# In this simulated dataset the true outcome depends on age and creatinine
# via a logistic link. Since logistic regression can already capture that
# linear relationship from the raw features, the engineered features (eGFR,
# ratios, log transforms) may add only a modest improvement — or none at all.
#
# In real clinical data, feature engineering often matters more: eGFR is a
# non-linear transform of creatinine that better reflects kidney function,
# and log-WBC handles the right skew common in lab values. The lesson is
# that engineered features encode domain knowledge the model cannot discover
# on its own from raw inputs — even if the benefit is small in this toy
# example.
