# =============================================================================
# Chapter 7, Exercise 4: Spot the Data Leakage
# A colleague reports AUC ~ 0.99. Find the leaking features, remove them,
# and re-evaluate.
# =============================================================================

library(tidyverse)
library(tidymodels)

# --- Simulate the clinical dataset (same as exercise) ---
set.seed(42)
n <- 800
ex_data <- tibble(
  age = rnorm(n, 65, 10),
  creatinine = rlnorm(n, 0, 0.5),
  hemoglobin = rnorm(n, 12, 2),
  wbc = rlnorm(n, 2, 0.4),
  icu = factor(
    rbinom(n, 1, plogis(-4 + 0.03 * age + 0.5 * creatinine)),
    labels = c("No", "Yes")
  )
)

# Leaked features (consequences of ICU admission, not causes)
ex_data <- ex_data %>%
  mutate(
    ventilator = ifelse(icu == "Yes", rbinom(n(), 1, 0.85), 0),
    sedation_score = ifelse(icu == "Yes", sample(1:10, n(), replace = TRUE), 0)
  )

set.seed(42)
folds <- vfold_cv(ex_data, v = 10, strata = icu)

lr_spec <- logistic_reg() %>%
  set_engine("glm")

# --- Step 1: reproduce the colleague's result ---
leaked_wf <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(recipe(icu ~ ., data = ex_data))

leaked_results <- fit_resamples(leaked_wf, resamples = folds,
                                metrics = metric_set(roc_auc))
cat("With leakage:\n")
print(collect_metrics(leaked_results))

# --- Step 2: identify and remove the leaking features ---
# ventilator: only ICU patients receive mechanical ventilation, so it is a
#   *consequence* of ICU admission, not a predictor available before admission.
# sedation_score: recorded only for ICU patients (0 for everyone else), so
#   the value directly encodes the outcome.
ex_data_clean <- ex_data %>%
  select(-ventilator, -sedation_score)

# --- Step 3: re-evaluate ---
folds_clean <- vfold_cv(ex_data_clean, v = 10, strata = icu)

clean_wf <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(recipe(icu ~ ., data = ex_data_clean))

clean_results <- fit_resamples(clean_wf, resamples = folds_clean,
                               metrics = metric_set(roc_auc))
cat("\nWithout leakage:\n")
print(collect_metrics(clean_results))

# --- Interpretation ---
# The AUC drops dramatically (from ~0.99 to something much more modest).
# The original near-perfect AUC was an artefact: ventilator and sedation_score
# are recorded *after* ICU admission and essentially encode the outcome.
# Including them is the ML equivalent of looking at the answer sheet.
# In clinical ML, always ask: "Would this variable be available at the time
# the prediction needs to be made?" If not, it must be excluded.
