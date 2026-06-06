# =============================================================================
# Chapter 8b, Exercise 1: Neural Network vs XGBoost on Tabular Data
# Fit both a neural network and XGBoost on the readmission dataset.
# Compare 5-fold cross-validated AUC.
# =============================================================================

library(tidyverse)
library(tidymodels)
library(keras3)

# --- Simulate readmission data (same as chapter) ---
set.seed(42)
n <- 1000

readmit_data <- tibble(
  age = rnorm(n, 68, 12),
  length_of_stay = rpois(n, 5) + 1,
  num_comorbidities = rpois(n, 3),
  prior_admissions = rpois(n, 1),
  discharge_hgb = rnorm(n, 11, 2),
  discharge_creatinine = rlnorm(n, 0.2, 0.5),
  has_diabetes = rbinom(n, 1, 0.35),
  has_chf = rbinom(n, 1, 0.25)
)

readmit_prob <- plogis(-3 + 0.02 * (readmit_data$age - 68) +
                         0.15 * readmit_data$prior_admissions +
                         0.1 * readmit_data$num_comorbidities +
                         0.3 * readmit_data$has_chf -
                         0.1 * readmit_data$discharge_hgb)
readmit_data$readmitted <- factor(rbinom(n, 1, readmit_prob),
                                   labels = c("No", "Yes"))

cat("Readmission rate:", mean(readmit_data$readmitted == "Yes"), "\n")

# --- XGBoost with 5-fold CV (using tidymodels) ---
set.seed(42)
folds <- vfold_cv(readmit_data, v = 5, strata = readmitted)

xgb_spec <- boost_tree(trees = 500, tree_depth = 4, learn_rate = 0.05,
                        min_n = 10) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(recipe(readmitted ~ ., data = readmit_data))

xgb_res <- fit_resamples(xgb_wf, resamples = folds,
                         metrics = metric_set(roc_auc))

xgb_metrics <- collect_metrics(xgb_res)
cat("\nXGBoost CV AUC:", xgb_metrics$mean, "+/-", xgb_metrics$std_err, "\n")

# --- Neural Network with 5-fold CV (manual loop) ---
x_all <- readmit_data %>% select(-readmitted) %>% as.matrix()
y_all <- as.numeric(readmit_data$readmitted == "Yes")

# Standardize features
x_mean <- apply(x_all, 2, mean)
x_sd   <- apply(x_all, 2, sd)
x_scaled <- scale(x_all, center = x_mean, scale = x_sd)

set.seed(42)
fold_ids <- vfold_cv(readmit_data, v = 5, strata = readmitted)
nn_aucs <- numeric(5)

for (i in seq_len(5)) {
  # Get train/validation indices
  train_idx <- fold_ids$splits[[i]] %>% analysis() %>% rownames() %>% as.integer()
  val_idx   <- fold_ids$splits[[i]] %>% assessment() %>% rownames() %>% as.integer()

  x_train <- x_scaled[train_idx, ]
  y_train <- y_all[train_idx]
  x_val   <- x_scaled[val_idx, ]
  y_val   <- y_all[val_idx]

  # Build neural network
  model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1, activation = "sigmoid")

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "binary_crossentropy",
    metrics = "AUC"
  )

  # Train with early stopping
  history <- model %>% fit(
    x_train, y_train,
    epochs = 50,
    batch_size = 32,
    validation_data = list(x_val, y_val),
    callbacks = list(
      callback_early_stopping(patience = 5, restore_best_weights = TRUE)
    ),
    verbose = 0
  )

  # Evaluate
  results <- model %>% evaluate(x_val, y_val, verbose = 0)
  nn_aucs[i] <- results[[2]]  # AUC metric
  cat(sprintf("  Fold %d: NN AUC = %.3f\n", i, nn_aucs[i]))
}

cat("\nNeural Network CV AUC:", mean(nn_aucs), "+/-", sd(nn_aucs) / sqrt(5), "\n")

# --- Comparison ---
cat("\n=== Comparison ===\n")
cat("XGBoost CV AUC:        ", round(xgb_metrics$mean, 3), "\n")
cat("Neural Network CV AUC: ", round(mean(nn_aucs), 3), "\n")

cat("\nInterpretation:\n")
cat("XGBoost typically matches or outperforms neural networks on tabular\n")
cat("clinical data. This is expected -- the Grinsztajn et al. (2022)\n")
cat("NeurIPS benchmark showed that tree-based models consistently\n")
cat("outperform neural networks on typical tabular datasets. Deep learning\n")
cat("excels on images, text, and sequences, not spreadsheets.\n")
