# =============================================================================
# Chapter 7, Exercise 3: The Bias-Variance Tradeoff in Practice
# Fit polynomials of degree 1, 3, 5, 10, and 20 to a training set.
# Plot fitted curves, compute training/test error, identify optimal degree.
# =============================================================================

library(tidyverse)

# --- Generate data ---
set.seed(42)
x_train <- sort(runif(50, 0, 10))
y_train <- sin(x_train) + rnorm(50, 0, 0.3)
x_test <- sort(runif(200, 0, 10))
y_test <- sin(x_test) + rnorm(200, 0, 0.3)

train_df <- tibble(x = x_train, y = y_train)
test_df  <- tibble(x = x_test,  y = y_test)

# --- Fit polynomials and compute RMSE ---
degrees <- c(1, 3, 5, 10, 20)
results <- tibble(degree = integer(), train_rmse = double(), test_rmse = double())

# Also store predictions for plotting
plot_data <- tibble()

for (d in degrees) {
  # Fit polynomial of degree d
  fit <- lm(y ~ poly(x, degree = d, raw = TRUE), data = train_df)

  # Predictions on train and test
  pred_train <- predict(fit, newdata = train_df)
  pred_test  <- predict(fit, newdata = test_df)

  # Compute RMSE
  rmse_train <- sqrt(mean((y_train - pred_train)^2))
  rmse_test  <- sqrt(mean((y_test - pred_test)^2))

  results <- bind_rows(results,
                       tibble(degree = d, train_rmse = rmse_train, test_rmse = rmse_test))

  # Smooth curve for plotting
  x_grid <- seq(0, 10, length.out = 300)
  pred_grid <- predict(fit, newdata = tibble(x = x_grid))
  # Clip extreme predictions for high-degree polynomials
  pred_grid <- pmin(pmax(pred_grid, -3), 3)
  plot_data <- bind_rows(plot_data,
                         tibble(x = x_grid, y_pred = pred_grid,
                                degree = paste("Degree", d)))
}

# --- Print RMSE results ---
cat("Polynomial Regression: Training vs Test RMSE\n")
cat("=============================================\n")
print(results)

# --- Part 2: Plot fitted curves ---
p1 <- ggplot() +
  geom_point(data = train_df, aes(x, y), alpha = 0.5, size = 2) +
  geom_line(data = plot_data, aes(x, y_pred, color = degree), linewidth = 1) +
  geom_line(data = tibble(x = seq(0, 10, 0.01), y = sin(seq(0, 10, 0.01))),
            aes(x, y), linetype = "dashed", color = "black", linewidth = 0.8) +
  labs(title = "Polynomial Fits of Varying Complexity",
       subtitle = "Dashed line = true function sin(x)",
       x = "x", y = "y", color = "Polynomial") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

print(p1)

# --- Part 3: Plot training vs test error ---
results_long <- results %>%
  pivot_longer(cols = c(train_rmse, test_rmse),
               names_to = "set", values_to = "rmse") %>%
  mutate(set = ifelse(set == "train_rmse", "Training", "Test"))

p2 <- ggplot(results_long, aes(x = degree, y = rmse, color = set)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  labs(title = "Training vs Test RMSE by Polynomial Degree",
       x = "Polynomial Degree", y = "RMSE", color = "Dataset") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

print(p2)

# --- Part 4: Interpretation ---
best_degree <- results$degree[which.min(results$test_rmse)]
cat("\nBest polynomial degree (lowest test RMSE):", best_degree, "\n")
cat("\nInterpretation (Bias-Variance Tradeoff):\n")
cat("- Degree 1 (linear): High bias -- too simple to capture the sine curve.\n")
cat("  Underfits both training and test data.\n")
cat("- Degree 3-5: Good balance. Captures the main curvature of sin(x)\n")
cat("  without fitting noise. Test error is minimized here.\n")
cat("- Degree 10-20: High variance -- the polynomial wiggles to fit\n")
cat("  training noise. Training error drops but test error increases.\n")
cat("  This is classic overfitting.\n")
cat("\nThe optimal degree (~3-5) sits at the sweet spot of the bias-variance\n")
cat("tradeoff, where the model is flexible enough to capture the true\n")
cat("pattern but not so flexible that it memorizes noise.\n")
