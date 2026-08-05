# =============================================================================
# Chapter 2 (Probability and Distributions) - Exercise 3: Poisson Distribution --- Emergency Department Visits
# =============================================================================

library(tidyverse)

lambda <- 8

# 1. P(X = 8)
prob_exactly_8 <- dpois(8, lambda = lambda)
cat("P(X = 8):", round(prob_exactly_8, 4), "\n")
# About 0.1396

# 2. P(X >= 12)
prob_12_or_more <- ppois(11, lambda = lambda, lower.tail = FALSE)
cat("P(X >= 12):", round(prob_12_or_more, 4), "\n")
# About 0.1121

# 3. Expected days with 0 cases
prob_zero <- dpois(0, lambda = lambda)
expected_zero_days <- 365 * prob_zero
cat("P(X = 0):", format(prob_zero, scientific = TRUE), "\n")
cat("Expected zero-case days per year:", round(expected_zero_days, 3), "\n")
# P(0) is very small (~0.000335), so about 0.12 days/year

# 4. Plot
k <- 0:20
poisson_data <- tibble(
  k = k,
  probability = dpois(k, lambda = lambda)
)

ggplot(poisson_data, aes(x = k, y = probability)) +
  geom_col(fill = "darkorange", alpha = 0.8) +
  geom_vline(xintercept = lambda, linetype = "dashed", color = "firebrick") +
  labs(
    title = "Poisson Distribution: Trauma Cases per Day",
    subtitle = expression(paste(lambda, " = 8")),
    x = "Number of Trauma Cases",
    y = "Probability"
  ) +
  scale_x_continuous(breaks = k) +
  theme_minimal()
