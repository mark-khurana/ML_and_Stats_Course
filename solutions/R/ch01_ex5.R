# =============================================================================
# Chapter 2 (Probability and Distributions) - Exercise 5: Central Limit Theorem --- Hands-On Simulation
# =============================================================================

library(tidyverse)

set.seed(123)
lambda <- 3
n_sim <- 5000

# 1. Raw Poisson data
raw_data <- rpois(10000, lambda = lambda)
ggplot(tibble(x = raw_data), aes(x = x)) +
  geom_histogram(
    binwidth = 1,
    fill = "steelblue",
    color = "white",
    alpha = 0.8
  ) +
  labs(title = "Raw Poisson(3) Data (right-skewed)", x = "Value", y = "Count") +
  theme_minimal()

# 2. CLT simulation
sample_sizes <- c(5, 15, 50, 200)

clt_data <- map_dfr(sample_sizes, function(n) {
  means <- replicate(n_sim, mean(rpois(n, lambda = lambda)))
  tibble(n = paste0("n = ", n), sample_mean = means)
})

clt_data$n <- factor(clt_data$n, levels = paste0("n = ", sample_sizes))

ggplot(clt_data, aes(x = sample_mean)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 40,
    fill = "steelblue",
    alpha = 0.7
  ) +
  facet_wrap(~n, scales = "free_y") +
  labs(
    title = "CLT: Distribution of Sample Means from Poisson(3)",
    x = "Sample Mean",
    y = "Density"
  ) +
  theme_minimal()

# 3. Overlay normal curve on n = 50
n50_means <- replicate(n_sim, mean(rpois(50, lambda = lambda)))
theoretical_sd <- sqrt(lambda / 50)

ggplot(tibble(x = n50_means), aes(x = x)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 40,
    fill = "steelblue",
    alpha = 0.7
  ) +
  stat_function(
    fun = dnorm,
    args = list(mean = lambda, sd = theoretical_sd),
    color = "firebrick",
    linewidth = 1.2
  ) +
  labs(
    title = "Sample Means (n=50) with Normal Approximation Overlay",
    subtitle = paste0(
      "Theoretical: N(",
      lambda,
      ", ",
      round(theoretical_sd^2, 4),
      ")"
    ),
    x = "Sample Mean",
    y = "Density"
  ) +
  theme_minimal()
