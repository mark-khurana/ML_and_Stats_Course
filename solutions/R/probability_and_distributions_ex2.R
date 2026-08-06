# =============================================================================
# Chapter 2 (Probability and Distributions) - Exercise 2: Binomial Distribution --- Vaccine Efficacy
# =============================================================================

library(tidyverse)

n <- 25
p <- 0.10

# 1. Expected number
expected <- n * p
cat("Expected breakthrough infections:", expected, "\n")
# 2.5

# 2. P(X = 0)
prob_zero <- dbinom(0, size = n, prob = p)
cat("P(X = 0):", round(prob_zero, 4), "\n")
# About 0.0718

# 3. P(X >= 5)
prob_five_or_more <- 1 - pbinom(4, size = n, prob = p)
# Equivalently: pbinom(4, size = n, prob = p, lower.tail = FALSE)
cat("P(X >= 5):", round(prob_five_or_more, 4), "\n")
# About 0.0980

# 4. Plot
outcomes <- tibble(
  k = 0:n,
  probability = dbinom(k, size = n, prob = p)
)

ggplot(outcomes, aes(x = k, y = probability)) +
  geom_col(aes(fill = k >= 5), alpha = 0.8) +
  scale_fill_manual(
    values = c("FALSE" = "steelblue", "TRUE" = "firebrick"),
    guide = "none"
  ) +
  geom_vline(xintercept = expected, linetype = "dashed") +
  labs(
    title = "Breakthrough Infections in 25 Vaccinated Individuals",
    subtitle = "Binomial(25, 0.10); red bars = 5 or more infections",
    x = "Number of Breakthrough Infections",
    y = "Probability"
  ) +
  scale_x_continuous(breaks = seq(0, 25, 2)) +
  theme_minimal()
