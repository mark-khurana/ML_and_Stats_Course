# =============================================================================
# Chapter 2 (Probability and Distributions) - Exercise 1: Z-Scores and the Normal Distribution
# =============================================================================

library(tidyverse)

mu <- 90
sigma <- 10
patient_value <- 115

# 1. Z-score
z_score <- (patient_value - mu) / sigma
cat("Z-score:", z_score, "\n")
# Z-score: 2.5

# 2. Proportion above 115
prop_above <- 1 - pnorm(patient_value, mean = mu, sd = sigma)
# Equivalently: pnorm(patient_value, mean = mu, sd = sigma, lower.tail = FALSE)
cat("Proportion above 115:", round(prop_above, 4), "\n")
# About 0.0062 or 0.62%

# 3. 95th percentile
percentile_95 <- qnorm(0.95, mean = mu, sd = sigma)
cat("95th percentile:", round(percentile_95, 1), "mg/dL\n")
# About 106.4 mg/dL

# 4. Plot
x <- seq(mu - 4 * sigma, mu + 4 * sigma, length.out = 300)
y <- dnorm(x, mean = mu, sd = sigma)
plot_data <- tibble(x = x, y = y)

ggplot(plot_data, aes(x = x, y = y)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_area(
    data = plot_data |> filter(x >= patient_value),
    fill = "firebrick",
    alpha = 0.4
  ) +
  geom_vline(
    xintercept = patient_value,
    linetype = "dashed",
    color = "firebrick"
  ) +
  annotate(
    "text",
    x = 118,
    y = 0.02,
    label = paste0("Glucose = ", patient_value, " mg/dL\nZ = ", z_score),
    color = "firebrick",
    hjust = 0
  ) +
  labs(
    title = "Distribution of Fasting Blood Glucose",
    x = "Fasting Blood Glucose (mg/dL)",
    y = "Density"
  ) +
  theme_minimal()
