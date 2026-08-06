# =============================================================================
# Chapter 2 (Probability and Distributions) - Exercise 4: Bayes' Theorem --- Interpreting a Cancer Screening Test
# =============================================================================

library(tidyverse)

sensitivity <- 0.87
specificity <- 0.95
prevalence <- 0.02

# 1. PPV
ppv <- (sensitivity * prevalence) /
  (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
cat("PPV (prevalence = 2%):", round(ppv, 4), "\n")
# About 0.2623 or 26.2%

# 2. NPV
npv <- (specificity * (1 - prevalence)) /
  (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)
cat("NPV (prevalence = 2%):", round(npv, 4), "\n")
# About 0.9972 or 99.7%

# 3. PPV with high-risk prevalence
prev_high <- 0.10
ppv_high_risk <- (sensitivity * prev_high) /
  (sensitivity * prev_high + (1 - specificity) * (1 - prev_high))
cat("PPV (prevalence = 10%):", round(ppv_high_risk, 4), "\n")
# About 0.6588 or 65.9%

# 4. Plot
prev_range <- seq(0.001, 0.30, by = 0.001)
ppv_curve <- (sensitivity * prev_range) /
  (sensitivity * prev_range + (1 - specificity) * (1 - prev_range))

plot_data <- tibble(prevalence = prev_range, PPV = ppv_curve)

ggplot(plot_data, aes(x = prevalence, y = PPV)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(
    data = tibble(prevalence = c(0.02, 0.10), PPV = c(ppv, ppv_high_risk)),
    color = "firebrick",
    size = 3
  ) +
  annotate(
    "text",
    x = 0.04,
    y = ppv - 0.03,
    label = paste0("General pop (2%): PPV = ", round(ppv * 100, 1), "%"),
    hjust = 0,
    color = "firebrick",
    size = 3.5
  ) +
  annotate(
    "text",
    x = 0.12,
    y = ppv_high_risk - 0.03,
    label = paste0(
      "High risk (10%): PPV = ",
      round(ppv_high_risk * 100, 1),
      "%"
    ),
    hjust = 0,
    color = "firebrick",
    size = 3.5
  ) +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  labs(
    title = "Mammography PPV Depends on Prevalence",
    subtitle = "Sensitivity = 87%, Specificity = 95%",
    x = "Prevalence",
    y = "Positive Predictive Value (PPV)"
  ) +
  theme_minimal()
