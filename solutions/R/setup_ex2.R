# =============================================================================
# Chapter 1 (Setup) - Exercise 2: Explore a Clinical Dataset
# =============================================================================

library(tidyverse)

set.seed(42)
n <- 200
clinical <- tibble(
  age = round(rnorm(n, mean = 55, sd = 12)),
  systolic_bp = round(100 + 0.8 * age + rnorm(n, sd = 10)),
  bmi = round(rnorm(n, mean = 27, sd = 5), 1)
)

ggplot(clinical, aes(x = age, y = systolic_bp)) +
  geom_point(alpha = 0.5, color = "darkblue") +
  geom_smooth(method = "lm", color = "firebrick", se = TRUE) +
  labs(
    title = "Age vs. Systolic Blood Pressure",
    subtitle = "Simulated clinical data (n = 200)",
    x = "Age (years)",
    y = "Systolic Blood Pressure (mmHg)"
  ) +
  theme_minimal()
