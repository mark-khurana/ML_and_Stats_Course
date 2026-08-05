# =============================================================================
# Chapter 1 (Setup) - Exercise 1: Verify Your Setup
# =============================================================================

library(tidyverse)

ggplot(iris, aes(x = Sepal.Width)) +
  geom_histogram(binwidth = 0.2, fill = "steelblue", color = "white") +
  labs(
    title = "Distribution of Sepal Width",
    x = "Sepal Width (cm)",
    y = "Count"
  ) +
  theme_minimal()
