# =============================================================================
# Chapter 9, Exercise 1: Bayes' Theorem in Practice (PPV Calculations)
# Calculate PPV at different prevalence levels and plot the relationship.
# =============================================================================

library(tidyverse)

# --- PPV function using Bayes' theorem ---
# PPV = (Sensitivity * Prevalence) /
#       (Sensitivity * Prevalence + (1 - Specificity) * (1 - Prevalence))
calculate_ppv <- function(sensitivity, specificity, prevalence) {
  numerator <- sensitivity * prevalence
  denominator <- numerator + (1 - specificity) * (1 - prevalence)
  return(numerator / denominator)
}

# --- Calculate NPV ---
calculate_npv <- function(sensitivity, specificity, prevalence) {
  numerator <- specificity * (1 - prevalence)
  denominator <- numerator + (1 - sensitivity) * prevalence
  return(numerator / denominator)
}

# --- Example: HIV rapid test (sensitivity = 99.7%, specificity = 99.5%) ---

# In a general population (prevalence ~ 0.4%)
ppv_general <- calculate_ppv(0.997, 0.995, 0.004)
npv_general <- calculate_npv(0.997, 0.995, 0.004)
cat("=== HIV Rapid Test (Sens=99.7%, Spec=99.5%) ===\n\n")
cat("General population (prevalence 0.4%):\n")
cat("  PPV:", round(ppv_general * 100, 1), "%\n")
cat("  NPV:", round(npv_general * 100, 1), "%\n")

# In an STI clinic population (prevalence ~ 5%)
ppv_clinic <- calculate_ppv(0.997, 0.995, 0.05)
npv_clinic <- calculate_npv(0.997, 0.995, 0.05)
cat("\nSTI clinic (prevalence 5%):\n")
cat("  PPV:", round(ppv_clinic * 100, 1), "%\n")
cat("  NPV:", round(npv_clinic * 100, 1), "%\n")

# In a population with known exposure (prevalence ~ 30%)
ppv_exposed <- calculate_ppv(0.997, 0.995, 0.30)
npv_exposed <- calculate_npv(0.997, 0.995, 0.30)
cat("\nKnown exposure (prevalence 30%):\n")
cat("  PPV:", round(ppv_exposed * 100, 1), "%\n")
cat("  NPV:", round(npv_exposed * 100, 1), "%\n")

# --- Plot PPV across a range of prevalences ---
prevalences <- seq(0.001, 0.5, by = 0.001)
ppvs <- sapply(prevalences, function(p) calculate_ppv(0.997, 0.995, p))
npvs <- sapply(prevalences, function(p) calculate_npv(0.997, 0.995, p))

plot_df <- tibble(
  prevalence = rep(prevalences, 2),
  value = c(ppvs, npvs),
  metric = rep(c("PPV", "NPV"), each = length(prevalences))
)

ggplot(plot_df, aes(x = prevalence * 100, y = value * 100, color = metric)) +
  geom_line(linewidth = 1.5) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("PPV" = "steelblue", "NPV" = "darkorange")) +
  labs(x = "Prevalence (%)",
       y = "Predictive Value (%)",
       title = "PPV and NPV Depend Heavily on Prevalence",
       subtitle = "HIV rapid test: Sensitivity=99.7%, Specificity=99.5%",
       color = "Metric") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

# --- Additional: Compare tests with different sensitivity/specificity ---
cat("\n\n=== Comparing Tests at 1% Prevalence ===\n")

tests <- tibble(
  Test = c("High Sens/Low Spec", "Balanced", "Low Sens/High Spec"),
  Sensitivity = c(0.99, 0.95, 0.80),
  Specificity = c(0.90, 0.95, 0.99)
)

for (i in seq_len(nrow(tests))) {
  ppv <- calculate_ppv(tests$Sensitivity[i], tests$Specificity[i], 0.01)
  npv <- calculate_npv(tests$Sensitivity[i], tests$Specificity[i], 0.01)
  cat(sprintf("%s (Sens=%.0f%%, Spec=%.0f%%): PPV=%.1f%%, NPV=%.1f%%\n",
              tests$Test[i], tests$Sensitivity[i]*100, tests$Specificity[i]*100,
              ppv*100, npv*100))
}

cat("\nKey takeaway: Even excellent tests have low PPV when prevalence is low.\n")
cat("This is why screening should target high-risk populations.\n")
