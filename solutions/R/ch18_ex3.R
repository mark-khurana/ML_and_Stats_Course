# =============================================================================
# Chapter 18 - Exercise 3: Subgroup Analysis and Meta-Regression in R
# =============================================================================

library(tidyverse)
library(meta)
library(metafor)

# --- Dataset ---
af_trials <- data.frame(
  study = c("TRAIL-1", "GUARD-AF", "SHIELD", "ORBIT-AF",
            "VENTURE", "COMPASS-AF", "PIONEER-2", "ATLAS-AF"),
  events_new  = c(28, 45, 112, 67, 33, 89, 52, 41),
  n_new       = c(1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100),
  events_warf = c(42, 58, 148, 84, 29, 102, 61, 53),
  n_warf      = c(1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100),
  region      = c("Europe", "North America", "Europe", "Asia",
                   "North America", "Europe", "Asia", "North America"),
  mean_age    = c(72, 68, 74, 65, 70, 71, 63, 69),
  pct_female  = c(38, 42, 35, 48, 40, 37, 52, 44)
)

# Compute effect sizes using metafor
es <- escalc(measure = "RR",
             ai = events_new, n1i = n_new,
             ci = events_warf, n2i = n_warf,
             data = af_trials)

# Base meta-analysis
m1 <- metabin(
  event.e = events_new, n.e = n_new,
  event.c = events_warf, n.c = n_warf,
  studlab = study, data = af_trials,
  sm = "RR", method.tau = "REML"
)

# --- Part (a): Subgroup analysis by region ---
cat("--- Part (a): Subgroup analysis by region ---\n\n")

# Using meta package
m_sub <- update(m1, subgroup = af_trials$region)
cat("Subgroup analysis results:\n")
print(summary(m_sub))

# Forest plot by subgroup
forest(m_sub,
       sortvar = TE,
       label.left = "Favours new anticoagulant",
       label.right = "Favours warfarin",
       col.diamond = "steelblue",
       print.tau2 = TRUE,
       print.I2 = TRUE,
       print.subgroup.name = TRUE,
       main = "Subgroup Analysis by Region")

# Using metafor for the test of subgroup differences
res_sub <- rma(yi, vi, mods = ~ region, data = es, method = "REML")
cat("\nTest for subgroup differences (metafor):\n")
print(summary(res_sub))

cat("\nDo treatment effects differ by region?\n")
cat("Test for moderation: QM =", round(res_sub$QM, 2),
    ", df =", res_sub$m, ", p =", round(res_sub$QMp, 4), "\n")
if (res_sub$QMp < 0.05) {
  cat("Yes, there is a statistically significant difference between regions.\n")
} else {
  cat("No, there is no statistically significant difference between regions.\n")
  cat("However, with only", nrow(es), "studies split across",
      length(unique(es$region)), "regions,\n")
  cat("statistical power for detecting subgroup differences is limited.\n")
}

# --- Part (b): Meta-regression with mean age ---
cat("\n--- Part (b): Meta-regression with mean age ---\n\n")

res_age <- rma(yi, vi, mods = ~ mean_age, data = es, method = "REML")
print(summary(res_age))

cat("\nInterpretation:\n")
cat("Coefficient for mean_age:", round(coef(res_age)["mean_age"], 4), "\n")
cat("p-value:", round(res_age$pval[2], 4), "\n")
cat("R^2 (proportion of heterogeneity explained):",
    round(max(0, res_age$R2), 1), "%\n")

if (res_age$pval[2] < 0.05) {
  cat("There IS a statistically significant relationship between mean age\n")
  cat("and treatment effect. Each 1-year increase in mean age is associated\n")
  cat("with a", round(coef(res_age)["mean_age"], 4),
      "change in log RR.\n")
} else {
  cat("There is NO statistically significant relationship between mean age\n")
  cat("and treatment effect (p =", round(res_age$pval[2], 3), ").\n")
}

# --- Part (c): Meta-regression with percentage female ---
cat("\n--- Part (c): Meta-regression with percentage female ---\n\n")

res_fem <- rma(yi, vi, mods = ~ pct_female, data = es, method = "REML")
print(summary(res_fem))

cat("\nInterpretation:\n")
cat("Coefficient for pct_female:", round(coef(res_fem)["pct_female"], 4), "\n")
cat("p-value:", round(res_fem$pval[2], 4), "\n")
cat("R^2:", round(max(0, res_fem$R2), 1), "%\n")

cat("\n*** ECOLOGICAL FALLACY WARNING ***\n")
cat("Even if there is an association between trial-level percentage female\n")
cat("and the treatment effect, this does NOT prove that individual women\n")
cat("respond differently to the treatment than individual men.\n")
cat("Trials with higher percentage female may differ in other ways:\n")
cat("  - Geographic region (cultural differences in enrolment)\n")
cat("  - Mean age (women often present with AF at older ages)\n")
cat("  - Comorbidity profiles\n")
cat("  - Trial methodology\n")
cat("Only an individual participant data (IPD) meta-analysis with a\n")
cat("treatment-by-sex interaction term can properly assess whether\n")
cat("sex modifies the treatment effect.\n")

# --- Part (d): Bubble plot of meta-regression on mean age ---
cat("\n--- Part (d): Bubble plot ---\n")

# Method 1: Using metafor's built-in bubble plot
regplot(res_age,
        xlab = "Mean Age (years)",
        ylab = "Log Risk Ratio",
        main = "Meta-Regression: Treatment Effect vs Mean Age",
        ci = TRUE,
        pi = TRUE,  # prediction interval
        col = "steelblue",
        bg = "lightblue",
        las = 1)

# Method 2: Using ggplot2 for more control
es$w_re <- 1 / (es$vi + res_age$tau2)
es$w_norm <- es$w_re / max(es$w_re)

# Predicted line from meta-regression
age_seq <- seq(min(es$mean_age) - 2, max(es$mean_age) + 2, length.out = 100)
pred <- predict(res_age, newmods = age_seq)

pred_df <- data.frame(
  mean_age = age_seq,
  pred = pred$pred,
  ci_lo = pred$ci.lb,
  ci_hi = pred$ci.ub,
  pi_lo = pred$pi.lb,
  pi_hi = pred$pi.ub
)

p_bubble <- ggplot() +
  # Prediction interval
  geom_ribbon(data = pred_df, aes(x = mean_age, ymin = pi_lo, ymax = pi_hi),
              fill = "grey90", alpha = 0.5) +
  # Confidence interval for regression line
  geom_ribbon(data = pred_df, aes(x = mean_age, ymin = ci_lo, ymax = ci_hi),
              fill = "steelblue", alpha = 0.2) +
  # Regression line
  geom_line(data = pred_df, aes(x = mean_age, y = pred),
            colour = "steelblue", linewidth = 1) +
  # Study points (bubbles)
  geom_point(data = es, aes(x = mean_age, y = yi, size = w_norm),
             colour = "steelblue", alpha = 0.7) +
  # Study labels
  geom_text(data = es, aes(x = mean_age, y = yi, label = study),
            size = 2.5, vjust = -1.5) +
  # Reference line at null
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  # Aesthetics
  scale_size_continuous(range = c(3, 10), guide = "none") +
  labs(x = "Mean Age (years)",
       y = "Log Risk Ratio",
       title = "Bubble Plot: Meta-Regression of Treatment Effect on Mean Age",
       subtitle = paste0("Slope = ", round(coef(res_age)["mean_age"], 4),
                        ", p = ", round(res_age$pval[2], 3),
                        "; bubble size proportional to study weight")) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold")
  )

print(p_bubble)

ggsave("bubble_plot_age.png", p_bubble, width = 8, height = 6, dpi = 300)

cat("\nBubble plot saved. Larger bubbles indicate more precise studies.\n")
cat("The regression line shows the relationship between mean age\n")
cat("and treatment effect, with shaded confidence and prediction intervals.\n")
