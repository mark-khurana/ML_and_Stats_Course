# Chapter 14, Exercise 2: Hierarchical Model for Multi-Site Drug Trial
# 12 hospitals, LDL cholesterol change, new statin vs standard care

library(brms)
library(tidyverse)

# ---- (a) Simulate data ----
set.seed(789)
n_hospitals <- 12
patients_per_hospital <- c(20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200)
true_grand_effect <- -25  # mg/dL reduction in LDL
tau_true <- 5             # between-hospital SD

hospital_effects <- rnorm(n_hospitals, 0, tau_true)

trial_data <- map2_dfr(1:n_hospitals, patients_per_hospital, function(j, nj) {
  tibble(
    hospital = factor(j),
    treatment = rep(0:1, length.out = nj),
    ldl_change = (true_grand_effect + hospital_effects[j]) * treatment +
      rnorm(nj, 0, 20)  # residual SD = 20 mg/dL
  )
})

cat("Data summary:\n")
cat("  Total patients:", nrow(trial_data), "\n")
cat("  Hospitals:", n_hospitals, "\n")
cat("  Patients per hospital:", paste(patients_per_hospital, collapse = ", "), "\n")
cat("  True grand effect:", true_grand_effect, "mg/dL\n")
cat("  True between-hospital SD:", tau_true, "mg/dL\n")

# ---- (b) Fit Bayesian hierarchical model ----
fit_hier <- brm(
  ldl_change ~ treatment + (treatment | hospital),
  data = trial_data,
  prior = c(
    prior(normal(0, 30), class = "b"),
    prior(normal(0, 30), class = "Intercept"),
    prior(cauchy(0, 5), class = "sd"),
    prior(exponential(0.05), class = "sigma")
  ),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 42,
  silent = 2,
  refresh = 0,
  control = list(adapt_delta = 0.95)
)

cat("\nHierarchical model summary:\n")
summary(fit_hier)

# Grand mean treatment effect
grand_mean <- fixef(fit_hier)["treatment", "Estimate"]
cat("\nGrand mean treatment effect:", round(grand_mean, 1), "mg/dL\n")

# ---- (c) Shrinkage plot ----
# Extract hospital-specific treatment effects (partial pooling)
ranefs <- ranef(fit_hier)$hospital[, , "treatment"]
partial_pool <- grand_mean + ranefs[, "Estimate"]

# No-pooling estimates (separate OLS per hospital)
no_pool <- trial_data %>%
  group_by(hospital) %>%
  summarise(
    no_pool_est = coef(lm(ldl_change ~ treatment))[2],
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    hospital_num = as.numeric(hospital),
    partial_pool_est = partial_pool
  )

# Plot shrinkage
ggplot(no_pool, aes(y = reorder(hospital, n))) +
  geom_point(aes(x = no_pool_est, colour = "No pooling"), size = 3) +
  geom_point(aes(x = partial_pool_est, colour = "Partial pooling"), size = 3) +
  geom_vline(xintercept = grand_mean, linetype = "dashed", colour = "grey40") +
  geom_segment(aes(x = no_pool_est, xend = partial_pool_est,
                   yend = reorder(hospital, n)),
               arrow = arrow(length = unit(0.15, "cm")),
               colour = "grey60") +
  annotate("text", x = grand_mean + 1, y = 12.5,
           label = paste0("Grand mean = ", round(grand_mean, 1)),
           hjust = 0, size = 3.5) +
  scale_colour_manual(values = c("No pooling" = "steelblue",
                                  "Partial pooling" = "firebrick")) +
  labs(x = "Treatment Effect (mg/dL change in LDL)",
       y = "Hospital (ordered by sample size)",
       colour = "Estimate Type",
       title = "Shrinkage in Hierarchical Model",
       subtitle = "Arrows show partial pooling pulling estimates toward the grand mean") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

# ---- (d) Compare hierarchical to separate OLS regressions ----
cat("\n=== Part (d): Comparison ===\n\n")
cat("Hospital  |  N   | No-Pooling | Partial Pooling | Shrinkage\n")
cat("----------|------|------------|-----------------|----------\n")

for (i in 1:nrow(no_pool)) {
  shrinkage <- abs(no_pool$no_pool_est[i] - no_pool$partial_pool_est[i])
  cat(sprintf("   %2d     | %3d  |   %6.1f   |     %6.1f      |  %5.1f\n",
              no_pool$hospital_num[i], no_pool$n[i],
              no_pool$no_pool_est[i], no_pool$partial_pool_est[i], shrinkage))
}

cat("\nThe hierarchical model is MORE APPROPRIATE because:\n")
cat("1. Small hospitals (n=20-30) have noisy OLS estimates that are\n")
cat("   shrunk toward the grand mean, reducing estimation error.\n")
cat("2. Large hospitals (n=150-200) retain their individual estimates\n")
cat("   since their data are informative enough.\n")
cat("3. The between-hospital SD (tau) is estimated from the data,\n")
cat("   quantifying the degree of heterogeneity across sites.\n")
cat("4. Hospital-by-hospital OLS ignores the shared structure --\n")
cat("   all hospitals are studying the same drug. The hierarchical\n")
cat("   model borrows strength across sites while allowing for\n")
cat("   genuine between-site variation.\n")
