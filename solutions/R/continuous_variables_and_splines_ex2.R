# =============================================================================
# Chapter 4, Exercise 2: Fitting and Interpreting Spline Models
# PBC dataset: bilirubin and transplant status
# =============================================================================

library(rms)
library(ggplot2)

# --- Load the PBC data ---
data(pbc, package = "survival")
pbc <- pbc[!is.na(pbc$trt), ]  # Remove incomplete cases

# Create a binary outcome: transplanted (status == 1) vs not
pbc$transplant <- as.numeric(pbc$status == 1)

# Set up datadist (required by rms)
dd <- datadist(pbc)
options(datadist = "dd")

# --- Task 1: Fit logistic regression with bilirubin as a linear term ---
fit_linear <- lrm(transplant ~ bili, data = pbc)
cat("=== Linear Model ===\n")
print(fit_linear)

# --- Task 2: Fit logistic regression with bilirubin using RCS (4 knots) ---
fit_rcs <- lrm(transplant ~ rcs(bili, 4), data = pbc)
cat("\n=== RCS Model (4 knots) ===\n")
print(fit_rcs)

# --- Task 3: Test for non-linearity ---
cat("\n=== ANOVA for RCS model (test of non-linearity) ===\n")
print(anova(fit_rcs))
# The ANOVA output shows:
#   - Total effect of bili (tests whether bili has ANY association)
#   - Nonlinear component (tests whether the relationship departs from linear)
# If the nonlinear p-value is significant, a linear term is insufficient.

# --- Task 4: Compare AIC ---
cat("\n=== AIC Comparison ===\n")
cat("Linear model AIC:", AIC(fit_linear), "\n")
cat("RCS model AIC:   ", AIC(fit_rcs), "\n")
cat("Lower AIC = better fit (penalised for complexity)\n")

# --- Task 5: Plot the relationship ---
p <- Predict(fit_rcs, bili)
pred_df <- as.data.frame(p)

plot_rcs <- ggplot(pred_df, aes(x = bili, y = yhat)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "#3498db") +
  geom_line(linewidth = 1.2, colour = "#2c3e50") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  labs(x = "Serum Bilirubin (mg/dL)",
       y = "Log-Odds of Transplant",
       title = "Non-linear effect of bilirubin on transplant (RCS, 4 knots)") +
  theme_minimal(base_size = 14)

print(plot_rcs)

# --- Task 6: Interpretation ---
cat("\n=== Interpretation ===\n")
cat("The plot shows the estimated log-odds of transplant as a function of\n")
cat("serum bilirubin. If the curve is not a straight line, the relationship\n")
cat("is non-linear, and a linear term would be insufficient.\n")
cat("\nThe ANOVA non-linearity test (above) formally tests this. If the\n")
cat("non-linear p-value is < 0.05, there is evidence against linearity.\n")
cat("\nCompare the AIC values: a lower AIC for the RCS model confirms\n")
cat("that the additional flexibility improves fit beyond what a linear\n")
cat("term provides.\n")
