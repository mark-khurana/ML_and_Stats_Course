# =============================================================================
# Chapter 4, Exercise 3: Categorisation vs Splines Head-to-Head
# Simulate U-shaped relationship and compare approaches
# =============================================================================

library(rms)
library(ggplot2)

set.seed(2025)
n <- 1000
x <- rnorm(n, mean = 50, sd = 10)

# True U-shaped relationship
logit_p <- -5 + 0.004 * (x - 50)^2
y <- rbinom(n, 1, plogis(logit_p))

sim_data <- data.frame(x = x, y = y)
dd <- datadist(sim_data)
options(datadist = "dd")

# --- Model 1: Categorise into quartiles ---
sim_data$x_cat <- cut(sim_data$x,
                       breaks = quantile(sim_data$x, c(0, 0.25, 0.5, 0.75, 1)),
                       include.lowest = TRUE)
fit_cat <- lrm(y ~ x_cat, data = sim_data)

# --- Model 2: Linear term ---
fit_lin <- lrm(y ~ x, data = sim_data)

# --- Model 3: RCS with 5 knots ---
fit_rcs <- lrm(y ~ rcs(x, 5), data = sim_data)

# --- Question (a): Compare AIC ---
cat("=== Question (a): AIC Comparison ===\n")
cat("Categorised (quartiles):", AIC(fit_cat), "\n")
cat("Linear:                 ", AIC(fit_lin), "\n")
cat("RCS (5 knots):          ", AIC(fit_rcs), "\n")
cat("\nThe RCS model should have the lowest AIC, indicating best fit.\n")
cat("The linear model has the worst fit because it cannot capture\n")
cat("the U-shape at all.\n")

# --- Question (b): Plot all three fitted curves with the true function ---
x_seq <- seq(min(sim_data$x), max(sim_data$x), length.out = 200)

# True curve
true_logit <- -5 + 0.004 * (x_seq - 50)^2
true_prob <- plogis(true_logit)

# RCS predictions
pred_rcs <- Predict(fit_rcs, x = x_seq)
rcs_prob <- plogis(as.data.frame(pred_rcs)$yhat)

# Linear predictions
pred_linear <- predict(fit_linear <- lrm(y ~ x, data = sim_data),
                        newdata = data.frame(x = x_seq), type = "fitted")

# Categorised predictions (need to assign each x_seq value to a category)
x_seq_cat <- cut(x_seq,
                  breaks = quantile(sim_data$x, c(0, 0.25, 0.5, 0.75, 1)),
                  include.lowest = TRUE)
pred_cat <- predict(fit_cat,
                     newdata = data.frame(x_cat = x_seq_cat),
                     type = "fitted")

plot_df <- data.frame(
  x = rep(x_seq, 4),
  probability = c(true_prob, pred_linear, pred_cat, rcs_prob),
  Model = rep(c("True relationship", "Linear", "Categorised (quartiles)",
                "RCS (5 knots)"), each = length(x_seq))
)

p <- ggplot(plot_df, aes(x = x, y = probability, colour = Model,
                          linetype = Model)) +
  geom_line(linewidth = 1.1) +
  scale_colour_manual(values = c("True relationship" = "black",
                                  "Linear" = "#3498db",
                                  "Categorised (quartiles)" = "#e74c3c",
                                  "RCS (5 knots)" = "#27ae60")) +
  scale_linetype_manual(values = c("True relationship" = "dashed",
                                    "Linear" = "solid",
                                    "Categorised (quartiles)" = "solid",
                                    "RCS (5 knots)" = "solid")) +
  labs(x = "Predictor (x)", y = "Predicted Probability",
       title = "Comparing modelling strategies for a U-shaped relationship") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

print(p)

cat("\n=== Question (b): Interpretation ===\n")
cat("The RCS model best recovers the true U-shaped relationship.\n")
cat("The linear model completely misses the U-shape (it fits a flat or\n")
cat("slightly sloped line). The categorised model captures some of the\n")
cat("pattern but in crude steps, losing precision.\n")

# --- Question (c): Change quartiles to tertiles ---
cat("\n=== Question (c): Tertile categorisation ===\n")
sim_data$x_tert <- cut(sim_data$x,
                         breaks = quantile(sim_data$x, c(0, 1/3, 2/3, 1)),
                         include.lowest = TRUE)
fit_tert <- lrm(y ~ x_tert, data = sim_data)
cat("AIC with tertiles: ", AIC(fit_tert), "\n")
cat("AIC with quartiles:", AIC(fit_cat), "\n")
cat("AIC with RCS:      ", AIC(fit_rcs), "\n")

cat("\nChanging from quartiles to tertiles changes the categorised model's\n")
cat("estimates. The AIC may differ, and the estimated effects will change\n")
cat("because different patients are grouped together. This demonstrates\n")
cat("that categorisation results are ARBITRARY and depend on cut-point\n")
cat("choice. The RCS model does not have this problem because it models\n")
cat("the continuous relationship directly.\n")

# Test for non-linearity in the RCS model
cat("\n=== Non-linearity test for RCS model ===\n")
print(anova(fit_rcs))
