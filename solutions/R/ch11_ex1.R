# Chapter 11, Exercise 1: Calibration Assessment
# Using the stroke mortality model from the chapter

library(rms)

# ---- Simulate the stroke data (same as chapter) ----
set.seed(2024)
n <- 1500

stroke_data <- data.frame(
  age = round(rnorm(n, 72, 12)),
  nihss = round(pmax(0, rnorm(n, 8, 6))),
  glucose = round(rnorm(n, 140, 50)),
  afib = rbinom(n, 1, 0.25),
  thrombolysis = rbinom(n, 1, 0.30)
)

lp <- -5 + 0.04 * stroke_data$age +
  0.12 * stroke_data$nihss +
  0.003 * stroke_data$glucose +
  0.3 * stroke_data$afib -
  0.5 * stroke_data$thrombolysis

stroke_data$death_30d <- rbinom(n, 1, plogis(lp))

# Fit the prediction model
dd <- datadist(stroke_data)
options(datadist = "dd")

fit <- lrm(death_30d ~ age + nihss + glucose + afib + thrombolysis,
           data = stroke_data, x = TRUE, y = TRUE)

pred_prob <- predict(fit, type = "fitted")

# ---- (a) Calibration plot using deciles of predicted risk ----
cat("=== Part (a): Calibration Plot ===\n")
val.prob(pred_prob, stroke_data$death_30d, m = 150, cex = 0.5,
         main = "Calibration Plot (Deciles): 30-Day Stroke Mortality")
# The model appears well-calibrated since the points cluster near the diagonal.
# This is expected because we are evaluating apparent performance on the
# training data.

# ---- (b) O:E ratio ----
cat("\n=== Part (b): O:E Ratio ===\n")
obs_rate <- mean(stroke_data$death_30d)
mean_pred <- mean(pred_prob)
oe_ratio <- obs_rate / mean_pred

cat("Observed event rate:", round(obs_rate, 3), "\n")
cat("Mean predicted probability:", round(mean_pred, 3), "\n")
cat("O:E ratio:", round(oe_ratio, 3), "\n")
# An O:E ratio near 1.0 indicates good calibration-in-the-large.
# The model's average predictions match the observed event rate.

# ---- (c) Calibration slope ----
cat("\n=== Part (c): Calibration Slope ===\n")
cal_model <- glm(stroke_data$death_30d ~ qlogis(pred_prob), family = binomial)
cal_slope <- coef(cal_model)[2]
cal_intercept <- coef(cal_model)[1]

cat("Calibration slope:", round(cal_slope, 3), "\n")
cat("Calibration intercept:", round(cal_intercept, 3), "\n")
# A calibration slope of 1.0 indicates no overfitting.
# The apparent slope is typically close to 1 on the training data.
# Values < 1 on new data would suggest overfitting (predictions too extreme).

# ---- (d) Bootstrap optimism correction ----
cat("\n=== Part (d): Bootstrap Optimism Correction ===\n")
set.seed(42)
val <- validate(fit, B = 200)

cat("Bootstrap Validation Results:\n")
print(val)

# Extract optimism-corrected C-statistic
dxy_corrected <- val["Dxy", "index.corrected"]
c_apparent <- fit$stats["C"]
c_corrected <- (dxy_corrected + 1) / 2

cat("\nApparent C-statistic:", round(c_apparent, 4), "\n")
cat("Optimism-corrected C-statistic:", round(c_corrected, 4), "\n")
cat("C-statistic decrease:", round(c_apparent - c_corrected, 4), "\n")

# Calibration slope from bootstrap
slope_corrected <- val["Slope", "index.corrected"]
cat("\nApparent calibration slope: 1.000\n")
cat("Optimism-corrected calibration slope:", round(slope_corrected, 4), "\n")
cat("Slope decrease:", round(1 - slope_corrected, 4), "\n")

# The C-statistic decreases slightly after optimism correction, reflecting
# the mild optimism in apparent performance. The calibration slope also
# decreases below 1, indicating some overfitting that inflates apparent
# performance.
