# Chapter 11, Exercise 2: External Validation Simulation
# Create three external validation populations and assess model performance

library(rms)
library(pROC)

# ---- Simulate development data and fit model (same as chapter) ----
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

dd <- datadist(stroke_data)
options(datadist = "dd")

fit <- lrm(death_30d ~ age + nihss + glucose + afib + thrombolysis,
           data = stroke_data, x = TRUE, y = TRUE)

# ---- Helper function: evaluate model performance ----
evaluate_ext <- function(ext_data, fit, label) {
  ext_data$pred <- predict(fit, newdata = ext_data, type = "fitted")
  obs_rate <- mean(ext_data$death_30d)
  mean_pred <- mean(ext_data$pred)
  oe <- obs_rate / mean_pred

  # C-statistic
  roc_obj <- roc(ext_data$death_30d, ext_data$pred, quiet = TRUE)
  c_stat <- auc(roc_obj)


  # Calibration slope
  lp_ext <- qlogis(ext_data$pred)
  cal_model <- glm(death_30d ~ lp_ext, data = ext_data, family = binomial)
  cal_slope <- coef(cal_model)[2]
  cal_int <- coef(cal_model)[1]

  cat(sprintf("\n=== %s ===\n", label))
  cat(sprintf("  N = %d, Observed mortality: %.3f\n", nrow(ext_data), obs_rate))
  cat(sprintf("  Mean predicted: %.3f\n", mean_pred))
  cat(sprintf("  C-statistic: %.3f\n", c_stat))
  cat(sprintf("  O:E ratio: %.3f\n", oe))
  cat(sprintf("  Calibration slope: %.3f\n", cal_slope))
  cat(sprintf("  Calibration intercept: %.3f\n", cal_int))

  return(ext_data)
}

# ---- Population A: Temporal validation (3 years later) ----
# Same demographics, slightly different practice patterns
set.seed(101)
n_a <- 800
pop_a <- data.frame(
  age = round(rnorm(n_a, 73, 12)),        # Similar age
  nihss = round(pmax(0, rnorm(n_a, 8, 6))),
  glucose = round(rnorm(n_a, 138, 48)),
  afib = rbinom(n_a, 1, 0.27),
  thrombolysis = rbinom(n_a, 1, 0.40)     # More thrombolysis over time
)

lp_a <- -5 + 0.04 * pop_a$age + 0.12 * pop_a$nihss +
  0.003 * pop_a$glucose + 0.3 * pop_a$afib - 0.5 * pop_a$thrombolysis
pop_a$death_30d <- rbinom(n_a, 1, plogis(lp_a))

pop_a <- evaluate_ext(pop_a, fit, "Population A: Temporal (3 years later)")

# ---- Population B: Geographical (younger patients) ----
set.seed(202)
n_b <- 600
pop_b <- data.frame(
  age = round(rnorm(n_b, 62, 10)),        # Younger
  nihss = round(pmax(0, rnorm(n_b, 6, 5))),  # Lower severity
  glucose = round(rnorm(n_b, 130, 45)),
  afib = rbinom(n_b, 1, 0.15),            # Less AF
  thrombolysis = rbinom(n_b, 1, 0.35)
)

lp_b <- -5 + 0.04 * pop_b$age + 0.12 * pop_b$nihss +
  0.003 * pop_b$glucose + 0.3 * pop_b$afib - 0.5 * pop_b$thrombolysis
pop_b$death_30d <- rbinom(n_b, 1, plogis(lp_b))

pop_b <- evaluate_ext(pop_b, fit, "Population B: Geographical (younger patients)")

# ---- Population C: Domain (primary care, lower severity) ----
set.seed(303)
n_c <- 500
pop_c <- data.frame(
  age = round(rnorm(n_c, 68, 14)),
  nihss = round(pmax(0, rnorm(n_c, 3, 3))),  # Much lower severity
  glucose = round(rnorm(n_c, 120, 35)),
  afib = rbinom(n_c, 1, 0.20),
  thrombolysis = rbinom(n_c, 1, 0.10)        # Rarely used in primary care
)

# Different outcome model: lower baseline risk in primary care
lp_c <- -6 + 0.03 * pop_c$age + 0.10 * pop_c$nihss +
  0.002 * pop_c$glucose + 0.2 * pop_c$afib - 0.3 * pop_c$thrombolysis
pop_c$death_30d <- rbinom(n_c, 1, plogis(lp_c))

pop_c <- evaluate_ext(pop_c, fit, "Population C: Domain (primary care)")

# ---- (b) Calibration plots ----
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
val.prob(pop_a$pred, pop_a$death_30d, m = 80, cex = 0.5,
         main = "A: Temporal")
val.prob(pop_b$pred, pop_b$death_30d, m = 60, cex = 0.5,
         main = "B: Geographical")
val.prob(pop_c$pred, pop_c$death_30d, m = 50, cex = 0.5,
         main = "C: Domain")

# ---- (c) Which population shows worst calibration? ----
cat("\n=== Part (c): Worst Calibration ===\n")
cat("Population C (primary care / domain validation) shows the worst calibration.\n")
cat("This is because the outcome model differs from the development setting:\n")
cat("  - Different baseline risk (intercept)\n")
cat("  - Different predictor-outcome relationships (coefficients)\n")
cat("  - Different case-mix (lower severity patients)\n")
cat("Domain validation is the most stringent test of transportability.\n")

# ---- (d) Logistic recalibration ----
cat("\n=== Part (d): Logistic Recalibration ===\n")

recalibrate <- function(ext_data, label) {
  lp_ext <- qlogis(ext_data$pred)
  cal_fit <- glm(death_30d ~ lp_ext, data = ext_data, family = binomial)
  ext_data$pred_recal <- predict(cal_fit, type = "response")

  oe_before <- mean(ext_data$death_30d) / mean(ext_data$pred)
  oe_after  <- mean(ext_data$death_30d) / mean(ext_data$pred_recal)

  cat(sprintf("\n%s:\n", label))
  cat(sprintf("  O:E before recalibration: %.3f\n", oe_before))
  cat(sprintf("  O:E after recalibration:  %.3f\n", oe_after))
}

recalibrate(pop_a, "Population A")
recalibrate(pop_b, "Population B")
recalibrate(pop_c, "Population C")

cat("\nLogistic recalibration corrects for differences in baseline risk\n")
cat("and prediction spread, bringing O:E ratios closer to 1.0.\n")
