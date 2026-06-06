# =============================================================================
# Chapter 18 - Exercise 2: Meta-Analysis from Scratch in R
# (Exercise 2 is Python-focused, but we provide an R version as well)
# Manual implementation of meta-analysis computations
# =============================================================================

library(tidyverse)

# --- Dataset (same as Exercise 1) ---
af_trials <- data.frame(
  study = c("TRAIL-1", "GUARD-AF", "SHIELD", "ORBIT-AF",
            "VENTURE", "COMPASS-AF", "PIONEER-2", "ATLAS-AF"),
  events_new  = c(28, 45, 112, 67, 33, 89, 52, 41),
  n_new       = c(1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100),
  events_warf = c(42, 58, 148, 84, 29, 102, 61, 53),
  n_warf      = c(1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100)
)

# --- Part (a): Compute log RR and variances by hand ---
af_trials$rr <- (af_trials$events_new / af_trials$n_new) /
                (af_trials$events_warf / af_trials$n_warf)
af_trials$log_rr <- log(af_trials$rr)
af_trials$var_log_rr <- (1/af_trials$events_new - 1/af_trials$n_new +
                         1/af_trials$events_warf - 1/af_trials$n_warf)
af_trials$se_log_rr <- sqrt(af_trials$var_log_rr)

cat("Log risk ratios and SEs:\n")
print(af_trials[, c("study", "log_rr", "se_log_rr", "rr")])

# --- Part (b): Fixed-effect inverse-variance method ---
w_fe <- 1 / af_trials$var_log_rr
pooled_fe <- sum(w_fe * af_trials$log_rr) / sum(w_fe)
se_fe <- sqrt(1 / sum(w_fe))
ci_fe_lo <- pooled_fe - 1.96 * se_fe
ci_fe_hi <- pooled_fe + 1.96 * se_fe
z_fe <- pooled_fe / se_fe
p_fe <- 2 * (1 - pnorm(abs(z_fe)))

cat("\n--- Fixed-effect meta-analysis (by hand) ---\n")
cat("Pooled log RR:", round(pooled_fe, 4), "\n")
cat("Pooled RR:", round(exp(pooled_fe), 4),
    "(95% CI:", round(exp(ci_fe_lo), 4), "-", round(exp(ci_fe_hi), 4), ")\n")
cat("z =", round(z_fe, 3), ", p =", round(p_fe, 4), "\n")

# --- Part (c): DerSimonian-Laird random-effects ---
k <- nrow(af_trials)

# Cochran's Q
Q <- sum(w_fe * (af_trials$log_rr - pooled_fe)^2)

# C constant
C_val <- sum(w_fe) - sum(w_fe^2) / sum(w_fe)

# Between-study variance
tau2 <- max(0, (Q - (k - 1)) / C_val)

# Random-effects weights
w_re <- 1 / (af_trials$var_log_rr + tau2)
pooled_re <- sum(w_re * af_trials$log_rr) / sum(w_re)
se_re <- sqrt(1 / sum(w_re))
ci_re_lo <- pooled_re - 1.96 * se_re
ci_re_hi <- pooled_re + 1.96 * se_re
z_re <- pooled_re / se_re
p_re <- 2 * (1 - pnorm(abs(z_re)))

cat("\n--- Random-effects meta-analysis (DerSimonian-Laird, by hand) ---\n")
cat("Pooled log RR:", round(pooled_re, 4), "\n")
cat("Pooled RR:", round(exp(pooled_re), 4),
    "(95% CI:", round(exp(ci_re_lo), 4), "-", round(exp(ci_re_hi), 4), ")\n")
cat("z =", round(z_re, 3), ", p =", round(p_re, 4), "\n")

# --- Part (d): Q, I-squared, tau-squared ---
p_Q <- 1 - pchisq(Q, df = k - 1)
I2 <- max(0, (Q - (k - 1)) / Q) * 100

cat("\n--- Heterogeneity statistics ---\n")
cat("Q =", round(Q, 2), ", df =", k - 1, ", p =", round(p_Q, 4), "\n")
cat("tau^2 =", round(tau2, 4), "\n")
cat("I^2 =", round(I2, 1), "%\n")

# Prediction interval
t_crit <- qt(0.975, df = k - 2)
pi_lo <- pooled_re - t_crit * sqrt(se_re^2 + tau2)
pi_hi <- pooled_re + t_crit * sqrt(se_re^2 + tau2)
cat("Prediction interval for RR: (", round(exp(pi_lo), 4),
    ",", round(exp(pi_hi), 4), ")\n")

# --- Part (e): Forest plot (base R, no packages) ---
af_trials$w_re <- w_re
af_trials$ci_lo <- exp(af_trials$log_rr - 1.96 * af_trials$se_log_rr)
af_trials$ci_hi <- exp(af_trials$log_rr + 1.96 * af_trials$se_log_rr)

par(mar = c(5, 10, 4, 2))
y_pos <- k:1

plot(af_trials$rr, y_pos,
     xlim = c(0.3, 2.0), ylim = c(-0.5, k + 0.5),
     pch = 15, cex = af_trials$w_re / max(af_trials$w_re) * 2,
     xlab = "Risk Ratio", ylab = "", yaxt = "n",
     main = "Forest Plot (Built from Scratch)",
     log = "x", col = "steelblue")

# Study CIs
segments(af_trials$ci_lo, y_pos, af_trials$ci_hi, y_pos, lwd = 1.5)

# Study labels
axis(2, at = y_pos, labels = af_trials$study, las = 1)

# Null line
abline(v = 1, lty = 2, col = "grey50")

# Pooled estimate diamond
diamond_x <- c(exp(ci_re_lo), exp(pooled_re), exp(ci_re_hi), exp(pooled_re))
diamond_y <- c(0, 0.3, 0, -0.3)
polygon(diamond_x, diamond_y, col = "steelblue", border = "darkblue")

# Pooled reference line
abline(v = exp(pooled_re), col = "steelblue", lty = 3, lwd = 1)

# --- Part (f): Funnel plot and Egger's test (by hand) ---
par(mar = c(5, 5, 4, 2))
plot(af_trials$log_rr, af_trials$se_log_rr,
     pch = 16, col = "steelblue", cex = 1.5,
     xlim = c(min(af_trials$log_rr) - 0.3, max(af_trials$log_rr) + 0.3),
     ylim = rev(c(0, max(af_trials$se_log_rr) * 1.1)),
     xlab = "Log Risk Ratio", ylab = "Standard Error",
     main = "Funnel Plot (Built from Scratch)")

abline(v = pooled_re, col = "grey50", lty = 2)

# Pseudo-CI region
se_seq <- seq(0, max(af_trials$se_log_rr) * 1.1, length.out = 100)
lines(pooled_re - 1.96 * se_seq, se_seq, lty = 3, col = "grey70")
lines(pooled_re + 1.96 * se_seq, se_seq, lty = 3, col = "grey70")

# Egger's test: weighted regression of standardised effect on precision
# Regression of yi/sei on 1/sei, weighted by 1/vi
# Equivalent: regress yi on sei, weighted by 1/vi
egger_fit <- lm(log_rr ~ se_log_rr, data = af_trials, weights = w_fe)
cat("\n--- Egger's test (by hand) ---\n")
cat("Intercept:", round(coef(egger_fit)[1], 4), "\n")
cat("SE:", round(summary(egger_fit)$coefficients[1, 2], 4), "\n")
cat("p-value:", round(summary(egger_fit)$coefficients[1, 4], 4), "\n")

if (summary(egger_fit)$coefficients[1, 4] < 0.05) {
  cat("Significant asymmetry detected.\n")
} else {
  cat("No significant asymmetry (limited power with", k, "studies).\n")
}
