# =============================================================================
# Chapter 18 - Exercise 1: Basic Meta-Analysis in R
# New anticoagulant vs warfarin for stroke prevention in atrial fibrillation
# =============================================================================

library(tidyverse)
library(meta)
library(metafor)

# --- Dataset ---
af_trials <- data.frame(
  study = c("TRAIL-1", "GUARD-AF", "SHIELD", "ORBIT-AF",
            "VENTURE", "COMPASS-AF", "PIONEER-2", "ATLAS-AF"),
  events_new = c(28, 45, 112, 67, 33, 89, 52, 41),
  n_new      = c(1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100),
  events_warf = c(42, 58, 148, 84, 29, 102, 61, 53),
  n_warf      = c(1200, 2500, 5400, 3100, 1800, 4200, 2800, 2100)
)

# --- Part (a): Compute RR and 95% CI for each trial ---
# Using metafor's escalc function
es <- escalc(measure = "RR",
             ai = events_new, n1i = n_new,
             ci = events_warf, n2i = n_warf,
             data = af_trials)

# Add human-readable columns
es$RR <- exp(es$yi)
es$ci_lo <- exp(es$yi - 1.96 * sqrt(es$vi))
es$ci_hi <- exp(es$yi + 1.96 * sqrt(es$vi))

cat("Individual trial risk ratios:\n")
print(es[, c("study", "RR", "ci_lo", "ci_hi")], digits = 3)

# By hand verification for first trial
rr_trail1 <- (28/1200) / (42/1200)
se_trail1 <- sqrt(1/28 - 1/1200 + 1/42 - 1/1200)
cat("\nManual check (TRAIL-1):")
cat("\n  RR =", round(rr_trail1, 3))
cat("\n  95% CI: (", round(exp(log(rr_trail1) - 1.96*se_trail1), 3),
    ",", round(exp(log(rr_trail1) + 1.96*se_trail1), 3), ")\n")

# --- Part (b): Random-effects meta-analysis ---
# Using the meta package
m1 <- metabin(
  event.e = events_new, n.e = n_new,
  event.c = events_warf, n.c = n_warf,
  studlab = study,
  data = af_trials,
  sm = "RR",
  method.tau = "REML",
  prediction = TRUE
)

cat("\n--- Random-effects meta-analysis ---\n")
print(summary(m1))

# Also using metafor for comparison
res <- rma(yi, vi, data = es, method = "REML")
cat("\n--- metafor results ---\n")
print(summary(res))

# --- Part (c): Forest plot ---
forest(m1,
       sortvar = TE,
       label.left = "Favours new anticoagulant",
       label.right = "Favours warfarin",
       col.diamond = "steelblue",
       col.square = "darkblue",
       print.tau2 = TRUE,
       print.I2 = TRUE,
       print.pval.Q = TRUE,
       prediction = TRUE,
       main = "New Anticoagulant vs Warfarin: Stroke Prevention in AF")

cat("\n--- Does the pooled effect favour the new anticoagulant? ---\n")
pooled_rr <- exp(m1$TE.random)
cat("Pooled RR:", round(pooled_rr, 3), "\n")
cat("95% CI: (", round(exp(m1$lower.random), 3), ",",
    round(exp(m1$upper.random), 3), ")\n")
if (pooled_rr < 1 & exp(m1$upper.random) < 1) {
  cat("Yes, the pooled effect significantly favours the new anticoagulant.\n")
} else if (pooled_rr < 1) {
  cat("The point estimate favours the new anticoagulant, but the CI includes 1.\n")
} else {
  cat("The pooled effect does not favour the new anticoagulant.\n")
}

# --- Part (d): I-squared and prediction interval ---
cat("\n--- Heterogeneity ---\n")
cat("I-squared:", round(m1$I2, 1), "%\n")
cat("tau-squared:", round(m1$tau2, 4), "\n")
cat("Q statistic:", round(m1$Q, 2), ", df =", m1$df.Q,
    ", p =", round(m1$pval.Q, 4), "\n")

cat("\n--- Prediction interval ---\n")
cat("Prediction interval for RR: (",
    round(exp(m1$lower.predict), 3), ",",
    round(exp(m1$upper.predict), 3), ")\n")

# Interpretation
if (m1$I2 < 25) {
  cat("I-squared suggests LOW heterogeneity.\n")
} else if (m1$I2 < 50) {
  cat("I-squared suggests MODERATE heterogeneity.\n")
} else if (m1$I2 < 75) {
  cat("I-squared suggests SUBSTANTIAL heterogeneity.\n")
} else {
  cat("I-squared suggests CONSIDERABLE heterogeneity.\n")
}

cat("\nThe prediction interval tells us the range within which the true\n")
cat("effect in a future study is expected to fall. If it crosses 1,\n")
cat("some settings might see no benefit from the new anticoagulant.\n")

# --- Part (e): Funnel plot and Egger's test ---
funnel(m1,
       xlab = "Risk Ratio (log scale)",
       studlab = TRUE,
       col = "steelblue",
       pch = 16,
       main = "Funnel Plot")

cat("\n--- Egger's test for funnel plot asymmetry ---\n")
egger <- metabias(m1, method.bias = "Egger")
print(egger)

if (egger$p.value < 0.05) {
  cat("Egger's test is significant (p < 0.05), suggesting potential\n")
  cat("publication bias or small-study effects.\n")
} else {
  cat("Egger's test is not significant, no strong evidence of\n")
  cat("publication bias. However, with only", length(af_trials$study),
      "studies,\n")
  cat("the test has limited power (recommended: >= 10 studies).\n")
}

# Trim-and-fill as additional check
tf <- trimfill(m1)
cat("\n--- Trim-and-fill ---\n")
cat("Imputed missing studies:", tf$k0, "\n")
cat("Adjusted pooled RR:", round(exp(tf$TE.random), 3), "\n")

# --- Part (f): Leave-one-out sensitivity analysis ---
cat("\n--- Leave-one-out sensitivity analysis ---\n")
l1o <- leave1out(res)
print(l1o)

# Check if removing any single study changes the conclusion
cat("\nIs the result robust to removing any single study?\n")
all_sig <- all(exp(l1o$ci.lb) < 1 | exp(l1o$ci.ub) > 1)
# Check if all leave-one-out CIs exclude 1 (if pooled is significant)
if (exp(res$ci.ub) < 1) {
  robust <- all(exp(l1o$ci.ub) < 1)
} else {
  robust <- TRUE
}

if (robust) {
  cat("Yes - no single study removal changes the direction or significance.\n")
} else {
  cat("No - removing at least one study changes the conclusion.\n")
  cat("The result may be driven by specific influential studies.\n")
}

# Influence diagnostics
inf <- influence(res)
plot(inf, main = "Influence Diagnostics")
