# =============================================================================
# Chapter 18 - Exercise 1: Fixed versus random effects, and why it mattered
# Intravenous magnesium after acute myocardial infarction (16 trials)
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(meta)     # metabin()
library(metafor)  # the dat.egger2001 dataset
library(tidyverse)

d <- dat.egger2001

# Note: in this dataset the control-arm event count is called `ci`, which reads
# confusingly next to "confidence interval". It is deaths in the control arm.
m <- metabin(
  event.e = ai, n.e = n1i,     # deaths / patients on magnesium
  event.c = ci, n.c = n2i,     # deaths / patients on control
  studlab = paste(study, year),
  data = d,
  sm = "RR",
  method.tau = "REML",         # REML rather than DerSimonian-Laird
  method.random.ci = "HK",     # Hartung-Knapp-Sidik-Jonkman interval
  prediction = TRUE
)

rr <- function(x) exp(x)

# -----------------------------------------------------------------------------
# (a) Both models
# -----------------------------------------------------------------------------
cat("=== (a) Fixed-effect vs random-effects ===\n")
cat(sprintf("Fixed effect   RR = %.3f (95%% CI %.3f to %.3f)\n",
            rr(m$TE.common), rr(m$lower.common), rr(m$upper.common)))
cat(sprintf("Random effects RR = %.3f (95%% CI %.3f to %.3f)\n",
            rr(m$TE.random), rr(m$lower.random), rr(m$upper.random)))
cat("\nSame 16 trials, same outcome. One model says magnesium does nothing;\n")
cat("the other says it roughly halves mortality.\n")

# -----------------------------------------------------------------------------
# (b) Where did the weight go?
# -----------------------------------------------------------------------------
w <- tibble(
  trial = m$studlab,
  n = d$n1i + d$n2i,
  fixed_pct = 100 * m$w.common / sum(m$w.common),
  random_pct = 100 * m$w.random / sum(m$w.random)
) |>
  arrange(desc(n))

cat("\n=== (b) Percentage weight under each model ===\n")
print(as.data.frame(w |> mutate(across(ends_with("pct"), \(x) round(x, 1)))),
      row.names = FALSE)

isis <- w |> filter(str_detect(trial, "ISIS"))
cat(sprintf(
  "\nISIS-4 has %.0f%% of all the patients but carries %.1f%% of the weight under\n",
  100 * isis$n / sum(w$n), isis$fixed_pct
))
cat(sprintf("the fixed-effect model and only %.1f%% under random effects.\n",
            isis$random_pct))
cat("\nWhy: random-effects weights are 1 / (within-study variance + tau^2).\n")
cat("ISIS-4's within-study variance is tiny, so adding tau^2 = ", round(m$tau2, 3),
    "\nswamps it and its weight collapses. A small trial's variance is already\n")
cat("large, so the same addition barely changes it. The effect is to level the\n")
cat("weights -- which hands the analysis to the 13 small trials.\n")

# -----------------------------------------------------------------------------
# (c) Heterogeneity: three statistics, three different questions
# -----------------------------------------------------------------------------
cat("\n=== (c) Heterogeneity ===\n")
cat(sprintf("tau^2 = %.3f  (tau = %.3f on the log-RR scale)\n", m$tau2, sqrt(m$tau2)))
cat(sprintf("I^2   = %.1f%%   Q test p = %.4f\n", 100 * m$I2, m$pval.Q))
cat(sprintf("Prediction interval: %.3f to %.3f\n",
            rr(m$lower.predict), rr(m$upper.predict)))
cat("\nThe PREDICTION INTERVAL is the one that answers 'how much does the effect\n")
cat("vary between settings'. I^2 is a ratio -- the share of the observed scatter\n")
cat("that is real rather than sampling noise -- and would rise if you simply ran\n")
cat("the same trials with more patients each. Note that the prediction interval\n")
cat("INCLUDES 1, while the confidence interval does not.\n")

# -----------------------------------------------------------------------------
# (d) What you would tell a guideline committee
# -----------------------------------------------------------------------------
cat("\n=== (d) Two one-sentence summaries ===\n")
cat("(i)  Fixed effect only: \"Pooling 62,607 patients across 16 randomised\n")
cat("     trials, intravenous magnesium had no effect on mortality after\n")
cat("     myocardial infarction (RR 1.01, 95% CI 0.95 to 1.06).\"\n\n")
cat("(ii) Random effects only: \"Pooling 16 randomised trials, intravenous\n")
cat("     magnesium reduced mortality after myocardial infarction by almost\n")
cat("     half (RR 0.51, 95% CI 0.36 to 0.74).\"\n\n")
cat("Both sentences are defensible from the same data, which is exactly why you\n")
cat("must report both models when they disagree, and why the prediction interval\n")
cat("and the funnel plot are not optional extras.\n")
