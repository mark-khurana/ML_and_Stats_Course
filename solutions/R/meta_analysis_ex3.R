# =============================================================================
# Chapter 18 - Exercise 3: Detecting the problem before the mega-trial
# What the magnesium evidence looked like in 1993, before ISIS-4 reported
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(meta)     # metabin(), funnel(), metabias()
library(metafor)  # the dat.egger2001 dataset

d <- dat.egger2001
pre <- subset(d, study != "ISIS-4")   # the 15 trials available before 1995

cat("Trials available pre-ISIS-4:", nrow(pre),
    "| total patients:", sum(pre$n1i + pre$n2i), "\n")
cat("Largest of them:", pre$study[which.max(pre$n1i + pre$n2i)],
    "with", max(pre$n1i + pre$n2i), "patients\n\n")

fit <- function(data) {
  metabin(event.e = ai, n.e = n1i, event.c = ci, n.c = n2i,
          studlab = paste(study, year), data = data, sm = "RR",
          method.tau = "REML", method.random.ci = "HK", prediction = TRUE)
}
m_pre <- fit(pre)
m_all <- fit(d)

# -----------------------------------------------------------------------------
# (a) What you would have concluded in 1993
# -----------------------------------------------------------------------------
cat("=== (a) The 15 trials, random effects ===\n")
cat(sprintf("RR = %.3f (95%% CI %.3f to %.3f)\n", exp(m_pre$TE.random),
            exp(m_pre$lower.random), exp(m_pre$upper.random)))
cat(sprintf("tau^2 = %.3f | I^2 = %.1f%%\n", m_pre$tau2, 100 * m_pre$I2))
cat(sprintf("Prediction interval: %.3f to %.3f\n",
            exp(m_pre$lower.predict), exp(m_pre$upper.predict)))
cat(sprintf("Fixed effect for comparison: RR = %.3f\n", exp(m_pre$TE.common)))
cat("\nOn this evidence you would have concluded that intravenous magnesium\n")
cat("roughly halves mortality after myocardial infarction, and the fixed and\n")
cat("random models AGREE, because without ISIS-4 there is no dominant large\n")
cat("trial to disagree with the small ones. That agreement is falsely reassuring.\n")
cat(sprintf("\nFor contrast, adding ISIS-4 later moves the fixed-effect estimate from\n"))
cat(sprintf("%.3f to %.3f.\n", exp(m_pre$TE.common), exp(m_all$TE.common)))

# -----------------------------------------------------------------------------
# (b) The funnel plot on the 15 trials
# -----------------------------------------------------------------------------
funnel(m_pre, xlab = "Risk ratio (log scale)",
       contour = c(0.9, 0.95, 0.99),
       col.contour = c("grey90", "grey80", "grey70"))
title(main = "Magnesium trials available before ISIS-4 (k = 15)")
legend("topright", c("p > 0.10", "p < 0.10", "p < 0.05"),
       fill = c("grey90", "grey80", "grey70"), bty = "n", cex = 0.8)

cat("\n=== (b) Funnel plot ===\n")
cat("Yes -- the asymmetry is clearly visible without the mega-trial. The lower\n")
cat("LEFT of the funnel (small trials showing benefit) is populated; the lower\n")
cat("RIGHT (small trials showing no benefit) is close to empty. The warning sign\n")
cat("was available years before ISIS-4 reported.\n")

# -----------------------------------------------------------------------------
# (c) The three asymmetry tests
# -----------------------------------------------------------------------------
cat("\n=== (c) Tests for funnel plot asymmetry (k = 15, so testing is allowed) ===\n")
for (test in c("Egger", "Harbord", "Peters")) {
  r <- metabias(m_pre, method.bias = test)
  if (is.null(r$p.value)) {
    cat(sprintf("  %-8s not performed (too few studies)\n", test))
  } else {
    cat(sprintf("  %-8s statistic %7.3f   p = %.4f\n", test, r$statistic, r$p.value))
  }
}
harbord_p <- metabias(m_pre, method.bias = "Harbord")$p.value

cat("\nWhich to trust: the outcome is BINARY (death), so prefer HARBORD or PETERS.\n")
cat("Egger's original test regresses the effect estimate on its standard error,\n")
cat("and for odds ratios and standardised mean differences those two quantities\n")
cat("are mathematically linked, which manufactures asymmetry and p-values that\n")
cat("are too small. Harbord's test fixes that correlation; Peters' test is the\n")
cat("most conservative of the three. Here all three agree, which is the easy case.\n")

# -----------------------------------------------------------------------------
# (d) The limitations paragraph you should have written in 1993
# -----------------------------------------------------------------------------
cat("\n=== (d) A two-sentence limitations paragraph ===\n")
cat(sprintf(
'"Fifteen trials totalling only %s patients (the largest randomising %s) suggest
 that intravenous magnesium substantially reduces mortality after myocardial
 infarction (RR %.2f, 95%% CI %.2f to %.2f); however the funnel plot is markedly
 asymmetric (Harbord p = %.3f) and the effect size falls as trial size rises, so
 we cannot exclude that small trials with null results are missing from the
 literature. The prediction interval spans %.2f to %.2f and therefore includes no
 effect, so a large pragmatic trial is needed before magnesium is adopted into
 routine practice."\n',
  format(sum(pre$n1i + pre$n2i), big.mark = ","),
  format(max(pre$n1i + pre$n2i), big.mark = ","),
  exp(m_pre$TE.random), exp(m_pre$lower.random), exp(m_pre$upper.random),
  harbord_p, exp(m_pre$lower.predict), exp(m_pre$upper.predict)))

cat("\nThat trial was ISIS-4: 58,050 patients, RR 1.06, no benefit.\n")
cat("\nOne last thing worth noticing about the 1993 evidence:\n")
cat(sprintf("  I^2 was only %.1f%% -- 'low heterogeneity' by the conventional bands --\n",
            100 * m_pre$I2))
cat("  and the fixed and random models broadly agreed. Both of the reassurances\n")
cat("  people usually look for were present. The two things that were NOT\n")
cat("  reassuring were the prediction interval crossing 1 and the asymmetric\n")
cat("  funnel, which is precisely why those deserve more attention than I^2.\n")
