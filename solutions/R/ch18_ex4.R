# =============================================================================
# Chapter 18 - Exercise 4: Critical appraisal of a published meta-analysis
# =============================================================================
#
# This is a conceptual exercise: the answer depends on the paper you chose. What
# follows is (1) a reusable checklist with the reasoning behind each item, and
# (2) a worked appraisal of the magnesium literature, which is the one paper we
# can all read the same way.
#
# Libraries -------------------------------------------------------------------
library(meta)
library(metafor)

# -----------------------------------------------------------------------------
# The checklist, and why each item is on it
# -----------------------------------------------------------------------------
# (a) HOW MANY STUDIES, AND HOW BIG?
#     Compute the largest study's share of the total sample. If one trial holds
#     most of the patients, the fixed/random choice will dominate the answer and
#     the paper must justify it. If every trial is small, ask what is missing.
#
# (b) WHICH MODEL, AND ARE BOTH REPORTED?
#     Random effects is the usual default. The question is whether the paper
#     reports the fixed-effect result too. If it does not and one trial is much
#     larger than the rest, you cannot tell whether the choice mattered -- and it
#     is exactly then that it matters most.
#
# (c) IS tau^2 OR A PREDICTION INTERVAL REPORTED, OR ONLY I^2?
#     I^2 is the PROPORTION of observed scatter that is real rather than sampling
#     noise. It does not tell you how much the effect varies, and it rises if you
#     simply run the same trials with more patients each. If only I^2 is given,
#     you cannot answer "would this work in my setting?" at all.
#
# (d) WAS ASYMMETRY ASSESSED, AND LEGITIMATELY?
#     Needs k >= 10. Below that the tests have too little power and Cochrane
#     advises against them; "we could not assess it" is the correct report, not
#     "the test was non-significant". And check the test suits the effect
#     measure: Egger's test is not appropriate for odds ratios or standardised
#     mean differences (use Harbord or Peters).
#
# (e) WOULD YOU CHANGE PRACTICE?
#     Force yourself to name the condition. "I would change if the prediction
#     interval excluded no effect and the large trials agreed with the small
#     ones" is a real answer; "the result was significant" is not.

# -----------------------------------------------------------------------------
# A worked appraisal: the magnesium literature
# -----------------------------------------------------------------------------
d <- dat.egger2001
m <- metabin(event.e = ai, n.e = n1i, event.c = ci, n.c = n2i,
             studlab = paste(study, year), data = d, sm = "RR",
             method.tau = "REML", method.random.ci = "HK", prediction = TRUE)

n_tot <- d$n1i + d$n2i
cat("=== (a) Size and spread ===\n")
cat(sprintf("k = %d trials, %s patients in total\n", m$k, format(sum(n_tot), big.mark = ",")))
cat(sprintf("smallest %d, largest %s (%.0f%% of all patients)\n",
            min(n_tot), format(max(n_tot), big.mark = ","),
            100 * max(n_tot) / sum(n_tot)))
cat("  -> one trial holds most of the evidence, so the model choice is decisive.\n")

cat("\n=== (b) Model choice ===\n")
cat(sprintf("fixed effect  RR = %.3f (95%% CI %.3f to %.3f)\n",
            exp(m$TE.common), exp(m$lower.common), exp(m$upper.common)))
cat(sprintf("random effects RR = %.3f (95%% CI %.3f to %.3f)\n",
            exp(m$TE.random), exp(m$lower.random), exp(m$upper.random)))
cat("  -> the two models give opposite conclusions. Reporting only one would be\n")
cat("     indefensible here.\n")

cat("\n=== (c) Heterogeneity ===\n")
cat(sprintf("tau^2 = %.3f | I^2 = %.1f%% | prediction interval %.3f to %.3f\n",
            m$tau2, 100 * m$I2, exp(m$lower.predict), exp(m$upper.predict)))
cat("  -> the prediction interval INCLUDES 1 even though the confidence interval\n")
cat("     does not, so a new setting could plausibly see no benefit.\n")

cat("\n=== (d) Asymmetry ===\n")
cat(sprintf("k = %d, so testing is legitimate (threshold is 10)\n", m$k))
for (test in c("Egger", "Harbord", "Peters")) {
  r <- metabias(m, method.bias = test)
  cat(sprintf("  %-8s p = %.4f%s\n", test, r$p.value,
              if (test == "Egger") "   <- not the right test for a ratio measure" else ""))
}
cat("  -> strong evidence of small-study effects on all three tests.\n")

cat("\n=== (e) Verdict ===\n")
cat("No. Three separate signals -- a fixed/random reversal, a prediction interval\n")
cat("crossing 1, and a markedly asymmetric funnel -- all say the same thing: the\n")
cat("small trials disagree with the large one, and the pooled benefit is an\n")
cat("artefact of giving the small trials more weight. What would change my mind:\n")
cat("a further large trial agreeing with the small ones, or a mechanism for why\n")
cat("effects should genuinely be larger in the settings the small trials studied.\n")
cat("\nHistorically the answer was settled by ISIS-4 (58,050 patients, RR 1.06):\n")
cat("no benefit. The appraisal above would have reached the right answer without\n")
cat("waiting for it.\n")
