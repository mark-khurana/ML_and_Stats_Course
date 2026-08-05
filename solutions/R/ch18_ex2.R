# =============================================================================
# Chapter 18 - Exercise 2: Meta-analysis from scratch
# Inverse-variance pooling by hand, checked against the meta package
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(meta)     # only for the check in part (d)
library(metafor)  # the dat.egger2001 dataset

d <- dat.egger2001

# -----------------------------------------------------------------------------
# (a) Log risk ratios and their variances, handling the zero cell
# -----------------------------------------------------------------------------
# The log risk ratio and its variance are
#     log RR = log( (a/(a+b)) / (c/(c+d)) )
#     Var    = 1/a - 1/(a+b) + 1/c - 1/(c+d)
# Bertschat 1989 recorded 0 deaths on magnesium. log(0) is undefined and 1/0 is
# infinite, so that trial cannot be used as it stands. The convention is to add
# a small continuity increment of 0.5 to the cells of the affected trial only --
# NOT to every trial, which would shift all 16 estimates.
zero_cell <- d$ai == 0 | d$ci == 0
cat("Trials with a zero cell:", paste(d$study[zero_cell], collapse = ", "), "\n\n")

incr <- ifelse(zero_cell, 0.5, 0)
a <- d$ai + incr
b <- d$n1i - d$ai + incr
c_ <- d$ci + incr
dd <- d$n2i - d$ci + incr

log_rr <- log((a / (a + b)) / (c_ / (c_ + dd)))
v <- 1 / a - 1 / (a + b) + 1 / c_ - 1 / (c_ + dd)

cat("=== (a) First few trials ===\n")
print(data.frame(trial = paste(d$study, d$year), log_rr = round(log_rr, 3),
                 var = round(v, 4), se = round(sqrt(v), 3))[1:5, ],
      row.names = FALSE)

# -----------------------------------------------------------------------------
# (b) Fixed-effect pooled estimate
# -----------------------------------------------------------------------------
w_fe <- 1 / v
te_fe <- sum(w_fe * log_rr) / sum(w_fe)
se_fe <- sqrt(1 / sum(w_fe))

cat(sprintf("\n=== (b) Fixed effect (inverse variance) ===\n"))
cat(sprintf("log RR = %+.4f (SE %.4f)  ->  RR = %.3f (95%% CI %.3f to %.3f)\n",
            te_fe, se_fe, exp(te_fe),
            exp(te_fe - 1.96 * se_fe), exp(te_fe + 1.96 * se_fe)))

# -----------------------------------------------------------------------------
# (c) tau^2 by DerSimonian-Laird, then the random-effects estimate
# -----------------------------------------------------------------------------
k <- length(log_rr)
Q <- sum(w_fe * (log_rr - te_fe)^2)
C <- sum(w_fe) - sum(w_fe^2) / sum(w_fe)
tau2_dl <- max(0, (Q - (k - 1)) / C)
I2 <- max(0, (Q - (k - 1)) / Q)

w_re <- 1 / (v + tau2_dl)
te_re <- sum(w_re * log_rr) / sum(w_re)
se_re <- sqrt(1 / sum(w_re))

cat(sprintf("\n=== (c) Random effects (DerSimonian-Laird) ===\n"))
cat(sprintf("Q = %.2f on %d df, p = %.4f\n", Q, k - 1, pchisq(Q, k - 1, lower.tail = FALSE)))
cat(sprintf("tau^2 = %.4f | I^2 = %.1f%%\n", tau2_dl, 100 * I2))
cat(sprintf("log RR = %+.4f  ->  RR = %.3f (95%% CI %.3f to %.3f)\n",
            te_re, exp(te_re),
            exp(te_re - 1.96 * se_re), exp(te_re + 1.96 * se_re)))

# -----------------------------------------------------------------------------
# (d) Check against the package, then switch to REML
# -----------------------------------------------------------------------------
# metabin() defaults to Mantel-Haenszel for the fixed-effect estimate with binary
# outcomes, so ask for "Inverse" to match what we just computed by hand.
chk_dl <- metabin(ai, n1i, ci, n2i, data = d, sm = "RR",
                  method = "Inverse", method.tau = "DL")
chk_reml <- metabin(ai, n1i, ci, n2i, data = d, sm = "RR",
                    method = "Inverse", method.tau = "REML")

cat("\n=== (d) Hand calculation vs metabin() ===\n")
cmp <- data.frame(
  quantity = c("fixed-effect RR", "random-effects RR", "tau^2"),
  by_hand = round(c(exp(te_fe), exp(te_re), tau2_dl), 4),
  metabin_DL = round(c(exp(chk_dl$TE.common), exp(chk_dl$TE.random), chk_dl$tau2), 4)
)
cmp$agrees <- ifelse(abs(cmp$by_hand - cmp$metabin_DL) < 5e-4, "yes", "NO")
print(cmp, row.names = FALSE)

cat(sprintf("\nSwitching to REML: tau^2 goes from %.4f (DL) to %.4f (REML),\n",
            chk_dl$tau2, chk_reml$tau2))
cat(sprintf("and the random-effects RR from %.3f to %.3f.\n",
            exp(chk_dl$TE.random), exp(chk_reml$TE.random)))
cat("\nWhy it moves: DerSimonian-Laird is a moment estimator and is known to\n")
cat("UNDERESTIMATE the between-study variance, particularly with few studies or\n")
cat("very unequal study sizes (Veroniki et al. 2016). A larger tau^2 levels the\n")
cat("weights further, so the pooled estimate moves further towards the small\n")
cat("trials, and the prediction interval widens. REML (or Paule-Mandel) is the\n")
cat("current recommendation; DL survives mainly because it was the default for\n")
cat("thirty years and needs no iteration.\n")
