# Exercise 1: Sample size calculation
# Pre-eclampsia model: prevalence 4%, 12 candidate predictors,
# an anticipated C-statistic of 0.72 from a published model in a
# comparable population.

library(pmsampsize)

# (a) Minimum sample size ----------------------------------------------------
# type = "b"      binary outcome
# cstatistic =    anticipated C-statistic; pmsampsize converts this into the
#                 Cox-Snell R-squared the criteria actually need
# parameters =    number of predictor PARAMETERS, not variables (a categorical
#                 predictor with k levels costs k - 1, a spline costs more)
ss <- pmsampsize(
  type = "b",
  cstatistic = 0.72,
  parameters = 12,
  prevalence = 0.04
)

cat("\nMinimum sample size:", ss$sample_size, "pregnancies\n")
cat("Minimum number of events:", ceiling(ss$events), "\n")
cat("Events per parameter:", round(ss$EPP, 2), "\n")

# (b) Which criterion is binding? -------------------------------------------
# pmsampsize's results table has one row per criterion and a "final" row that
# is simply the largest of them. The binding criterion is whichever row the
# final row was taken from.
res <- ss$results_table
print(res)

crit_n <- res[rownames(res) != "Final", "Samp_size"]
binding <- names(which.max(crit_n))
cat("\nSample size demanded by each criterion:\n")
print(crit_n)
cat("\nBinding criterion:", binding, "->", max(crit_n), "pregnancies\n")

# (c) Comparison with the 10-EPV rule of thumb -----------------------------
# The old rule asks only for 10 events per parameter, and says nothing about
# how precisely the model must be estimated or how much it may overfit.
epv_events <- 10 * 12
epv_n <- ceiling(epv_events / 0.04)

cat("\n--- 10 events per variable rule ---\n")
cat("Events required:", epv_events, "\n")
cat("Implied sample size at 4% prevalence:", epv_n, "\n")

cat("\n--- Riley criteria (pmsampsize) ---\n")
cat("Events required:", ceiling(ss$events), "\n")
cat("Sample size:", ss$sample_size, "\n")

cat(
  "\nRiley / EPV ratio:",
  round(ss$sample_size / epv_n, 2),
  "times the EPV recommendation\n"
)

# Note what the low prevalence does. At 4%, events are expensive: every extra
# event costs 25 pregnancies. That is why the required sample size is large
# even though the number of parameters is modest, and it is the argument for
# reducing the candidate predictor list before recruitment rather than after.
