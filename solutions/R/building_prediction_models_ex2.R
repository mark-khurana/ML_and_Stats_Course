# Exercise 2: Missing data simulation
# Compare complete case analysis, single mean imputation, and multiple
# imputation on the simulated Framingham cohort from the chapter.

library(mice)

# --- The cohort, exactly as in the chapter ---------------------------------
set.seed(2024)
n <- 2000

framingham <- data.frame(
  age = round(runif(n, 30, 74)),
  male = rbinom(n, 1, 0.48),
  sbp = round(rnorm(n, 130, 18)),
  total_chol = round(rnorm(n, 210, 38)),
  hdl_chol = round(rnorm(n, 52, 15)),
  smoking = rbinom(n, 1, 0.22),
  diabetes = rbinom(n, 1, 0.08),
  bp_treatment = rbinom(n, 1, 0.15)
)

# The true coefficients. Because we simulated the data we know them, which is
# what makes this comparison possible at all.
truth <- c(
  "(Intercept)" = -7.5, age = 0.06, male = 0.4, sbp = 0.012,
  total_chol = 0.005, hdl_chol = -0.02, smoking = 0.5,
  diabetes = 0.7, bp_treatment = 0.3
)

lp <- with(
  framingham,
  -7.5 + 0.06 * age + 0.4 * male + 0.012 * sbp + 0.005 * total_chol -
    0.02 * hdl_chol + 0.5 * smoking + 0.7 * diabetes + 0.3 * bp_treatment
)
framingham$cvd_10yr <- rbinom(n, 1, plogis(lp))
cat("Cohort:", n, "patients,", sum(framingham$cvd_10yr), "events\n")

model_formula <- cvd_10yr ~ age + male + sbp + total_chol + hdl_chol +
  smoking + diabetes + bp_treatment

# --- Make two predictors missing, under MAR -------------------------------
# The mechanism matters more than the amount. Missingness in total_chol is made
# to depend on the OUTCOME and on age -- both recorded, so this is MAR, not
# MNAR. That choice is deliberate: if missingness depended only on the
# predictors already in the model, complete case analysis would still be
# unbiased for the coefficients, and there would be nothing to see.
punch_holes <- function(df, intercept) {
  set.seed(7)
  p_chol <- plogis(intercept + 1.0 * df$cvd_10yr + 0.03 * (df$age - 52))
  p_hdl <- plogis(intercept + 0.2 + 0.9 * df$smoking + 0.02 * (df$sbp - 130))
  df$total_chol[runif(nrow(df)) < p_chol] <- NA
  df$hdl_chol[runif(nrow(df)) < p_hdl] <- NA
  df
}

incomplete <- punch_holes(framingham, intercept = -2.2)

cat("Missing total_chol:", sum(is.na(incomplete$total_chol)), "\n")
cat("Missing hdl_chol: ", sum(is.na(incomplete$hdl_chol)), "\n")
cat(
  "Complete rows:", sum(complete.cases(incomplete)),
  sprintf("(%.0f%% of the cohort discarded by a complete case analysis)\n",
          100 * mean(!complete.cases(incomplete)))
)

# --- The three approaches, plus the full data as a benchmark --------------
fit_full <- glm(model_formula, data = framingham, family = binomial)
fit_cca <- glm(model_formula, data = incomplete, family = binomial)

mean_imputed <- incomplete
for (v in c("total_chol", "hdl_chol")) {
  mean_imputed[[v]][is.na(mean_imputed[[v]])] <- mean(mean_imputed[[v]], na.rm = TRUE)
}
fit_mean <- glm(model_formula, data = mean_imputed, family = binomial)

# 20 imputations, predictive mean matching, pooled with Rubin's rules
imp <- mice(incomplete, m = 20, method = "pmm", seed = 42, printFlag = FALSE)
pool_obj <- pool(with(
  imp,
  glm(cvd_10yr ~ age + male + sbp + total_chol + hdl_chol + smoking +
        diabetes + bp_treatment, family = binomial)
))
pooled <- summary(pool_obj)
mi_est <- setNames(pooled$estimate, pooled$term)
mi_se <- setNames(pooled$std.error, pooled$term)
# lambda is the share of the pooled variance that comes from the missing data
# (the between-imputation part). It is precisely what single imputation drops.
mi_lambda <- setNames(pool_obj$pooled$lambda, pool_obj$pooled$term)

# --- (a) Which approach lands closest to the truth? ----------------------
comparison <- data.frame(
  truth = truth,
  full_data = coef(fit_full),
  complete_case = coef(fit_cca),
  mean_imputation = coef(fit_mean),
  multiple_imputation = mi_est[names(truth)]
)
cat("\n--- Coefficient estimates ---\n")
print(round(comparison, 4))

# Two error measures, and the difference between them is the point. Distance
# from the truth mixes up two things: damage done by the missing data, and the
# sampling noise that was already in this cohort of 2000. Distance from the
# full-data estimates isolates the first.
slopes <- rownames(comparison) != "(Intercept)"
mae <- function(x) mean(abs(x - comparison$truth[slopes]))
mae_full <- function(x) mean(abs(x - comparison$full_data[slopes]))

cat("\n--- Mean absolute error across the 8 slopes ---\n")
err <- data.frame(
  vs_truth = sapply(comparison[slopes, -1], mae),
  vs_full_data = sapply(comparison[slopes, -1], mae_full)
)
print(round(err, 5))

cat("\nRanked by distance from the full-data estimates:\n")
print(round(sort(err$vs_full_data[-1] |> setNames(rownames(err)[-1])), 5))

# --- (b) What happens to the standard errors? ---------------------------
se_table <- data.frame(
  complete_case = summary(fit_cca)$coefficients[, "Std. Error"],
  mean_imputation = summary(fit_mean)$coefficients[, "Std. Error"],
  multiple_imputation = mi_se[names(truth)]
)
cat("\n--- Standard errors ---\n")
print(round(se_table, 5))

cat("\nMean imputation SE as a percentage of the multiple-imputation SE,\n")
cat("for the two variables that were actually imputed:\n")
for (v in c("total_chol", "hdl_chol")) {
  cat(sprintf(
    "  %-11s %.1f%%\n", v,
    100 * se_table[v, "mean_imputation"] / se_table[v, "multiple_imputation"]
  ))
}

# What single imputation actually discards, as a number. Rubin's rules split the
# pooled variance into a within-imputation part (ordinary sampling uncertainty)
# and a between-imputation part (uncertainty about the guesses themselves).
# Single imputation sets the second to zero by construction.
cat("\nShare of the pooled MI variance that comes from the imputation itself:\n")
for (v in c("total_chol", "hdl_chol")) {
  cat(sprintf("  %-11s %.1f%%\n", v, 100 * mi_lambda[[v]]))
}

# Two opposing effects on the mean-imputation standard error, which is why the
# percentages above are close to 100 at this fraction of missing data:
#   1. it ignores the imputation uncertainty just quantified  -> SE too small
#   2. it flattens the variable's spread, and less spread in a
#      predictor means less information about its coefficient  -> SE too large
# The first grows with the fraction missing; the second is the reason the two
# can briefly cancel. Neither makes the SE trustworthy.
cat(sprintf(
  "\nSD of total_chol: %.1f observed -> %.1f after mean imputation\n",
  sd(incomplete$total_chol, na.rm = TRUE), sd(mean_imputed$total_chol)
))

# --- The same comparison with far more missing data ---------------------
# At 12-16% missing the three approaches disagree modestly. Raise the
# missingness to roughly half and the false precision of single imputation
# becomes impossible to miss.
heavy <- punch_holes(framingham, intercept = -0.4)
heavy_mean <- heavy
for (v in c("total_chol", "hdl_chol")) {
  heavy_mean[[v]][is.na(heavy_mean[[v]])] <- mean(heavy_mean[[v]], na.rm = TRUE)
}
imp_h <- mice(heavy, m = 20, method = "pmm", seed = 42, printFlag = FALSE)
pool_h <- pool(with(
  imp_h,
  glm(cvd_10yr ~ age + male + sbp + total_chol + hdl_chol + smoking +
        diabetes + bp_treatment, family = binomial)
))
pooled_h <- summary(pool_h)
lambda_h <- setNames(pool_h$pooled$lambda, pool_h$pooled$term)

cat(sprintf(
  "\n--- With %.0f%% of total_chol and %.0f%% of hdl_chol missing ---\n",
  100 * mean(is.na(heavy$total_chol)), 100 * mean(is.na(heavy$hdl_chol))
))
s_cca_h <- summary(glm(model_formula, data = heavy, family = binomial))$coefficients
s_mean_h <- summary(glm(model_formula, data = heavy_mean, family = binomial))$coefficients
for (v in c("total_chol", "hdl_chol")) {
  cat(sprintf("%-11s truth %+.4f\n", v, truth[[v]]))
  cat(sprintf(
    "  complete case   %+.4f (SE %.5f)\n  mean imputation %+.4f (SE %.5f)\n  MI              %+.4f (SE %.5f)\n",
    s_cca_h[v, 1], s_cca_h[v, 2], s_mean_h[v, 1], s_mean_h[v, 2],
    pooled_h$estimate[pooled_h$term == v],
    pooled_h$std.error[pooled_h$term == v]
  ))
  cat(sprintf(
    "  -> mean imputation's SE is %.0f%% of MI's, and %.0f%% of MI's variance\n     for this coefficient now comes from the imputation (was %.0f%%)\n",
    100 * s_mean_h[v, 2] / pooled_h$std.error[pooled_h$term == v],
    100 * lambda_h[[v]], 100 * mi_lambda[[v]]
  ))
}

cat("
Conclusions
-----------
(a) Complete case analysis is the clear loser. It is the furthest from the
    full-data estimates, and it is biased by construction here: because
    missingness depends on the outcome, the retained rows under-represent
    patients who had an event, and coefficients such as smoking are distorted
    well beyond sampling noise. It also discards a quarter of the cohort, so
    its standard errors are the widest of the three.

    Mean imputation and multiple imputation give similar point estimates at
    this fraction of missing data. Note also that measuring against the true
    values alone is misleading: the diabetes coefficient is far from its true
    0.7 in every column, including the full-data one, because 2000 patients
    and 235 events cannot pin it down. That error is sampling noise, not
    missing-data handling.

(b) Filling every gap with the mean asserts that those values were measured
    rather than guessed, so nothing in the model widens to reflect the
    guessing. Rubin's rules make visible exactly what is being thrown away:
    the printed lambda says what share of the pooled uncertainty comes from
    the imputation itself, and single imputation sets that share to zero.

    That does not always show up as a smaller standard error, and the output
    above is a good reminder to check rather than assume. Two effects pull in
    opposite directions -- ignoring the imputation uncertainty makes the
    standard error too small, while flattening the variable's spread (SD 37.3
    to 35.0) makes it too large -- and at 12-16% missing they nearly cancel,
    leaving mean imputation within 2% of the multiple-imputation standard
    error. Raise the missingness to roughly half and the first effect wins
    outright: mean imputation's standard error for total_chol is about two
    thirds of the honest one.

    Watch the share itself rather than the ratio, because the share is the
    part that behaves predictably: it climbs from under a fifth to two thirds
    or more as the missingness grows. The fair statement is not that single
    imputation always looks more precise, but that its uncertainty is
    unaccounted for, and the size of what it ignores grows with the amount you
    imputed.

    Which method lands nearest the truth in any one dataset is luck; all of
    them sit within a standard error of each other. The missing variance
    component is systematic.

    That is the more serious problem: a biased estimate with an honest
    confidence interval announces its own uncertainty, whereas a spuriously
    precise one invites a confident claim about a coefficient the data cannot
    support. Multiple imputation exists to keep that uncertainty visible.

    One caveat: multiple imputation assumes MAR, which holds here by
    construction. If the highest cholesterol values were missing precisely
    because they were high (MNAR), no method here would recover them, and the
    honest response would be a sensitivity analysis.
")
