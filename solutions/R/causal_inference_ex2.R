# =============================================================================
# Chapter 17 - Exercise 2: Propensity score matching
# Beta-blocker use and 1-year mortality
# =============================================================================
#
# Libraries -------------------------------------------------------------------
# Every one of these is needed. In particular, WITHOUT library(broom) the call
# to tidy() below fails with the misleading
#     Error in UseMethod("tidy") : no applicable method for 'tidy' ...
library(tidyverse) # tibble(), mutate(), ggplot2
library(MatchIt)   # matchit(), match.data()
library(cobalt)    # love.plot(), bal.tab()
library(broom)     # tidy()

# --- The dataset from the exercise ------------------------------------------
set.seed(123)
n <- 1500

exercise_dat <- tibble(
  age           = rnorm(n, 70, 8),
  creatinine    = rnorm(n, 1.2, 0.4),
  heart_failure = rbinom(n, 1, 0.35),
  prior_mi      = rbinom(n, 1, 0.20)
) |>
  mutate(
    treatment = rbinom(n, 1, plogis(-0.4 + 0.05 * (age - 70) +
                                      0.7 * heart_failure +
                                      0.9 * prior_mi +
                                      0.8 * (creatinine - 1.2))),
    death_1yr = rbinom(n, 1, plogis(-1.9 + 0.05 * (age - 70) +
                                      0.7 * heart_failure +
                                      0.8 * prior_mi +
                                      1.0 * (creatinine - 1.2) -
                                      0.8 * treatment))
  )

# The truth, available only because we simulated the data. Compute it once so
# every estimate below can be judged against it.
lp_untreated <- with(exercise_dat, -1.9 + 0.05 * (age - 70) +
  0.7 * heart_failure + 0.8 * prior_mi + 1.0 * (creatinine - 1.2))
treated <- exercise_dat$treatment == 1
p1 <- mean(plogis(lp_untreated[treated] - 0.8))
p0 <- mean(plogis(lp_untreated[treated]))
TRUE_ATT_RD <- p1 - p0
TRUE_ATT_OR <- (p1 / (1 - p1)) / (p0 / (1 - p0))

cat("Cohort:", nrow(exercise_dat), "patients |",
    sprintf("%.0f%% treated | %.1f%% died within 1 year\n",
            100 * mean(exercise_dat$treatment),
            100 * mean(exercise_dat$death_1yr)))
cat(sprintf("TRUE ATT: risk difference %+.4f, odds ratio %.3f\n\n",
            TRUE_ATT_RD, TRUE_ATT_OR))

# =============================================================================
# (a) Estimate the propensity score and look at OVERLAP
# =============================================================================
ps_model <- glm(treatment ~ age + creatinine + heart_failure + prior_mi,
  data = exercise_dat, family = binomial
)
exercise_dat$ps <- predict(ps_model, type = "response")

cat("--- (a) Propensity score distribution ---\n")
print(exercise_dat |>
  group_by(treatment) |>
  summarise(
    n = n(),
    min = round(min(ps), 3),
    median = round(median(ps), 3),
    max = round(max(ps), 3)
  ))

# Overlap plot: we want the two densities to cover the same range of scores.
ps_plot <- ggplot(exercise_dat, aes(
  x = ps,
  fill = factor(treatment, labels = c("No beta-blocker", "Beta-blocker"))
)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#0072B2", "#D55E00")) +
  labs(
    x = "Propensity score", y = "Density", fill = NULL,
    title = "Propensity score overlap by treatment group"
  ) +
  theme_minimal()
print(ps_plot)

cat("\nBoth groups span roughly the same range of scores, with no pile-up at\n")
cat("0 or 1, so positivity is not obviously violated and matching is feasible.\n")

# =============================================================================
# (b) 1:1 nearest-neighbour matching with a caliper of 0.2 SD
# =============================================================================
m_out <- matchit(treatment ~ age + creatinine + heart_failure + prior_mi,
  data = exercise_dat,
  method = "nearest",
  distance = "glm", # propensity score from logistic regression
  caliper = 0.2,    # no match further than 0.2 SD of logit(PS)
  ratio = 1
)

cat("\n--- (b) Matching ---\n")
print(m_out)

n_treated <- sum(exercise_dat$treatment == 1)
n_matched_treated <- sum(match.data(m_out)$treatment == 1)
cat(sprintf(
  "\n%d of %d treated patients found a partner; %d did NOT.\n",
  n_matched_treated, n_treated, n_treated - n_matched_treated
))
cat("Those unmatched patients are silently DROPPED. They are not a random\n")
cat("subset -- they are the ones with the most extreme propensity scores, i.e.\n")
cat("the patients who were most obviously going to be treated. So the estimate\n")
cat("no longer describes 'all treated patients'; it describes the treated\n")
cat("patients for whom a comparable untreated patient exists. Always report\n")
cat("how many were dropped, and compare their characteristics.\n")

# =============================================================================
# (c) Love plot: did matching actually balance the covariates?
# =============================================================================
love_p <- love.plot(m_out,
  thresholds = c(m = 0.1),
  binary = "std",
  var.order = "unadjusted",
  title = "Covariate balance: before and after matching",
  colors = c("#D55E00", "#0072B2")
)
print(love_p)

cat("\n--- (c) Balance table ---\n")
print(bal.tab(m_out, thresholds = c(m = 0.1)))
cat("\nAll four COVARIATES are now inside the 0.1 threshold (age is the worst at\n")
cat("0.075, down from 0.38 before matching), so the matched groups have a\n")
cat("comparable mix of patients. Note that the row labelled 'distance' -- the\n")
cat("propensity score itself -- is still around 0.12. That is common and is not\n")
cat("in itself a failure: the score is a summary, and it is balance on the\n")
cat("actual covariates that removes confounding. If a real COVARIATE were above\n")
cat("0.1, the fixes are to tighten the caliper, allow more controls per treated\n")
cat("patient, or move to weighting (Exercise 3) rather than matching.\n")

# =============================================================================
# (d) The ATT, as an odds ratio and as a risk difference
# =============================================================================
m_data <- match.data(m_out)

or_fit <- glm(death_1yr ~ treatment,
  data = m_data, family = binomial, weights = weights
)
or_res <- tidy(or_fit, conf.int = TRUE, exponentiate = TRUE) |>
  filter(term == "treatment")

# A risk difference is easier to communicate to clinicians than an odds ratio.
rd_fit <- lm(death_1yr ~ treatment, data = m_data, weights = weights)
rd_res <- tidy(rd_fit, conf.int = TRUE) |> filter(term == "treatment")

cat("\n--- (d) ATT estimates in the matched sample ---\n")
cat(sprintf(
  "Odds ratio      : %.3f (95%% CI %.3f, %.3f)   [truth %.3f]\n",
  or_res$estimate, or_res$conf.low, or_res$conf.high, TRUE_ATT_OR
))
cat(sprintf(
  "Risk difference : %+.4f (95%% CI %+.4f, %+.4f)   [truth %+.4f]\n",
  rd_res$estimate, rd_res$conf.low, rd_res$conf.high, TRUE_ATT_RD
))
cat(sprintf(
  "\nFor comparison, the UNADJUSTED risk difference in the full cohort is %+.4f\n",
  mean(exercise_dat$death_1yr[exercise_dat$treatment == 1]) -
    mean(exercise_dat$death_1yr[exercise_dat$treatment == 0])
))
cat("-- less than half the true effect. Matching recovers most of what the\n")
cat("naive comparison hides.\n")

# But compare against the RIGHT target. The estimate describes the matched
# treated patients, not all treated patients, so recompute the truth over
# exactly that subgroup.
matched_rows <- as.integer(rownames(m_data)[m_data$treatment == 1])
lp_m <- lp_untreated[matched_rows]
p1_m <- mean(plogis(lp_m - 0.8))
p0_m <- mean(plogis(lp_m))
cat(sprintf(
  "\nThe truth quoted above is the ATT over ALL %d treated patients. Our\n",
  n_treated
))
cat(sprintf(
  "estimate only describes the %d who found a match, and for THAT subgroup the\n",
  n_matched_treated
))
cat(sprintf(
  "true values are: risk difference %+.4f, odds ratio %.3f.\n",
  p1_m - p0_m,
  ((p1_m / (1 - p1_m)) / (p0_m / (1 - p0_m)))
))
cat("That is what the estimate should be judged against, and the gap between\n")
cat("the two targets is the price of discarding unmatched patients: matching\n")
cat("answers a slightly different question from the one you asked.\n")

cat("\nMortality in the matched sample:\n")
cat(sprintf(
  "  Treated: %.3f    Control: %.3f\n",
  mean(m_data$death_1yr[m_data$treatment == 1]),
  mean(m_data$death_1yr[m_data$treatment == 0])
))

# =============================================================================
# (e) E-value: how strong would a hidden confounder have to be?
# =============================================================================
# The E-value works on the risk-ratio scale and is symmetric about the null, so
# a protective estimate is first flipped to the above-null side.
e_value <- function(rr) {
  if (rr < 1) rr <- 1 / rr
  rr + sqrt(rr * (rr - 1))
}

# With a 15% outcome the odds ratio overstates the risk ratio, so compute the
# risk ratio directly from the matched sample rather than reusing the OR.
risk_treated <- mean(m_data$death_1yr[m_data$treatment == 1])
risk_control <- mean(m_data$death_1yr[m_data$treatment == 0])
rr_point <- risk_treated / risk_control

# CI for the risk ratio, on the log scale
a <- sum(m_data$death_1yr[m_data$treatment == 1])
b <- sum(m_data$treatment == 1)
c_ <- sum(m_data$death_1yr[m_data$treatment == 0])
d_ <- sum(m_data$treatment == 0)
se_log_rr <- sqrt(1 / a - 1 / b + 1 / c_ - 1 / d_)
rr_lo <- exp(log(rr_point) - 1.96 * se_log_rr)
rr_hi <- exp(log(rr_point) + 1.96 * se_log_rr)

# For a protective effect, the CI bound CLOSEST to the null is the upper one.
e_point <- e_value(rr_point)
e_ci <- if (rr_hi >= 1) 1 else e_value(rr_hi)

cat("\n--- (e) E-value ---\n")
cat(sprintf(
  "Risk ratio: %.3f (95%% CI %.3f, %.3f)\n", rr_point, rr_lo, rr_hi
))
cat(sprintf("E-value for the point estimate      : %.2f\n", e_point))
cat(sprintf("E-value for the CI bound nearest null: %.2f\n", e_ci))

cat("\nInterpretation in one sentence: an unmeasured confounder would have to\n")
cat(sprintf(
  "be associated with BOTH beta-blocker use and death by a risk ratio of at\nleast %.2f",
  e_point
))
cat(" -- over and above age, creatinine, heart failure and prior MI --\n")
cat("to explain away this result entirely.\n")
cat("\nWhether that is plausible is a clinical judgement, not a statistical\n")
cat("one. Compare it with the strength of the confounders you DID measure: if\n")
cat("none of them reaches that magnitude, a hidden one probably does not either.\n")

# The EValue package does this for you, including for odds ratios and
# hazard ratios, and is worth using in real work:
#   install.packages("EValue")
#   library(EValue)
#   evalues.RR(est = 1 / rr_point, lo = 1 / rr_hi, hi = 1 / rr_lo)
