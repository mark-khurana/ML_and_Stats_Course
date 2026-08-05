# =============================================================================
# Chapter 17 - Exercise 5: Do the adjusted methods really recover the truth?
# One dataset is not evidence of unbiasedness. Repeat the whole simulation.
# =============================================================================
#
# Libraries -------------------------------------------------------------------
library(tidyverse) # tibble(), ggplot2
# Everything below uses base R glm() only, so the simulation stays fast and
# has no package dependencies beyond tidyverse for the plot.

# --- The data-generating process --------------------------------------------
# ONE binary confounder, `frail`, which raises BOTH the chance of treatment and
# the risk of death. The treatment has a known protective effect.
TRUTH_LOG_ODDS <- -0.8

simulate_cohort <- function(n = 2000) {
  frail <- rbinom(n, 1, 0.4)
  # Frail patients are much more likely to be treated (confounding by indication)
  treat <- rbinom(n, 1, plogis(-0.4 + 1.6 * frail))
  death <- rbinom(n, 1, plogis(-0.7 + 1.0 * frail + TRUTH_LOG_ODDS * treat))
  data.frame(frail, treat, death)
}

# The TRUE marginal risk difference, on the scale everything below reports.
# We get it by brute force: a huge cohort, both counterfactuals computed exactly.
set.seed(99)
big <- data.frame(frail = rbinom(2e6, 1, 0.4))
TRUE_RD <- mean(plogis(-0.7 + 1.0 * big$frail + TRUTH_LOG_ODDS)) -
  mean(plogis(-0.7 + 1.0 * big$frail))
cat(sprintf("TRUE marginal risk difference: %+.4f\n", TRUE_RD))
cat(sprintf("(built from a conditional log-odds of %+.2f)\n\n", TRUTH_LOG_ODDS))

# --- The three estimators, each returning a marginal risk difference ---------
contrast_from <- function(model, d) {
  mean(predict(model, transform(d, treat = 1), type = "response")) -
    mean(predict(model, transform(d, treat = 0), type = "response"))
}

est_naive <- function(d) {
  contrast_from(glm(death ~ treat, data = d, family = binomial), d)
}

est_ipw <- function(d) {
  ps <- predict(glm(treat ~ frail, data = d, family = binomial), type = "response")
  p_marg <- mean(d$treat)
  sw <- ifelse(d$treat == 1, p_marg / ps, (1 - p_marg) / (1 - ps))
  # quasibinomial silences a harmless non-integer-successes warning; the point
  # estimate is identical to binomial
  msm <- glm(death ~ treat, data = d, family = quasibinomial, weights = sw)
  contrast_from(msm, d)
}

est_gcomp <- function(d) {
  contrast_from(glm(death ~ treat * frail, data = d, family = binomial), d)
}

# =============================================================================
# (a) One dataset, three estimates
# =============================================================================
set.seed(42)
dat <- simulate_cohort()

cat("--- (a) A single dataset (n = 2000) ---\n")
cat(sprintf("Treatment rate: %.1f%% of non-frail, %.1f%% of frail patients\n",
            100 * mean(dat$treat[dat$frail == 0]),
            100 * mean(dat$treat[dat$frail == 1])))
one <- c(naive = est_naive(dat), ipw = est_ipw(dat), gcomp = est_gcomp(dat))
for (nm in names(one)) {
  cat(sprintf("  %-6s %+.4f   (error vs truth: %+.4f)\n",
              nm, one[[nm]], one[[nm]] - TRUE_RD))
}
cat("\nOn this one sample the adjusted estimates look good -- but a single\n")
cat("sample cannot distinguish an unbiased estimator from a lucky one.\n")

# =============================================================================
# (b) Repeat the whole simulation 500 times
# =============================================================================
set.seed(2024)
R <- 500
sims <- map_dfr(seq_len(R), function(i) {
  d <- simulate_cohort()
  tibble(
    rep = i,
    Naive = est_naive(d),
    IPW = est_ipw(d),
    `G-computation` = est_gcomp(d)
  )
})

long <- sims |>
  pivot_longer(-rep, names_to = "method", values_to = "estimate") |>
  mutate(method = factor(method, levels = c("Naive", "IPW", "G-computation")))

p <- ggplot(long, aes(x = estimate, fill = method)) +
  geom_density(alpha = 0.55, colour = NA) +
  geom_vline(xintercept = TRUE_RD, linetype = "dashed",
             colour = "#b02a2a", linewidth = 0.8) +
  annotate("text", x = TRUE_RD, y = Inf, vjust = 1.6, hjust = -0.05,
           label = sprintf("truth = %+.3f", TRUE_RD),
           colour = "#b02a2a", fontface = "bold", size = 3.4) +
  scale_fill_manual(values = c("#D55E00", "#0072B2", "#009E73")) +
  labs(
    x = "Estimated marginal risk difference", y = "Density", fill = NULL,
    title = sprintf("Sampling distribution over %d simulated cohorts", R)
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")
print(p)

# =============================================================================
# (c) Bias
# =============================================================================
summary_tab <- long |>
  group_by(method) |>
  summarise(
    mean_estimate = mean(estimate),
    bias = mean(estimate) - TRUE_RD,
    sd = sd(estimate),
    rmse = sqrt(mean((estimate - TRUE_RD)^2)),
    .groups = "drop"
  )

cat(sprintf("\n--- (c) Over %d replications ---\n", R))
print(as.data.frame(summary_tab), digits = 4)

cat("\nRead the `bias` column: it is the average error, and averaging over 500\n")
cat("cohorts is what lets us see it. The naive estimator's bias is large and\n")
cat("in a consistent direction -- it is not noise, it is a systematic failure.\n")
cat("IPW and g-computation have bias close to zero: they are centred on the\n")
cat("truth, which is what 'unbiased' means and what one dataset could never\n")
cat("have shown us.\n")

# =============================================================================
# (d) Spread, and why the tightest estimator is not automatically the best
# =============================================================================
cat("\n--- (d) Spread ---\n")
tightest <- summary_tab$method[which.min(summary_tab$sd)]
cat(sprintf("Smallest standard deviation: %s\n", tightest))
cat("\nThe naive estimator is typically the TIGHTEST of the three, and it is\n")
cat("also the only one that is wrong. That is the whole point: precision\n")
cat("measures how consistently an estimator returns the same answer, not\n")
cat("whether that answer is right. A biased estimator can be beautifully\n")
cat("precise -- reliably wrong.\n")
cat("\nThe quantity that combines both is the root mean squared error (RMSE)\n")
cat("in the table above, which penalises bias and variance together. On RMSE\n")
cat("the adjusted methods win comfortably despite being noisier.\n")
cat("\nOne striking detail: IPW and g-computation give IDENTICAL numbers here,\n")
cat("to every decimal place, in every replication. That is not a coincidence\n")
cat("and not a bug. With a single binary confounder, both models are\n")
cat("SATURATED -- `treat * frail` has one parameter for each of the four\n")
cat("treatment-by-frailty cells, and the propensity model likewise reproduces\n")
cat("the observed treatment rate in each cell exactly. Both estimators then\n")
cat("reduce to the same non-parametric calculation: take the observed death\n")
cat("rate in each of the four cells and re-average it over the frailty\n")
cat("distribution. There is nothing left for them to disagree about.\n")
cat("\nThey come apart as soon as a model has to make an assumption -- with\n")
cat("continuous confounders, non-linear effects, or omitted interactions. Then\n")
cat("IPW is at risk from a wrong TREATMENT model and extreme weights, and\n")
cat("g-computation from a wrong OUTCOME model. Neither is universally better;\n")
cat("they fail in different circumstances, which is why agreement between them\n")
cat("is informative and why doubly robust estimators combine the two.\n")
cat("\nTry it: change `frail` to a continuous variable, or fit the outcome model\n")
cat("without the interaction, and the two columns will separate.\n")
