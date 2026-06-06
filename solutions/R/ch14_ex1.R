# Chapter 14, Exercise 1: Bayesian Logistic Regression for ICU Mortality
# 500 ICU admissions with age, APACHE II, ventilation status, 28-day mortality

library(brms)
library(tidyverse)
library(bayesplot)

# ---- (a) Simulate dataset ----
set.seed(101)
n <- 500

icu_data <- tibble(
  age = round(rnorm(n, 62, 15)),
  apache = round(rnorm(n, 18, 7)),
  ventilated = rbinom(n, 1, 0.35)
)

# True model: mortality increases with age, APACHE, and ventilation
lp <- -4.5 + 0.02 * icu_data$age +
  0.12 * icu_data$apache +
  0.8 * icu_data$ventilated

icu_data$mortality <- rbinom(n, 1, plogis(lp))

cat("Dataset summary:\n")
cat("  N:", n, "\n")
cat("  Mortality rate:", mean(icu_data$mortality), "\n")
cat("  Mean age:", round(mean(icu_data$age), 1), "\n")
cat("  Mean APACHE:", round(mean(icu_data$apache), 1), "\n")
cat("  % Ventilated:", round(mean(icu_data$ventilated) * 100, 1), "%\n")

# ---- (b) Fit Bayesian logistic regression ----
fit_icu <- brm(
  mortality ~ age + apache + ventilated,
  data = icu_data,
  family = bernoulli(link = "logit"),
  prior = c(
    prior(normal(0, 2.5), class = "b"),       # weakly informative on log-odds
    prior(normal(0, 5), class = "Intercept")
  ),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 42,
  silent = 2,
  refresh = 0
)

cat("\nModel summary:\n")
summary(fit_icu)

# ---- (c) Prior predictive check ----
cat("\n=== Part (c): Prior Predictive Check ===\n")

# Fit with priors only (no data influence)
fit_prior <- brm(
  mortality ~ age + apache + ventilated,
  data = icu_data,
  family = bernoulli(link = "logit"),
  prior = c(
    prior(normal(0, 2.5), class = "b"),
    prior(normal(0, 5), class = "Intercept")
  ),
  sample_prior = "only",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 42,
  silent = 2,
  refresh = 0
)

# Simulate from prior predictive
pp_prior <- posterior_predict(fit_prior)
prior_mort_rates <- rowMeans(pp_prior)

cat("Prior predictive mortality rates:\n")
cat("  Mean:", round(mean(prior_mort_rates), 3), "\n")
cat("  SD:", round(sd(prior_mort_rates), 3), "\n")
cat("  Range:", round(min(prior_mort_rates), 3), "to",
    round(max(prior_mort_rates), 3), "\n")
cat("The priors allow mortality rates from near 0 to near 1,\n")
cat("which covers all plausible ICU mortality rates. The priors\n")
cat("are appropriately weakly informative.\n")

# ---- (d) Posterior odds ratios with 95% credible intervals ----
cat("\n=== Part (d): Posterior Odds Ratios ===\n")

post <- as_draws_df(fit_icu)

or_age <- exp(post$b_age)
or_apache <- exp(post$b_apache)
or_vent <- exp(post$b_ventilated)

cat(sprintf("OR Age:        %.3f [%.3f, %.3f]\n",
            mean(or_age), quantile(or_age, 0.025), quantile(or_age, 0.975)))
cat(sprintf("OR APACHE:     %.3f [%.3f, %.3f]\n",
            mean(or_apache), quantile(or_apache, 0.025), quantile(or_apache, 0.975)))
cat(sprintf("OR Ventilated: %.3f [%.3f, %.3f]\n",
            mean(or_vent), quantile(or_vent, 0.025), quantile(or_vent, 0.975)))

# Plot posterior OR distributions
mcmc_areas(fit_icu, pars = c("b_age", "b_apache", "b_ventilated"),
           prob = 0.95, prob_outer = 0.99,
           transformations = exp) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
  labs(title = "Posterior Odds Ratios (95% CrI)",
       x = "Odds Ratio") +
  theme_minimal(base_size = 13)

# ---- (e) P(OR_APACHE > 1.10 | data) ----
cat("\n=== Part (e): P(OR_APACHE > 1.10 | data) ===\n")

prob_or_gt_110 <- mean(or_apache > 1.10)
cat(sprintf("P(OR_APACHE > 1.10 | data) = %.3f\n", prob_or_gt_110))
cat("\nInterpretation: There is a", round(prob_or_gt_110 * 100, 1),
    "% posterior probability\n")
cat("that each unit increase in APACHE score increases the odds of\n")
cat("28-day mortality by more than 10%. This is a direct probability\n")
cat("statement about the parameter -- something only Bayesian inference\n")
cat("can provide.\n")
