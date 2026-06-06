# Chapter 13, Exercise 2: Prior Sensitivity Analysis
# Phase II antibiotic trial: 21/30 patients achieve clinical cure

library(tidyverse)

y <- 21  # successes
n <- 30  # total patients

# Define three priors
priors <- list(
  list(name = "Beta(1,1) - Uninformative", a = 1, b = 1),
  list(name = "Beta(5,5) - Weakly informative", a = 5, b = 5),
  list(name = "Beta(20,10) - Informative", a = 20, b = 10)
)

theta <- seq(0, 1, length.out = 500)

# ---- Part (a): Posterior mean and 95% credible interval ----
cat("=== Part (a): Posterior Summaries ===\n\n")

plot_df <- tibble()

for (p in priors) {
  a_post <- p$a + y
  b_post <- p$b + n - y
  post_mean <- a_post / (a_post + b_post)
  ci <- qbeta(c(0.025, 0.975), a_post, b_post)

  cat(sprintf("Prior: %s\n", p$name))
  cat(sprintf("  Prior mean: %.3f\n", p$a / (p$a + p$b)))
  cat(sprintf("  Posterior: Beta(%d, %d)\n", a_post, b_post))
  cat(sprintf("  Posterior mean: %.3f\n", post_mean))
  cat(sprintf("  95%% credible interval: [%.3f, %.3f]\n\n", ci[1], ci[2]))

  plot_df <- bind_rows(plot_df,
    tibble(
      theta = theta,
      Density = dbeta(theta, a_post, b_post),
      Prior = p$name
    )
  )
}

# ---- Part (b): Plot all three posteriors ----
cat("=== Part (b): Posterior Plot ===\n")

ggplot(plot_df, aes(x = theta, y = Density, colour = Prior)) +
  geom_line(linewidth = 1.2) +
  geom_vline(xintercept = y / n, linetype = "dashed", colour = "grey40") +
  annotate("text", x = y / n + 0.02, y = max(plot_df$Density) * 0.9,
           label = paste0("MLE = ", round(y / n, 3)),
           hjust = 0, size = 3.5) +
  labs(x = expression(theta ~ "(cure rate)"),
       y = "Density",
       title = "Posterior Distributions Under Different Priors",
       subtitle = "21/30 patients cured in Phase II trial") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

# ---- Part (c): Which prior has most influence? ----
cat("=== Part (c): Which Prior Has Most Influence? ===\n\n")
cat("The Beta(5,5) prior has the MOST influence on the posterior.\n\n")
cat("Why? The effective sample size of a Beta(a,b) prior is a + b.\n")
cat("  - Beta(1,1):   effective n = 2  (trivial influence)\n")
cat("  - Beta(5,5):   effective n = 10 (modest influence)\n")
cat("  - Beta(20,10): effective n = 30 (substantial influence)\n\n")
cat("However, while Beta(20,10) has the largest effective sample size,\n")
cat("its prior mean (0.667) is close to the data (21/30 = 0.700),\n")
cat("so the prior and data 'agree', and the posterior is not pulled\n")
cat("far from the MLE.\n\n")
cat("Beta(5,5) has a prior mean of 0.500, which is DIFFERENT from\n")
cat("the data. It pulls the posterior mean toward 0.5, producing the\n")
cat("most visible shift from the MLE. With n=30 observed data, even\n")
cat("an effective sample size of 10 produces noticeable pull when\n")
cat("the prior and data disagree.\n\n")
cat("Key insight: prior influence depends on BOTH the effective sample\n")
cat("size AND how much the prior disagrees with the data.\n")
