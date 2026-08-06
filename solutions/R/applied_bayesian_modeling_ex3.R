# Chapter 14, Exercise 3: Prior Sensitivity for Rare Events
# New surgical technique: 0 adverse events in 40 patients

# ---- (a) Compute posterior distributions under three priors ----
cat("=== Part (a): Posteriors Under Three Priors ===\n\n")

y <- 0   # adverse events
n <- 40  # total patients

priors <- list(
  list(name = "Beta(1,1) - Uniform", a = 1, b = 1),
  list(name = "Beta(0.5,0.5) - Jeffreys", a = 0.5, b = 0.5),
  list(name = "Beta(1,9) - Informative (~10%)", a = 1, b = 9)
)

theta <- seq(0, 0.3, length.out = 500)

par(mfrow = c(1, 1))
plot(NULL, xlim = c(0, 0.2), ylim = c(0, 50),
     xlab = "Adverse Event Rate", ylab = "Density",
     main = "Posterior Distributions: 0/40 Adverse Events")

colors <- c("steelblue", "firebrick", "forestgreen")

for (i in seq_along(priors)) {
  p <- priors[[i]]
  a_post <- p$a + y
  b_post <- p$b + n - y

  post_mean <- a_post / (a_post + b_post)
  ci <- qbeta(c(0.025, 0.975), a_post, b_post)
  upper_95 <- qbeta(0.95, a_post, b_post)

  cat(sprintf("Prior: %s\n", p$name))
  cat(sprintf("  Posterior: Beta(%.1f, %.1f)\n", a_post, b_post))
  cat(sprintf("  Posterior mean: %.4f (%.2f%%)\n", post_mean, post_mean * 100))
  cat(sprintf("  95%% credible interval: [%.4f, %.4f]\n", ci[1], ci[2]))
  cat(sprintf("  95%% upper credible bound: %.4f (%.2f%%)\n\n",
              upper_95, upper_95 * 100))

  lines(theta, dbeta(theta, a_post, b_post), col = colors[i], lwd = 2)
}

legend("topright", sapply(priors, function(p) p$name),
       col = colors, lwd = 2, cex = 0.8)

# ---- (b) Summary table ----
cat("\n=== Part (b): Summary Table ===\n\n")
cat("Prior              | Post Mean | 95% Upper Bound\n")
cat("-------------------|-----------|----------------\n")

for (p in priors) {
  a_post <- p$a + y
  b_post <- p$b + n - y
  post_mean <- a_post / (a_post + b_post)
  upper_95 <- qbeta(0.95, a_post, b_post)
  cat(sprintf("%-19s| %7.4f   | %7.4f (%.1f%%)\n",
              p$name, post_mean, upper_95, upper_95 * 100))
}

# ---- (c) Why Bayesian estimates are more useful ----
cat("\n=== Part (c): Why Bayesian Estimates Are More Useful ===\n\n")

cat("The frequentist point estimate is 0/40 = 0 (0%).\n\n")

cat("This is PROBLEMATIC for regulatory decision-making because:\n\n")

cat("1. ZERO IS NOT CREDIBLE: Just because no adverse events were\n")
cat("   observed in 40 patients does not mean the true rate is zero.\n")
cat("   The 'rule of 3' (frequentist) gives an upper bound of 3/40 = 7.5%,\n")
cat("   but this is ad hoc and does not provide a full distribution.\n\n")

cat("2. BAYESIAN ESTIMATES ARE HONEST: Each prior gives a non-zero\n")
cat("   estimate of the adverse event rate, which is more realistic.\n")
cat("   Even with the Jeffreys prior (minimal information), the\n")
cat("   posterior mean is about 1.2%, acknowledging that rare events\n")
cat("   can occur even if none were observed.\n\n")

cat("3. UPPER CREDIBLE BOUNDS: Regulators need worst-case estimates.\n")
cat("   The 95% upper credible bound provides a direct probability\n")
cat("   statement: 'There is a 95% probability that the true adverse\n")
cat("   event rate is below X%'. This is exactly what is needed for\n")
cat("   risk-benefit assessments.\n\n")

cat("4. PRIOR INFORMATION IS VALUABLE: If similar surgical procedures\n")
cat("   have known complication rates (~10%), the informative prior\n")
cat("   Beta(1,9) incorporates this, giving a more realistic estimate.\n")
cat("   The frequentist approach ignores all prior knowledge.\n\n")

cat("5. DECISION SUPPORT: The full posterior distribution allows\n")
cat("   calculation of quantities like P(rate < 5% | data), which\n")
cat("   directly supports regulatory decisions.\n")

# Compute P(rate < 5% | data) for each prior
cat("\n  P(rate < 5% | data):\n")
for (p in priors) {
  a_post <- p$a + y
  b_post <- p$b + n - y
  prob_lt_5 <- pbeta(0.05, a_post, b_post)
  cat(sprintf("    %s: %.3f\n", p$name, prob_lt_5))
}
