# Chapter 13, Exercise 3: Interpreting MCMC Diagnostics
# Conceptual exercise about convergence problems

# Given diagnostics for a Bayesian logistic regression treatment effect:
#   - R-hat: 1.08
#   - Bulk ESS: 150
#   - Tail ESS: 85
#   - Trace plot: two chains exploring different regions

# ---- Part (a): Is there evidence of convergence problems? ----
cat("=== Part (a): Evidence of Convergence Problems ===\n\n")

cat("YES, there is clear evidence of convergence problems.\n")
cat("ALL THREE diagnostics are concerning:\n\n")

cat("1. R-hat = 1.08 (threshold: < 1.01)\n")
cat("   - R-hat compares between-chain to within-chain variance.\n")
cat("   - A value of 1.08 far exceeds the 1.01 threshold.\n")
cat("   - This means chains are NOT sampling from the same distribution.\n")
cat("   - The trace plot confirms this: chains explore different regions.\n\n")

cat("2. Bulk ESS = 150 (recommended: > 400 per chain, i.e., > 1600 total)\n")
cat("   - Bulk ESS estimates the effective number of independent samples\n")
cat("     for estimating posterior means and medians.\n")
cat("   - 150 is far too low for reliable posterior summaries.\n")
cat("   - High autocorrelation or poor mixing causes low ESS.\n\n")

cat("3. Tail ESS = 85 (recommended: > 400)\n")
cat("   - Tail ESS measures the effective samples for tail quantiles\n")
cat("     (e.g., 2.5th and 97.5th percentiles for credible intervals).\n")
cat("   - 85 is dangerously low -- credible intervals will be unreliable.\n")
cat("   - Tail ESS is almost always lower than bulk ESS, so if bulk is\n")
cat("     already too low, tail ESS will be even worse.\n\n")

cat("4. Trace plot: chains exploring different regions\n")
cat("   - This is the most visually obvious sign of non-convergence.\n")
cat("   - Well-behaved chains should overlap ('hairy caterpillar').\n")
cat("   - Chains stuck in different regions indicate multimodality or\n")
cat("     insufficient sampling to traverse the parameter space.\n")

# ---- Part (b): Steps to fix these problems ----
cat("\n=== Part (b): Steps to Fix These Problems ===\n\n")

cat("1. INCREASE WARMUP AND ITERATIONS:\n")
cat("   - Current chains may not have had enough warmup to find\n")
cat("     the typical set. Try warmup = 2000-4000, iter = 4000-8000.\n\n")

cat("2. INCREASE adapt_delta (Stan/brms):\n")
cat("   - Set control = list(adapt_delta = 0.95 or 0.99).\n")
cat("   - This makes the sampler take smaller steps, reducing\n")
cat("     divergent transitions at the cost of slower sampling.\n\n")

cat("3. CHECK THE MODEL SPECIFICATION:\n")
cat("   - Priors may be too vague or conflicting with the likelihood.\n")
cat("   - Use prior predictive checks to ensure priors are reasonable.\n")
cat("   - Consider stronger (more informative) priors if justified.\n\n")

cat("4. REPARAMETERISE THE MODEL:\n")
cat("   - Some parameterisations create difficult posterior geometries.\n")
cat("   - For hierarchical models, use non-centred parameterisation.\n")
cat("   - Standardise predictors to improve sampler efficiency.\n\n")

cat("5. SIMPLIFY THE MODEL:\n")
cat("   - If the model is too complex for the data, consider\n")
cat("     removing weak predictors or reducing random effects.\n\n")

cat("6. RUN MORE CHAINS:\n")
cat("   - Running 6-8 chains (instead of 4) with different\n")
cat("     starting values helps diagnose multimodality.\n")

# ---- Part (c): Trust the posterior summaries? ----
cat("\n=== Part (c): Can We Trust the Posterior Summaries? ===\n\n")

cat("NO, absolutely not. The posterior summaries should NOT be trusted.\n\n")

cat("Reasons:\n")
cat("1. With R-hat = 1.08, the chains have not converged to the same\n")
cat("   distribution. Any summary (mean, median, CI) is averaging\n")
cat("   across different distributions -- the result is meaningless.\n\n")

cat("2. With bulk ESS = 150, the posterior mean is estimated from\n")
cat("   only ~150 effective independent samples. The Monte Carlo\n")
cat("   error is too large for the estimates to be precise.\n\n")

cat("3. With tail ESS = 85, the 95% credible interval is based on\n")
cat("   only ~85 effective tail samples. The interval bounds could\n")
cat("   shift substantially with additional sampling.\n\n")

cat("4. Reporting results from a non-converged model is a serious\n")
cat("   methodological error. The correct action is to fix the\n")
cat("   convergence issues BEFORE interpreting any results.\n\n")

cat("Rule: If R-hat > 1.01 or ESS < 400, do NOT interpret results.\n")
cat("Fix the model first.\n")
