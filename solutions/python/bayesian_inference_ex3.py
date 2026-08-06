# Chapter 13, Exercise 3: Interpreting MCMC Diagnostics
# Conceptual exercise about convergence problems

# Given diagnostics for a Bayesian logistic regression treatment effect:
#   - R-hat: 1.08
#   - Bulk ESS: 150
#   - Tail ESS: 85
#   - Trace plot: two chains exploring different regions

# ---- Part (a): Is there evidence of convergence problems? ----
print("=== Part (a): Evidence of Convergence Problems ===\n")

print("YES, there is clear evidence of convergence problems.")
print("ALL THREE diagnostics are concerning:\n")

print("1. R-hat = 1.08 (threshold: < 1.01)")
print("   - R-hat compares between-chain to within-chain variance.")
print("   - A value of 1.08 far exceeds the 1.01 threshold.")
print("   - This means chains are NOT sampling from the same distribution.")
print("   - The trace plot confirms this: chains explore different regions.\n")

print("2. Bulk ESS = 150 (recommended: > 400 per chain, i.e., > 1600 total)")
print("   - Bulk ESS estimates the effective number of independent samples")
print("     for estimating posterior means and medians.")
print("   - 150 is far too low for reliable posterior summaries.")
print("   - High autocorrelation or poor mixing causes low ESS.\n")

print("3. Tail ESS = 85 (recommended: > 400)")
print("   - Tail ESS measures effective samples for tail quantiles")
print("     (2.5th and 97.5th percentiles for credible intervals).")
print("   - 85 is dangerously low -- credible intervals will be unreliable.")
print("   - Tail ESS is almost always lower than bulk ESS.\n")

print("4. Trace plot: chains exploring different regions")
print("   - Well-behaved chains should overlap ('hairy caterpillar').")
print("   - Chains in different regions indicate multimodality or")
print("     insufficient sampling.\n")

# ---- Part (b): Steps to fix ----
print("=== Part (b): Steps to Fix These Problems ===\n")

print("1. INCREASE WARMUP AND ITERATIONS:")
print("   Try warmup=2000-4000, draws=4000-8000.\n")

print("2. INCREASE target_accept (PyMC) / adapt_delta (Stan):")
print("   Set target_accept=0.95 or 0.99 to reduce divergences.\n")

print("3. CHECK MODEL SPECIFICATION:")
print("   - Are priors too vague or conflicting with the likelihood?")
print("   - Run prior predictive checks.")
print("   - Consider stronger priors if justified.\n")

print("4. REPARAMETERISE THE MODEL:")
print("   - Use non-centred parameterisation for hierarchical models.")
print("   - Standardise predictors.\n")

print("5. SIMPLIFY THE MODEL:")
print("   - Remove weak predictors or reduce random effects.\n")

print("6. RUN MORE CHAINS:")
print("   - 6-8 chains with different starting values to diagnose multimodality.\n")

# ---- Part (c): Trust the posterior summaries? ----
print("=== Part (c): Can We Trust the Posterior Summaries? ===\n")

print("NO, absolutely not. The posterior summaries should NOT be trusted.\n")

print("Reasons:")
print("1. With R-hat = 1.08, chains have not converged to the same")
print("   distribution. Any summary averages across different")
print("   distributions -- the result is meaningless.\n")

print("2. With bulk ESS = 150, the posterior mean is estimated from")
print("   only ~150 effective samples. Monte Carlo error is too large.\n")

print("3. With tail ESS = 85, the 95% credible interval is based on")
print("   ~85 effective tail samples. Bounds could shift substantially.\n")

print("4. Reporting results from a non-converged model is a serious")
print("   methodological error.\n")

print("Rule: If R-hat > 1.01 or ESS < 400, do NOT interpret results.")
print("Fix the model first.")
