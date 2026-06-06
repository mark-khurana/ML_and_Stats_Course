# Chapter 14, Exercise 3: Prior Sensitivity for Rare Events
# New surgical technique: 0 adverse events in 40 patients

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

y = 0   # adverse events
n = 40  # total patients

priors = [
    ("Beta(1,1) - Uniform", 1, 1),
    ("Beta(0.5,0.5) - Jeffreys", 0.5, 0.5),
    ("Beta(1,9) - Informative (~10%)", 1, 9),
]

theta = np.linspace(0, 0.25, 500)
colors = ["steelblue", "firebrick", "forestgreen"]

# ---- (a) Compute posterior distributions ----
print("=== Part (a): Posteriors Under Three Priors ===\n")

fig, ax = plt.subplots(figsize=(8, 5))

for (name, a0, b0), color in zip(priors, colors):
    a_post = a0 + y
    b_post = b0 + n - y
    post_mean = a_post / (a_post + b_post)
    ci = beta.ppf([0.025, 0.975], a_post, b_post)
    upper_95 = beta.ppf(0.95, a_post, b_post)

    print(f"Prior: {name}")
    print(f"  Posterior: Beta({a_post}, {b_post})")
    print(f"  Posterior mean: {post_mean:.4f} ({post_mean*100:.2f}%)")
    print(f"  95% credible interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  95% upper credible bound: {upper_95:.4f} ({upper_95*100:.2f}%)")
    print()

    ax.plot(theta, beta.pdf(theta, a_post, b_post),
            color=color, lw=2, label=name)

ax.set_xlabel("Adverse Event Rate")
ax.set_ylabel("Density")
ax.set_title("Posterior Distributions: 0/40 Adverse Events")
ax.legend(fontsize=9)
ax.set_xlim(0, 0.2)
plt.tight_layout()
plt.savefig("ch14_ex3_rare_events.png", dpi=150)
plt.show()

# ---- (b) Summary table ----
print("=== Part (b): Summary Table ===\n")
print(f"{'Prior':<30s} | {'Post Mean':>9s} | {'95% Upper Bound':>15s}")
print("-" * 60)

for name, a0, b0 in priors:
    a_post = a0 + y
    b_post = b0 + n - y
    post_mean = a_post / (a_post + b_post)
    upper_95 = beta.ppf(0.95, a_post, b_post)
    print(f"{name:<30s} | {post_mean:>9.4f} | {upper_95:>9.4f} ({upper_95*100:.1f}%)")

# ---- (c) Why Bayesian estimates are more useful ----
print("\n=== Part (c): Why Bayesian Estimates Are More Useful ===\n")

print("The frequentist point estimate is 0/40 = 0 (0%).\n")

print("This is PROBLEMATIC for regulatory decision-making because:\n")

print("1. ZERO IS NOT CREDIBLE: Just because no adverse events were")
print("   observed in 40 patients does not mean the true rate is zero.")
print("   The 'rule of 3' gives an upper bound of 3/40 = 7.5%, but")
print("   this is ad hoc and does not provide a full distribution.\n")

print("2. BAYESIAN ESTIMATES ARE HONEST: Each prior gives a non-zero")
print("   estimate, which is more realistic. Even the Jeffreys prior")
print("   yields ~1.2%, acknowledging rare events can occur.\n")

print("3. UPPER CREDIBLE BOUNDS: Regulators need worst-case estimates.")
print("   The 95% upper bound provides a direct probability statement:")
print("   'There is 95% probability the true rate is below X%'.\n")

print("4. PRIOR INFORMATION IS VALUABLE: Known complication rates from")
print("   similar procedures can be formally incorporated. The frequentist")
print("   approach ignores all prior knowledge.\n")

print("5. DECISION SUPPORT: The full posterior enables quantities like")
print("   P(rate < 5% | data), directly supporting regulatory decisions.\n")

print("  P(rate < 5% | data):")
for name, a0, b0 in priors:
    a_post = a0 + y
    b_post = b0 + n - y
    prob_lt_5 = beta.cdf(0.05, a_post, b_post)
    print(f"    {name}: {prob_lt_5:.3f}")
