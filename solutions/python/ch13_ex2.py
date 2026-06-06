# Chapter 13, Exercise 2: Prior Sensitivity Analysis
# Phase II antibiotic trial: 21/30 patients achieve clinical cure

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

y, n = 21, 30  # successes and total
theta = np.linspace(0, 1, 500)

# Define three priors
priors = [
    ("Beta(1,1) - Uninformative", 1, 1),
    ("Beta(5,5) - Weakly informative", 5, 5),
    ("Beta(20,10) - Informative", 20, 10),
]

# ---- Part (a): Posterior mean and 95% credible interval ----
print("=== Part (a): Posterior Summaries ===\n")

fig, ax = plt.subplots(figsize=(8, 5))

for name, a0, b0 in priors:
    a_post = a0 + y
    b_post = b0 + n - y
    post_mean = a_post / (a_post + b_post)
    ci = beta.ppf([0.025, 0.975], a_post, b_post)

    print(f"Prior: {name}")
    print(f"  Prior mean: {a0 / (a0 + b0):.3f}")
    print(f"  Posterior: Beta({a_post}, {b_post})")
    print(f"  Posterior mean: {post_mean:.3f}")
    print(f"  95% credible interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print()

    # ---- Part (b): Plot all three posteriors ----
    ax.plot(theta, beta.pdf(theta, a_post, b_post), label=name, linewidth=1.5)

ax.axvline(y / n, linestyle="--", color="grey", linewidth=1)
ax.text(y / n + 0.02, ax.get_ylim()[1] * 0.1,
        f"MLE = {y/n:.3f}", fontsize=9, color="grey")
ax.set_xlabel(r"$\theta$ (cure rate)")
ax.set_ylabel("Density")
ax.set_title("Posterior Distributions Under Different Priors\n"
             "21/30 patients cured in Phase II trial")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("ch13_ex2_posteriors.png", dpi=150)
plt.show()

# ---- Part (c): Which prior has the most influence? ----
print("=== Part (c): Which Prior Has Most Influence? ===\n")
print("The Beta(5,5) prior has the MOST influence on the posterior.")
print()
print("Why? The effective sample size of a Beta(a,b) prior is a + b.")
print("  - Beta(1,1):   effective n = 2  (trivial influence)")
print("  - Beta(5,5):   effective n = 10 (modest influence)")
print("  - Beta(20,10): effective n = 30 (substantial influence)")
print()
print("However, while Beta(20,10) has the largest effective sample size,")
print("its prior mean (0.667) is close to the data (21/30 = 0.700),")
print("so the prior and data 'agree', and the posterior is not pulled")
print("far from the MLE.")
print()
print("Beta(5,5) has a prior mean of 0.500, which DISAGREES with the")
print("data. It pulls the posterior mean toward 0.5, producing the most")
print("visible shift from the MLE. With n=30 observed data, even an")
print("effective sample size of 10 produces noticeable pull when the")
print("prior and data disagree.")
print()
print("Key insight: prior influence depends on BOTH the effective sample")
print("size AND how much the prior disagrees with the data.")
