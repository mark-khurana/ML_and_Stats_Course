# Chapter 13, Exercise 1: Diagnostic Test Updating
# Sequential Bayesian updating with mammogram and ultrasound

import matplotlib.pyplot as plt

# ---- Part (a): Posterior probability after positive mammogram ----
print("=== Part (a): Positive Mammogram ===")

sens_mammo = 0.90   # Sensitivity of mammogram
spec_mammo = 0.92   # Specificity of mammogram
prevalence = 0.02   # Prior probability (prevalence in women aged 50-59)

# Apply Bayes' theorem:
# P(cancer | +) = P(+ | cancer) * P(cancer) / P(+)
p_pos = sens_mammo * prevalence + (1 - spec_mammo) * (1 - prevalence)
post_mammo = (sens_mammo * prevalence) / p_pos

print(f"Prior (prevalence):  {prevalence}")
print(f"P(positive test):    {p_pos:.4f}")
print(f"P(cancer | positive mammogram): {post_mammo:.4f}")
print(f"\nDespite 90% sensitivity and 92% specificity, the posterior")
print(f"probability is only about {post_mammo*100:.1f}% because the")
print(f"prior probability (prevalence) is very low at 2%.")

# ---- Part (b): Sequential update with positive ultrasound ----
print("\n=== Part (b): Sequential Update with Positive Ultrasound ===")

sens_us = 0.95   # Sensitivity of ultrasound
spec_us = 0.85   # Specificity of ultrasound

# The posterior from (a) becomes the new prior
prior_us = post_mammo
p_pos_us = sens_us * prior_us + (1 - spec_us) * (1 - prior_us)
post_us = (sens_us * prior_us) / p_pos_us

print(f"Prior (posterior from mammogram): {prior_us:.4f}")
print(f"P(cancer | pos mammogram AND pos ultrasound): {post_us:.4f}")
print(f"\nAfter two positive tests, the posterior probability rises to about {post_us*100:.1f}%.")

# ---- Part (c): What sequential updating illustrates ----
print("\n=== Part (c): What Sequential Updating Illustrates ===")
print("1. NATURAL SEQUENTIAL UPDATING: In the Bayesian framework,")
print("   today's posterior becomes tomorrow's prior. Each new piece")
print("   of evidence updates our belief incrementally.")
print()
print("2. THE PRIOR MATTERS: Starting from a low prevalence (2%),")
print(f"   even a sensitive test only raises the probability to ~{post_mammo*100:.0f}%.")
print(f"   A second positive test raises it further to ~{post_us*100:.0f}%.")
print()
print("3. CLINICAL REASONING IS BAYESIAN: Clinicians intuitively do")
print("   this -- they order confirmatory tests precisely because a")
print("   single screening test in a low-prevalence population is")
print("   insufficient. The Bayesian framework formalises this logic.")
print()
print("4. COHERENCE: The Bayesian approach naturally handles sequential")
print("   evidence without the multiple-testing corrections that")
print("   frequentist methods would require.")

# ---- Visualise the updating process ----
stages = ["Prior\n(prevalence)", "After\nmammogram (+)", "After\nultrasound (+)"]
probs = [prevalence, post_mammo, post_us]
colors = ["steelblue", "goldenrod", "firebrick"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(stages, probs, color=colors, edgecolor="none", width=0.6)
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{p*100:.1f}%", ha="center", fontsize=11)
ax.set_ylabel("P(Cancer)")
ax.set_title("Sequential Bayesian Updating\nBreast Cancer Diagnosis")
ax.set_ylim(0, max(probs) * 1.3)
plt.tight_layout()
plt.savefig("ch13_ex1_updating.png", dpi=150)
plt.show()
