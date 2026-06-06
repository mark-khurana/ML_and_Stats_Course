# =============================================================================
# Chapter 9, Exercise 1: Bayes' Theorem in Practice (PPV Calculations)
# Calculate PPV at different prevalence levels and plot the relationship.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


# --- PPV function using Bayes' theorem ---
def calculate_ppv(sensitivity, specificity, prevalence):
    """PPV = (Sens * Prev) / (Sens * Prev + (1-Spec) * (1-Prev))"""
    numerator = sensitivity * prevalence
    denominator = numerator + (1 - specificity) * (1 - prevalence)
    return numerator / denominator


def calculate_npv(sensitivity, specificity, prevalence):
    """NPV = (Spec * (1-Prev)) / (Spec * (1-Prev) + (1-Sens) * Prev)"""
    numerator = specificity * (1 - prevalence)
    denominator = numerator + (1 - sensitivity) * prevalence
    return numerator / denominator


# --- Example: HIV rapid test (sensitivity = 99.7%, specificity = 99.5%) ---
print("=== HIV Rapid Test (Sens=99.7%, Spec=99.5%) ===\n")

# General population (prevalence ~ 0.4%)
ppv_general = calculate_ppv(0.997, 0.995, 0.004)
npv_general = calculate_npv(0.997, 0.995, 0.004)
print(f"General population (prevalence 0.4%):")
print(f"  PPV: {ppv_general*100:.1f}%")
print(f"  NPV: {npv_general*100:.1f}%")

# STI clinic (prevalence ~ 5%)
ppv_clinic = calculate_ppv(0.997, 0.995, 0.05)
npv_clinic = calculate_npv(0.997, 0.995, 0.05)
print(f"\nSTI clinic (prevalence 5%):")
print(f"  PPV: {ppv_clinic*100:.1f}%")
print(f"  NPV: {npv_clinic*100:.1f}%")

# Known exposure (prevalence ~ 30%)
ppv_exposed = calculate_ppv(0.997, 0.995, 0.30)
npv_exposed = calculate_npv(0.997, 0.995, 0.30)
print(f"\nKnown exposure (prevalence 30%):")
print(f"  PPV: {ppv_exposed*100:.1f}%")
print(f"  NPV: {npv_exposed*100:.1f}%")

# --- Plot PPV and NPV across a range of prevalences ---
prevalences = np.linspace(0.001, 0.5, 500)
ppvs = [calculate_ppv(0.997, 0.995, p) for p in prevalences]
npvs = [calculate_npv(0.997, 0.995, p) for p in prevalences]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(prevalences * 100, np.array(ppvs) * 100, lw=2, color="steelblue",
        label="PPV")
ax.plot(prevalences * 100, np.array(npvs) * 100, lw=2, color="darkorange",
        label="NPV")
ax.axhline(y=50, linestyle="--", color="grey", alpha=0.7)
ax.set_xlabel("Prevalence (%)")
ax.set_ylabel("Predictive Value (%)")
ax.set_title("PPV and NPV Depend Heavily on Prevalence\n"
             "HIV rapid test: Sens=99.7%, Spec=99.5%")
ax.legend()
plt.tight_layout()
plt.show()

# --- Additional: Compare tests with different sensitivity/specificity ---
print("\n=== Comparing Tests at 1% Prevalence ===")

tests = [
    ("High Sens/Low Spec", 0.99, 0.90),
    ("Balanced",           0.95, 0.95),
    ("Low Sens/High Spec", 0.80, 0.99),
]

for name, sens, spec in tests:
    ppv = calculate_ppv(sens, spec, 0.01)
    npv = calculate_npv(sens, spec, 0.01)
    print(f"{name} (Sens={sens*100:.0f}%, Spec={spec*100:.0f}%): "
          f"PPV={ppv*100:.1f}%, NPV={npv*100:.1f}%")

print("\nKey takeaway: Even excellent tests have low PPV when prevalence is low.")
print("This is why screening should target high-risk populations.")
