# =============================================================================
# Chapter 17 - Exercise 1: DAG Construction (Conceptual)
# Studying the relationship between ACE inhibitor use and Acute Kidney Injury
# =============================================================================

# This is a conceptual exercise. The answers are provided as detailed comments
# with optional code to illustrate DAG concepts using the dagitty package.

# --- Part (a): List at least five relevant variables ---
#
# 1. Baseline kidney function (eGFR / serum creatinine)
#    - Patients with worse kidney function are more likely to receive ACE inhibitors
#      (for renoprotective effects) AND are at higher risk of AKI.
#
# 2. Heart failure severity
#    - Heart failure is a major indication for ACE inhibitors AND independently
#      increases AKI risk (through haemodynamic changes).
#
# 3. Diabetes
#    - Diabetes is an indication for ACE inhibitors (nephroprotection) AND a risk
#      factor for AKI.
#
# 4. Age
#    - Older patients are more likely to be on ACE inhibitors and are at higher
#      risk of AKI.
#
# 5. Hypertension
#    - Primary indication for ACE inhibitors and can contribute to kidney injury.
#
# 6. Concomitant nephrotoxic drugs (e.g., NSAIDs, contrast agents)
#    - May be prescribed alongside or instead of ACE inhibitors and directly
#      cause AKI.
#
# 7. Volume status / dehydration
#    - Dehydration increases AKI risk and may influence whether ACE inhibitors
#      are held or continued.
#
# 8. Proteinuria
#    - An indication for ACE inhibitors and a marker of kidney disease severity
#      (related to AKI risk).

# --- Part (b): Draw a DAG ---
# We use the dagitty package to encode the DAG programmatically.

# install.packages("dagitty")
library(dagitty)

dag <- dagitty('dag {
  ACEi [exposure]
  AKI [outcome]
  eGFR [confounder]
  HeartFailure [confounder]
  Diabetes [confounder]
  Age [confounder]
  Hypertension [confounder]
  NephrotoxicDrugs [confounder]
  VolumeStatus [confounder]
  ICUAdmission [collider]

  Age -> ACEi

  Age -> AKI
  Age -> eGFR
  Age -> HeartFailure
  Age -> Diabetes

  eGFR -> ACEi
  eGFR -> AKI

  HeartFailure -> ACEi
  HeartFailure -> AKI
  HeartFailure -> VolumeStatus

  Diabetes -> ACEi
  Diabetes -> AKI
  Diabetes -> eGFR

  Hypertension -> ACEi
  Hypertension -> AKI

  NephrotoxicDrugs -> AKI

  VolumeStatus -> AKI

  ACEi -> AKI

  # Collider: ICU Admission is caused by both ACEi use patterns
  # and by AKI itself (patients with AKI go to ICU)
  ACEi -> ICUAdmission
  AKI -> ICUAdmission
}')

# Visualise the DAG (if running interactively)
# plot(dag)

# --- Part (c): Minimal sufficient adjustment set ---
# Using dagitty to compute the adjustment set automatically
adj_set <- adjustmentSets(dag, exposure = "ACEi", outcome = "AKI", type = "minimal")
cat("Minimal sufficient adjustment sets:\n")
print(adj_set)

# Explanation:
# The minimal adjustment set includes confounders that block all backdoor paths
# from ACEi to AKI. Based on our DAG, this includes:
# - Age, eGFR, HeartFailure, Diabetes, Hypertension
# These are the common causes of both ACEi use and AKI.
# We do NOT need to adjust for VolumeStatus (it's not a confounder of ACEi->AKI
# unless there's a direct path from ACEi to VolumeStatus).
# We do NOT need to adjust for NephrotoxicDrugs (it only affects AKI, not ACEi).

# --- Part (d): Identify a collider ---
#
# ICU Admission is a COLLIDER: it is caused by both ACEi use (patients on
# ACE inhibitors who develop complications may be admitted to ICU) and AKI
# (AKI itself is a reason for ICU admission).
#
# What happens if we adjust for ICU Admission?
# Conditioning on a collider OPENS a spurious path between ACEi and AKI.
# This would create "collider bias" (also known as Berkson's bias in clinical
# settings). Even if ACEi had no effect on AKI, adjusting for ICU admission
# would create an artificial association because:
# - Among ICU patients, if they were NOT admitted for AKI, they were more
#   likely admitted for ACEi-related reasons (and vice versa).
# - This induces a negative correlation between the two causes of the collider.
#
# Practical lesson: Do NOT adjust for variables that are consequences of both
# the exposure and the outcome (or that lie on causal paths from both).

cat("\n--- Summary ---\n")
cat("Key confounders to adjust for: Age, eGFR, Heart Failure, Diabetes, Hypertension\n")
cat("Collider to AVOID adjusting for: ICU Admission\n")
cat("Mediators to consider carefully: Volume status changes caused by ACEi\n")
