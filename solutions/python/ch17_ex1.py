# =============================================================================
# Chapter 17 - Exercise 1: DAG Construction (Conceptual)
# Studying the relationship between ACE inhibitor use and Acute Kidney Injury
# =============================================================================

# This is a conceptual exercise. The answers are provided as detailed comments
# with optional code to illustrate DAG concepts.

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
# We represent the DAG as an adjacency list and visualise with networkx.

import networkx as nx
import matplotlib.pyplot as plt

# Define the DAG
dag = nx.DiGraph()

# Add edges representing causal relationships
edges = [
    # Age affects many variables
    ("Age", "ACEi"), ("Age", "AKI"), ("Age", "eGFR"),
    ("Age", "HeartFailure"), ("Age", "Diabetes"),
    # eGFR confounds ACEi-AKI
    ("eGFR", "ACEi"), ("eGFR", "AKI"),
    # Heart failure confounds ACEi-AKI
    ("HeartFailure", "ACEi"), ("HeartFailure", "AKI"),
    ("HeartFailure", "VolumeStatus"),
    # Diabetes confounds ACEi-AKI
    ("Diabetes", "ACEi"), ("Diabetes", "AKI"), ("Diabetes", "eGFR"),
    # Hypertension confounds ACEi-AKI
    ("Hypertension", "ACEi"), ("Hypertension", "AKI"),
    # Other causes of AKI
    ("NephrotoxicDrugs", "AKI"),
    ("VolumeStatus", "AKI"),
    # Treatment -> Outcome (causal effect of interest)
    ("ACEi", "AKI"),
    # Collider: ICU admission caused by both ACEi complications and AKI
    ("ACEi", "ICUAdmission"), ("AKI", "ICUAdmission"),
]

dag.add_edges_from(edges)

# Visualise the DAG
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(dag, seed=42, k=2)
nx.draw(dag, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=9, font_weight='bold',
        arrows=True, arrowsize=20, edge_color='gray')
plt.title("DAG: ACE Inhibitor Use and Acute Kidney Injury")
plt.tight_layout()
plt.savefig("dag_acei_aki.png", dpi=300)
plt.show()

# --- Part (c): Minimal sufficient adjustment set ---
#
# To estimate the causal effect of ACEi on AKI, we need to block all
# backdoor paths (non-causal paths from ACEi to AKI).
#
# Backdoor paths from ACEi to AKI:
# 1. ACEi <- Age -> AKI
# 2. ACEi <- eGFR -> AKI
# 3. ACEi <- HeartFailure -> AKI
# 4. ACEi <- Diabetes -> AKI
# 5. ACEi <- Diabetes -> eGFR -> AKI
# 6. ACEi <- Hypertension -> AKI
# 7. ACEi <- Age -> eGFR -> AKI
# 8. ACEi <- Age -> HeartFailure -> AKI
# 9. ACEi <- Age -> Diabetes -> AKI
# 10. ACEi <- HeartFailure -> VolumeStatus -> AKI
#
# Minimal sufficient adjustment set:
# {Age, eGFR, HeartFailure, Diabetes, Hypertension}
#
# This blocks all backdoor paths. Note:
# - NephrotoxicDrugs only affects AKI (not a confounder), so no need to adjust.
# - VolumeStatus is blocked by conditioning on HeartFailure (the path goes
#   through HeartFailure).

print("Minimal sufficient adjustment set:")
print("  {Age, eGFR, HeartFailure, Diabetes, Hypertension}")

# --- Part (d): Identify a collider ---
#
# ICU Admission is a COLLIDER in the DAG:
#   ACEi -> ICUAdmission <- AKI
#
# Both ACEi use (through complications or monitoring) and AKI (as a
# critical condition) can cause ICU admission.
#
# What happens if we adjust for ICU Admission?
#
# Conditioning on a collider OPENS a spurious path between its causes.
# This creates "collider bias" (Berkson's bias):
#
# - Among ICU patients, if a patient was NOT admitted because of AKI,
#   they were more likely admitted due to ACEi-related issues (and vice versa).
# - This induces an artificial negative association between ACEi and AKI,
#   even if no true causal relationship exists.
#
# Practical lesson: Never adjust for variables that are consequences of both
# the exposure and the outcome. Restricting the analysis to ICU patients
# only would similarly introduce collider bias.

print("\nCollider identified: ICU Admission")
print("  ACEi -> ICUAdmission <- AKI")
print("  Adjusting for ICU admission would OPEN a spurious path (collider bias)")

print("\n--- Summary ---")
print("Key confounders to adjust for: Age, eGFR, Heart Failure, Diabetes, Hypertension")
print("Collider to AVOID adjusting for: ICU Admission")
print("Mediators to consider: Volume status changes caused by ACEi")
