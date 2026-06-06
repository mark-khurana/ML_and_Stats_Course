# =============================================================================
# Chapter 17 - Exercise 4: Target Trial Emulation (Conceptual)
# Early vs delayed metformin initiation and 5-year cardiovascular events
# =============================================================================

# This is a conceptual exercise. All answers are provided as detailed comments.

# --- Part (a): Complete target trial protocol table ---
#
# Protocol Component     | Target Trial                          | Observational Emulation
# -----------------------|---------------------------------------|---------------------------------------
# Eligibility criteria   | Adults aged 30-80 with new T2DM       | Same criteria in EHR: first T2DM
#                        | diagnosis, no prior CVD, no prior      | ICD-10 E11 code, no prior MACE,
#                        | metformin use, eGFR >= 30, no          | no metformin dispensing before index,
#                        | contraindications to metformin.        | eGFR >= 30 within prior 6 months.
#                        |                                       |
# Treatment strategies   | Strategy 1: Initiate metformin        | Same. Defined by first metformin
#                        | within 3 months of T2DM diagnosis.    | prescription fill date relative to
#                        | Strategy 2: Do not initiate within     | T2DM diagnosis date.
#                        | 3 months (delayed: 3-12 months).      |
#                        |                                       |
# Treatment assignment   | Random assignment at T2DM diagnosis.  | Not random. Adjusted via IPTW or
#                        |                                       | PS matching on baseline confounders.
#                        |                                       |
# Start of follow-up     | Date of randomisation (= date of      | Date of T2DM diagnosis (time zero).
# (time zero)            | T2DM diagnosis).                      | All eligible patients enter at this
#                        |                                       | date, regardless of treatment timing.
#                        |                                       |
# Outcome                | First MACE (MI, stroke, or CV death)  | Same. MACE via ICD-10 codes.
#                        | within 5 years.                       | Censored at 5 years, loss to
#                        |                                       | follow-up, or non-CV death.
#                        |                                       |
# Causal contrast        | ITT: effect of assignment to early    | ITT: compare groups as assigned.
#                        | vs delayed. Per-protocol: effect of   | Per-protocol: clone-censor-weight.
#                        | adhering to assignment.                |
#                        |                                       |
# Analysis plan          | Cox PH model; ITT primary.            | IPTW-weighted Cox PH for ITT.
#                        |                                       | Clone-censor-weight for per-protocol.

# --- Part (b): Sources of immortal time bias ---
#
# 1. SURVIVAL REQUIREMENT FOR CLASSIFICATION:
#    To be classified as a "delayed initiator," a patient must survive at
#    least 3 months without starting metformin. This period is "immortal
#    time" -- the patient cannot die (by definition) during this window.
#    This artificially inflates survival in the delayed group.
#
# 2. MISCLASSIFICATION OF PERSON-TIME:
#    If follow-up starts at T2DM diagnosis but treatment classification
#    depends on future behavior, person-time before treatment initiation
#    is misclassified. The period before a patient's first metformin
#    prescription is attributed to the wrong treatment group.
#
# 3. EXCLUSION OF EARLY DEATHS:
#    Patients who die before initiating metformin may be excluded from
#    the analysis entirely, creating survivorship bias.
#
# HOW TARGET TRIAL EMULATION AVOIDS THIS:
# - Aligns time zero with eligibility (T2DM diagnosis), not treatment
# - Uses clone-censor-weight: clones each patient into both strategies,
#   censors when they deviate, and applies IPCW to correct for the
#   informative censoring introduced by this approach.

# --- Part (c): New-user, active comparator design ---
#
# NEW-USER (INCIDENT USER) DESIGN:
# - Includes patients only at the point of first treatment decision
# - Avoids prevalent user bias (selective survival of tolerators)
# - Captures early effects including initial side effects
# - Provides a clear "time zero" aligned with treatment decision
#
# ACTIVE COMPARATOR DESIGN:
# - Compares early metformin vs delayed metformin (not vs no treatment)
# - Patients in both groups are deemed to need metformin
# - Reduces confounding by indication: the comparison is about timing,
#   not about treatment vs no treatment
# - Improves positivity: most T2DM patients can plausibly be in either
#   timing group
# - Reduces healthy user bias: both groups seek treatment
# - Mimics the clinically relevant question

# --- Part (d): Unmeasured confounders and sensitivity analysis ---
#
# POTENTIAL UNMEASURED CONFOUNDERS:
# 1. Diabetes severity trajectory (HbA1c trends)
# 2. Health literacy and medication adherence behaviour
# 3. Physician quality / prescribing culture
# 4. Socioeconomic status / insurance coverage
# 5. Diet, exercise, and lifestyle factors
# 6. Patient preferences and shared decision-making
#
# SENSITIVITY ANALYSIS APPROACHES:
# 1. E-value: quantify minimum confounder strength to nullify the result
# 2. Quantitative bias analysis: use external data to model bias
# 3. Negative control outcomes: test outcomes unrelated to metformin
# 4. Negative control exposures: test unrelated treatments
# 5. Instrumental variable analysis: e.g., physician preference

print("=" * 70)
print("Exercise 4: Target Trial Emulation (Conceptual)")
print("=" * 70)
print()
print("All answers are provided as comments in this script.")
print("Review the comments for:")
print("  (a) Complete target trial protocol table")
print("  (b) Sources of immortal time bias in a naive analysis")
print("  (c) How new-user, active comparator design addresses confounding")
print("  (d) Unmeasured confounders and sensitivity analysis approaches")
print()
print("Key takeaway: Target trial emulation forces explicit specification")
print("of all protocol components, making assumptions transparent and")
print("reducing common biases like immortal time bias and prevalent user bias.")
