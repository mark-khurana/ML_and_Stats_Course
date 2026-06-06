# =============================================================================
# Chapter 17 - Exercise 4: Target Trial Emulation (Conceptual)
# Early vs delayed metformin initiation and 5-year cardiovascular events
# =============================================================================

# This is a conceptual exercise. All answers are provided as detailed comments.

# --- Part (a): Complete target trial protocol table ---
#
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Protocol component         | Target trial                                | Observational emulation                   |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Eligibility criteria       | Adults aged 30-80 with new T2DM diagnosis,  | Same criteria applied to EHR database:    |
# |                            | no prior CVD, no prior metformin use,        | first T2DM diagnosis code (ICD-10 E11),   |
# |                            | eGFR >= 30, no contraindications to          | no prior MACE, no metformin dispensing     |
# |                            | metformin (e.g., severe renal impairment).   | before index date, eGFR >= 30 in prior    |
# |                            |                                             | 6 months.                                 |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Treatment strategies       | Strategy 1: Initiate metformin within       | Same. Defined by first metformin          |
# |                            | 3 months of T2DM diagnosis.                 | prescription fill date relative to T2DM   |
# |                            | Strategy 2: Do not initiate metformin       | diagnosis date.                           |
# |                            | within 3 months (delayed, 3-12 months).     |                                           |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Treatment assignment       | Random assignment at time of T2DM           | Not random. Adjusted using IPTW or        |
# |                            | diagnosis.                                  | propensity score matching, conditioning   |
# |                            |                                             | on baseline confounders.                  |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Start of follow-up         | Date of randomisation (= date of T2DM       | Date of T2DM diagnosis (time zero).       |
# | (time zero)                | diagnosis).                                 | All eligible patients enter at this date. |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Outcome                    | First major adverse cardiovascular event     | Same. MACE defined by ICD-10 codes for    |
# |                            | (MACE): composite of MI, stroke, or CV      | MI (I21), stroke (I63, I64), or CV death  |
# |                            | death within 5 years.                       | (cause of death codes). Censored at 5     |
# |                            |                                             | years, loss to follow-up, or death from   |
# |                            |                                             | non-CV causes.                            |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Causal contrast            | Intention-to-treat: effect of being          | Intention-to-treat: compare early vs      |
# |                            | assigned to early vs delayed initiation.     | delayed initiation regardless of          |
# |                            | Per-protocol: effect of adhering to the      | subsequent adherence. Per-protocol:       |
# |                            | assigned strategy.                           | use clone-censor-weight approach.         |
# +----------------------------+---------------------------------------------+-------------------------------------------+
# | Analysis plan              | Cox proportional hazards model.              | IPTW-weighted Cox PH model for ITT.       |
# |                            | ITT analysis is primary.                     | For per-protocol: clone each patient      |
# |                            | Per-protocol as sensitivity analysis.        | into both strategies, censor when they    |
# |                            |                                             | deviate, and apply IPCW to correct for    |
# |                            |                                             | informative censoring.                    |
# +----------------------------+---------------------------------------------+-------------------------------------------+

# --- Part (b): Sources of immortal time bias in a naive analysis ---
#
# In a naive analysis comparing "early initiators" (metformin within 3 months)
# vs "delayed initiators" (3-12 months), several sources of immortal time
# bias can arise:
#
# 1. SURVIVAL REQUIREMENT FOR CLASSIFICATION:
#    To be classified as a "delayed initiator," a patient must survive at least
#    3 months (to reach the delayed window). The time between T2DM diagnosis
#    and 3 months is "immortal" for the delayed group -- they cannot experience
#    the outcome during this period by definition. This artificially inflates
#    survival in the delayed group.
#
# 2. MISCLASSIFICATION OF PERSON-TIME:
#    If follow-up starts at T2DM diagnosis for both groups but treatment
#    classification depends on future events (whether and when metformin is
#    started), person-time before treatment initiation is misclassified.
#    A patient who starts metformin at month 2 contributes 2 months of
#    "untreated" time that is credited to the "early" group.
#
# 3. EXCLUSION OF NON-INITIATORS WHO DIE EARLY:
#    Patients who die before 3 months and never start metformin may be excluded
#    entirely if the study requires treatment initiation. This creates
#    survivorship bias.
#
# HOW TARGET TRIAL EMULATION AVOIDS THIS:
# - Time zero is aligned with eligibility (T2DM diagnosis), not treatment start
# - The clone-censor-weight approach assigns each patient to BOTH strategies
#   at time zero, censors them when they deviate from their assigned strategy,
#   and uses inverse probability of censoring weights (IPCW) to correct for
#   the informative censoring

# --- Part (c): How new-user, active comparator design addresses confounding ---
#
# The NEW-USER (INCIDENT USER) DESIGN:
# - Only includes patients at the TIME OF FIRST metformin prescription
#   (or the decision point: T2DM diagnosis)
# - Avoids "prevalent user bias" where including patients already on metformin
#   selectively includes those who tolerated and responded to the drug
#   (survivors of the initial treatment period)
# - Captures early effects (including side effects) that might be missed
#   in prevalent user analyses
#
# The ACTIVE COMPARATOR DESIGN:
# - Compares early metformin vs delayed metformin (not metformin vs no treatment)
# - Patients choosing delayed initiation are more comparable to early initiators
#   than patients who never initiate (who may differ fundamentally in disease
#   severity, healthcare access, or physician preferences)
# - Reduces confounding by indication because both groups are deemed to need
#   metformin -- the question is only about TIMING
# - Makes the positivity assumption more plausible: most patients have a
#   realistic probability of being in either group
# - Mimics the clinical question: "Should I start metformin now or wait?"

# --- Part (d): Unmeasured confounders and sensitivity analysis ---
#
# POTENTIAL UNMEASURED CONFOUNDERS:
#
# 1. HbA1c trajectory / diabetes severity:
#    Patients with more rapidly rising HbA1c may receive metformin earlier.
#    If severity also affects CVD risk, this confounds the comparison.
#    (May be partially captured in EHR but with measurement timing issues.)
#
# 2. Patient health literacy and adherence behaviour:
#    Patients who seek care promptly and fill prescriptions early may also
#    engage in other health-promoting behaviours (exercise, diet) that
#    independently reduce CVD risk. This is a form of "healthy user bias."
#
# 3. Physician prescribing patterns / quality of care:
#    Physicians who prescribe metformin early may also provide better
#    overall cardiovascular risk management.
#
# 4. Socioeconomic status / insurance:
#    Patients with better insurance or higher SES may fill prescriptions
#    faster and have better access to follow-up care.
#
# 5. Diet, exercise, and lifestyle factors:
#    Rarely captured in EHR data but strongly associated with both
#    treatment initiation patterns and CVD outcomes.
#
# SENSITIVITY ANALYSIS APPROACHES:
#
# 1. E-VALUE:
#    Calculate the E-value for the primary estimate to quantify how strong
#    an unmeasured confounder would need to be (in terms of associations with
#    both treatment and outcome) to fully explain away the observed effect.
#
# 2. QUANTITATIVE BIAS ANALYSIS:
#    Use external data (e.g., surveys with lifestyle data) to estimate the
#    likely magnitude of confounding by specific unmeasured factors. Apply
#    the bias formula to adjust the estimate.
#
# 3. NEGATIVE CONTROL OUTCOMES:
#    Test outcomes that should NOT be affected by metformin timing (e.g.,
#    accidental injuries). If an association is found, it suggests residual
#    confounding.
#
# 4. NEGATIVE CONTROL EXPOSURES:
#    Test treatments that share the same confounding structure but should
#    not affect CVD (e.g., timing of proton pump inhibitor initiation).
#
# 5. INSTRUMENTAL VARIABLE ANALYSIS:
#    If a valid instrument exists (e.g., physician preference for early
#    prescribing), IV analysis can estimate the causal effect even with
#    unmeasured confounding.

cat("Exercise 4 is a conceptual exercise.\n")
cat("All answers are provided as detailed comments in this script.\n")
cat("Review the comments for:\n")
cat("  (a) Complete target trial protocol table\n")
cat("  (b) Sources of immortal time bias\n")
cat("  (c) How new-user, active comparator design helps\n")
cat("  (d) Unmeasured confounders and sensitivity approaches\n")
