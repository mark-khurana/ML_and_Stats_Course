# =============================================================================
# Chapter 17 - Exercise 1: Build a DAG and derive the adjustment set
# ACE inhibitor use and acute kidney injury (AKI) in hospitalised patients
# =============================================================================
#
# Libraries -------------------------------------------------------------------
# install.packages("dagitty")
library(dagitty) # encode a DAG, derive adjustment sets

# -----------------------------------------------------------------------------
# (a) Relevant variables, and the causal ROLE of each
# -----------------------------------------------------------------------------
# The role matters more than the list: it decides whether you adjust or not.
#
# CONFOUNDERS (arrow into BOTH ACEi and AKI) -> MUST adjust
#   1. Baseline kidney function (eGFR / creatinine)
#      Worse kidney function is a reason to prescribe an ACE inhibitor
#      (renoprotection) AND an independent risk factor for AKI.
#   2. Heart failure
#      A major indication for ACE inhibitors AND independently raises AKI risk
#      through haemodynamic changes.
#   3. Age
#      Older patients are more likely to be on an ACE inhibitor and are at
#      higher risk of AKI.
#   4. Diabetes / hypertension
#      Both are indications for ACE inhibitors and both raise AKI risk.
#
# MEDIATOR (on the causal path ACEi -> ... -> AKI) -> do NOT adjust for a
# total effect
#   5. Renal perfusion pressure
#      Part of HOW an ACE inhibitor precipitates AKI is by reducing glomerular
#      perfusion pressure. Adjust for it and you remove part of the very effect
#      you are trying to measure.
#
# COMPETING CAUSE (arrow into AKI only) -> adjusting is optional, harmless,
# and may improve precision, but it is NOT needed to remove bias
#   6. Concomitant nephrotoxic drugs (NSAIDs, contrast)
#      Only a cause of AKI, not of ACEi prescribing (in this DAG).
#
# COLLIDER (arrow in from BOTH) -> NEVER adjust
#   7. ICU admission
#      Patients are admitted to ICU because of ACEi-related complications AND
#      because of AKI itself. Conditioning on it manufactures an association.

# -----------------------------------------------------------------------------
# (b) Encode the DAG
# -----------------------------------------------------------------------------
# IMPORTANT: the text inside dagitty('dag { ... }') is NOT R code. It is
# dagitty's own DAG language, and it has NO comment syntax -- putting a `#`
# inside the quotes is a syntax error:
#     Error: SyntaxError: Expected "-", "--", ... but "#" found.
# Keep all explanation outside the quotes, as ordinary R comments like these.

aki_dag <- dagitty('dag {
  ACEi              [exposure]
  AKI               [outcome]

  Age               -> ACEi
  Age               -> AKI
  Age               -> BaselineEGFR
  BaselineEGFR      -> ACEi
  BaselineEGFR      -> AKI
  HeartFailure      -> ACEi
  HeartFailure      -> AKI
  Diabetes          -> ACEi
  Diabetes          -> AKI
  Diabetes          -> BaselineEGFR
  Hypertension      -> ACEi
  Hypertension      -> AKI

  NephrotoxicDrugs  -> AKI

  ACEi              -> AKI
  ACEi              -> RenalPerfusion
  RenalPerfusion    -> AKI

  ACEi              -> ICUAdmission
  AKI               -> ICUAdmission
}')

# Visualise it (needs coordinates to look tidy; the browser tool at
# https://dagitty.net is easier for drawing by hand)
# plot(graphLayout(aki_dag))

# -----------------------------------------------------------------------------
# (c) Minimal sufficient adjustment set for the TOTAL effect
# -----------------------------------------------------------------------------
cat("--- (c) Minimal adjustment set for the TOTAL effect of ACEi on AKI ---\n")
print(adjustmentSets(aki_dag, type = "minimal", effect = "total"))

# Note what dagitty returns and what it leaves out:
#   INCLUDED: Age, BaselineEGFR, HeartFailure, Diabetes, Hypertension
#             -- the five confounders, which between them close every
#                backdoor path.
#   EXCLUDED: RenalPerfusion  (a mediator -- adjusting removes part of the
#                              effect we want)
#             ICUAdmission     (a collider -- adjusting CREATES bias)
#             NephrotoxicDrugs (only causes AKI, so it opens no backdoor path;
#                              harmless to include, unnecessary for validity)
#
# dagitty will also list the conditional independences your DAG implies. Some
# of these are testable in real data, which is a rare chance to check a causal
# assumption empirically rather than just asserting it.
cat("\nFirst few implied conditional independences (testable in real data):\n")
print(head(impliedConditionalIndependencies(aki_dag), 5))

# -----------------------------------------------------------------------------
# (d) The collider, demonstrated on simulated data
# -----------------------------------------------------------------------------
# We simulate a small version of the DAG in which the ACE inhibitor has NO
# effect whatsoever on AKI (true effect = 0), then estimate the association
# three ways.

set.seed(42)
n <- 20000

# One confounder, for clarity: baseline kidney disease
ckd <- rbinom(n, 1, 0.35)

# ACEi is prescribed more often in CKD (confounding by indication)
acei <- rbinom(n, 1, plogis(-0.5 + 1.2 * ckd))

# AKI depends on CKD but NOT AT ALL on ACEi -- the true effect is zero
aki <- rbinom(n, 1, plogis(-2.0 + 1.5 * ckd + 0.0 * acei))

# ICU admission is caused by BOTH ACEi and AKI: the collider
icu <- rbinom(n, 1, plogis(-2.0 + 1.0 * acei + 2.0 * aki))

dat <- data.frame(ckd, acei, aki, icu)

lo <- function(fit) coef(fit)["acei"]

fit_unadj <- glm(aki ~ acei, data = dat, family = binomial)
fit_conf <- glm(aki ~ acei + ckd, data = dat, family = binomial)
fit_collider <- glm(aki ~ acei + ckd + icu, data = dat, family = binomial)

cat("\n--- (d) Log-odds of ACEi on AKI (TRUE value = 0.000) ---\n")
cat(sprintf("Unadjusted (confounded by CKD)      : %+.3f\n", lo(fit_unadj)))
cat(sprintf("Adjusted for the confounder (CKD)   : %+.3f   <- correct\n", lo(fit_conf)))
cat(sprintf("ALSO adjusted for ICU (a collider) : %+.3f   <- bias re-introduced\n", lo(fit_collider)))

# And the other common form of collider bias: restricting the analysis to a
# collider-defined subgroup, e.g. running the study in an ICU cohort only.
fit_icu_only <- glm(aki ~ acei + ckd, data = subset(dat, icu == 1), family = binomial)
cat(sprintf("Restricted to ICU patients only     : %+.3f   <- same bias\n", lo(fit_icu_only)))

cat("\nInterpretation:\n")
cat("Adjusting for CKD removes the confounding and recovers the truth (0).\n")
cat("Adding ICU admission -- or studying only ICU patients -- pushes the\n")
cat("estimate NEGATIVE, inventing a protective effect for a drug that does\n")
cat("nothing. Why: among ICU patients, someone who is NOT on an ACE inhibitor\n")
cat("probably got there because of their AKI, so 'no ACEi' starts to predict\n")
cat("AKI. The association is real inside the ICU and absent outside it.\n")
cat("\nPractical lesson: never adjust for, or select on, a variable that is a\n")
cat("consequence of both the exposure and the outcome.\n")
