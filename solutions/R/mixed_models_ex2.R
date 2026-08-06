# =============================================================================
# Chapter 6b, Exercise 2: Fit and interpret a random-intercept model
# Recreate the BP data, add a binary treatment, fit a random-intercept LMM,
# compute the ICC, and compare the treatment SE against ordinary lm().
# =============================================================================

library(tidyverse)   # data wrangling and simulation
library(lme4)        # lmer() for linear mixed models
# lmerTest adds p-values to lmer(); use it if available, otherwise fall back
# to estimates + profile confidence intervals from base lme4.
has_lmerTest <- requireNamespace("lmerTest", quietly = TRUE)
if (has_lmerTest) suppressMessages(library(lmerTest))

set.seed(42)

# --- Recreate the chapter's longitudinal, clustered BP dataset --------------
n_clinic  <- 8     # 8 clinics
n_patient <- 25    # 25 patients per clinic
n_visit   <- 5     # 5 visits per patient

clinic_effect  <- rnorm(n_clinic, 0, 6)              # clinic baselines (SD 6)
patient_effect <- rnorm(n_clinic * n_patient, 0, 8)  # patient baselines (SD 8)

# Add a simulated binary treatment, assigned at the PATIENT level (a patient is
# either treated or not, and keeps that assignment across all their visits).
# True treatment effect built into the simulation: -5 mmHg.
treatment_by_patient <- rbinom(n_clinic * n_patient, 1, 0.5)

bp <- expand_grid(
  clinic  = 1:n_clinic,
  patient = 1:n_patient,
  visit   = 1:n_visit
) |>
  mutate(
    patient_id = interaction(clinic, patient),   # unique patient label
    pid_index  = as.integer(patient_id),
    treatment  = treatment_by_patient[pid_index],
    sbp = 135 +
          clinic_effect[clinic] +
          patient_effect[pid_index] +
          -1.5 * (visit - 1) +      # downward drift per visit
          -5   * treatment +        # true treatment effect: -5 mmHg
          rnorm(n(), 0, 5)          # residual noise
  )

cat("Dataset:", nrow(bp), "rows |",
    n_clinic, "clinics x", n_patient, "patients x", n_visit, "visits\n\n")

# --- Fit the random-intercept model -----------------------------------------
# Fixed effects: visit + treatment. Random intercepts for clinic and for
# patient nested within clinic.
m_ri <- lmer(sbp ~ visit + treatment + (1 | clinic/patient_id), data = bp)

cat("=== Random-intercept model summary ===\n")
print(summary(m_ri))

# --- (a) Treatment fixed effect and its interpretation ----------------------
coefs   <- summary(m_ri)$coefficients
beta_trt <- coefs["treatment", "Estimate"]
se_trt   <- coefs["treatment", "Std. Error"]

cat("\n=== (a) Treatment fixed effect ===\n")
cat(sprintf("Estimate: %.3f mmHg   SE: %.3f\n", beta_trt, se_trt))
if (has_lmerTest && "Pr(>|t|)" %in% colnames(coefs)) {
  cat(sprintf("p-value (lmerTest): %.4g\n", coefs["treatment", "Pr(>|t|)"]))
} else {
  ci <- confint(m_ri, parm = "treatment", method = "Wald")
  cat(sprintf("95%% CI (Wald): %.3f to %.3f\n", ci[1], ci[2]))
}
cat("Interpretation: after accounting for visit and the clustering of visits\n")
cat("  within patients and patients within clinics, being on treatment is\n")
cat(sprintf(
  "  associated with a %.2f mmHg change in systolic BP (a reduction),\n",
  beta_trt))
cat("  holding visit number constant.\n")

# --- (b) Intraclass correlation ---------------------------------------------
vc <- as.data.frame(VarCorr(m_ri))
var_clinic  <- vc$vcov[vc$grp == "clinic"]
var_patient <- vc$vcov[vc$grp == "patient_id:clinic"]
var_resid   <- vc$vcov[vc$grp == "Residual"]
var_total   <- var_clinic + var_patient + var_resid

icc_clinic  <- var_clinic / var_total
icc_patient <- var_patient / var_total
# Proportion of variance at or above the patient level (clinic + patient):
icc_cluster <- (var_clinic + var_patient) / var_total

cat("\n=== (b) Variance components and ICC ===\n")
cat(sprintf("Between-clinic variance : %.2f\n", var_clinic))
cat(sprintf("Between-patient variance: %.2f\n", var_patient))
cat(sprintf("Residual variance       : %.2f\n", var_resid))
cat(sprintf("Clinic-level ICC : %.3f\n", icc_clinic))
cat(sprintf("Patient-level ICC: %.3f\n", icc_patient))
cat(sprintf(
  "Plain English: about %.0f%% of the total variation in blood pressure lies\n",
  100 * icc_cluster))
cat(sprintf(
  "  between clinics and patients rather than within a patient's own visits\n"))
cat(sprintf(
  "  (clinic alone accounts for roughly %.0f%%).\n", 100 * icc_clinic))

# --- (c) Compare with ordinary lm() that ignores clustering -----------------
m_lm <- lm(sbp ~ visit + treatment, data = bp)
lm_coefs <- summary(m_lm)$coefficients
se_trt_lm <- lm_coefs["treatment", "Std. Error"]

cat("\n=== (c) Ordinary lm() ignoring clustering ===\n")
cat(sprintf("Treatment estimate (lm)  : %.3f\n", lm_coefs["treatment", "Estimate"]))
cat(sprintf("Treatment SE (lm)        : %.3f\n", se_trt_lm))
cat(sprintf("Treatment SE (mixed)     : %.3f\n", se_trt))
cat(sprintf("Ratio mixed/lm           : %.2f\n", se_trt / se_trt_lm))
cat("\nWhich is larger, and why it matters:\n")
cat("  The MIXED-model SE is larger. Treatment is a BETWEEN-PATIENT variable\n")
cat("  (constant across a patient's 5 visits), so the real amount of\n")
cat("  independent information about treatment is roughly the number of\n")
cat("  patients, not the 1000 rows. Ordinary lm() pretends all 1000 rows are\n")
cat("  independent, so it reports an SE that is too small -- an over-optimistic\n")
cat("  confidence interval and p-value. Ignoring the clustering would make the\n")
cat("  treatment effect look more certain than the data can support.\n")
