# =============================================================================
# Chapter 6b, Exercise 3: Random intercept vs random slope
# Extend the Exercise-2 model with a clinic-specific visit slope and compare
# the random-intercept and random-slope models with a likelihood-ratio test.
# =============================================================================

library(tidyverse)   # data wrangling and simulation
library(lme4)        # lmer() and anova() likelihood-ratio test
has_lmerTest <- requireNamespace("lmerTest", quietly = TRUE)
if (has_lmerTest) suppressMessages(library(lmerTest))

set.seed(42)

# --- Recreate the chapter's BP dataset with a binary treatment --------------
n_clinic  <- 8
n_patient <- 25
n_visit   <- 5

clinic_effect  <- rnorm(n_clinic, 0, 6)
patient_effect <- rnorm(n_clinic * n_patient, 0, 8)
treatment_by_patient <- rbinom(n_clinic * n_patient, 1, 0.5)

bp <- expand_grid(
  clinic  = 1:n_clinic,
  patient = 1:n_patient,
  visit   = 1:n_visit
) |>
  mutate(
    patient_id = interaction(clinic, patient),
    pid_index  = as.integer(patient_id),
    treatment  = treatment_by_patient[pid_index],
    # NOTE: the data are simulated with a SINGLE shared visit slope (-1.5),
    # i.e. no genuine clinic-to-clinic variation in the trend.
    sbp = 135 + clinic_effect[clinic] + patient_effect[pid_index] +
          -1.5 * (visit - 1) + -5 * treatment + rnorm(n(), 0, 5)
  )

# --- Random-intercept model (from Exercise 2) -------------------------------
m_ri <- lmer(sbp ~ visit + treatment + (1 | clinic/patient_id), data = bp)

# --- Random-slope model: let the visit trend vary by clinic -----------------
# (visit | clinic) gives each clinic its own intercept AND its own visit slope.
m_rs <- lmer(sbp ~ visit + treatment + (visit | clinic) + (1 | clinic:patient_id),
             data = bp)

cat("=== Random-slope model summary ===\n")
print(summary(m_rs))

# --- (a) Likelihood-ratio test comparing the two models ---------------------
# anova() on lmer objects refits both models with ML and performs the LRT.
cat("\n=== (a) Likelihood-ratio test (random intercept vs random slope) ===\n")
lrt <- anova(m_ri, m_rs)
print(lrt)

p_lrt <- lrt$`Pr(>Chisq)`[2]

# --- (b) Interpretation ------------------------------------------------------
cat("\n=== (b) Does a clinic-specific trend improve the model? ===\n")
cat(sprintf("LRT p-value: %.4g\n", p_lrt))
if (p_lrt < 0.05) {
  cat("The random slope significantly improves fit: the visit trend genuinely\n")
  cat("differs between clinics.\n")
} else {
  cat("The random slope does NOT significantly improve fit (p > 0.05).\n")
  cat("Clinically: there is no evidence that blood pressure changes at\n")
  cat("different rates across clinics -- the single shared downward trend is\n")
  cat("adequate. This is expected, because the data were simulated with one\n")
  cat("common visit slope. The lesson: add random slopes only when the data\n")
  cat("(and clinical sense) support them; needless complexity can also trigger\n")
  cat("convergence warnings.\n")
}
