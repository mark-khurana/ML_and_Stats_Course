# =============================================================================
# Chapter 9b, Exercise 1: Permutation importance and the wrong-reason model
# Add a leaky variable (discharge_disposition) and see it dominate the ranking.
# =============================================================================

library(ranger)
library(dplyr)
library(tibble)
library(ggplot2)
library(pROC)

# --- Re-create the readmission data ------------------------------------------
set.seed(42)
n <- 1500
dat <- tibble(
  age              = rnorm(n, 68, 12),
  length_of_stay   = rpois(n, 5) + 1,
  num_comorbidities= rpois(n, 3),
  prior_admissions = rpois(n, 1),
  discharge_hb     = rnorm(n, 11, 2),
  discharge_creat  = rlnorm(n, 0.2, 0.5)
)
# True risk depends mostly on prior admissions and comorbidities
lin <- -3 + 0.45 * dat$prior_admissions + 0.20 * dat$num_comorbidities +
        0.015 * (dat$age - 68) - 0.05 * dat$discharge_hb
readmit_num <- rbinom(n, 1, plogis(lin))
dat$readmit <- factor(readmit_num, labels = c("No", "Yes"))

# --- Add a LEAKY variable: discharge_disposition -----------------------------
# Where the patient was discharged TO is recorded AFTER the clinical course has
# played out. Patients who go on to be readmitted were far more likely to have
# been sent to a skilled-nursing facility / rehab (a marker that the team already
# judged them frail), whereas those who did well went home. The disposition is a
# CONSEQUENCE of the underlying risk (and of information not available at the
# moment we would actually make a prediction), not a cause of readmission.
disp <- ifelse(
  readmit_num == 1,
  sample(c("SNF", "Rehab", "Home"), n, replace = TRUE, prob = c(0.55, 0.30, 0.15)),
  sample(c("SNF", "Rehab", "Home"), n, replace = TRUE, prob = c(0.08, 0.12, 0.80))
)
dat$discharge_disposition <- factor(disp)

# --- Re-fit the random forest with the leaky variable included ---------------
rf <- ranger(readmit ~ ., data = dat, probability = TRUE,
             num.trees = 500, seed = 42)

# --- Permutation importance, computed directly (as in the chapter) -----------
auc_for <- function(prob) {
  as.numeric(pROC::auc(dat$readmit, prob,
                       levels = c("No", "Yes"), direction = "<"))
}
baseline_auc <- auc_for(predict(rf, dat)$predictions[, "Yes"])

predictors <- setdiff(names(dat), "readmit")
set.seed(1)
importance <- sapply(predictors, function(v) {
  drops <- replicate(10, {
    shuffled <- dat
    shuffled[[v]] <- sample(shuffled[[v]]) # break this variable only
    baseline_auc - auc_for(predict(rf, shuffled)$predictions[, "Yes"])
  })
  mean(drops)
})

imp_df <- tibble(Variable = predictors, Importance = importance) |>
  arrange(desc(Importance))

cat("=== Permutation importance (drop in AUC when shuffled) ===\n")
print(as.data.frame(imp_df), row.names = FALSE)

leak_rank <- which(imp_df$Variable == "discharge_disposition")
cat(sprintf("\nBaseline AUC (with leak): %.3f\n", baseline_auc))
cat(sprintf("discharge_disposition ranks #%d of %d predictors.\n",
            leak_rank, nrow(imp_df)))

# --- Plot (saved to a temp file, no display needed) --------------------------
p <- ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#2E86AB") +
  coord_flip() +
  labs(x = NULL, y = "Drop in AUC when shuffled",
       title = "Permutation importance with a leaky variable") +
  theme_minimal(base_size = 13)
out <- file.path(tempdir(), "ch09b_ex1_importance.png")
ggsave(out, p, width = 7, height = 4, dpi = 100)
cat("Plot saved to:", out, "\n")

# =============================================================================
# INTERPRETATION
#
# 1) Where does the leaky variable rank?
#    discharge_disposition rockets to the TOP of the permutation-importance
#    ranking -- shuffling it collapses the AUC far more than shuffling any
#    genuine clinical predictor. It looks like the single "best" variable.
#
# 2) Why is a high rank here a WARNING, not a discovery?
#    Permutation importance only tells you how much the MODEL leans on a
#    variable to reproduce the observed outcome -- not whether that variable is
#    usable or causal. discharge_disposition is recorded at (or after) the very
#    event we are trying to predict and is a downstream MARKER of the risk the
#    clinical team already perceived. A model that leans on it will look
#    brilliant in development and then fail in deployment, because at true
#    prediction time (before discharge decisions are finalised) the value is
#    unavailable or not yet meaningful. A variable that dominates the ranking
#    for no plausible clinical reason should trigger a hunt for leakage, not a
#    celebration. "Too good to be true" usually is.
#
# 3) Real-world variables that behave like this leak:
#    - Discharge destination / disposition codes (as here).
#    - Palliative-care or hospice referral flags.
#    - "Do not resuscitate" orders entered late in the stay.
#    - Number of consults, ICU transfers, or rapid-response calls during the
#      index stay (consequences of deterioration).
#    - Medications started for complications (e.g. vasopressors, broad-spectrum
#      antibiotics) that postdate the predictor cut-off.
#    - Billing/DRG codes finalised after the outcome is known.
#    - Timestamps or ward names that proxy for how sick a patient was.
#    Each is associated with the outcome because it is a CONSEQUENCE of the
#    illness, not a baseline predictor available when the model must act.
# =============================================================================
