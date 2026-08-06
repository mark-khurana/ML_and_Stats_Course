# =============================================================================
# Chapter 8, Exercise 3: Interpreting a Clinical Prediction Model
# Conceptual exercise: write answers as comments.
# =============================================================================

# =============================================================================
# Part 1: Non-technical summary for a hospital quality committee
# =============================================================================
#
# "We developed a computer-based prediction model that estimates each patient's
# risk of being readmitted to the hospital within 30 days of discharge. The
# model uses information routinely collected during hospitalisation -- such as
# lab values, age, prior admissions, and existing medical conditions -- to
# assign each patient a risk score between 0% and 100%. In cross-validated
# testing, the model correctly distinguished between patients who were and
# were not readmitted approximately [AUC]% of the time. This tool could help
# care teams focus discharge planning resources on the patients at highest
# risk, potentially reducing readmission rates and associated penalties."

# =============================================================================
# Part 2: Explaining variable importance to a clinical audience
# =============================================================================
#
# "The model identified 'number of prior admissions' and 'discharge creatinine'
# as the two variables that contribute most to the model's predictions. This
# means that knowing these values provides the most useful information for
# distinguishing patients who will be readmitted from those who will not.
#
# WHAT THIS MEANS:
# - Patients with more prior admissions tend to have higher predicted risk.
# - Patients with elevated discharge creatinine (indicating impaired kidney
#   function) also tend to have higher predicted risk.
# - These findings align with clinical intuition: patients with a history of
#   recurrent hospitalisations and those with renal impairment are known to
#   be at elevated risk.
#
# WHAT THIS DOES NOT MEAN:
# - Variable importance does NOT imply causation. We cannot say that reducing
#   creatinine at discharge will reduce readmission risk. The model identifies
#   associations, not causes.
# - A variable with high importance may be a proxy for something else. For
#   example, 'prior admissions' may reflect underlying disease severity,
#   social determinants, or healthcare access patterns rather than being
#   a direct cause of readmission.
# - We should not intervene on these variables based on importance alone.
#   Clinical trials or causal inference methods would be needed to establish
#   whether modifying these factors actually changes outcomes."

# =============================================================================
# Part 3: Validation strategy before clinical deployment
# =============================================================================
#
# PROPOSED VALIDATION PLAN:
#
# 1. Temporal validation: Test the model on data from a time period AFTER the
#    training data (e.g., train on 2020-2022, validate on 2023-2024). This
#    tests whether the model's performance holds over time, as clinical
#    practices and patient populations may shift.
#
# 2. External validation: Apply the model to data from a different hospital
#    system. A model developed at one institution may not generalise to
#    another due to differences in patient demographics, coding practices,
#    discharge protocols, and local disease patterns.
#
# 3. Subgroup analysis: Evaluate performance across key demographic groups
#    (age, sex, race/ethnicity, insurance status). A model that performs
#    well overall but poorly for specific populations could worsen existing
#    health disparities.
#
# 4. Calibration assessment: Verify that predicted probabilities match
#    observed readmission rates. A model that says "30% risk" should be
#    right about 30% of the time across all risk levels.
#
# 5. Prospective pilot: Before full deployment, run the model in parallel
#    alongside current practice (silent mode) to monitor performance in
#    real-time without affecting clinical decisions.
#
# RISKS OF SKIPPING EXTERNAL VALIDATION:
# - The model may be overfit to idiosyncrasies of the development data
#   (specific EMR system, local coding conventions, patient mix).
# - Performance reported from internal validation (even cross-validation)
#   tends to be optimistically biased.
# - Deploying an unvalidated model could misallocate resources, either
#   missing high-risk patients (false negatives) or overwhelming care
#   teams with false alarms (false positives).
# - Regulatory and ethical risks: deploying a model without adequate
#   validation may violate institutional policies and could cause patient
#   harm.

cat("This exercise is conceptual. See the comments in this file for the\n")
cat("complete answers to all three parts.\n")
