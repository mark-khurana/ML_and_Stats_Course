# =============================================================================
# Chapter 8b, Exercise 3: Critical Appraisal of a Deep Learning Study
# Evaluate a DL paper against the CLAIM or TRIPOD+AI checklist.
# This is a conceptual/guided exercise -- the template below provides
# the framework for appraising any DL study.
# =============================================================================

# =============================================================================
# INSTRUCTIONS:
# Find a recent (2024 or later) paper applying deep learning to a clinical
# task in your area of interest. Use this template to evaluate it.
# =============================================================================

# =============================================================================
# Question 1: Was the model externally validated?
# =============================================================================
#
# Look for:
# - Was the model tested on data from a DIFFERENT institution, time period,
#   or geographic region than the training data?
# - If yes, how did external performance compare to internal (e.g., was there
#   a drop in AUC)?
#
# Example answer:
# "The model was validated on data from Hospital B after training on Hospital A.
#  Internal AUC was 0.92; external AUC dropped to 0.84 -- a 0.08 decrease.
#  This is consistent with the systematic review finding that 81% of DL models
#  show decreased accuracy on external datasets."
#
# Red flag: If NO external validation was performed, the results should be
# treated as preliminary.

# =============================================================================
# Question 2: Were subgroup analyses reported?
# =============================================================================
#
# Look for:
# - Performance broken down by age, sex, race/ethnicity
# - Any mention of fairness or equity analysis
# - Performance in clinically important subgroups
#
# Example answer:
# "The paper reported AUC by sex (male: 0.90, female: 0.87) but did not
#  report performance by race/ethnicity or age group. Given the known issue
#  of fairness non-transferability across sites, this is a significant
#  omission."

# =============================================================================
# Question 3: Was the model compared to a simpler baseline?
# =============================================================================
#
# Look for:
# - Comparison against logistic regression, random forest, or XGBoost
# - If the DL model only marginally outperforms the baseline, the added
#   complexity may not be justified
#
# Example answer:
# "The paper compared DL (AUC 0.91) to logistic regression (AUC 0.85) and
#  random forest (AUC 0.88). The improvement over RF is modest (0.03)."

# =============================================================================
# Question 4: Were training data, code, and model weights shared?
# =============================================================================
#
# Look for:
# - Public dataset or data sharing agreement
# - Code repository (GitHub, GitLab)
# - Pretrained model weights available
#
# Example answer:
# "Code was shared on GitHub. Training data is from a private hospital
#  system. Model weights were not released."

# =============================================================================
# Question 5: How close is this model to clinical deployment?
# =============================================================================
#
# Consider:
# - Has it been externally validated across multiple sites?
# - Has it been tested prospectively?
# - Has a clinical workflow been designed?
# - Has a regulatory pathway been identified?
# - Has a monitoring plan been described?
#
# Example answer:
# "This model is at an early research stage. Before clinical deployment,
#  I would want to see:
#  (1) multi-site external validation,
#  (2) a prospective pilot study,
#  (3) subgroup analysis across demographics,
#  (4) calibration assessment,
#  (5) a monitoring plan for model drift."

print("This exercise is a guided template for critical appraisal.")
print("See the comments for the framework to evaluate any DL paper.")
print("Students should find their own paper and fill in the answers.")
