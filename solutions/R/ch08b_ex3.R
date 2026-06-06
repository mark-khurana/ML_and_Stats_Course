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
# treated as preliminary. Internal CV alone is insufficient for clinical claims.

# =============================================================================
# Question 2: Were subgroup analyses reported?
# =============================================================================
#
# Look for:
# - Performance broken down by age, sex, race/ethnicity
# - Any mention of fairness or equity analysis
# - Performance in clinically important subgroups (e.g., patients with
#   comorbidities, different disease severity)
#
# Example answer:
# "The paper reported AUC by sex (male: 0.90, female: 0.87) but did not
#  report performance by race/ethnicity or age group. Given the known issue
#  of fairness non-transferability across sites (Nature Medicine 2024),
#  this is a significant omission."

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
#  random forest (AUC 0.88). The improvement over RF is modest (0.03),
#  raising questions about whether the DL model's added complexity and
#  reduced interpretability are justified."

# =============================================================================
# Question 4: Were training data, code, and model weights shared?
# =============================================================================
#
# Look for:
# - Public dataset or data sharing agreement
# - Code repository (GitHub, GitLab)
# - Pretrained model weights available for download
# - FAIR data principles
#
# Example answer:
# "Code was shared on GitHub. The training data is from a private hospital
#  system and is not publicly available, though the authors describe a
#  data sharing agreement. Model weights were not released."

# =============================================================================
# Question 5: How close is this model to clinical deployment?
# =============================================================================
#
# Consider:
# - Has it been externally validated across multiple sites?
# - Has it been tested prospectively (not just retrospectively)?
# - Has a clinical workflow been designed for how it would be used?
# - Has a regulatory pathway been identified (e.g., FDA 510(k), EU MDR)?
# - Has a monitoring plan for post-deployment performance been described?
# - What are the potential failure modes and harms?
#
# Example answer:
# "This model is at an early research stage. While the internal results are
#  promising, the model has been validated at only one external site with a
#  modest sample. Before clinical deployment, I would want to see:
#  (1) multi-site external validation,
#  (2) a prospective pilot study in clinical workflow,
#  (3) subgroup analysis across demographics,
#  (4) calibration assessment,
#  (5) a monitoring plan for detecting model drift over time."

cat("This exercise is a guided template for critical appraisal.\n")
cat("See the comments for the framework to evaluate any DL paper.\n")
cat("Students should find their own paper and fill in the answers.\n")
