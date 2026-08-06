# =============================================================================
# Chapter 6b, Exercise 1: Spot the clustering (conceptual)
# Identify clusters, longitudinal structure, and the risk of ignoring both.
# =============================================================================

# This is a conceptual exercise. The answer is written as structured comments.
# For each study, state (a) the clusters, (b) whether the data are also
# longitudinal, and (c) what goes wrong if you use ordinary regression that
# ignores the clustering.

# -----------------------------------------------------------------------------
# (a) A trial of a new inhaler recruiting asthma patients from 15 GP practices.
# -----------------------------------------------------------------------------
#   Clusters:     The 15 GP practices. Patients are nested within practices;
#                 patients from the same practice share staff, prescribing
#                 habits, catchment population and case mix, so their outcomes
#                 resemble one another more than outcomes from other practices.
#   Longitudinal? Not as described. Each patient contributes (presumably) a
#                 single outcome, so the data are CLUSTERED but not longitudinal.
#                 (It would become longitudinal if each patient were measured
#                 repeatedly over time.)
#   Ignoring it:  Ordinary regression treats all patients as independent and so
#                 overstates the amount of information. Because patients within
#                 a practice are positively correlated (ICC > 0), the standard
#                 error on the inhaler effect is too small, the confidence
#                 interval too narrow and the p-value too small -- you risk
#                 declaring the inhaler effective when the evidence does not
#                 support it (inflated type-I error). The fix is a random
#                 intercept for practice: outcome ~ treatment + (1 | practice).

# -----------------------------------------------------------------------------
# (b) A study measuring fasting glucose monthly for a year in 200 diabetics.
# -----------------------------------------------------------------------------
#   Clusters:     The 200 patients. The repeated monthly readings are nested
#                 within each patient; a patient who runs high tends to run high
#                 at every visit.
#   Longitudinal? YES. Each patient has 12 monthly measurements ordered in
#                 time (200 x 12 = 2400 rows), so this is a classic repeated-
#                 measures / longitudinal design.
#   Ignoring it:  You have 2400 rows but NOT 2400 independent observations --
#                 you effectively have ~200 patients' worth of independent
#                 information. Ordinary regression would give standard errors
#                 that are far too small and p-values that are far too
#                 impressive, and it could not separate within-patient change
#                 over time from between-patient differences. A mixed model with
#                 a random intercept (and possibly a random slope for time) per
#                 patient handles this correctly.

# -----------------------------------------------------------------------------
# (c) A multi-site ICU study of a sepsis bundle across 6 hospitals,
#     one outcome per patient.
# -----------------------------------------------------------------------------
#   Clusters:     The 6 hospitals. Patients are nested within hospital and share
#                 protocols, staffing and case mix.
#   Longitudinal? NO. There is a single outcome per patient, so the data are
#                 cross-sectional but clustered.
#   Ignoring it:  Patients within the same hospital are correlated, so ordinary
#                 regression again gives standard errors that are too small and
#                 overstates the significance of the sepsis-bundle effect.
#                 A caution specific to this study: with only 6 clusters there
#                 are very few hospitals from which to estimate the between-
#                 hospital variance, so a random-effects estimate of that
#                 variance will itself be imprecise (some analysts prefer to
#                 treat so few sites as fixed effects, or to use a cluster-robust
#                 approach). The clustering must still be accounted for either
#                 way -- ignoring it is not an option.

print("Exercise 1 is conceptual -- see the commented answers above.")
