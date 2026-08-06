# =============================================================================
# Chapter 6c, Exercise 1: Classify the mechanism (conceptual)
# Decide MCAR / MAR / MNAR for three clinical scenarios, with justification.
# =============================================================================

import pandas as pd

# This exercise is conceptual: the answers are reasoned below as structured
# comments. No data analysis is required, but we print the answers so the
# script produces visible output when run.

# -----------------------------------------------------------------------------
# (a) A weighing scale was out of service for two weeks; everyone seen in that
#     window has missing weight.
#
#     ANSWER: MCAR (Missing Completely At Random).
#     JUSTIFICATION: The equipment failure is external to the patients -- it has
#     nothing to do with their weight or any of their characteristics, so the
#     missing records are a genuinely random subset of the cohort.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (b) Depression-score questionnaires are returned less often by male patients,
#     but among men the chance of returning is unrelated to the score.
#
#     ANSWER: MAR (Missing At Random).
#     JUSTIFICATION: Missingness depends only on sex, an OBSERVED variable, and
#     is unrelated to the depression score itself once sex is accounted for, so
#     the mechanism is recoverable by conditioning on sex in the imputation.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (c) Patients with the most severe symptoms are too unwell to complete a
#     quality-of-life survey, and severity is not otherwise recorded.
#
#     ANSWER: MNAR (Missing Not At Random).
#     JUSTIFICATION: The probability of missingness depends on the unmeasured
#     severity, which is exactly what the missing quality-of-life value
#     reflects, and no observed variable captures it -- so the missingness
#     depends on the missing value itself.
# -----------------------------------------------------------------------------

answers = pd.DataFrame({
    "scenario": ["(a) scale out of service",
                 "(b) fewer returns from men, unrelated to score",
                 "(c) sickest skip QoL survey, severity unrecorded"],
    "mechanism": ["MCAR", "MAR", "MNAR"],
})

print("=== Exercise 1: Missingness mechanism classification ===\n")
print(answers.to_string(index=False))
print("\nKey idea: the mechanism is defined by what the CHANCE of being missing")
print("depends on -- nothing (MCAR), observed variables (MAR), or the missing")
print("value itself (MNAR).")
