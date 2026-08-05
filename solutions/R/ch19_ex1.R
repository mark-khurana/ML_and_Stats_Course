# =============================================================================
# Chapter 19 - Exercise 4: TRIPOD+AI Checklist (Conceptual)
# Completed for Capstone 1: Cardiovascular Risk Prediction Model
# =============================================================================

# This exercise asks you to complete the TRIPOD+AI checklist for one of the
# capstone projects. Below is a completed checklist for Capstone 1:
# "Development and external validation of a cardiovascular risk prediction
# model using Framingham methodology, validated in NHANES data."

checklist <- data.frame(
  Item = c(
    "1. Title",
    "2. Abstract",
    "3a. Background/rationale",
    "3b. Objectives",
    "4a. Source of data",
    "4b. Dates of study",
    "5a. Key eligibility criteria",
    "5b. Treatments received",
    "6a. Outcome definition",
    "6b. Outcome timing",
    "6c. Blinding of outcome",
    "7. Predictors",
    "8. Sample size",
    "9. Missing data",
    "10a. Statistical analysis: model development",
    "10b. Model specification",
    "10c. Predictor selection",
    "10d. Model performance measures",
    "11. Risk groups",
    "12a. Internal validation",
    "12b. External validation",
    "13. Fairness assessment (AI-specific)",
    "14. Results: participants",
    "15. Results: model development",
    "16. Results: model performance",
    "17. Results: model updating",
    "18. Discussion: interpretation",
    "19. Discussion: limitations",
    "20. Discussion: implications",
    "21. Supplementary: code and data",
    "AI-1. Data preprocessing",
    "AI-2. Hyperparameter tuning",
    "AI-3. Model explainability",
    "AI-4. Software and hardware"
  ),
  Section = c(
    "Title page",
    "Abstract",
    "Introduction",
    "Introduction",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Methods",
    "Results",
    "Results",
    "Results",
    "Results",
    "Discussion",
    "Discussion",
    "Discussion",
    "Supplement",
    "Methods/Supplement",
    "Methods/Supplement",
    "Results/Supplement",
    "Methods"
  ),
  How_Addressed = c(
    # 1. Title
    "'Development and external validation of a logistic regression model
    for predicting 10-year cardiovascular disease risk: a Framingham-NHANES
    study.' Identifies study type (development + external validation) and
    modelling approach.",

    # 2. Abstract
    "Structured abstract with: objective, study design, data sources,
    outcome (10-year CVD), predictors, sample sizes, C-statistic and
    calibration results for both development and validation cohorts.",

    # 3a. Background
    "Existing CVD risk models (Framingham, QRISK, SCORE2) and their
    limitations. Gap: need for updated validation in contemporary US
    population.",

    # 3b. Objectives
    "To develop a 10-year CVD risk prediction model using Framingham
    data and externally validate it in NHANES 2017-2020.",

    # 4a. Source of data
    "Development: Framingham Heart Study teaching dataset (riskCommunicator
    R package). Validation: NHANES 2017-2020 (nhanesA package). Both are
    publicly available.",

    # 4b. Dates
    "Framingham: original cohort with follow-up through 2005. NHANES:
    2017-2020 survey cycles.",

    # 5a. Eligibility
    "Adults aged 30-74, free of CVD at baseline, with complete data on
    key predictors. Exclude prior MI, stroke, heart failure.",

    # 5b. Treatments
    "Not applicable (prediction model, not treatment comparison).
    However, blood pressure treatment status is included as a predictor.",

    # 6a. Outcome definition
    "10-year cardiovascular event: composite of MI, coronary death,
    stroke, or heart failure requiring hospitalisation.",

    # 6b. Outcome timing
    "10-year follow-up from baseline examination. Censored at death
    from non-CVD causes or loss to follow-up.",

    # 6c. Blinding
    "Not applicable for retrospective analysis. Outcome ascertainment
    in Framingham was by adjudication committee blinded to risk factors.",

    # 7. Predictors
    "Pre-specified based on established Framingham model: age, sex, total
    cholesterol, HDL cholesterol, systolic blood pressure, blood pressure
    treatment status, smoking, diabetes. No data-driven selection.",

    # 8. Sample size
    "Sample size justified using Riley et al. (2020) criteria via
    pmsampsize package: 8 predictors, anticipated outcome prevalence 10%,
    target R-squared 0.15, requiring minimum ~800 events.",

    # 9. Missing data
    "Development cohort: <5% missing. Validation: up to 12% for some
    laboratory values. Handled by multiple imputation (MICE, 50 datasets,
    20 iterations). Complete-case sensitivity analysis performed.",

    # 10a. Model development
    "Logistic regression for 10-year risk. Pre-specified predictors,
    no variable selection. Continuous predictors modelled with restricted
    cubic splines (4 knots for age and SBP; 3 knots for cholesterol)
    based on prior literature.",

    # 10b. Model specification
    "Full model equation provided in supplementary materials. Coefficients
    table with 95% CIs in main text.",

    # 10c. Predictor selection
    "All predictors pre-specified from clinical knowledge. No stepwise
    selection. Rationale provided for each predictor.",

    # 10d. Performance measures
    "Discrimination: C-statistic (Harrell's concordance) with 95% CI.
    Calibration: calibration plot (observed vs predicted), calibration
    slope, calibration-in-the-large. Both reported for development
    (bootstrap-corrected) and external validation.",

    # 11. Risk groups
    "Patients categorised into low (<5%), moderate (5-10%), high (10-20%),
    and very high (>20%) 10-year CVD risk groups. Classification tables
    provided.",

    # 12a. Internal validation
    "500 bootstrap resamples for optimism-corrected C-statistic and
    calibration slope. Apparent and corrected performance reported.",

    # 12b. External validation
    "Model applied to NHANES data without recalibration. C-statistic
    and calibration plot in external data. Recalibration of intercept
    also explored.",

    # 13. Fairness
    "C-statistic and calibration reported separately by sex (male/female)
    and by race/ethnicity (non-Hispanic White, non-Hispanic Black,
    Hispanic, Asian). Differences in performance >0.05 C-statistic
    flagged for discussion.",

    # 14. Participants
    "Flow diagram showing sample selection: patients screened, excluded
    (with reasons), and included in development and validation cohorts.
    Table 1 with baseline characteristics stratified by CVD status.",

    # 15. Model development
    "Full regression table with coefficients, standard errors, odds
    ratios, 95% CIs, and p-values. Presented as Table 2.",

    # 16. Model performance
    "Table 3: discrimination (C-statistic) and calibration metrics for
    development (apparent and corrected) and external validation.
    Figures: calibration plots (development and validation), decision
    curve analysis.",

    # 17. Model updating
    "Recalibration of intercept in NHANES improved calibration-in-the-
    large from X to Y. Full model updating was explored but not
    recommended due to limited improvement.",

    # 18. Interpretation
    "Model performance compared with published Framingham, QRISK3, and
    PCE models. Clinical utility assessed via decision curve analysis
    across clinically relevant threshold probabilities (5-20%).",

    # 19. Limitations
    "Limitations discussed: (1) Framingham cohort may not represent
    contemporary diverse populations; (2) NHANES validation limited by
    shorter follow-up and self-reported outcomes; (3) Missing data
    assumptions; (4) Ecological fallacy in subgroup performance
    assessment.",

    # 20. Implications
    "Clinical implications for CVD risk assessment. Recommendation for
    recalibration before use in non-US populations. Need for prospective
    validation.",

    # 21. Code and data
    "R code provided via GitHub repository [URL]. Framingham data
    available via riskCommunicator package. NHANES data publicly available.
    Reproducible via renv for package management.",

    # AI-1. Data preprocessing
    "Not applicable (logistic regression, not ML/AI). However, data
    cleaning steps, outlier handling, and transformation of predictors
    (restricted cubic splines) described in supplement.",

    # AI-2. Hyperparameter tuning
    "Not applicable for logistic regression. Number of knots for RCS
    pre-specified based on Harrell's recommendations.",

    # AI-3. Model explainability
    "Not applicable (logistic regression is inherently interpretable).
    Coefficient interpretation provided. Nomogram included for clinical
    use.",

    # AI-4. Software
    "R version 4.4.1. Packages: rms (v6.8), survival (v3.7), mice
    (v3.16), gtsummary (v1.7), pmsampsize (v1.1), riskCommunicator
    (v1.0), nhanesA (v1.1). Full renv.lock file in repository."
  ),

  Applicable = c(
    "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
    "Partially (no treatment comparison)",
    "Yes", "Yes",
    "Partially (retrospective)",
    "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
    "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes",
    "No (not ML)", "No (not ML)", "No (not ML)", "Yes"
  ),

  stringsAsFactors = FALSE
)

# Print the checklist
cat("=" * 70, "\n")
cat("TRIPOD+AI Checklist - Capstone 1: CV Risk Prediction Model\n")
cat("=" * 70, "\n\n")

for (i in 1:nrow(checklist)) {
  cat("ITEM:", checklist$Item[i], "\n")
  cat("  Section:", checklist$Section[i], "\n")
  cat("  Applicable:", checklist$Applicable[i], "\n")
  cat("  How addressed:", trimws(checklist$How_Addressed[i]), "\n\n")
}

# Summary
cat("\n=== SUMMARY ===\n")
n_applicable <- sum(checklist$Applicable == "Yes")
n_partial <- sum(grepl("Partially", checklist$Applicable))
n_na <- sum(checklist$Applicable == "No (not ML)")

cat("Items fully addressed:", n_applicable, "\n")
cat("Items partially applicable:", n_partial, "\n")
cat("Items not applicable (ML-specific):", n_na, "\n")
cat("Total items:", nrow(checklist), "\n")

cat("\nNote: AI-specific items (AI-1 through AI-3) are not applicable because\n")
cat("this capstone uses logistic regression, not machine learning. If a\n")
cat("gradient boosted model or neural network were used instead, these\n")
cat("items would need to be addressed with details on feature engineering,\n")
cat("hyperparameter search strategy, and SHAP/LIME explanations.\n")
