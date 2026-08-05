# =============================================================================
# Chapter 19 - Exercise 4: TRIPOD+AI Checklist (Conceptual)
# Completed for Capstone 1: Cardiovascular Risk Prediction Model
# =============================================================================

# This exercise asks you to complete the TRIPOD+AI checklist for one of the
# capstone projects. Below is a completed checklist for Capstone 1:
# "Development and external validation of a cardiovascular risk prediction
# model using Framingham methodology, validated in NHANES data."

checklist = [
    {
        "item": "1. Title",
        "section": "Title page",
        "applicable": "Yes",
        "how": (
            "'Development and external validation of a logistic regression "
            "model for predicting 10-year cardiovascular disease risk: a "
            "Framingham-NHANES study.' Identifies study type (development + "
            "external validation) and modelling approach."
        )
    },
    {
        "item": "2. Abstract",
        "section": "Abstract",
        "applicable": "Yes",
        "how": (
            "Structured abstract with: objective, study design, data sources, "
            "outcome (10-year CVD), predictors, sample sizes, C-statistic and "
            "calibration results for both development and validation cohorts."
        )
    },
    {
        "item": "3a. Background/rationale",
        "section": "Introduction",
        "applicable": "Yes",
        "how": (
            "Existing CVD risk models (Framingham, QRISK, SCORE2) and their "
            "limitations. Gap: need for updated validation in contemporary "
            "US population."
        )
    },
    {
        "item": "3b. Objectives",
        "section": "Introduction",
        "applicable": "Yes",
        "how": (
            "To develop a 10-year CVD risk prediction model using Framingham "
            "data and externally validate it in NHANES 2017-2020."
        )
    },
    {
        "item": "4a. Source of data",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Development: Framingham Heart Study teaching dataset "
            "(riskCommunicator R package). Validation: NHANES 2017-2020 "
            "(nhanesA package). Both publicly available."
        )
    },
    {
        "item": "4b. Dates of study",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Framingham: original cohort with follow-up through 2005. "
            "NHANES: 2017-2020 survey cycles."
        )
    },
    {
        "item": "5a. Key eligibility criteria",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Adults aged 30-74, free of CVD at baseline, with complete data "
            "on key predictors. Exclude prior MI, stroke, heart failure."
        )
    },
    {
        "item": "5b. Treatments received",
        "section": "Methods",
        "applicable": "Partially",
        "how": (
            "Not applicable (prediction model, not treatment comparison). "
            "However, BP treatment status included as a predictor."
        )
    },
    {
        "item": "6a. Outcome definition",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "10-year CVD event: composite of MI, coronary death, stroke, "
            "or heart failure requiring hospitalisation."
        )
    },
    {
        "item": "6b. Outcome timing",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "10-year follow-up from baseline examination. Censored at "
            "death from non-CVD causes or loss to follow-up."
        )
    },
    {
        "item": "6c. Blinding of outcome",
        "section": "Methods",
        "applicable": "Partially",
        "how": (
            "Retrospective analysis. Framingham outcome adjudication was "
            "by committee blinded to risk factors."
        )
    },
    {
        "item": "7. Predictors",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Pre-specified: age, sex, total cholesterol, HDL cholesterol, "
            "systolic BP, BP treatment, smoking, diabetes. No data-driven "
            "selection."
        )
    },
    {
        "item": "8. Sample size",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Justified using Riley et al. (2020) criteria via pmsampsize: "
            "8 predictors, prevalence 10%, target R^2 0.15, requiring "
            "minimum ~800 events."
        )
    },
    {
        "item": "9. Missing data",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Development: <5% missing. Validation: up to 12%. Handled by "
            "MICE (50 datasets, 20 iterations). Complete-case sensitivity "
            "analysis performed."
        )
    },
    {
        "item": "10a. Model development",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Logistic regression for 10-year risk. Pre-specified predictors. "
            "Continuous predictors with restricted cubic splines."
        )
    },
    {
        "item": "10b. Model specification",
        "section": "Methods",
        "applicable": "Yes",
        "how": "Full model equation in supplementary materials."
    },
    {
        "item": "10c. Predictor selection",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "All predictors pre-specified from clinical knowledge. "
            "No stepwise selection."
        )
    },
    {
        "item": "10d. Performance measures",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Discrimination: C-statistic with 95% CI. Calibration: "
            "calibration plot, calibration slope, calibration-in-the-large."
        )
    },
    {
        "item": "11. Risk groups",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Low (<5%), moderate (5-10%), high (10-20%), very high (>20%) "
            "10-year CVD risk categories."
        )
    },
    {
        "item": "12a. Internal validation",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "500 bootstrap resamples for optimism-corrected C-statistic "
            "and calibration slope."
        )
    },
    {
        "item": "12b. External validation",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Model applied to NHANES without recalibration. C-statistic "
            "and calibration reported. Recalibration of intercept explored."
        )
    },
    {
        "item": "13. Fairness assessment (AI-specific)",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Performance reported by sex and race/ethnicity. Differences "
            ">0.05 C-statistic flagged."
        )
    },
    {
        "item": "14. Results: participants",
        "section": "Results",
        "applicable": "Yes",
        "how": (
            "Flow diagram with sample selection. Table 1 stratified by "
            "CVD event status."
        )
    },
    {
        "item": "15. Results: model development",
        "section": "Results",
        "applicable": "Yes",
        "how": (
            "Full regression table with coefficients, ORs, 95% CIs, "
            "and p-values."
        )
    },
    {
        "item": "16. Results: model performance",
        "section": "Results",
        "applicable": "Yes",
        "how": (
            "Table of discrimination and calibration metrics. Calibration "
            "plots and decision curve analysis figures."
        )
    },
    {
        "item": "17. Results: model updating",
        "section": "Results",
        "applicable": "Yes",
        "how": (
            "Recalibration of intercept in NHANES. Full model updating "
            "explored but not recommended."
        )
    },
    {
        "item": "18. Discussion: interpretation",
        "section": "Discussion",
        "applicable": "Yes",
        "how": (
            "Comparison with Framingham, QRISK3, PCE models. Clinical "
            "utility via decision curve analysis."
        )
    },
    {
        "item": "19. Discussion: limitations",
        "section": "Discussion",
        "applicable": "Yes",
        "how": (
            "Framingham may not represent diverse populations. NHANES "
            "validation limited by follow-up. Missing data assumptions."
        )
    },
    {
        "item": "20. Discussion: implications",
        "section": "Discussion",
        "applicable": "Yes",
        "how": (
            "Clinical implications for CVD risk assessment. Need for "
            "recalibration in non-US populations."
        )
    },
    {
        "item": "21. Code and data",
        "section": "Supplement",
        "applicable": "Yes",
        "how": (
            "Code on GitHub. Data publicly available via R packages. "
            "Reproducible via renv/conda."
        )
    },
    {
        "item": "AI-1. Data preprocessing",
        "section": "Methods/Supplement",
        "applicable": "No (logistic regression)",
        "how": (
            "Not applicable for logistic regression. Data cleaning and "
            "spline transformations described in supplement."
        )
    },
    {
        "item": "AI-2. Hyperparameter tuning",
        "section": "Methods/Supplement",
        "applicable": "No (logistic regression)",
        "how": (
            "Not applicable. Number of RCS knots pre-specified based on "
            "Harrell's recommendations."
        )
    },
    {
        "item": "AI-3. Model explainability",
        "section": "Results/Supplement",
        "applicable": "No (logistic regression)",
        "how": (
            "Not applicable (logistic regression is inherently "
            "interpretable). Nomogram provided."
        )
    },
    {
        "item": "AI-4. Software and hardware",
        "section": "Methods",
        "applicable": "Yes",
        "how": (
            "Python 3.11 with scikit-learn, lifelines, statsmodels, "
            "tableone. Full requirements.txt in repository."
        )
    },
]

# --- Print the checklist ---
print("=" * 70)
print("TRIPOD+AI Checklist")
print("Capstone 1: Cardiovascular Risk Prediction Model")
print("=" * 70)

for entry in checklist:
    print(f"\nITEM: {entry['item']}")
    print(f"  Section: {entry['section']}")
    print(f"  Applicable: {entry['applicable']}")
    print(f"  How addressed: {entry['how']}")

# --- Summary ---
n_yes = sum(1 for e in checklist if e['applicable'] == 'Yes')
n_partial = sum(1 for e in checklist if 'Partially' in e['applicable'])
n_no = sum(1 for e in checklist if e['applicable'].startswith('No'))

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"Items fully addressed:               {n_yes}")
print(f"Items partially applicable:          {n_partial}")
print(f"Items not applicable (ML-specific):  {n_no}")
print(f"Total items:                         {len(checklist)}")

print("""
Note on AI-specific items:
AI-specific items (AI-1 through AI-3) are not applicable because this
capstone uses logistic regression, not machine learning. If a gradient
boosted model or neural network were used, these items would require:
  - AI-1: Feature engineering pipeline, normalization, encoding
  - AI-2: Hyperparameter search strategy (grid, random, Bayesian)
  - AI-3: SHAP values, partial dependence plots, or LIME explanations

The TRIPOD+AI checklist is available from the EQUATOR Network:
https://www.equator-network.org/reporting-guidelines/tripod-ai/

Key points for completing TRIPOD+AI:
1. NEVER report AUROC/C-statistic without calibration
2. Always include a calibration plot
3. Fairness assessment across demographic subgroups is now required
4. Code and data availability statements are mandatory
5. For AI/ML models, explainability methods must be described
""")
