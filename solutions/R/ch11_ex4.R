# Chapter 11, Exercise 4: Complete Evaluation of a Published Model
# Example: QRISK3 Cardiovascular Risk Prediction Model
#
# This exercise is primarily a literature review exercise. Below we provide
# a structured answer based on published validation data for QRISK3.

# ---- Part (a): C-statistic with confidence interval ----
cat("=== Part (a): C-statistic ===\n")
cat("QRISK3 (Hippisley-Cox et al., 2017, BMJ) was developed to predict\n")
cat("10-year cardiovascular disease risk.\n\n")
cat("Development cohort:\n")
cat("  C-statistic (women): 0.880 (95% CI: 0.878-0.882)\n")
cat("  C-statistic (men):   0.858 (95% CI: 0.856-0.860)\n\n")
cat("Validation cohort (separate 25% held out):\n")
cat("  C-statistic (women): 0.879 (95% CI: 0.876-0.882)\n")
cat("  C-statistic (men):   0.858 (95% CI: 0.855-0.861)\n")
cat("The discrimination is excellent and stable between development\n")
cat("and validation sets.\n")

# ---- Part (b): Calibration ----
cat("\n=== Part (b): Calibration ===\n")
cat("QRISK3 provides calibration plots in the original publication.\n")
cat("In the validation cohort:\n")
cat("  - The calibration was generally good, with predicted risks\n")
cat("    closely matching observed event rates across deciles.\n")
cat("  - Some overestimation of risk was observed at higher risk\n")
cat("    levels, particularly in older age groups.\n")
cat("  - External validations in different populations (e.g., outside\n")
cat("    the UK) have shown variable calibration, often with\n")
cat("    overestimation in lower-risk populations.\n")

# ---- Part (c): Decision curve analysis ----
cat("\n=== Part (c): Decision Curve Analysis ===\n")
cat("The original QRISK3 paper does not include decision curve analysis.\n\n")
cat("Clinically relevant threshold range for statin initiation:\n")
cat("  - UK NICE guidelines: 10% 10-year CVD risk threshold\n")
cat("  - US ACC/AHA guidelines: 7.5% threshold\n")
cat("  - A decision curve would be most informative in the 5-20%\n")
cat("    threshold range, where the decision to start statins is\n")
cat("    most uncertain.\n")
cat("  - Below 5%, most clinicians would not start statins\n")
cat("  - Above 20%, most clinicians would start statins regardless\n")

# ---- Part (d): External validation ----
cat("\n=== Part (d): External Validation ===\n")
cat("QRISK3 has been validated in multiple populations:\n")
cat("  - Internal-external: UK CPRD data (separate time period)\n")
cat("  - Geographic: Various European populations\n")
cat("  - Ethnic subgroups: South Asian, Black, Chinese populations\n")
cat("  - Temporal: Validated across different calendar periods\n\n")
cat("Key findings from external validations:\n")
cat("  - Discrimination generally remains good (C > 0.80)\n")
cat("  - Calibration can deteriorate in non-UK populations\n")
cat("  - May overestimate risk in populations with lower baseline\n")
cat("    cardiovascular event rates\n")

# ---- Part (e): Recommendation for implementation ----
cat("\n=== Part (e): Recommendation ===\n")
cat("Recommendation depends on the clinical setting:\n\n")
cat("FOR a UK primary care setting:\n")
cat("  - YES, recommend implementation. QRISK3 was developed and\n")
cat("    validated in UK primary care, has excellent discrimination,\n")
cat("    good calibration, and is already integrated into UK\n")
cat("    clinical guidelines.\n\n")
cat("FOR a non-UK setting:\n")
cat("  - CONDITIONAL recommendation. Would first require:\n")
cat("    1. External validation in the local population\n")
cat("    2. Assessment of calibration (likely needs recalibration)\n")
cat("    3. Decision curve analysis at locally relevant thresholds\n")
cat("    4. Comparison with locally developed risk scores\n")
cat("  - If calibration is poor, logistic recalibration should be\n")
cat("    considered before implementation.\n")
