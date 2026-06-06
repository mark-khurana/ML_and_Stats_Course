# Chapter 11, Exercise 4: Complete Evaluation of a Published Model
# Example: QRISK3 Cardiovascular Risk Prediction Model
#
# This exercise is primarily a literature review exercise. Below we provide
# a structured answer based on published validation data for QRISK3.

# ---- Part (a): C-statistic with confidence interval ----
print("=== Part (a): C-statistic ===")
print("QRISK3 (Hippisley-Cox et al., 2017, BMJ) was developed to predict")
print("10-year cardiovascular disease risk.")
print()
print("Development cohort:")
print("  C-statistic (women): 0.880 (95% CI: 0.878-0.882)")
print("  C-statistic (men):   0.858 (95% CI: 0.856-0.860)")
print()
print("Validation cohort (separate 25% held out):")
print("  C-statistic (women): 0.879 (95% CI: 0.876-0.882)")
print("  C-statistic (men):   0.858 (95% CI: 0.855-0.861)")
print("The discrimination is excellent and stable between development")
print("and validation sets.")

# ---- Part (b): Calibration ----
print("\n=== Part (b): Calibration ===")
print("QRISK3 provides calibration plots in the original publication.")
print("In the validation cohort:")
print("  - Calibration was generally good, with predicted risks closely")
print("    matching observed event rates across deciles.")
print("  - Some overestimation at higher risk levels, particularly in")
print("    older age groups.")
print("  - External validations outside the UK have shown variable")
print("    calibration, often with overestimation in lower-risk populations.")

# ---- Part (c): Decision curve analysis ----
print("\n=== Part (c): Decision Curve Analysis ===")
print("The original QRISK3 paper does not include decision curve analysis.")
print()
print("Clinically relevant threshold range for statin initiation:")
print("  - UK NICE guidelines: 10% 10-year CVD risk threshold")
print("  - US ACC/AHA guidelines: 7.5% threshold")
print("  - A decision curve would be most informative in the 5-20%")
print("    threshold range, where the statin decision is most uncertain.")
print("  - Below 5%, most clinicians would not start statins")
print("  - Above 20%, most clinicians would start statins regardless")

# ---- Part (d): External validation ----
print("\n=== Part (d): External Validation ===")
print("QRISK3 has been validated in multiple populations:")
print("  - Internal-external: UK CPRD data (separate time period)")
print("  - Geographic: Various European populations")
print("  - Ethnic subgroups: South Asian, Black, Chinese populations")
print("  - Temporal: Validated across different calendar periods")
print()
print("Key findings from external validations:")
print("  - Discrimination generally remains good (C > 0.80)")
print("  - Calibration can deteriorate in non-UK populations")
print("  - May overestimate risk in populations with lower baseline")
print("    cardiovascular event rates")

# ---- Part (e): Recommendation ----
print("\n=== Part (e): Recommendation ===")
print("Recommendation depends on the clinical setting:")
print()
print("FOR a UK primary care setting:")
print("  YES, recommend implementation. QRISK3 was developed and")
print("  validated in UK primary care, has excellent discrimination,")
print("  good calibration, and is integrated into UK clinical guidelines.")
print()
print("FOR a non-UK setting:")
print("  CONDITIONAL recommendation. Would first require:")
print("    1. External validation in the local population")
print("    2. Assessment of calibration (likely needs recalibration)")
print("    3. Decision curve analysis at locally relevant thresholds")
print("    4. Comparison with locally developed risk scores")
print("  If calibration is poor, logistic recalibration should be")
print("  considered before implementation.")
