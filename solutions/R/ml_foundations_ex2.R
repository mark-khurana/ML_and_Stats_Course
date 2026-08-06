# =============================================================================
# Chapter 7, Exercise 2: Feature Engineering Challenge
# Engineer clinically meaningful features from raw variables for a diabetes
# prediction model. Implement using tidymodels recipes.
# =============================================================================

library(tidyverse)
library(tidymodels)

# --- Simulate raw clinical data ---
set.seed(42)
n <- 500

raw_data <- tibble(
  height_cm = rnorm(n, 170, 10),
  weight_kg = rnorm(n, 80, 15),
  sbp = rnorm(n, 130, 18),
  dbp = rnorm(n, 82, 12),
  fasting_glucose = rnorm(n, 105, 25),
  hba1c = rnorm(n, 5.8, 0.8),
  age = rnorm(n, 55, 12),
  sex = sample(c("Male", "Female"), n, replace = TRUE),
  waist_circumference = rnorm(n, 95, 12),
  hip_circumference = rnorm(n, 100, 10),
  total_cholesterol = rnorm(n, 200, 40),
  hdl = rnorm(n, 50, 15),
  ldl = rnorm(n, 120, 35),
  triglycerides = rnorm(n, 150, 60),
  diabetes = factor(rbinom(n, 1, 0.3), labels = c("No", "Yes"))
)

# --- Part 1: Clinically meaningful engineered features ---
# 1. BMI = weight_kg / (height_m)^2
#    WHY: Standard obesity measure; strong risk factor for Type 2 diabetes.
#
# 2. Pulse Pressure = sbp - dbp
#    WHY: Reflects arterial stiffness; associated with cardiovascular risk
#         and metabolic syndrome.
#
# 3. Mean Arterial Pressure (MAP) = dbp + (sbp - dbp) / 3
#    WHY: Measures average perfusion pressure; linked to vascular health.
#
# 4. Waist-to-Hip Ratio (WHR) = waist_circumference / hip_circumference
#    WHY: Central adiposity is a stronger predictor of insulin resistance
#         than BMI alone.
#
# 5. Non-HDL Cholesterol = total_cholesterol - hdl
#    WHY: Captures all atherogenic lipoproteins; recommended by guidelines
#         as a secondary target in diabetes management.
#
# 6. Triglyceride-to-HDL Ratio = triglycerides / hdl
#    WHY: A proxy for insulin resistance; high TG/HDL ratio is associated
#         with increased diabetes risk.
#
# 7. LDL/HDL Ratio = ldl / hdl
#    WHY: Captures atherogenic dyslipidemia profile common in diabetes.

# --- Part 2: Implement feature engineering with recipes ---
diabetes_recipe <- recipe(diabetes ~ ., data = raw_data) %>%
  # BMI: weight / height_m^2
  step_mutate(
    height_m = height_cm / 100,
    bmi = weight_kg / height_m^2,
    # Pulse pressure
    pulse_pressure = sbp - dbp,
    # Mean arterial pressure
    map = dbp + (sbp - dbp) / 3,
    # Waist-to-hip ratio
    whr = waist_circumference / hip_circumference,
    # Non-HDL cholesterol
    non_hdl = total_cholesterol - hdl,
    # Triglyceride-to-HDL ratio
    tg_hdl_ratio = triglycerides / hdl,
    # LDL/HDL ratio
    ldl_hdl_ratio = ldl / hdl
  ) %>%
  # Remove intermediate and redundant variables
  step_rm(height_m) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

# Prepare (bake) the recipe to see the result
prepped <- prep(diabetes_recipe)
engineered_data <- bake(prepped, new_data = NULL)

cat("Original variables:", ncol(raw_data) - 1, "\n")
cat("Engineered dataset columns:", ncol(engineered_data) - 1, "\n")
cat("\nColumn names:\n")
print(names(engineered_data))

# --- Part 3: Discuss redundant features ---
# After engineering:
# - height_cm and weight_kg may become redundant once BMI is computed
#   (though height or weight alone may still carry predictive signal).
# - sbp and dbp are partially captured by pulse_pressure and MAP,
#   though keeping them may still be useful for tree-based models.
# - waist_circumference and hip_circumference are largely captured by WHR.
# - total_cholesterol is partly captured by non_hdl (since non_hdl = TC - HDL).
# - Individual lipids (hdl, ldl, triglycerides) overlap with the ratios,
#   but a LASSO or tree model can sort out which representation is most useful.
#
# In practice, include both raw and engineered features and let a
# regularised model (LASSO, elastic net) or tree-based model perform
# implicit feature selection.
