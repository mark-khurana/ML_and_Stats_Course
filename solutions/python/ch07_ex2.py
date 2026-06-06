# =============================================================================
# Chapter 7, Exercise 2: Feature Engineering Challenge
# Engineer clinically meaningful features from raw variables for a diabetes
# prediction model. Implement using pandas and sklearn.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Simulate raw clinical data ---
np.random.seed(42)
n = 500

raw_data = pd.DataFrame({
    'height_cm': np.random.normal(170, 10, n),
    'weight_kg': np.random.normal(80, 15, n),
    'sbp': np.random.normal(130, 18, n),
    'dbp': np.random.normal(82, 12, n),
    'fasting_glucose': np.random.normal(105, 25, n),
    'hba1c': np.random.normal(5.8, 0.8, n),
    'age': np.random.normal(55, 12, n),
    'sex': np.random.choice(['Male', 'Female'], n),
    'waist_circumference': np.random.normal(95, 12, n),
    'hip_circumference': np.random.normal(100, 10, n),
    'total_cholesterol': np.random.normal(200, 40, n),
    'hdl': np.random.normal(50, 15, n),
    'ldl': np.random.normal(120, 35, n),
    'triglycerides': np.random.normal(150, 60, n),
})
y = np.random.binomial(1, 0.3, n)

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
# 4. Waist-to-Hip Ratio (WHR) = waist / hip
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

# --- Part 2: Implement feature engineering ---
df = raw_data.copy()

# Compute engineered features
df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
df['pulse_pressure'] = df['sbp'] - df['dbp']
df['map'] = df['dbp'] + (df['sbp'] - df['dbp']) / 3
df['whr'] = df['waist_circumference'] / df['hip_circumference']
df['non_hdl'] = df['total_cholesterol'] - df['hdl']
df['tg_hdl_ratio'] = df['triglycerides'] / df['hdl']
df['ldl_hdl_ratio'] = df['ldl'] / df['hdl']

# Encode sex as binary
df['sex_male'] = (df['sex'] == 'Male').astype(int)
df = df.drop(columns=['sex'])

print(f"Original features: {raw_data.shape[1]}")
print(f"Engineered features: {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")

# Scale numeric features
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(f"\nFirst few rows of engineered data:")
print(df_scaled.head())

# --- Part 3: Discuss redundant features ---
# After engineering:
# - height_cm and weight_kg may become redundant once BMI is computed
#   (though they may still carry independent predictive signal).
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
