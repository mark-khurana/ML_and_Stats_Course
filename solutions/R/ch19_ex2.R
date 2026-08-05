# =============================================================================
# Chapter 19 - Exercise 3: Write a Statistical Methods Section
# Retrospective cohort study: SGLT2i vs DPP4i and MACE in T2DM
# =============================================================================

# This is a conceptual exercise. The statistical methods section is provided
# as a multi-line character string that could be included in a manuscript.

methods_section <- '
STATISTICAL METHODS

Study Design and Population
This was a retrospective cohort study of 5,000 adults with type 2 diabetes
mellitus identified from the hospital electronic health records (EHR) database
between 1 January 2015 and 31 December 2023. Patients were eligible if they
had a new prescription for either an SGLT2 inhibitor or a DPP-4 inhibitor, with
no prior use of the comparator drug class. Patients with a history of major
adverse cardiovascular events (MACE) prior to the index date were excluded.
The index date (time zero) was defined as the date of first prescription of
the study drug, consistent with a new-user, active comparator design to
minimise immortal time bias and confounding by indication.

Primary and Secondary Outcomes
The primary outcome was time to first MACE, defined as a composite of
myocardial infarction (ICD-10: I21), ischaemic stroke (ICD-10: I63), or
cardiovascular death (underlying cause of death codes I00-I99). Patients were
followed from the index date until the first MACE event, death from non-
cardiovascular causes, loss to follow-up, end of the study period
(31 December 2023), or 5 years after the index date, whichever occurred first.

Sample Size
With 5,000 patients and an anticipated event rate of 8% over 5 years in the
DPP-4 inhibitor group, the study had approximately 80% power to detect a
hazard ratio of 0.70 or smaller at a two-sided alpha of 0.05, assuming a
1:1 treatment group ratio and accounting for 10% loss to follow-up.

Descriptive Statistics
Baseline characteristics were summarised as means (SD) for normally
distributed continuous variables, medians (IQR) for skewed continuous
variables, and frequencies (percentages) for categorical variables.
Standardised mean differences (SMDs) were used to compare baseline
characteristics between treatment groups, with an absolute SMD < 0.1
indicating adequate balance; p-values were not used for baseline comparisons
in accordance with current recommendations.

Propensity Score Estimation and Matching
The propensity score -- the probability of receiving an SGLT2 inhibitor
versus a DPP-4 inhibitor -- was estimated using multivariable logistic
regression. Covariates included age, sex, body mass index (BMI), glycated
haemoglobin (HbA1c), estimated glomerular filtration rate (eGFR), history
of cardiovascular disease, hypertension, and smoking status. These covariates
were selected a priori based on clinical knowledge and a directed acyclic
graph (DAG) encoding assumed causal relationships. Propensity scores were
used for 1:1 nearest-neighbour matching without replacement, using a caliper
of 0.2 standard deviations of the logit of the propensity score. Covariate
balance after matching was assessed using SMDs, with all covariates required
to achieve an absolute SMD < 0.1. Propensity score overlap was assessed
visually using density plots.

Primary Analysis
The primary analysis estimated the average treatment effect on the treated
(ATT) using a Cox proportional hazards regression model fitted to the matched
cohort, with SGLT2 inhibitor use as the sole covariate. The proportional
hazards assumption was tested using scaled Schoenfeld residuals and
log-log survival plots. If the assumption was violated, a time-varying
coefficient or restricted mean survival time (RMST) analysis was planned
as an alternative. Hazard ratios (HRs) with 95% confidence intervals (CIs)
were reported.

Missing Data
Missing covariate data ranged from 2% (age) to 15% (BMI). Missingness was
assumed to be missing at random (MAR) conditional on observed variables.
Multiple imputation by chained equations (MICE) was performed with 50
imputed datasets and 20 iterations per dataset. The imputation model
included all analysis variables (covariates, exposure, outcome indicator,
and the Nelson-Aalen cumulative hazard estimate) to ensure compatibility
with the substantive analysis model. Propensity score estimation and
matching were performed within each imputed dataset, and results were
pooled using Rubin rules. A complete-case analysis was performed as a
sensitivity analysis.

Sensitivity Analyses
Five pre-specified sensitivity analyses were conducted: (1) complete-case
analysis; (2) inverse probability of treatment weighting (IPTW) with
stabilised weights as an alternative to matching; (3) inclusion of
additional covariates (income quintile, number of medications) in the
propensity score model; (4) restriction to patients with at least
12 months of follow-up; and (5) E-value calculation to quantify the
minimum strength of association an unmeasured confounder would need with
both the treatment and the outcome to explain away the observed association.

Multiple Comparisons
As this study had a single pre-specified primary outcome, no adjustment
for multiple comparisons was applied to the primary analysis. Secondary
outcomes were interpreted with appropriate caution as hypothesis-generating.

Software
All analyses were conducted using R version 4.4.1 (R Foundation for
Statistical Computing, Vienna, Austria) with the following packages:
MatchIt (v4.5.5) for propensity score matching, cobalt (v4.5.1) for
balance assessment, survival (v3.7-0) for Cox regression, mice (v3.16.0)
for multiple imputation, and survey (v4.4-2) for IPTW analyses. The
random seed was set to 42 for reproducibility. Analysis code is available
at [repository URL]. Two-sided p-values < 0.05 were considered
statistically significant.
'

# Print the methods section
cat(methods_section)

# --- Checklist verification ---
cat("\n\n=== CHECKLIST: Key Items Addressed ===\n")
cat("1. Study design and population:           YES\n")
cat("2. Primary and secondary outcomes:        YES\n")
cat("3. Sample size / power calculation:        YES\n")
cat("4. Descriptive statistics approach:        YES\n")
cat("5. Primary analysis model:                YES\n")
cat("6. Assumptions and how checked:           YES (PH assumption)\n")
cat("7. Missing data (extent & handling):       YES (MICE, 50 datasets)\n")
cat("8. Sensitivity analyses:                  YES (5 pre-specified)\n")
cat("9. Multiple comparisons:                  YES (single primary)\n")
cat("10. Software and versions:                YES\n")
cat("11. Active comparator, new-user design:   YES\n")
cat("12. DAG for covariate selection:          YES\n")
cat("13. Balance assessment (SMDs):            YES\n")
cat("14. E-value for unmeasured confounding:   YES\n")
