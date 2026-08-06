# =============================================================================
# Chapter 9b, Exercise 3: Explaining one patient to a patient
# SHAP waterfall plots for one high-risk and one low-risk patient.
# =============================================================================

library(xgboost)
library(shapviz)
library(dplyr)
library(tibble)
library(ggplot2)

# --- Re-create data and fit an XGBoost model ---------------------------------
set.seed(42)
n <- 1500
dat <- tibble(
  age              = rnorm(n, 68, 12),
  length_of_stay   = rpois(n, 5) + 1,
  num_comorbidities= rpois(n, 3),
  prior_admissions = rpois(n, 1),
  discharge_hb     = rnorm(n, 11, 2),
  discharge_creat  = rlnorm(n, 0.2, 0.5)
)
lin <- -3 + 0.45 * dat$prior_admissions + 0.20 * dat$num_comorbidities +
        0.015 * (dat$age - 68) - 0.05 * dat$discharge_hb
dat$readmit <- rbinom(n, 1, plogis(lin))

X <- dat %>% select(-readmit) %>% as.matrix()
xgb <- xgb.train(
  params = list(objective = "binary:logistic",
                max_depth = 4, learning_rate = 0.1),
  data = xgb.DMatrix(X, label = dat$readmit),
  nrounds = 100, verbose = 0
)
sv <- shapviz(xgb, X_pred = X, X = as.data.frame(X))

# --- Pick one high-risk and one low-risk patient by predicted probability ----
preds <- predict(xgb, xgb.DMatrix(X))
hi <- which.max(preds)   # highest predicted readmission risk
lo <- which.min(preds)   # lowest predicted readmission risk

cat(sprintf("High-risk patient: row %d, predicted risk = %.1f%%\n",
            hi, 100 * preds[hi]))
print(as.data.frame(X)[hi, , drop = FALSE])
cat(sprintf("\nLow-risk patient:  row %d, predicted risk = %.1f%%\n",
            lo, 100 * preds[lo]))
print(as.data.frame(X)[lo, , drop = FALSE])

# --- Waterfall plots (saved to temp files) -----------------------------------
p_hi <- sv_waterfall(sv, row_id = hi) +
  labs(title = "High-risk patient: building the prediction from baseline up")
p_lo <- sv_waterfall(sv, row_id = lo) +
  labs(title = "Low-risk patient: building the prediction from baseline up")
out_hi <- file.path(tempdir(), "ch09b_ex3_waterfall_high.png")
out_lo <- file.path(tempdir(), "ch09b_ex3_waterfall_low.png")
ggsave(out_hi, p_hi, width = 7, height = 4.5, dpi = 100)
ggsave(out_lo, p_lo, width = 7, height = 4.5, dpi = 100)
cat("\nWaterfalls saved to:\n  ", out_hi, "\n  ", out_lo, "\n")

# --- Which characteristic contributed most for the high-risk patient? --------
shap_hi <- get_shap_values(sv)[hi, ]
top_feat <- names(shap_hi)[which.max(abs(shap_hi))]
cat(sprintf("\nLargest contributor for the high-risk patient: %s (SHAP = %+.3f)\n",
            top_feat, shap_hi[which.max(abs(shap_hi))]))

# =============================================================================
# INTERPRETATION
#
# 1) Plain-language explanations from each waterfall:
#
#    HIGH-RISK patient (to the patient):
#    "Our tool starts everyone at the average readmission risk. For you it moved
#     UP mainly because you have had several previous hospital admissions and a
#     number of ongoing health conditions, which together point to a higher
#     chance of coming back within 30 days. A couple of your other results
#     nudged the estimate down a little, but not enough to change the picture."
#
#    LOW-RISK patient (to the patient):
#    "Starting from the average, your estimate moved DOWN because you have had
#     few or no previous admissions and few ongoing conditions. That is why the
#     tool puts your 30-day readmission risk below average."
#
# 2) Which characteristic contributed most for the high-risk patient, and would
#    intervening on it necessarily reduce risk?
#    For this patient the largest single contributor is prior_admissions (read
#    the printed top contributor above; it is a marker of high baseline risk).
#    It is tempting to conclude "reduce that variable and the risk falls" -- but
#    that is a CAUSAL claim the SHAP value does NOT support. SHAP only reports
#    how the MODEL used this patient's data; prior admissions (like a
#    comorbidity count) is a MARKER of underlying frailty and unstable disease,
#    not plausibly the direct cause of the next readmission. You cannot
#    "intervene" on a count of past events, and even the underlying frailty it
#    proxies would need a genuine causal-inference study (not an explanation
#    plot) to know whether any action actually lowers risk. Explanations
#    describe association learned by the model, never an intervention effect
#    (see the chapter's health warning: "Never read a SHAP value as a causal
#    effect").
# =============================================================================
