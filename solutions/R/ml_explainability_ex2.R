# =============================================================================
# Chapter 9b, Exercise 2: Reading a SHAP beeswarm
# Fit XGBoost, compute TreeSHAP, draw the beeswarm, and read it clinically.
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

# --- Compute SHAP values once (fast exact TreeSHAP) --------------------------
sv <- shapviz(xgb, X_pred = X, X = as.data.frame(X))

# --- Global SHAP importance ordering (mean |SHAP|) ---------------------------
mean_abs <- colMeans(abs(get_shap_values(sv)))
imp_order <- sort(mean_abs, decreasing = TRUE)
cat("=== Global SHAP importance (mean |SHAP value|) ===\n")
print(round(imp_order, 4))
cat("\nTop two variables:", paste(names(imp_order)[1:2], collapse = ", "), "\n")

# --- Beeswarm summary plot (saved to a temp file) ----------------------------
p <- sv_importance(sv, kind = "beeswarm") +
  labs(title = "Global SHAP summary: which variables matter, and how",
       x = "SHAP value (left = lowers risk, right = raises it)")
out <- file.path(tempdir(), "ch09b_ex2_beeswarm.png")
ggsave(out, p, width = 7, height = 4.5, dpi = 100)
cat("Beeswarm saved to:", out, "\n")

# =============================================================================
# INTERPRETATION
#
# 1) Which two variables are globally most important?
#    Read them off the printed ranking above. prior_admissions is clearly the
#    single most important variable; the second slot is taken by another genuine
#    risk driver -- here discharge_hb (with num_comorbidities and age close
#    behind). All of the top variables are ones that actually enter the
#    simulated true risk, while the two variables that do NOT (length_of_stay
#    and discharge_creat) fall to the bottom. Reassuring: the model relies on
#    clinically sensible signals. (The exact 2nd place can differ by backend --
#    e.g. num_comorbidities in the scikit-learn version -- because discharge_hb
#    has a wide spread and num_comorbidities a narrow one; trust the printout.)
#
# 2) For the TOP variable (prior_admissions), do high values push the
#    prediction UP or DOWN?
#    HIGH values (red dots) sit on the RIGHT -- more prior admissions push the
#    predicted readmission risk UP; few prior admissions (blue) push it down.
#    This is clinically sensible: a history of admissions is a well-known marker
#    of frailty and unstable disease, so higher predicted risk is expected.
#
# 3) A variable with NO clear left-right colour pattern:
#    discharge_creat (and, to a lesser extent, length_of_stay) shows red and
#    blue dots mixed on both sides with SHAP values tightly clustered near zero.
#    We built creatinine as pure noise (it enters neither the true risk nor a
#    correlation), so the model has found no consistent signal in it. A scrambled
#    colour pattern with small SHAP values means the variable is essentially
#    unused; a scrambled pattern with LARGE SHAP values would instead flag a
#    complex, non-monotonic or interaction-driven effect worth investigating.
# =============================================================================
