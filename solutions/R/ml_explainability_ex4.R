# =============================================================================
# Chapter 9b, Exercise 4: PDP versus ALE with correlated predictors
# Make creatinine rise with age, then compare PDP and ALE for creatinine.
# =============================================================================

library(ranger)
library(pdp)     # partial() for partial dependence
library(iml)     # FeatureEffect(..., method = "ale") for ALE
library(dplyr)
library(tibble)
library(ggplot2)

# --- Re-create data, but make discharge_creat CORRELATED with age ------------
set.seed(42)
n <- 1500
dat <- tibble(
  age              = rnorm(n, 68, 12),
  length_of_stay   = rpois(n, 5) + 1,
  num_comorbidities= rpois(n, 3),
  prior_admissions = rpois(n, 1),
  discharge_hb     = rnorm(n, 11, 2)
)
# Creatinine now RISES WITH AGE (very strong correlation, corr ~ 0.98) plus a
# little noise. Crucially it is NOT part of the true risk -- it is a proxy for
# age. The tighter the correlation, the more the PDP is forced to extrapolate.
dat$discharge_creat <- 0.4 + 0.02 * dat$age + rnorm(n, 0, 0.05)

cat(sprintf("Correlation(age, discharge_creat) = %.2f\n",
            cor(dat$age, dat$discharge_creat)))

# True risk depends on age (and prior admissions, comorbidities), NOT creatinine
lin <- -3 + 0.45 * dat$prior_admissions + 0.20 * dat$num_comorbidities +
        0.05 * (dat$age - 68) - 0.05 * dat$discharge_hb
dat$readmit <- factor(rbinom(n, 1, plogis(lin)), labels = c("No", "Yes"))

rf <- ranger(readmit ~ ., data = dat, probability = TRUE,
             num.trees = 500, seed = 42)

predictors <- setdiff(names(dat), "readmit")
pred_fun <- function(object, newdata) {
  predict(object, newdata)$predictions[, "Yes"]
}

# --- Partial dependence (PDP) of predicted risk on creatinine ----------------
pd <- partial(rf, pred.var = "discharge_creat", pred.fun = pred_fun,
              train = as.data.frame(dat), grid.resolution = 25)
# pdp with a per-row pred.fun returns one row per (grid point, obs);
# aggregate to the mean prediction at each creatinine grid value.
pd_curve <- pd %>%
  group_by(discharge_creat) %>%
  summarise(pdp = mean(yhat), .groups = "drop")

p_pdp <- ggplot(pd_curve, aes(discharge_creat, pdp)) +
  geom_line(linewidth = 1.2, colour = "#2E86AB") +
  labs(title = "PDP: predicted risk vs discharge creatinine",
       x = "Discharge creatinine (mg/dL)", y = "Average predicted risk") +
  theme_minimal(base_size = 13)
out_pdp <- file.path(tempdir(), "ch09b_ex4_pdp.png")
ggsave(out_pdp, p_pdp, width = 7, height = 4, dpi = 100)

# --- ALE plot for the same variable (iml) ------------------------------------
predictor <- Predictor$new(
  rf, data = as.data.frame(dat[predictors]), y = dat$readmit,
  predict.function = pred_fun
)
ale <- FeatureEffect$new(predictor, feature = "discharge_creat", method = "ale")
p_ale <- plot(ale) +
  labs(title = "ALE: predicted risk vs discharge creatinine")
out_ale <- file.path(tempdir(), "ch09b_ex4_ale.png")
ggsave(out_ale, p_ale, width = 7, height = 4, dpi = 100)

cat("PDP saved to:", out_pdp, "\n")
cat("ALE saved to:", out_ale, "\n")

# --- Quantify the slope of each curve for comparison -------------------------
pdp_slope <- coef(lm(pdp ~ discharge_creat, data = pd_curve))[2]
ale_df <- ale$results
ale_slope <- coef(lm(.value ~ discharge_creat, data = ale_df))[2]
cat(sprintf("\nPDP slope (risk per mg/dL creatinine): %+.4f\n", pdp_slope))
cat(sprintf("ALE slope (centred effect per mg/dL): %+.4f\n", ale_slope))

# =============================================================================
# INTERPRETATION
#
# 1) Do the two curves agree?
#    No. The PDP shows a steep UPWARD slope, making creatinine look strongly
#    risk-increasing. The ALE curve rises too but is SUBSTANTIALLY FLATTER (its
#    slope is roughly a third to a half of the PDP's -- see the printed slopes),
#    indicating that once age is accounted for, creatinine's own conditional
#    effect is much smaller than the PDP suggests.
#
# 2) Why is the PDP misleading here?
#    We built creatinine as a pure PROXY for age: they are strongly correlated
#    and only AGE truly drives risk. A PDP works by forcing EVERY patient to a
#    given creatinine value while leaving their real age untouched -- so to draw
#    the point at "high creatinine" it averages predictions for IMPOSSIBLE
#    patients (young people with an old person's creatinine). Because age and
#    creatinine travel together in the real data, the model learned to read
#    creatinine partly as a stand-in for age; when the PDP breaks that link it
#    smears age's genuine effect onto the creatinine axis, producing a spurious
#    upward slope. ALE avoids this by measuring how the prediction CHANGES
#    within small, realistic windows of creatinine (where age is roughly
#    constant) and accumulating those local changes -- so it never evaluates
#    impossible combinations and reports creatinine's much smaller independent
#    effect. With correlated clinical predictors, trust the ALE.
# =============================================================================
