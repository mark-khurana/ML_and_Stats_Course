# Exercise 3: Variable selection
# Compare a pre-specified model, forward stepwise selection, and LASSO on the
# simulated Framingham cohort, then check how stable each selection is across
# 100 bootstrap samples.
#
# Runtime is about a minute: the bootstrap refits both selection procedures
# 100 times, cross-validating the LASSO penalty inside each one.

library(MASS) # stepAIC
library(glmnet) # cv.glmnet

# --- The cohort, exactly as in the chapter ---------------------------------
set.seed(2024)
n <- 2000

framingham <- data.frame(
  age = round(runif(n, 30, 74)),
  male = rbinom(n, 1, 0.48),
  sbp = round(rnorm(n, 130, 18)),
  total_chol = round(rnorm(n, 210, 38)),
  hdl_chol = round(rnorm(n, 52, 15)),
  smoking = rbinom(n, 1, 0.22),
  diabetes = rbinom(n, 1, 0.08),
  bp_treatment = rbinom(n, 1, 0.15)
)
lp <- with(
  framingham,
  -7.5 + 0.06 * age + 0.4 * male + 0.012 * sbp + 0.005 * total_chol -
    0.02 * hdl_chol + 0.5 * smoking + 0.7 * diabetes + 0.3 * bp_treatment
)
framingham$cvd_10yr <- rbinom(n, 1, plogis(lp))

REAL <- setdiff(names(framingham), "cvd_10yr")

cat("Events:", sum(framingham$cvd_10yr), "of", n, "\n")

# --- Helpers ---------------------------------------------------------------
forward_stepwise <- function(data, candidates) {
  upper <- as.formula(paste("~", paste(candidates, collapse = " + ")))
  fit <- stepAIC(
    glm(cvd_10yr ~ 1, data = data, family = binomial),
    scope = list(lower = ~1, upper = upper),
    direction = "forward", trace = 0
  )
  setdiff(names(coef(fit)), "(Intercept)")
}

lasso_selected <- function(data, candidates, s = "lambda.min", nfolds = 10) {
  X <- as.matrix(data[candidates])
  cvfit <- cv.glmnet(X, data$cvd_10yr, family = "binomial",
                     alpha = 1, nfolds = nfolds)
  b <- coef(cvfit, s = s)[-1, 1]
  names(b)[b != 0]
}

# --- (a) The three approaches on the pre-specified predictors -------------
cat("\n=== Part (a): candidates are the 8 pre-specified predictors ===\n")

fit_all <- glm(cvd_10yr ~ ., data = framingham, family = binomial)
cat("\nPre-specified model (all 8 predictors):\n")
print(round(summary(fit_all)$coefficients[, c(1, 2, 4)], 4))

set.seed(1)
step_a <- forward_stepwise(framingham, REAL)
lasso_min_a <- lasso_selected(framingham, REAL, "lambda.min")
lasso_1se_a <- lasso_selected(framingham, REAL, "lambda.1se")

cat("\nSelected sets:\n")
cat("  pre-specified      (8):", paste(REAL, collapse = ", "), "\n")
cat(sprintf("  forward stepwise   (%d): %s\n", length(step_a),
            paste(sort(step_a), collapse = ", ")))
cat(sprintf("  LASSO lambda.min   (%d): %s\n", length(lasso_min_a),
            paste(sort(lasso_min_a), collapse = ", ")))
cat(sprintf("  LASSO lambda.1se   (%d): %s\n", length(lasso_1se_a),
            paste(sort(lasso_1se_a), collapse = ", ")))

cat("
With 8 genuinely predictive variables and 235 events there is nothing for
selection to remove: every predictor is significant, so stepwise keeps all
eight and so does the LASSO at lambda.min. This is the easy case, and the
honest conclusion is that selection added nothing -- the pre-specified model
was already the answer. Note that lambda.1se, the deliberately conservative
choice, drops real predictors.
")

# --- (b) The realistic case: candidates that do not belong ---------------
# Selection only becomes interesting when the candidate list contains
# variables with no relationship to the outcome, which is the usual situation
# when a list is drawn up from "everything we measured".
set.seed(99)
noise <- as.data.frame(matrix(rnorm(n * 12), n, 12))
names(noise) <- paste0("noise", 1:12)
wide <- cbind(framingham, noise)
CANDIDATES <- c(REAL, names(noise))

cat("\n=== Part (b): 8 real predictors + 12 pure-noise candidates ===\n")

set.seed(1)
step_b <- forward_stepwise(wide, CANDIDATES)
lasso_min_b <- lasso_selected(wide, CANDIDATES, "lambda.min")
lasso_1se_b <- lasso_selected(wide, CANDIDATES, "lambda.1se")

report <- function(label, selected) {
  cat(sprintf(
    "  %-18s kept %2d  (%d of 8 real, %d of 12 noise)\n",
    label, length(selected), sum(selected %in% REAL),
    sum(grepl("noise", selected))
  ))
}
cat("\n")
report("forward stepwise", step_b)
report("LASSO lambda.min", lasso_min_b)
report("LASSO lambda.1se", lasso_1se_b)
cat("\n  stepwise kept these noise variables:",
    paste(sort(grep("noise", step_b, value = TRUE)), collapse = ", "), "\n")

# --- (c) Stability across 100 bootstrap samples -------------------------
cat("\n=== Part (c): selection frequency across 100 bootstrap samples ===\n")

set.seed(2025)
B <- 100
counts <- list(
  stepwise = setNames(numeric(length(CANDIDATES)), CANDIDATES),
  lasso_min = setNames(numeric(length(CANDIDATES)), CANDIDATES)
)
sizes <- list(stepwise = numeric(B), lasso_min = numeric(B))
signatures <- list(stepwise = character(B), lasso_min = character(B))

for (b in seq_len(B)) {
  boot <- wide[sample(n, n, replace = TRUE), ]
  picks <- list(
    stepwise = forward_stepwise(boot, CANDIDATES),
    # nfolds = 5 inside the loop purely to keep the runtime reasonable
    lasso_min = lasso_selected(boot, CANDIDATES, "lambda.min", nfolds = 5)
  )
  for (m in names(picks)) {
    counts[[m]][picks[[m]]] <- counts[[m]][picks[[m]]] + 1
    sizes[[m]][b] <- length(picks[[m]])
    signatures[[m]][b] <- paste(sort(picks[[m]]), collapse = "|")
  }
}

freq <- data.frame(
  variable = CANDIDATES,
  truth = ifelse(CANDIDATES %in% REAL, "real", "noise"),
  stepwise_pct = 100 * counts$stepwise / B,
  lasso_pct = 100 * counts$lasso_min / B,
  row.names = NULL
)
freq <- freq[order(freq$truth, -freq$stepwise_pct), ]
cat("\nHow often each candidate was selected (%):\n")
print(freq, row.names = FALSE)

cat("\nSummary across the 100 bootstrap samples:\n")
for (m in c("stepwise", "lasso_min")) {
  real_pct <- mean(counts[[m]][REAL]) / B * 100
  noise_pct <- mean(counts[[m]][grepl("noise", CANDIDATES)]) / B * 100
  cat(sprintf(
    "  %-10s median size %.0f | real predictors kept %.0f%% of the time | noise kept %.0f%% | %d distinct models\n",
    m, median(sizes[[m]]), real_pct, noise_pct, length(unique(signatures[[m]]))
  ))
}

cat("
Conclusions
-----------
(a) When every candidate belongs in the model, selection has no work to do and
    both methods keep everything. Selection cannot improve on a well-chosen
    pre-specified list; the most it can do is leave it alone.

(b) Add candidates that do not belong and both methods start letting them in.
    Forward stepwise admits several noise variables, because a variable enters
    on whether it improves AIC in this particular sample, and with 12 chances
    some noise variable always looks helpful. The LASSO at lambda.min is no
    better here -- lambda.min optimises prediction error, not variable
    recovery, so it keeps most of the noise with small coefficients. Only
    lambda.1se is clean of noise, and it pays by dropping real predictors.

(c) The bootstrap frequencies are the real finding. Stepwise produced 96
    different models in 100 resamples of the same patients, and the LASSO 75.
    Real predictors were kept 88% of the time by stepwise and 97% by the
    LASSO; noise variables 37% and 75%. Even the strongest real predictors
    (age, hdl_chol, diabetes) are selected every time, so the instability is
    concentrated exactly where it matters -- the weaker predictors, where you
    would actually want the method's advice.

    One subtlety worth naming, because it cuts against the obvious reading.
    noise10 is selected 75% of the time by stepwise and 99% by the LASSO,
    which looks like a reliable predictor. It is not: it happens to correlate
    with the outcome in this particular cohort, and every bootstrap sample is
    drawn from that same cohort, so the fluke is reproduced. Bootstrap
    stability therefore measures robustness to resampling these patients, not
    to collecting new ones -- it is a lower bound on the instability a fresh
    dataset would reveal.

    That is what makes a paper reporting one stepwise-selected model
    misleading. The list of retained variables is presented as a finding --
    'these are the predictors of risk' -- when a different sample of the same
    patients would have produced a different list, and a genuinely new sample
    a different one again. The p-values and confidence intervals are also
    wrong, because they take no account of the searching that preceded them.
    The defensible options are to pre-specify the predictors on clinical
    grounds and keep them all, or to use a penalised model and report the
    whole procedure rather than the variables that happened to survive it.
")
