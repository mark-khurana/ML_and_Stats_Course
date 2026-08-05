# install_packages.R — Install all R packages needed for this course.
#
# Open RStudio and run:
#   source("R/install_packages.R")
#
# This may take 10-15 minutes the first time (brms/rstanarm compile Stan).

pkgs <- c(
  # Core tidyverse ecosystem
  "tidyverse", # dplyr, ggplot2, tidyr, readr, purrr, ...
  "broom", # tidy model output
  "patchwork", # combine ggplot panels
  "here", # project-relative file paths

  # Regression & modeling frameworks
  "rms", # Regression Modeling Strategies (Harrell)
  "glmnet", # Lasso / ridge / elastic net
  "tidymodels", # Unified ML framework
  "caret", # Classification And REgression Training
  "MASS", # Modern Applied Statistics (veninger/Ripley)

  # Survival analysis
  "survival", # Cox models, Kaplan-Meier
  "survminer", # Survival curve visualization
  "tidycmprsk", # Competing risks
  "survey", # Complex survey designs

  # Mixed models & missing data
  "lme4", # Linear mixed-effects models
  "lmerTest", # p-values for lme4
  "mice", # Multiple imputation

  # Trees, ensembles & kernels
  "ranger", # Fast random forests
  "rpart", # Recursive partitioning
  "rpart.plot", # Plotting rpart trees
  "xgboost", # Gradient boosted trees
  "kernlab", # Kernel methods (SVM etc.)
  "mlbench", # ML benchmark datasets

  # Clustering
  "cluster", # k-medoids (PAM), hierarchical
  "mclust", # Model-based clustering
  "dbscan", # Density-based clustering

  # Dimensionality reduction
  "uwot", # UMAP
  "Rtsne", # t-SNE

  # Model evaluation & explainability
  "pROC", # ROC curves / AUC
  "PRROC", # Precision-recall curves
  "dcurves", # Decision curve analysis
  "vip", # Variable importance (archived from CRAN; r-universe repo added below)
  "pdp", # Partial dependence plots
  "shapviz", # SHAP values visualization
  "pmsampsize", # Prediction model sample size

  # Bayesian
  "brms", # Bayesian regression via Stan
  "rstanarm", # Pre-compiled Bayesian models
  "bayesplot", # Bayesian diagnostic plots

  # Meta-analysis
  "metafor", # Meta-analysis
  "meta", # Meta-analysis (alternative interface)
  "netmeta", # Network meta-analysis

  # Causal inference & mediation
  "WeightIt", # Propensity score weighting
  "MatchIt", # Propensity score matching
  "cobalt", # Covariate balance tables/plots
  "marginaleffects", # Marginal effects & contrasts
  "dagitty", # DAG analysis
  "EValue", # E-values for unmeasured confounding
  "regmedint", # Causal mediation with exposure-mediator interaction (CRAN)

  # Reporting
  "gtsummary", # Publication-quality Table 1
  "CMAverse", # Causal mediation analysis (installed from GitHub below)

  # Deep learning
  "keras3", # Keras interface to TensorFlow

  # Utilities
  "remotes" # Install packages from GitHub
)

GREEN <- "\033[32m"
RED <- "\033[31m"
BOLD <- "\033[1m"
DIM <- "\033[2m"
RESET <- "\033[0m"

github_pkgs <- list(
  CMAverse = "bs1125/CMAverse"
)

cran_pkgs <- pkgs[!pkgs %in% names(github_pkgs)]
to_install <- cran_pkgs[!sapply(cran_pkgs, requireNamespace, quietly = TRUE)]

if (length(to_install) > 0) {
  cat(sprintf(
    "\n%sInstalling %d CRAN package(s)…%s\n\n",
    BOLD,
    length(to_install),
    RESET
  ))
  rspm <- Sys.getenv("RSPM", unset = "")
  cran_repo <- if (nzchar(rspm)) rspm else "https://cloud.r-project.org"
  repos <- c(
    "https://bgreenwell.r-universe.dev",
    cran_repo
  )
  install.packages(to_install, repos = repos)
}

for (pkg in names(github_pkgs)) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("\n%sInstalling %s from GitHub…%s\n", BOLD, pkg, RESET))
    remotes::install_github(github_pkgs[[pkg]])
  }
}

cat(sprintf("\n%sChecking R packages…%s\n\n", BOLD, RESET))
cat(sprintf(
  "  R %s%s%s\n\n",
  DIM,
  paste0(R.version$major, ".", R.version$minor),
  RESET
))

failures <- character(0)
for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    failures <- c(failures, pkg)
    cat(sprintf("  %s✗%s  %s — not installed\n", RED, RESET, pkg))
    next
  }
  ver <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NULL)
  ver_label <- if (!is.null(ver)) sprintf(" %s%s%s", DIM, ver, RESET) else ""
  cat(sprintf("  %s✓%s  %s%s\n", GREEN, RESET, pkg, ver_label))
}

cat("\n")
if (length(failures) > 0) {
  cat(sprintf(
    "%s%s%d package(s) failed.%s\n",
    RED,
    BOLD,
    length(failures),
    RESET
  ))
  quit(status = 1)
} else {
  cat(sprintf(
    "%s%sAll %d packages OK.%s\n\n",
    GREEN,
    BOLD,
    length(pkgs),
    RESET
  ))
}
